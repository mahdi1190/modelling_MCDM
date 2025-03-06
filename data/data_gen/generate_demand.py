#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Read configuration from JSON file.
# It is assumed that config_demand.json exists in the same directory as this script.
config_path = os.path.join(os.path.dirname(__file__), 'config_demand.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# For one-year simulation, use the start and end dates from the config.
one_year_start = pd.to_datetime(config["simulation"]["hourly"]["start_date"])
one_year_end = pd.to_datetime(config["simulation"]["hourly"]["end_date"])
one_year_hours = pd.date_range(start=one_year_start, end=one_year_end, freq='H')

# For multi-year simulation, we want 30 years plus 1 extra month.
years = config["simulation"]["years"]
multi_year_start = one_year_start
# End date: 30 years + 1 month after start_date, then subtract one hour.
multi_year_end = one_year_start + pd.DateOffset(years=years, months=1) - pd.DateOffset(hours=1)
multi_year_hours = pd.date_range(start=multi_year_start, end=multi_year_end, freq='H')


# Define functions for seasonal and daily cycles.
def seasonal_factor(timestamp, amplitude, phase_shift):
    """Return a seasonal multiplier based on the day of the year."""
    day_of_year = timestamp.timetuple().tm_yday
    return 1 + amplitude * np.sin(2 * np.pi * (day_of_year / 365.0) + phase_shift)

def daily_cycle(hour, amplitude, peak_hour):
    """Return a daily multiplier based on the hour."""
    return 1 + amplitude * np.cos(2 * np.pi * ((hour - peak_hour) / 24.0))

def add_downtime(timestamps, maintenance, emergency):
    """
    Generate a boolean array for downtime.
    Scheduled maintenance is added first, then random emergency downtime events.
    """
    downtime_flag = np.zeros(len(timestamps), dtype=bool)
    years_in_data = np.unique([ts.year for ts in timestamps])
    
    # Scheduled maintenance downtime for each year.
    for yr in years_in_data:
        maint_start = datetime(yr, 1, 1) + timedelta(days=maintenance["start_day_of_year"] - 1)
        maint_end = maint_start + timedelta(hours=maintenance["duration_hours"])
        downtime_flag |= ((timestamps >= maint_start) & (timestamps < maint_end))
        
    # Emergency downtime events for each year.
    for yr in years_in_data:
        year_timestamps = [ts for ts in timestamps if ts.year == yr]
        num_events = emergency["number_of_events"]
        for _ in range(num_events):
            rand_hour = random.randint(0, len(year_timestamps) - 1)
            rand_start = year_timestamps[0] + timedelta(hours=rand_hour)
            duration = random.randint(emergency["min_duration_hours"], emergency["max_duration_hours"])
            rand_end = rand_start + timedelta(hours=duration)
            downtime_flag |= ((timestamps >= rand_start) & (timestamps < rand_end))
    return downtime_flag

# --- NEW: Helper to generate a common grid event trigger ---
def grid_event_trigger(grid_cfg):
    """Return True if a grid event is triggered based on the trigger probability."""
    return random.random() < grid_cfg.get("trigger_probability", 0.1)

# --- Modified grid functions accepting scaling_factor and event_trigger ---
def grid_reduction_request(ts, grid_cfg, baseline_load, scaling_factor, event_trigger):
    """
    Compute the grid reduction request (in MWh) for a given timestamp.
    If an event is triggered, the plant reduces a fraction (random between min and max) of its
    (scaled) baseline load, with daily, seasonal multipliers and noise.
    """
    if event_trigger:
        amplitude = grid_cfg["daily_cycle"]["amplitude"]
        peak_hour = grid_cfg["daily_cycle"]["peak_hour"]
        daily = 1 + amplitude * np.cos(2 * np.pi * ((ts.hour - peak_hour) / 24.0))
        seasonal = grid_cfg["winter_multiplier"] if ts.month in [12, 1, 2] else 1.0
        noise = np.random.normal(0, grid_cfg["noise_std"])
        ramp_down_pct = random.uniform(grid_cfg.get("min_ramp_down_percentage", 0.2),
                                       grid_cfg.get("max_ramp_down_percentage", 0.5))
        adjusted_load = baseline_load * scaling_factor
        potential_reduction = ramp_down_pct * adjusted_load * daily * seasonal + noise
        reduction = max(0, potential_reduction)
        return reduction if reduction >= 0.1 else 0
    else:
        return 0

def grid_reduction_price(ts, grid_cfg, scaling_factor, event_trigger):
    """
    Compute the grid reduction price (currency per kWh) for a given timestamp.
    Uses the same event trigger for consistency.
    """
    if event_trigger:
        amplitude = grid_cfg["daily_cycle"]["amplitude"]
        peak_hour = grid_cfg["daily_cycle"]["peak_hour"]
        daily = 1 + amplitude * np.cos(2 * np.pi * ((ts.hour - peak_hour) / 24.0))
        seasonal = grid_cfg["winter_multiplier"] if ts.month in [12, 1, 2] else 1.0
        noise = np.random.normal(0, grid_cfg["price_noise_std"])
        price = scaling_factor * grid_cfg["base_reduction_price"] * daily * seasonal + noise
        return max(0, price) if price >= 0.001 else 0
    else:
        return 0

def grid_penalty_rate(ts, grid_cfg, scaling_factor, event_trigger):
    """
    Compute the grid penalty rate (currency per kWh) for a given timestamp.
    Uses the same event trigger.
    """
    if event_trigger:
        amplitude = grid_cfg["daily_cycle"]["amplitude"]
        peak_hour = grid_cfg["daily_cycle"]["peak_hour"]
        daily = 1 + amplitude * np.cos(2 * np.pi * ((ts.hour - peak_hour) / 24.0))
        seasonal = grid_cfg["winter_multiplier"] if ts.month in [12, 1, 2] else 1.0
        noise = np.random.normal(0, grid_cfg["penalty_noise_std"])
        penalty = scaling_factor * grid_cfg["base_penalty_rate"] * daily * seasonal + noise
        return max(0, penalty) if penalty >= 0.001 else 0
    else:
        return 0

# --- Updated generate_hourly_data using the common event trigger and scaling factor ---
def generate_hourly_data(timestamps, config, eb_allowed=False):
    """
    Generate an hourly dataset based on configuration.
    Returns a DataFrame with columns: heat, elec, cool, request, revenue, penalty, downtime.
    """
    thermal_cfg = config["demands"]["thermal"]
    elec_cfg = config["demands"]["electricity"]
    refrig_cfg = config["demands"]["refrigeration"]
    grid_cfg = config["grid_reduction"]
    maintenance_cfg = config["downtime"]["maintenance"]
    emergency_cfg = config["downtime"]["emergency"]
    
    df = pd.DataFrame(index=timestamps)
    
    # Thermal demand (MWh/h)
    df['heat'] = [
        thermal_cfg["average_MWh_per_hour"] * seasonal_factor(ts, thermal_cfg["seasonality"]["amplitude"],
                                                              thermal_cfg["seasonality"]["phase_shift"]) +
        np.random.normal(0, thermal_cfg["noise_std"])
        for ts in timestamps
    ]
    
    # Electricity demand (MWh/h)
    df['elec'] = [
        elec_cfg["average_MWh_per_hour"] * daily_cycle(ts.hour, elec_cfg["daily_cycle"]["amplitude"],
                                                        elec_cfg["daily_cycle"]["peak_hour"]) +
        np.random.normal(0, elec_cfg["noise_std"])
        for ts in timestamps
    ]
    
    # Refrigeration demand (MWh/h)
    df['cool'] = [
        refrig_cfg["average_MWh_per_hour"] * daily_cycle(ts.hour, refrig_cfg["daily_cycle"]["amplitude"],
                                                          refrig_cfg["daily_cycle"]["peak_hour"]) +
        np.random.normal(0, refrig_cfg["noise_std"])
        for ts in timestamps
    ]
    
    # Baseline load from electricity config.
    baseline_load = elec_cfg.get("average_MWh_per_hour", 4.0)
    # When eb_allowed is True, assume electrification increases the purchased load.
    scaling_factor = 3.0 if eb_allowed else 1.0
    
    # Generate a common event trigger for each timestamp.
    event_triggers = [grid_event_trigger(grid_cfg) for _ in timestamps]
    
    # Use the same event trigger for request, revenue, and penalty.
    df['request'] = [
        grid_reduction_request(ts, grid_cfg, baseline_load, scaling_factor, event_triggers[i])
        for i, ts in enumerate(timestamps)
    ]
    df['revenue'] = [
        grid_reduction_price(ts, grid_cfg, scaling_factor, event_triggers[i])
        for i, ts in enumerate(timestamps)
    ]
    df['penalty'] = [
        grid_penalty_rate(ts, grid_cfg, scaling_factor, event_triggers[i])
        for i, ts in enumerate(timestamps)
    ]
    
    # Downtime flags.
    downtime = add_downtime(timestamps, maintenance_cfg, emergency_cfg)
    df['downtime'] = downtime
    
    # During downtime: reduce demands to baseline values.
    df.loc[downtime, 'heat'] *= thermal_cfg.get("downtime_baseline", 0)
    df.loc[downtime, 'elec'] *= elec_cfg.get("downtime_baseline", 0)
    df.loc[downtime, 'cool'] *= refrig_cfg.get("downtime_baseline", 0)
    
    return df

def ensure_dir(directory):
    """Ensure that the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    # Ensure output directory exists.
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    ensure_dir(output_dir)
    
    # Generate one-year hourly dataset (8760 hours).
    one_year_data = generate_hourly_data(one_year_hours, config, eb_allowed=False)
    one_year_data.index.name = 'timestamp'
    one_year_output_path = os.path.join(output_dir, "demands_hourly_year.csv")
    one_year_data.to_csv(one_year_output_path)
    print(f"One-year hourly demand dataset saved as '{one_year_output_path}'.")
    
    # Generate multi-year hourly dataset (30 years + 1 month) with eb_allowed flag.
    # Change eb_allowed to True if the plant has electrified.
    multi_year_data = generate_hourly_data(multi_year_hours, config, eb_allowed=True)
    multi_year_data.index.name = 'timestamp'
    multi_year_output_path = os.path.join(output_dir, "demands.csv")
    multi_year_data.to_csv(multi_year_output_path)
    print(f"30-year hourly demand dataset (with extra month) saved as '{multi_year_output_path}'.")
    
    # Generate monthly aggregated dataset for the multi-year dataset.
    monthly_index = pd.date_range(start="2025-01-01", periods=361, freq='MS')
    monthly_data = multi_year_data.resample('MS').sum().reindex(monthly_index)
    monthly_data.index.name = 'month'
    monthly_output_path = os.path.join(output_dir, "demands_monthly_30.csv")
    monthly_data.to_csv(monthly_output_path)
    print(f"Monthly demand dataset saved as '{monthly_output_path}'.")
