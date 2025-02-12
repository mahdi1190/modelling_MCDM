import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Read configuration from JSON file
with open('config_demand.json', 'r') as f:
    config = json.load(f)

# For one-year simulation, use the start and end dates from the config.
one_year_start = pd.to_datetime(config["simulation"]["hourly"]["start_date"])
one_year_end = pd.to_datetime(config["simulation"]["hourly"]["end_date"])
one_year_hours = pd.date_range(start=one_year_start, end=one_year_end, freq='H')

# For multi-year simulation, we want 30 years plus 1 extra month.
years = config["simulation"]["years"]
multi_year_start = one_year_start
# End date: 30 years + 1 month after start_date, then subtract one hour to complete the series.
multi_year_end = one_year_start + pd.DateOffset(years=years, months=1) - pd.DateOffset(hours=1)
multi_year_hours = pd.date_range(start=multi_year_start, end=multi_year_end, freq='H')

# Define functions for seasonal and daily cycles.
def seasonal_factor(timestamp, amplitude, phase_shift):
    day_of_year = timestamp.timetuple().tm_yday
    return 1 + amplitude * np.sin(2 * np.pi * (day_of_year / 365.0) + phase_shift)

def daily_cycle(hour, amplitude, peak_hour):
    return 1 + amplitude * np.cos(2 * np.pi * ((hour - peak_hour) / 24.0))

def add_downtime(timestamps, maintenance, emergency):
    downtime_flag = np.zeros(len(timestamps), dtype=bool)
    # For each year, add scheduled maintenance downtime.
    for yr in np.unique([ts.year for ts in timestamps]):
        maint_start = datetime(yr, 1, 1) + timedelta(days=maintenance["start_day_of_year"] - 1)
        maint_end = maint_start + timedelta(hours=maintenance["duration_hours"])
        downtime_flag |= ((timestamps >= maint_start) & (timestamps < maint_end))
    # For each year, add emergency downtime events.
    for yr in np.unique([ts.year for ts in timestamps]):
        year_timestamps = [ts for ts in timestamps if ts.year == yr]
        num_events = emergency["number_of_events"]
        for _ in range(num_events):
            rand_start = year_timestamps[0] + timedelta(hours=random.randint(0, len(year_timestamps)-1))
            duration = random.randint(emergency["min_duration_hours"], emergency["max_duration_hours"])
            rand_end = rand_start + timedelta(hours=duration)
            downtime_flag |= ((timestamps >= rand_start) & (timestamps < rand_end))
    return downtime_flag

def grid_reduction_request(ts, elec_avg, grid_cfg):
    # Only trigger a reduction request with a given probability.
    if random.random() < grid_cfg.get("trigger_probability", 0.1):
        amplitude = grid_cfg["daily_cycle"]["amplitude"]
        peak_hour = grid_cfg["daily_cycle"]["peak_hour"]
        daily = 1 + amplitude * np.cos(2 * np.pi * ((ts.hour - peak_hour) / 24.0))
        seasonal = grid_cfg["winter_multiplier"] if ts.month in [12, 1, 2] else 1.0
        noise = np.random.normal(0, grid_cfg["noise_std"])
        base_request = grid_cfg["base_request_MWh"]
        reduction = base_request * daily * seasonal + noise
        reduction = max(0, reduction)
        return reduction if reduction >= 0.1 else 0
    else:
        return 0

def grid_reduction_price(ts, grid_cfg):
    # Only provide a price if a reduction request is triggered (same probability)
    if random.random() < grid_cfg.get("trigger_probability", 0.1):
        amplitude = grid_cfg["daily_cycle"]["amplitude"]
        peak_hour = grid_cfg["daily_cycle"]["peak_hour"]
        daily = 1 + amplitude * np.cos(2 * np.pi * ((ts.hour - peak_hour) / 24.0))
        seasonal = grid_cfg["winter_multiplier"] if ts.month in [12, 1, 2] else 1.0
        noise = np.random.normal(0, grid_cfg["price_noise_std"])
        price = grid_cfg["base_reduction_price"] * daily * seasonal + noise
        return max(0, price) if price >= 0.001 else 0
    else:
        return 0

def grid_penalty_rate(ts, grid_cfg):
    # Only provide a penalty if a reduction request is triggered.
    if random.random() < grid_cfg.get("trigger_probability", 0.1):
        amplitude = grid_cfg["daily_cycle"]["amplitude"]
        peak_hour = grid_cfg["daily_cycle"]["peak_hour"]
        daily = 1 + amplitude * np.cos(2 * np.pi * ((ts.hour - peak_hour) / 24.0))
        seasonal = grid_cfg["winter_multiplier"] if ts.month in [12, 1, 2] else 1.0
        noise = np.random.normal(0, grid_cfg["penalty_noise_std"])
        penalty = grid_cfg["base_penalty_rate"] * daily * seasonal + noise
        return max(0, penalty) if penalty >= 0.001 else 0
    else:
        return 0

# Generate hourly dataset for a given set of timestamps.
def generate_hourly_data(timestamps, config):
    thermal_cfg = config["demands"]["thermal"]
    elec_cfg = config["demands"]["electricity"]
    refrig_cfg = config["demands"]["refrigeration"]
    grid_cfg = config["grid_reduction"]
    maintenance_cfg = config["downtime"]["maintenance"]
    emergency_cfg = config["downtime"]["emergency"]
    
    df = pd.DataFrame(index=timestamps)
    
    # Thermal demand (MWh/h)
    df['heat'] = [
        thermal_cfg["average_MWh_per_hour"] * seasonal_factor(ts, thermal_cfg["seasonality"]["amplitude"], thermal_cfg["seasonality"]["phase_shift"]) +
        np.random.normal(0, thermal_cfg["noise_std"])
        for ts in timestamps
    ]
    
    # Electricity demand (MWh/h)
    df['elec'] = [
        elec_cfg["average_MWh_per_hour"] * daily_cycle(ts.hour, elec_cfg["daily_cycle"]["amplitude"], elec_cfg["daily_cycle"]["peak_hour"]) +
        np.random.normal(0, elec_cfg["noise_std"])
        for ts in timestamps
    ]
    
    # Refrigeration demand (MWh/h)
    df['cool'] = [
        refrig_cfg["average_MWh_per_hour"] * daily_cycle(ts.hour, refrig_cfg["daily_cycle"]["amplitude"], refrig_cfg["daily_cycle"]["peak_hour"]) +
        np.random.normal(0, refrig_cfg["noise_std"])
        for ts in timestamps
    ]
    
    # Grid reduction request (MWh/h)
    df['request'] = [
        grid_reduction_request(ts, elec_cfg["average_MWh_per_hour"], grid_cfg)
        for ts in timestamps
    ]
    
    # Grid reduction price (currency per kWh) and penalty rate (currency per kWh)
    df['revenue'] = [
        grid_reduction_price(ts, grid_cfg)
        for ts in timestamps
    ]
    df['penalty'] = [
        grid_penalty_rate(ts, grid_cfg)
        for ts in timestamps
    ]
    
    # Add downtime flags.
    downtime = add_downtime(timestamps, maintenance_cfg, emergency_cfg)
    df['downtime'] = downtime
    
    # During downtime: thermal demand becomes 0; electricity and refrigeration drop to a baseline fraction.
    df.loc[downtime, 'heat'] *= thermal_cfg.get("downtime_baseline", 0)
    df.loc[downtime, 'elec'] *= elec_cfg.get("downtime_baseline", 0)
    df.loc[downtime, 'cool'] *= refrig_cfg.get("downtime_baseline", 0)
    
    return df

# Generate one-year hourly dataset (8760 hours)
one_year_data = generate_hourly_data(one_year_hours, config)
one_year_data.index.name = 'timestamp'
one_year_data.to_csv("data/demands_hourly_year.csv")
print("One-year hourly demand dataset saved as 'data/demands_hourly_year.csv'.")

# Generate multi-year hourly dataset (30 years + 1 month)
multi_year_data = generate_hourly_data(multi_year_hours, config)
multi_year_data.index.name = 'timestamp'
multi_year_data.to_csv("data/demands.csv")
print("30-year hourly demand dataset (with extra month) saved as 'data/demands.csv'.")

# Generate monthly aggregated dataset for the multi-year dataset.
# Sum all hourly values in each month.
monthly_index = pd.date_range(start="2025-01-01", periods=361, freq='MS')
monthly_data = multi_year_data.resample('MS').sum().reindex(monthly_index)
monthly_data.index.name = 'month'
monthly_data.to_csv("data/demands_monthly_30.csv")
print("Monthly demand dataset saved as 'demands_monthly_30.csv'.")
