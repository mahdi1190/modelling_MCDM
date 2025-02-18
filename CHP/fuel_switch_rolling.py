from pyomo.environ import *
import pandas as pd
import os
import numpy as np
from datetime import datetime
import sys
from fuel_switch_ccs_hourly import pyomomodel  # Your hourly model function
from solver_options import get_solver

# ------------------------------------------------------------------
# Fixed base-case name; when set to "base_case", the script will use the folder:
# results/base_case/
BASE_NAME = "base_case"

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct paths to input data files
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'demands.csv'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.csv'))

# Read the input CSV files (for hourly simulation, assume one year = 8760 hours)
demands = pd.read_csv(demands_path, nrows=8761)
markets = pd.read_csv(markets_path)

# Optional: Additional arrays needed for objective computation
shortfall_penalty = demands["penalty"].to_numpy()
reward = demands["revenue"].to_numpy()
request = demands["request"].to_numpy()
resin_per_tonne = 100  # Example value; set appropriately

# Hourly demand arrays
electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()

# Hourly model parameters
CHP_capacity = 4000
energy_ratio = 0.22
hours_per_year = 8760
time_limit = 90  # in seconds

# The monthly model’s variables are expected in the folder:
base_dir = os.path.join(os.path.dirname(__file__), "results", BASE_NAME)
if not os.path.exists(base_dir):
    raise Exception("Results folder for base_case not found. Please run the monthly model first.")

# ------------------------------------------------------------------
# Functions to read monthly variables and extract investment decisions
# ------------------------------------------------------------------

def read_monthly_variables():
    """Read the monthly model’s variables CSV file."""
    filepath = os.path.join(base_dir, f"{BASE_NAME}_monthly.csv")
    df = pd.read_csv(filepath)
    return df

def check_eb_active(year, total_months=360, no_intervals=30):
    """
    Determine if Electric Boiler is active for the given interval (year)
    by mapping the year to a month index and reading the 'active_eb' column.
    """
    df = read_monthly_variables()
    month_index = int(year * (total_months / no_intervals))
    if "active_eb" not in df.columns:
        print("Column 'active_eb' not found in monthly variables file.")
        return False
    try:
        decision = float(df.loc[month_index, "active_eb"])
    except Exception as e:
        print(f"Error reading 'active_eb' at month index {month_index}: {e}")
        return False
    return decision >= 0.5

def check_h2_active(year, total_months=360, no_intervals=30):
    """
    Determine if H2 Blending is active for the given interval (year)
    by reading the 'active_h2_blending' column.
    """
    df = read_monthly_variables()
    month_index = int(year * (total_months / no_intervals))
    if "active_h2_blending" not in df.columns:
        print("Column 'active_h2_blending' not found in monthly variables file.")
        return False
    try:
        decision = float(df.loc[month_index, "active_h2_blending"])
    except Exception as e:
        print(f"Error reading 'active_h2_blending' at month index {month_index}: {e}")
        return False
    return decision >= 0.5

def check_ccs_active(year, total_months=360, no_intervals=30):
    """
    Determine if CCS is active for the given interval (year)
    by reading the 'active_ccs' column.
    """
    df = read_monthly_variables()
    month_index = int(year * (total_months / no_intervals))
    if "active_ccs" not in df.columns:
        print("Column 'active_ccs' not found in monthly variables file.")
        return False
    try:
        decision = float(df.loc[month_index, "active_ccs"])
    except Exception as e:
        print(f"Error reading 'active_ccs' at month index {month_index}: {e}")
        return False
    return decision >= 0.5

def get_market_data(year):
    """Retrieve market data slices for the given year."""
    start_hour = year * hours_per_year
    end_hour = start_hour + hours_per_year
    start_month = year * 12
    end_month = start_month + 12
    yearly_data = {
        'NG_market': markets["Natural Gas Price ($/kWh)"].to_numpy()[start_hour:end_hour],
        'electricity_market': markets["Electricity Price ($/kWh)"].to_numpy()[start_hour:end_hour],
        'carbon_market': markets["Carbon Credit Price ($/tonne CO2)"].to_numpy()[start_month:end_month],
        'H2_market': markets["Hydrogen Price ($/kWh)"].to_numpy()[start_hour:end_hour],
        'BM_market': markets["Biomass Price ($/kWh)"].to_numpy()[start_hour:end_hour],
        'electricity_market_sold': markets["Electricity Price ($/kWh)"].to_numpy()[start_hour:end_hour] * 0.6,
        'heat_market_sold': markets["Natural Gas Price ($/kWh)"].to_numpy()[start_hour:end_hour] * 0.6,
    }
    return yearly_data

# ------------------------------------------------------------------
# Function to compute objective function components from the solved model
# ------------------------------------------------------------------

def compute_objective_components(model, market_data, interval_index):
    """
    Compute objective function components from the solved hourly model.
    For variables defined on INTERVALS (e.g. fuel_blend_ng), we use the value for the current simulation year (i.e. interval_index).
    NOTE: This function assumes that external arrays (reward, shortfall_penalty, resin_per_tonne) are defined.
    """
    HOURS = list(model.HOURS)
    # For variables that are defined on HOURS, use the hourly value.
    elec_cost = sum((value(model.purchased_electricity[h]) + value(model.heat_to_elec[h])) * market_data['electricity_market'][h] for h in HOURS)
    elec_sold = sum(value(model.electricity_over_production[h]) * market_data['electricity_market_sold'][h] for h in HOURS)
    heat_sold = sum(market_data['heat_market_sold'][h] * value(model.heat_over_production[h]) for h in HOURS)
    
    # For variables defined on INTERVALS, use the value for the current interval (i.e. the simulation year)
    fuel_cost_NG = sum(value(model.fuel_blend_ng[interval_index]) * market_data['NG_market'][h] * value(model.fuel_consumed[h]) for h in HOURS)
    fuel_cost_H2 = sum(value(model.fuel_blend_h2[interval_index]) * market_data['H2_market'][h] * value(model.fuel_consumed[h]) for h in HOURS)
    fuel_cost_BM = sum(value(model.fuel_blend_biomass[interval_index]) * market_data['BM_market'][h] * value(model.fuel_consumed[h]) for h in HOURS)
    
    # For interval-indexed variables used in objective
    INTERVALS = list(model.INTERVALS)
    carbon_cost = sum(value(model.carbon_credits[i]) * market_data['carbon_market'][i] for i in INTERVALS)
    carbon_sold = sum(value(model.credits_sold[i]) * market_data['carbon_market'][i] for i in INTERVALS)
    
    production_revenue = sum(value(model.production_output[h]) for h in HOURS) * resin_per_tonne
    ancillary_revenue = sum(reward[h] * value(model.elec_reduction[h]) for h in HOURS) * 1E6
    shortfall_penalty_total = sum(shortfall_penalty[h] * value(model.grid_reduction_shortfall[h]) for h in HOURS)
    
    transport_storage_cost = sum(value(model.transport_cost[i]) for i in INTERVALS)
    
    total_costs = (fuel_cost_NG + fuel_cost_H2 + fuel_cost_BM +
                   elec_cost + carbon_cost +
                   transport_storage_cost + shortfall_penalty_total)
    total_revenues = elec_sold + heat_sold + carbon_sold + ancillary_revenue + production_revenue
    
    return {
        "elec_cost": elec_cost,
        "elec_sold": elec_sold,
        "heat_sold": heat_sold,
        "fuel_cost_NG": fuel_cost_NG,
        "fuel_cost_H2": fuel_cost_H2,
        "fuel_cost_BM": fuel_cost_BM,
        "carbon_cost": carbon_cost,
        "carbon_sold": carbon_sold,
        "production_revenue": production_revenue,
        "ancillary_revenue": ancillary_revenue,
        "shortfall_penalty_total": shortfall_penalty_total,
        "transport_storage_cost": transport_storage_cost,
        "total_costs": total_costs,
        "total_revenues": total_revenues,
        "objective_value": total_costs - total_revenues
    }

# ------------------------------------------------------------------
# Functions for saving aggregated hourly model variables
# ------------------------------------------------------------------

def save_to_csv(data, file_path, header=True):
    """Save a DataFrame to a CSV file using append mode."""
    data.to_csv(file_path, mode='a', header=header, index=False)

def save_all_hourly_model_variables_aggregated(model, year, results_folder, interval_index):
    """
    Save every model variable from the hourly model into one aggregated CSV file.
    In the resulting CSV, each column corresponds to one model variable and each row is an hour.
    For variables indexed by HOURS, we use the hourly value.
    For variables indexed by INTERVALS, we use the value for the current simulation interval (i.e. interval_index).
    Scalar variables are replicated for every hour.
    Also, objective function components (computed from the solved model) are added as separate columns.
    """
    HOURS = list(model.HOURS)
    aggregated_data = {}
    aggregated_data["Year"] = [year] * len(HOURS)
    aggregated_data["Time"] = HOURS

    # Helper: if a variable is indexed by INTERVALS, use the current simulation interval.
    for var_component in model.component_objects(Var, active=True):
        var_name = var_component.name
        col_values = []
        if not var_component.is_indexed():
            scalar_val = value(var_component)
            col_values = [scalar_val] * len(HOURS)
        else:
            keys = list(var_component.keys())
            keys_set = set(keys)
            try:
                model_hours = set(model.HOURS)
            except:
                model_hours = None
            try:
                model_intervals = set(model.INTERVALS)
            except:
                model_intervals = None
            
            if model_hours is not None and keys_set == model_hours:
                for h in HOURS:
                    col_values.append(value(var_component[h]))
            elif model_intervals is not None and keys_set == model_intervals:
                # Instead of mapping each hour, use the value at the current interval (simulation year)
                for h in HOURS:
                    col_values.append(value(var_component[interval_index]))
            else:
                print(f"Skipping variable '{var_name}' due to unsupported indexing.")
                continue
        aggregated_data[var_name] = col_values

    # Compute objective function components and add them as columns.
    # We pass the current market data and interval_index.
    obj_components = compute_objective_components(model, get_market_data(year), interval_index)
    for comp_name, comp_value in obj_components.items():
        aggregated_data[comp_name] = [comp_value] * len(HOURS)
    
    df = pd.DataFrame(aggregated_data)
    filepath = os.path.join(results_folder, f"{BASE_NAME}_hourly.csv")
    header_flag = not os.path.exists(filepath)
    save_to_csv(df, filepath, header=header_flag)
    print(f"Saved aggregated hourly model variables for Year {year} to {filepath}")

# ------------------------------------------------------------------
# Rolling Horizon Simulation: Year-by-Year with Warm-Start
# ------------------------------------------------------------------

def create_run_directory():
    """Creates a unique directory for each rolling horizon run based on the current timestamp."""
    base_dir_run = os.path.join(os.path.dirname(__file__), "runs")
    if not os.path.exists(base_dir_run):
        os.makedirs(base_dir_run)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir_run, f'run_{timestamp}')
    os.makedirs(run_dir)
    return run_dir

def run_30_year_simulation():
    # We use the fixed base folder (results/base_case) for aggregated hourly outputs.
    results_folder = base_dir
    total_years = 30
    # In the monthly model, total_months=360 and no_intervals=30 so each interval covers 12 months.
    # In the rolling horizon, each simulated year corresponds to one interval.
    warm_start_values = {}
    
    for year in range(total_years):
        print(f"\nRunning hourly simulation for Year {year + 1}...")
        eb_allowed = check_eb_active(year, total_months=360, no_intervals=30)
        h2_allowed = check_h2_active(year, total_months=360, no_intervals=30)
        ccs_allowed = check_ccs_active(year, total_months=360, no_intervals=30)
        market_data = get_market_data(year)
        
        # Run the hourly model.
        model = pyomomodel(
            total_hours=hours_per_year,  # 8760 hours per year
            time_limit=time_limit,
            CHP_capacity=CHP_capacity,
            energy_ratio=energy_ratio,
            eb_allowed=eb_allowed,
            h2_allowed=h2_allowed,
            ccs_allowed=ccs_allowed,
            market_data=market_data,
            warm_start_values=warm_start_values
        )
        
        # For variables indexed on INTERVALS, we use the current simulation year (which corresponds to the interval).
        interval_index = year  # (Assumes simulation year is in the domain of model.INTERVALS)
        
        # Save every model variable (and objective components) into one aggregated CSV.
        save_all_hourly_model_variables_aggregated(model, year, results_folder, interval_index)
        
        # Optionally update warm_start_values for the next year.
        warm_start_values = {
            'electricity_production': {h: model.electricity_production[h].value for h in model.HOURS},
            # ... update additional warm-start variables as needed ...
        }
    
    print("\nRolling horizon simulation complete.")

if __name__ == '__main__':
    run_30_year_simulation()
