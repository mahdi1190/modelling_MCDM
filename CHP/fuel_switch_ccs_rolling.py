#!/usr/bin/env python
import pandas as pd
import os
import numpy as np
from datetime import datetime
import sys
from fuel_switch_ccs_hourly import pyomomodel  # Your model function; must now accept demand_data and market_data
from solver_options import get_solver
from pyomo.environ import value, Var, Expression  # Needed for saving results

# --------------------------
# Global Parameters
# --------------------------
BASE_NAME = "B"         # Used for naming results folder/files
hours_per_year = 8760           # Number of hours per simulation year
total_years = 30                # Total simulation years (the CSV files should have total_years*8760 rows)
time_limit = 90                 # Time limit for solver (seconds)
CHP_capacity = 15            # Example CHP capacity value
energy_ratio = 0.25             # Example energy ratio

# --------------------------
# Helper Functions: Data Ingestion
# --------------------------
def get_yearly_demand_data(year, demands_df, hours_per_year=8760):
    """
    Slices the full demand DataFrame for the specified simulation year.
    """
    year = 0
    start = year * hours_per_year
    end = (year + 1) * hours_per_year
    df_year = demands_df.iloc[start:end]
    if df_year.empty:
        raise ValueError(f"No demand data found for year {year}.")
    return {
        "penalty": df_year["penalty"].to_numpy(),
        "revenue": df_year["revenue"].to_numpy(),
        "request": df_year["request"].to_numpy(),
        "elec": df_year["elec"].to_numpy(),
        "heat": df_year["heat"].to_numpy(),
        "cool": df_year["cool"].to_numpy()
    }

def get_yearly_market_data(year, markets_df, hours_per_year=8760):
    """
    Slices the full market DataFrame for the specified simulation year.
    """
    start = year * hours_per_year
    end = (year + 1) * hours_per_year
    df_year = markets_df.iloc[start:end]
    if df_year.empty:
        raise ValueError(f"No market data found for year {year}.")
    unit_conv = 1E3  # Conversion factor used in your model 
    return {
        "electricity_market": df_year["Electricity Price ($/kWh)"].to_numpy() * unit_conv,
        "electricity_market_sold": df_year["Electricity Price ($/kWh)"].to_numpy(),
        "carbon_market": df_year["Carbon Credit Price ($/tonne CO2)"].to_numpy(),
        "NG_market": df_year["Natural Gas Price ($/kWh)"].to_numpy() * unit_conv,
        "heat_market_sold": df_year["Natural Gas Price ($/kWh)"].to_numpy(),
        "H2_market": df_year["Hydrogen Price ($/kWh)"].to_numpy() * unit_conv,
        "BM_market": df_year["Biomass Price ($/kWh)"].to_numpy() * unit_conv,
        "em_bm": df_year["Biomass Carbon Intensity (kg CO2/kWh)"].to_numpy(),
        "em_h2": df_year["Hydrogen Carbon Intensity (kg CO2/kWh)"].to_numpy(),
        "em_ng": df_year["Natural Gas Carbon Intensity (kg CO2/kWh)"].to_numpy(),
        "em_elec": df_year["Grid Carbon Intensity (kg CO2/kWh)"].to_numpy(),
        "max_co2_emissions": df_year["Effective Carbon Credit Cap"].to_numpy() / 12,
        "margin": df_year["Input Margin ($/tonne) PV"].to_numpy(),
        "labour": df_year["Fixed Cost ($/tonne)"].to_numpy(),
    }

# --------------------------
# Helper Functions: Monthly Variables & Investment Flags
# --------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "results", BASE_NAME)

def read_monthly_variables():
    """Read the monthly model’s variables CSV file."""
    filepath = os.path.join(BASE_DIR, f"{BASE_NAME}_monthly.csv")
    df = pd.read_csv(filepath)
    return df

def check_eb_active(year, total_months=360, no_intervals=30):
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

# --------------------------
# Helper Functions: Unique Filename Generator & Saving Model Results
# --------------------------
def get_unique_filepath(folder, base_name, extension=".csv"):
    """Return a unique filepath in folder. If base_name+extension exists, append an underscore and counter."""
    base_filepath = os.path.join(folder, f"{base_name}{extension}")
    if not os.path.exists(base_filepath):
        return base_filepath
    counter = 2
    while True:
        new_filepath = os.path.join(folder, f"{base_name}_{counter}{extension}")
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1

def save_to_csv(data, file_path, header=True):
    """Append a DataFrame to a CSV file."""
    data.to_csv(file_path, mode='a', header=header, index=False)

def save_all_hourly_model_variables_aggregated(model, year, output_filepath, hours_per_year=8760):
    HOURS = list(model.HOURS)
    aggregated_data = {"Year": [year] * len(HOURS), "Time": HOURS}

    def hour_to_interval(h):
        return int(h // (hours_per_year / len(model.INTERVALS)))
    
    # Process all Var components
    for var_component in model.component_objects(Var, active=True):
        var_name = var_component.name
        col_values = []
        try:
            # If the variable is indexed, then use its keys.
            if var_component.is_indexed():
                if set(var_component.keys()) == set(model.HOURS):
                    for h in HOURS:
                        col_values.append(value(var_component[h]))
                elif set(var_component.keys()) == set(model.INTERVALS):
                    for h in HOURS:
                        idx = hour_to_interval(h)
                        col_values.append(value(var_component[idx]))
                else:
                    print(f"Skipping variable '{var_name}' due to unsupported indexing.")
                    continue
            else:
                # Scalar variable: replicate its value across all hours.
                scalar_val = value(var_component)
                col_values = [scalar_val] * len(HOURS)
        except Exception as e:
            print(f"Error processing variable '{var_name}': {e}")
            continue
        aggregated_data[var_name] = col_values

    # Process all Expression components
    for expr_component in model.component_objects(Expression, active=True):
        expr_name = expr_component.name
        col_values = []
        try:
            # Check if the expression is indexed.
            if expr_component.is_indexed():
                if set(expr_component.keys()) == set(model.HOURS):
                    for h in HOURS:
                        col_values.append(value(expr_component[h]))
                elif set(expr_component.keys()) == set(model.INTERVALS):
                    for h in HOURS:
                        idx = hour_to_interval(h)
                        col_values.append(value(expr_component[idx]))
                else:
                    print(f"Skipping expression '{expr_name}' due to unsupported indexing.")
                    continue
            else:
                # Scalar expression: replicate its value across all hours.
                scalar_val = value(expr_component)
                col_values = [scalar_val] * len(HOURS)
        except Exception as e:
            print(f"Error processing expression '{expr_name}': {e}")
            continue
        aggregated_data[expr_name] = col_values

    df_out = pd.DataFrame(aggregated_data)
    header_flag = not os.path.exists(output_filepath)
    save_to_csv(df_out, output_filepath, header=header_flag)
    print(f"Saved aggregated hourly model variables for Year {year} to {output_filepath}")


def create_run_directory():
    """Creates a unique directory for each rolling horizon run."""
    base_dir_run = os.path.join(os.path.dirname(__file__), "runs")
    if not os.path.exists(base_dir_run):
        os.makedirs(base_dir_run)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir_run, f'run_{timestamp}')
    os.makedirs(run_dir)
    return run_dir

# --------------------------
# Rolling Horizon Simulation
# --------------------------
def run_30_year_simulation():
    results_folder = os.path.join(os.path.dirname(__file__), "results", BASE_NAME)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Generate one unique output file for this entire simulation run.
    output_filepath = get_unique_filepath(results_folder, f"{BASE_NAME}_hourly")
    
    # Read the full CSV files once at the start.
    current_dir = os.path.dirname(__file__)
    demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'demands_hourly_year.csv'))
    markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.csv'))
    
    print("Reading full demand data...")
    full_demand_df = pd.read_csv(demands_path)
    print("Reading full market data...")
    full_market_df = pd.read_csv(markets_path)
    
    warm_start_values = {}
    
    for year in range(total_years):
        print(f"\nRunning hourly simulation for Year {year+1}...")
        try:
            # Determine investment flags using monthly variables
            eb_allowed = check_eb_active(year, total_months=360, no_intervals=30)
            h2_allowed = check_h2_active(year, total_months=360, no_intervals=30)
            ccs_allowed = check_ccs_active(year, total_months=360, no_intervals=30)
            print(f"Year {year}: Investment flags - EB: {eb_allowed}, H2: {h2_allowed}, CCS: {ccs_allowed}")
            
            # Slice the full DataFrames for the current simulation year.
            demand_data = get_yearly_demand_data(year, full_demand_df, hours_per_year)
            market_data = get_yearly_market_data(year, full_market_df, hours_per_year)
        except Exception as e:
            print(f"Error reading data for year {year}: {e}")
            print("Ending simulation.")
            break
        
        # Run the Pyomo model for the current year using the sliced data.
        model = pyomomodel(
            total_hours=hours_per_year,
            time_limit=time_limit,
            CHP_capacity=CHP_capacity,
            energy_ratio=energy_ratio,
            eb_allowed=int(eb_allowed),
            h2_allowed=int(h2_allowed),
            ccs_allowed=int(ccs_allowed),
            demand_data=demand_data,
            market_data=market_data,
            warm_start_values=warm_start_values
        )
        
        # Save the aggregated hourly model variables into the same output file.
        save_all_hourly_model_variables_aggregated(model, year, output_filepath)
        
        # Update warm_start_values for warm starting the next simulation year.
        warm_start_values = {
            'electricity_production': {h: model.electricity_production[h].value for h in model.HOURS},
            # Add additional warm-start variables here if needed.
        }
    
    print("\nRolling horizon simulation complete.")
    print(f"All results saved in: {output_filepath}")

# --------------------------
# Main Execution
# --------------------------
if __name__ == '__main__':
    run_30_year_simulation()
