from pyomo.environ import *
import pandas as pd
import dash
import locale
locale.setlocale(locale.LC_ALL, '')
import os
import numpy as np
from datetime import datetime

import sys
# Add the config directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
# Import the solver options
from solver_options import get_solver
global markets, markets_monthly

# Initialize Dash app
last_mod_time = 0

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct paths to the data files by correctly moving up one directory to 'modelling_MCDM'
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'demands.csv'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.csv'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.xlsx'))  # Corrected path
load_shedding_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'load_shedding.xlsx'))  # Corrected path

# Directory paths
investment_log_dir = os.path.abspath(os.path.join(current_dir, '..', 'investment_log', "eb_investment.csv"))
investment_data = pd.read_csv(investment_log_dir)

investment_log_dir2 = os.path.abspath(os.path.join(current_dir, '..', 'investment_log', "h2_investment.csv"))
investment_data2 = pd.read_csv(investment_log_dir2)


# Read the Excel files
demands = pd.read_excel(demands_path, nrows=10000)
markets = pd.read_excel(markets_path)
load_shedding = pd.read_excel(load_shedding_path, nrows=10000)
#Jack Gower is not great
shortfall_penalty = load_shedding["penalty"].to_numpy()
reward = load_shedding["revenue"].to_numpy()
request = load_shedding["request"].to_numpy()
electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()
CHP_capacity = 4000

energy_ratio = 0.22

total_hours = 8400
time_limit = 90
# Create a simple model

def check_eb_active(year):
    investment_row = investment_data[investment_data['Interval'] == year]
    if not investment_row.empty:
        if investment_row.iloc[0]['Active EB'] == 1.0:
            print("EB Active")
            return True
    print("EB Inactive")
    return False

def check_h2_active(year):
    investment_row2 = investment_data2[investment_data2['Interval'] == year]
    if not investment_row2.empty:
        if investment_row2.iloc[0]['Active H2 Blending'] == 1.0:
            print("Active H2 Blending")
            return True
    print("H2 Inactive")
    return False

# Function to get market data for the corresponding year
def get_market_data(year):
    # Adjust the market data indices for the given year
    yearly_data = {
        'NG_market': markets["nat_gas"].to_numpy()[(year * 8760):(year * 8760) + 8760],
        'electricity_market': markets["elec"].to_numpy()[(year * 8760):(year * 8760) + 8760],
        'carbon_market': markets["carbon"].to_numpy()[(year * 12):(year * 12) + 12],
        'H2_market': markets["hydrogen"].to_numpy()[(year * 8760):(year * 8760) + 8760],
        'BM_market': markets["biomass"].to_numpy()[(year * 8760):(year * 8760) + 8760],
        'electricity_market_sold': markets["elec_sold"].to_numpy()[(year * 8760):(year * 8760) + 8760],
        'heat_market_sold': markets["nat_gas_sold"].to_numpy()[(year * 8760):(year * 8760) + 8760],
    }
    return yearly_data

# Initialize arrays to store results
total_costs = np.zeros(30)  # Array to store total costs for each year
demand_profiles = []  # List to store demand profiles as dictionaries of arrays

# Run the model over 30 years, one year at a time
# Dictionary to store warm start values
warm_start_values = {}
def create_run_directory():
    """Creates a unique directory for each run based on the current timestamp."""
    base_dir = 'runs'
    # Ensure the main directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create a unique subfolder with a timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(run_dir)
    return run_dir

def save_to_csv(data, file_path, header=True):
    """Save data to a CSV file with append mode."""
    if os.path.exists(file_path):
        header = False  # Prevent writing headers if file already exists
    data.to_csv(file_path, mode='a', header=header, index=False)

def save_variables(model, run_dir, year):
    """Save all variables into CSV files in the given run directory."""
    # Save time-indexed variables (HOURS)
    hours_data = pd.DataFrame({
        'Year': [year] * len(model.HOURS),
        'Hour': model.HOURS,
        'ElectricityProduction': [model.electricity_production[h].value for h in model.HOURS],
        'HeatProduction': [model.heat_production[h].value for h in model.HOURS],
        'PurchasedElectricity': [model.purchased_electricity[h].value for h in model.HOURS],
        'HeatOverProduction': [model.heat_over_production[h].value for h in model.HOURS],
        'ElectricityOverProduction': [model.electricity_over_production[h].value for h in model.HOURS],
        'CO2Emissions': [model.co2_emissions[h].value for h in model.HOURS],
        # Add more time-indexed variables as needed
    })
    save_to_csv(hours_data, os.path.join(run_dir, 'time_indexed_results.csv'))

    # Save interval-indexed variables (INTERVALS)
    interval_data = pd.DataFrame({
        'Year': [year] * len(model.INTERVALS),
        'Interval': model.INTERVALS,
        'FuelBlendNG': [model.fuel_blend_ng[i].value for i in model.INTERVALS],
        'FuelBlendH2': [model.fuel_blend_h2[i].value for i in model.INTERVALS],
        'FuelBlendBiomass': [model.fuel_blend_biomass[i].value for i in model.INTERVALS],
        'CarbonCredits': [model.carbon_credits[i].value for i in model.INTERVALS],
        'CreditsPurchased': [model.credits_purchased[i].value for i in model.INTERVALS],
        'CreditsEarned': [model.credits_earned[i].value for i in model.INTERVALS],
        'CreditsSold': [model.credits_sold[i].value for i in model.INTERVALS],
        'CreditsHeld': [model.credits_held[i].value for i in model.INTERVALS],
        'CreditsUsedToOffset': [model.credits_used_to_offset[i].value for i in model.INTERVALS],
        # Add more interval-indexed variables as needed
    })

    save_to_csv(interval_data, os.path.join(run_dir, 'interval_indexed_results.csv'))

    # Save yearly summary data
    yearly_data = pd.DataFrame({
        'Year': [year],
        'TotalCost': [model.objective()],
        # Add other summary statistics if needed
    })
    save_to_csv(yearly_data, os.path.join(run_dir, 'yearly_summary.csv'))
import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_run_directory():
    """Creates a unique directory for each run based on the current timestamp."""
    base_dir = 'runs'
    # Ensure the main directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create a unique subfolder with a timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(run_dir)
    return run_dir

def save_to_csv(data, file_path, header=True):
    """Save data to a CSV file with append mode."""
    if os.path.exists(file_path):
        header = False  # Prevent writing headers if file already exists
    data.to_csv(file_path, mode='a', header=header, index=False)

def save_variables(model, run_dir, year):
    """Save all variables into CSV files in the given run directory."""
    # Save time-indexed variables (HOURS)
    hours_data = pd.DataFrame({
        'Year': [year] * len(model.HOURS),
        'Hour': model.HOURS,
        'ElectricityProduction': [model.electricity_production[h].value for h in model.HOURS],
        'HeatProduction': [model.heat_production[h].value for h in model.HOURS],
        'PurchasedElectricity': [model.purchased_electricity[h].value for h in model.HOURS],
        'HeatOverProduction': [model.heat_over_production[h].value for h in model.HOURS],
        'ElectricityOverProduction': [model.electricity_over_production[h].value for h in model.HOURS],
        'CO2Emissions': [model.co2_emissions[h].value for h in model.HOURS],
        'HeatStored': [model.heat_stored[h].value for h in model.HOURS],
        'HeatWithdrawn': [model.heat_withdrawn[h].value for h in model.HOURS],
        'ElecReductionByCHP': [model.elec_reduction_by_CHP[h].value for h in model.HOURS],
        'HeatReductionByCHP': [model.heat_reduction_by_CHP[h].value for h in model.HOURS],
        'GridReductionShortfall': [model.grid_reduction_shortfall[h].value for h in model.HOURS],
        'HeatToElec': [model.heat_to_elec[h].value for h in model.HOURS],
        'UsefulHeat': [model.useful_heat[h].value for h in model.HOURS],
        'UsefulElec': [model.useful_elec[h].value for h in model.HOURS],
        'HeatToPlant': [model.heat_to_plant[h].value for h in model.HOURS],
        'ElecToPlant': [model.elec_to_plant[h].value for h in model.HOURS],
        'RefrigerationProduced': [model.refrigeration_produced[h].value for h in model.HOURS],
        'HeatUsedForCooling': [model.heat_used_for_cooling[h].value for h in model.HOURS],
        'ElecUsedForCooling': [model.elec_used_for_cooling[h].value for h in model.HOURS],
        'RampRate': [model.ramp_rate[h].value for h in model.HOURS],
        'ProductionOutput': [model.production_output[h].value for h in model.HOURS],
        # Add other time-indexed variables as needed
    })
    save_to_csv(hours_data, os.path.join(run_dir, 'time_indexed_results.csv'))

    # Save interval-indexed variables (INTERVALS)
    interval_data = pd.DataFrame({
        'Year': [year] * len(model.INTERVALS),
        'Interval': model.INTERVALS,
        'FuelBlendNG': [model.fuel_blend_ng[i].value for i in model.INTERVALS],
        'FuelBlendH2': [model.fuel_blend_h2[i].value for i in model.INTERVALS],
        'FuelBlendBiomass': [model.fuel_blend_biomass[i].value for i in model.INTERVALS],
        'CarbonCredits': [model.carbon_credits[i].value for i in model.INTERVALS],
        'CreditsPurchased': [model.credits_purchased[i].value for i in model.INTERVALS],
        'CreditsEarned': [model.credits_earned[i].value for i in model.INTERVALS],
        'CreditsSold': [model.credits_sold[i].value for i in model.INTERVALS],
        'CreditsHeld': [model.credits_held[i].value for i in model.INTERVALS],
        'CreditsUsedToOffset': [model.credits_used_to_offset[i].value for i in model.INTERVALS],
        'TotalEmissionsPerInterval': [model.total_emissions_per_interval[i].value for i in model.INTERVALS],
        'ExceedsCap': [model.exceeds_cap[i].value for i in model.INTERVALS],
        'EmissionsDifference': [model.emissions_difference[i].value for i in model.INTERVALS],
        'BelowCap': [model.below_cap[i].value for i in model.INTERVALS],
        'InvestTime': [model.invest_time[i].value for i in model.INTERVALS],
        'ActiveH2Blending': [model.active_h2_blending[i].value for i in model.INTERVALS],
        # Add other interval-indexed variables as needed
    })
    save_to_csv(interval_data, os.path.join(run_dir, 'interval_indexed_results.csv'))

    # Save other scalar or binary variables if any, e.g., investment decisions or activation flags
    scalar_data = pd.DataFrame({
        'Year': [year],
        'InvestH2': [model.invest_h2.value],
        'UseElecForHeat': [model.use_elec_for_heat.value],
        # Add other scalar variables if needed
    })
    save_to_csv(scalar_data, os.path.join(run_dir, 'scalar_results.csv'))

    # Save yearly summary data
    yearly_data = pd.DataFrame({
        'Year': [year],
        'TotalCost': [model.objective()],
        # Add other summary statistics if needed
    })
    save_to_csv(yearly_data, os.path.join(run_dir, 'yearly_summary.csv'))

def run_30_year_simulation():
    global warm_start_values
    run_dir = create_run_directory()  # Create a unique directory for this run

    for year in range(30):
        print(f"Running model for year {year + 1}")

        # Check if the electric boiler is allowed this year
        eb_allowed = check_eb_active(year)
        h2_allowed = check_h2_active(year)

        # Get the market data for the current year
        market_data = get_market_data(year)

        # Run the model for the current year with the current state and warm start values
        model = pyomomodel(eb_allowed=eb_allowed, market_data=market_data, warm_start_values=warm_start_values, h2_allowed=h2_allowed)

        # Extract relevant results from the model
        model_cost = model.objective()
        total_costs[year] = model_cost

        # Save results after each year into the unique run folder
        save_variables(model, run_dir, year)

        # Save current year's results to use as warm start for the next year
        warm_start_values = {
            'electricity_production': {h: model.electricity_production[h].value for h in model.HOURS},
            'heat_production': {h: model.heat_production[h].value for h in model.HOURS},
            'fuel_blend_ng': {i: model.fuel_blend_ng[i].value for i in model.INTERVALS},
            'fuel_blend_h2': {i: model.fuel_blend_h2[i].value for i in model.INTERVALS},
            'fuel_blend_biomass': {i: model.fuel_blend_biomass[i].value for i in model.INTERVALS},
            'heat_stored': {h: model.heat_stored[h].value for h in model.HOURS},
            'co2_emissions': {h: model.co2_emissions[h].value for h in model.HOURS},
            'carbon_credits': {i: model.carbon_credits[i].value for i in model.INTERVALS},
            'credits_purchased': {i: model.credits_purchased[i].value for i in model.INTERVALS},
            'credits_earned': {i: model.credits_earned[i].value for i in model.INTERVALS},
            'credits_sold': {i: model.credits_sold[i].value for i in model.INTERVALS},
            'credits_held': {i: model.credits_held[i].value for i in model.INTERVALS},
            'credits_used_to_offset': {i: model.credits_used_to_offset[i].value for i in model.INTERVALS},
            'electricity_over_production': {h: model.electricity_over_production[h].value for h in model.HOURS},
            'heat_over_production': {h: model.heat_over_production[h].value for h in model.HOURS},
            'purchased_electricity': {h: model.purchased_electricity[h].value for h in model.HOURS},
            # Add other variables as needed
        }

        # Save demand and production profiles as NumPy arrays
        demand_profile = {
            'electricity_production': np.array([model.electricity_production[h]() for h in model.HOURS]),
            'heat_production': np.array([model.heat_production[h]() for h in model.HOURS]),
            'cold_production': np.array([model.refrigeration_produced[h]() for h in model.HOURS]),
            'purchased_electricity': np.array([model.purchased_electricity[h]() for h in model.HOURS])
        }
        demand_profiles.append(demand_profile)


# Modify the pyomomodel function to handle dynamic market data and electric boiler decisions

def pyomomodel(total_hours = total_hours, time_limit = time_limit, CHP_capacity=CHP_capacity, energy_ratio = energy_ratio, eb_allowed=False, h2_allowed=False, market_data=None, warm_start_values=None):
    # Create model
    model = ConcreteModel()
    # Use market data for the year
    NG_market = market_data['NG_market']
    electricity_market = market_data['electricity_market']
    carbon_market = market_data['carbon_market']
    H2_market = market_data['H2_market'] * 0.1
    BM_market = market_data['BM_market']
    electricity_market_sold = market_data['electricity_market_sold']
    heat_market_sold = market_data['heat_market_sold']
    # -------------- Parameters --------------
    # Time periods (e.g., hours in a day)
    HOURS = np.arange(total_hours)
    model.HOURS = Set(initialize=HOURS)

    no_intervals = 12
    intervals_time = int(total_hours / no_intervals)
    INTERVALS = np.arange(no_intervals)  # Four 6-hour intervals in a day
    model.INTERVALS = Set(initialize=INTERVALS)

    # Storage
    storage_efficiency = 0.5 # %
    withdrawal_efficiency = 0.5 # %
    max_storage_capacity = 1000 #kW
    heat_storage_loss_factor = 0.95  # %/timestep

    # Refrigeration
    COP_h = 2
    COP_e = 1

    # CHP params
    capital_cost_per_kw = 1000 # $/kw
    fuel_energy = 1  # kW
    max_ramp_rate = 400  #kW/timestep

    TEMP = 700
    PRES = 50
    # Efficiency coefficients
    ng_coeffs = {"elec": [0.25, 0.025, 0.001], "thermal": [0.2, 0.05, 0.001]}  # Placeholder
    h2_coeffs = {"elec": [0.18, 0.02, 0.0015], "thermal": [0.15, 0.04, 0.0012]}  # Placeholder


    h2_capex = 100000  # Example value, adjust based on your case


    # Co2 params
    co2_per_unit_ng = 0.18  # kg CO2 per kW of fuel
    co2_per_unit_bm = 0.1
    co2_per_unit_h2 = 0.01
    co2_per_unit_elec = 0.22  # kg CO2 per kW of electricity
    max_co2_emissions = markets["cap"]  # kg CO2
    M = 1E5
    # -------------- Decision Variables --------------

    # CHP System Variables
    max_electricity_production = max(electricity_demand) * 1.2  # Allow a small buffer over demand
    max_heat_production = max(heat_demand) * 1.2
    max_refrigeration = max(refrigeration_demand) * 1.2

    model.electricity_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_electricity_production))
    model.heat_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_heat_production))

    model.fuel_consumed = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 3))

    model.electrical_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(1, 1))
    model.thermal_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(1, 1))

    # Fuel variables
    model.fuel_blend_ng = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_h2 = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_biomass = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))

    #Investment decision variables
    model.invest_h2 = Var(within=Binary)  # Decision to invest in H2 system
    model.invest_time = Var(model.INTERVALS, within=Binary)  # Decision for when to invest
    model.active_h2_blending = Var(model.INTERVALS, within=Binary)  # To activate H2 blending after investment

    # Plant Supply and Useful Energy
    model.heat_to_plant = Var(model.HOURS, within=NonNegativeReals)
    model.elec_to_plant = Var(model.HOURS, within=NonNegativeReals)
    model.useful_heat = Var(model.HOURS, within=NonNegativeReals)
    model.useful_elec = Var(model.HOURS, within=NonNegativeReals)


    # Market and Storage
    model.purchased_electricity = Var(model.HOURS, within=NonNegativeReals)
    model.heat_stored = Var(model.HOURS, within=NonNegativeReals)
    model.heat_withdrawn = Var(model.HOURS, within=NonNegativeReals)

    # Overproduction and Ramp Rate
    model.heat_over_production = Var(model.HOURS, within=NonNegativeReals)
    model.electricity_over_production = Var(model.HOURS, within=NonNegativeReals)
    model.ramp_rate = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_ramp_rate))

    # Refrigeration
    model.refrigeration_produced = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_refrigeration))
    model.heat_used_for_cooling = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.elec_used_for_cooling = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))

    # Variables
    model.co2_emissions = Var(model.HOURS, within=NonNegativeReals)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)
    model.carbon_credits = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_purchased = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_earned = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions)))
    model.credits_sold = Var(model.INTERVALS, within=NonNegativeReals)
    model.exceeds_cap = Var(model.INTERVALS, within=Binary)
    model.credits_held = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions)))
    model.emissions_difference = Var(model.INTERVALS, domain=Reals, bounds=(-max(max_co2_emissions),max(max_co2_emissions)))
    model.credits_used_to_offset = Var(model.INTERVALS, within=NonNegativeReals,bounds=(0, max(max_co2_emissions)))
    model.below_cap = Var(model.INTERVALS, within=Binary)

    model.production_output = Var(model.HOURS, within=NonNegativeReals)

    # Flexibility variables for ancillary market participation
    # Flexibility variables for load shedding
    model.elec_reduction_by_CHP = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.heat_reduction_by_CHP = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)* 1.2))

    # Penalty variable for shortfall in grid reduction request
    model.grid_reduction_shortfall = Var(model.HOURS, within=NonNegativeReals)

    # Electricity to heat conversion
    model.heat_to_elec = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)))
    model.heat_to_elec_allowed = Var(within=Binary)
    # Binary variable to activate/deactivate the electricity-to-heat feature
    model.use_elec_for_heat = Var(within=Binary, initialize=1)  # Activated by default

    eb_bin = 0
    # Set the binary variable based on the external `eb_allowed` parameter
    if eb_allowed:
       eb_bin = 1
    else:
       eb_bin = 0
       energy_ratio = 0

    if h2_allowed:
       h2_bin = 1
    else:
       h2_bin = 0

    # Conversion efficiency factor
    elec_to_heat_efficiency = 0.9  # Example: 90% efficiency for converting electricity to heat
        # Example: Set initial values for variables if warm start values are provided
# Set initial values for variables if warm start values are provided
    if warm_start_values:
        for h in model.HOURS:
            # Initialize electricity and heat production variables
            if 'electricity_production' in warm_start_values:
                model.electricity_production[h].value = warm_start_values['electricity_production'].get(h, 0)
            if 'heat_production' in warm_start_values:
                model.heat_production[h].value = warm_start_values['heat_production'].get(h, 0)
            if 'heat_stored' in warm_start_values:
                model.heat_stored[h].value = warm_start_values['heat_stored'].get(h, 0)
            if 'co2_emissions' in warm_start_values:
                model.co2_emissions[h].value = warm_start_values['co2_emissions'].get(h, 0)
            if 'electricity_over_production' in warm_start_values:
                model.electricity_over_production[h].value = warm_start_values['electricity_over_production'].get(h, 0)
            if 'heat_over_production' in warm_start_values:
                model.heat_over_production[h].value = warm_start_values['heat_over_production'].get(h, 0)
            if 'purchased_electricity' in warm_start_values:
                model.purchased_electricity[h].value = warm_start_values['purchased_electricity'].get(h, 0)

        for i in model.INTERVALS:
            # Initialize fuel blend variables
            if 'fuel_blend_ng' in warm_start_values:
                model.fuel_blend_ng[i].value = warm_start_values['fuel_blend_ng'].get(i, 0)
            if 'fuel_blend_h2' in warm_start_values:
                model.fuel_blend_h2[i].value = warm_start_values['fuel_blend_h2'].get(i, 0)
            if 'fuel_blend_biomass' in warm_start_values:
                model.fuel_blend_biomass[i].value = warm_start_values['fuel_blend_biomass'].get(i, 0)

            # Initialize carbon credit variables
            if 'carbon_credits' in warm_start_values:
                model.carbon_credits[i].value = warm_start_values['carbon_credits'].get(i, 0)
            if 'credits_purchased' in warm_start_values:
                model.credits_purchased[i].value = warm_start_values['credits_purchased'].get(i, 0)
            if 'credits_earned' in warm_start_values:
                model.credits_earned[i].value = warm_start_values['credits_earned'].get(i, 0)
            if 'credits_sold' in warm_start_values:
                model.credits_sold[i].value = warm_start_values['credits_sold'].get(i, 0)
            if 'credits_held' in warm_start_values:
                model.credits_held[i].value = warm_start_values['credits_held'].get(i, 0)
            if 'credits_used_to_offset' in warm_start_values:
                model.credits_used_to_offset[i].value = warm_start_values['credits_used_to_offset'].get(i, 0)
 # -------------- Constraints --------------

    # Heat Balance
    def heat_reduction_rule(model, h):
        return model.heat_reduction_by_CHP[h] == (model.elec_reduction_by_CHP[h] * (1-energy_ratio))
    model.heat_balance_reduction = Constraint(model.HOURS, rule=heat_reduction_rule)

    # Overproduction of Heat
    def heat_over_production_rule(model, h):
        return model.heat_over_production[h] == model.heat_production[h] - model.useful_heat[h]
    model.heat_over_production_constraint = Constraint(model.HOURS, rule=heat_over_production_rule)

    # Useful Heat
    def useful_heat_rule(model, h):
        return model.useful_heat[h] == model.heat_to_plant[h] - (model.heat_withdrawn[h] * withdrawal_efficiency) + (model.heat_used_for_cooling[h] / COP_h)
    model.useful_heat_constraint = Constraint(model.HOURS, rule=useful_heat_rule)

    # ======== Electricity-Related Constraints ========
    # Useful Electricity
    def useful_elec_rule(model, h):
        return model.useful_elec[h] == model.elec_to_plant[h] + (model.elec_used_for_cooling[h] / COP_e)
    model.useful_elec_constraint = Constraint(model.HOURS, rule=useful_elec_rule)

    # Overproduction of Electricity
    def elec_over_production_rule(model, h):
        return model.electricity_over_production[h] == model.electricity_production[h] - model.useful_elec[h]
    model.elec_over_production_constraint = Constraint(model.HOURS, rule=elec_over_production_rule)

    # ======== CHP and Fuel-Related Constraints ========

    # CHP Capacity
    def capacity_rule(model, h):
        return CHP_capacity >= (model.heat_production[h] + model.heat_stored[h]) + model.electricity_production[h]
    model.capacity_constraint = Constraint(model.HOURS, rule=capacity_rule)

    # Fuel Consumption
    def fuel_consumed_rule(model, h):
        return fuel_energy * model.fuel_consumed[h] * (1 - energy_ratio) * 1 == model.heat_production[h]
    model.fuel_consumed_rule = Constraint(model.HOURS, rule=fuel_consumed_rule)

    def electrical_efficiency_rule(model, h):
        efficiency_adjustment_times_CHP = (
            (model.fuel_blend_ng[h] + model.fuel_blend_biomass[h]) * 
            (ng_coeffs["elec"][0] * CHP_capacity + ng_coeffs["elec"][1] * model.electricity_production[h] + ng_coeffs["elec"][2] * TEMP * CHP_capacity) +
            model.fuel_blend_h2[h] * 
            (h2_coeffs["elec"][0] * CHP_capacity + h2_coeffs["elec"][1] * model.electricity_production[h] + h2_coeffs["elec"][2] * TEMP * CHP_capacity)
        )
        return model.electrical_efficiency[h] * CHP_capacity == efficiency_adjustment_times_CHP

    #model.electrical_efficiency_constraint = Constraint(model.HOURS, rule=electrical_efficiency_rule)

    def thermal_efficiency_rule(model, h):
        efficiency_adjustment_times_CHP = (
            (model.fuel_blend_ng[h] + model.fuel_blend_biomass[h]) * 
            (ng_coeffs["thermal"][0] * CHP_capacity + ng_coeffs["thermal"][1] * model.heat_production[h] + ng_coeffs["thermal"][2] * TEMP * CHP_capacity) +
            model.fuel_blend_h2[h] * 
            (h2_coeffs["thermal"][0] * CHP_capacity + h2_coeffs["thermal"][1] * model.heat_production[h] + h2_coeffs["thermal"][2] * TEMP * CHP_capacity)
        )
        return model.thermal_efficiency[h] * CHP_capacity == efficiency_adjustment_times_CHP

    #model.thermal_efficiency_constraint = Constraint(model.HOURS, rule=thermal_efficiency_rule)

    # Constraint to ensure that the sum of the fuel blend percentages equals 1.
    def fuel_blend_rule(model, i):
        return model.fuel_blend_ng[i] + model.fuel_blend_h2[i] * h2_bin + model.fuel_blend_biomass[i] == 1
    model.fuel_blend_constraint = Constraint(model.INTERVALS, rule=fuel_blend_rule)

    # ======== Ramping Constraints ========

    # Ramp Up
    def ramp_up_rule(model, h):
        if h == 0:
            return Constraint.Skip  # Skip the first hour
        return model.heat_production[h] - model.heat_production[h - 1] <= model.ramp_rate[h]
    model.ramp_up_constraint = Constraint(model.HOURS, rule=ramp_up_rule)

    # Ramp Down
    def ramp_down_rule(model, h):
        if h == 0:
            return Constraint.Skip  # Skip the first hour
        return model.heat_production[h - 1] - model.heat_production[h] <= model.ramp_rate[h]
    model.ramp_down_constraint = Constraint(model.HOURS, rule=ramp_down_rule)

    # Max Ramp
    def max_ramp_rule(model, h):
        return model.ramp_rate[h] <= max_ramp_rate
    model.max_ramp_constraint = Constraint(model.HOURS, rule=max_ramp_rule)

    # ======== Storage Constraints ========
    # Heat Storage Dynamics
    # Simplified storage dynamics by consolidating rules
    def storage_dynamics_rule(model, h):
        if h == 0:
            return model.heat_stored[h] == 0
        return model.heat_stored[h] == model.heat_stored[h - 1] * heat_storage_loss_factor + \
            (model.heat_over_production[h] * storage_efficiency) - \
            (model.heat_withdrawn[h] / withdrawal_efficiency)
    model.storage_dynamics = Constraint(model.HOURS, rule=storage_dynamics_rule)


    # Heat Storage Capacity
    def storage_capacity_rule(model, h):
        return model.heat_stored[h] <= max_storage_capacity
    model.storage_capacity = Constraint(model.HOURS, rule=storage_capacity_rule)

    # ======== Refrigeration Constraints ========

    # Refrigeration Balance
    def refrigeration_balance_rule(model, h):
        return model.refrigeration_produced[h] == (model.elec_used_for_cooling[h] * COP_e) + (model.heat_used_for_cooling[h] * COP_h)
    model.refrigeration_balance = Constraint(model.HOURS, rule=refrigeration_balance_rule)

    # Refrigeration Demand
    def refrigeration_demand_rule(model, h):
        return model.refrigeration_produced[h] == refrigeration_demand[h]
    model.refrigeration_demand_con = Constraint(model.HOURS, rule=refrigeration_demand_rule)

    # Energy Ratio
    def energy_ratio_rule(model, h):
        return model.electricity_production[h] == (model.heat_production[h] * energy_ratio) * 1
    model.energy_ratio_constraint = Constraint(model.HOURS, rule=energy_ratio_rule)

# ======== CO2 Constraints ========
    def co2_emissions_rule(model, h):
        i = h // (len(model.HOURS) // len(model.INTERVALS))  # Map month to interval
        return model.co2_emissions[i] == (
            co2_per_unit_ng * model.fuel_blend_ng[i] * model.fuel_consumed[i] +
            co2_per_unit_h2 * model.fuel_blend_h2[i] * model.fuel_consumed[i] +
            co2_per_unit_bm * model.fuel_blend_biomass[i] * model.fuel_consumed[i] +
            co2_per_unit_elec * (model.purchased_electricity[i] + model.heat_to_elec[i])
        )
    model.co2_emissions_constraint = Constraint(model.HOURS, rule=co2_emissions_rule)

    # Total Emissions Per Interval Rule
    def total_emissions_per_interval_rule(model, i):
        start = i * intervals_time
        end = (i + 1) * intervals_time
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[h] for h in range(start, end))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    # Carbon Credits Needed Rule
    def carbon_credits_needed_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] >= model.total_emissions_per_interval[i] - max_co2_emissions[i*no_intervals]
    model.carbon_credits_needed_constraint = Constraint(model.INTERVALS, rule=carbon_credits_needed_rule)

    # Carbon Credits Earned Rule
    def carbon_credits_earned_rule(model, i):
        if i == 0:
            return model.credits_earned[i] == 0  # For the first interval
        return model.credits_earned[i] == (max_co2_emissions[i*no_intervals] - model.total_emissions_per_interval[i] + model.credits_used_to_offset[i]) * (1 - model.below_cap[i])
    model.carbon_credits_earned_constraint = Constraint(model.INTERVALS, rule=carbon_credits_earned_rule)

    # Carbon Credits Purchased Rule
    def carbon_credits_purchased_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] <= M * model.below_cap[i]
    model.carbon_credits_purchased_con = Constraint(model.INTERVALS, rule=carbon_credits_purchased_rule)

    # Credits Unheld Limit Rule
    def credits_unheld_limit_rule(model, i):
        if i == 0:
            return model.credits_sold[i] == 0  # For the first interval
        return model.credits_sold[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_used_to_offset[i]
    model.credits_unheld_limit = Constraint(model.INTERVALS, rule=credits_unheld_limit_rule)

    # Credits Held Dynamics Rule
    def credits_held_dynamics_rule(model, i):
        if i == 0:
            return model.credits_held[i] == model.credits_earned[i]  # For the first interval
        return model.credits_held[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_sold[i] - model.credits_used_to_offset[i]
    model.credits_held_dynamics = Constraint(model.INTERVALS, rule=credits_held_dynamics_rule)

    # Below Cap Rule
    def below_cap_rule(model, i):
        return model.total_emissions_per_interval[i] - max_co2_emissions[i*no_intervals] <= M * model.below_cap[i]
    model.below_cap_con = Constraint(model.INTERVALS, rule=below_cap_rule)

    # Force Zero Rule
    def force_0_rule(model, i):
        if i == 0:
            return model.credits_used_to_offset[i] == 0
        return Constraint.Skip
    model.force_0_rule_con = Constraint(model.INTERVALS, rule=force_0_rule)


    # Production Output Constraint
    def production_output_rule(model, h):
        # Assuming a linear relationship for simplicity; you can modify this as needed
        production_output_ratio = 0.8  # Example production-output-to-energy ratio
        return model.production_output[h] == production_output_ratio * (heat_demand[h] + electricity_demand[h] - model.elec_reduction_by_CHP[h] - model.heat_reduction_by_CHP[h])
    model.production_output_constraint = Constraint(model.HOURS, rule=production_output_rule)

    # -------------- Anchillary market participation --------------

    plant_flexibility = 0.5
    # Modified constraint to allow shortfall in grid reduction
    # Modified constraint to allow shortfall in grid reduction for each hour
    def grid_call_constraint(model, h):
        return model.elec_reduction_by_CHP[h] + model.grid_reduction_shortfall[h] >= request[h]
    model.grid_call_constraint = Constraint(model.HOURS, rule=grid_call_constraint)

        # Constraint to limit the maximum reduction based on plant flexibility
    def flexibility_constraint(model, h):
        return model.elec_reduction_by_CHP[h] <= plant_flexibility * electricity_demand[h]
    model.flexibility_constraint = Constraint(model.HOURS, rule=flexibility_constraint)

    def elec_for_heat_conversion_rule(model, h):
        # Only allow electricity for heat if the feature is activated
        return model.heat_production[h] >= model.useful_heat[h] 
    model.elec_for_heat_conversion_constraint = Constraint(model.HOURS, rule=elec_for_heat_conversion_rule)

    # Electricity Demand with Additional Heat Conversion
    def updated_elec_demand_balance_rule(model, h):
        return model.elec_to_plant[h] + model.purchased_electricity[h] == electricity_demand[h] - model.elec_reduction_by_CHP[h]
    model.updated_elec_demand_rule = Constraint(model.HOURS, rule=updated_elec_demand_balance_rule)

    # Modify the heat demand rule to account for electricity conversion to heat
    def updated_heat_demand_balance_rule(model, h):
        return model.heat_to_plant[h] + model.heat_to_elec[h] == heat_demand[h] - model.heat_reduction_by_CHP[h] + elec_to_heat_efficiency 
    model.updated_heat_demand_rule = Constraint(model.HOURS, rule=updated_heat_demand_balance_rule)

    def eb_activation_rule(model, h):
        return model.heat_to_elec[h] <= (eb_bin * M) 
    model.eb_activation_constraint = Constraint(model.HOURS, rule=eb_activation_rule)
    # -------------- Objective Function --------------

    def objective_rule(model):
        elec_cost = sum((model.purchased_electricity[h] + model.heat_to_elec[h]) * electricity_market[h] for h in model.HOURS)
        elec_sold = sum(model.electricity_over_production[h] * electricity_market_sold[h] for h in model.HOURS)
        heat_sold = sum((heat_market_sold[h] * model.heat_over_production[h]) for h in model.HOURS)
        fuel_cost_NG = sum(model.fuel_blend_ng[i] * NG_market[h] * model.fuel_consumed[h] for h in model.HOURS for i in model.INTERVALS if h // intervals_time == i)
        fuel_cost_H2 = sum(model.fuel_blend_h2[i] * H2_market[h] * model.fuel_consumed[h] for h in model.HOURS for i in model.INTERVALS if h // intervals_time == i)
        fuel_cost_BM = sum(model.fuel_blend_biomass[i] * BM_market[h] * model.fuel_consumed[h] for h in model.HOURS for i in model.INTERVALS if h // intervals_time == i)
        carbon_cost = sum(model.carbon_credits[i] * carbon_market[i] for i in model.INTERVALS)
        carbon_sold = sum(model.credits_sold[i] * carbon_market[i] for i in model.INTERVALS) * 0.9

        production_revenue = sum(model.production_output[h] for h in model.HOURS)
        ancillary_revenue = sum(reward[h] * (model.elec_reduction_by_CHP[h]) for h in model.HOURS) * 1000
        all_shortfall_penalty = sum(shortfall_penalty[h] * model.grid_reduction_shortfall[h] for h in model.HOURS) * 0



        return (
            (fuel_cost_NG + fuel_cost_H2 + fuel_cost_BM) + elec_cost + carbon_cost + all_shortfall_penalty 
            - (elec_sold + heat_sold + carbon_sold + production_revenue + ancillary_revenue)
        )
    
    model.objective = Objective(rule=objective_rule, sense=minimize)
    # -------------- Solver --------------
    solver = get_solver(time_limit)  # Use the imported solver configuration
    solver.solve(model, tee=True, symbolic_solver_labels=False)
    return model




# Run the Dash app
if __name__ == '__main__':
    run_30_year_simulation()

