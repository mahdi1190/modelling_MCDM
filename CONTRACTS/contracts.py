from pyomo.environ import *
import pandas as pd
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
# Import the solver options
from solver_options import get_solver
current_dir = os.path.dirname(__file__)
# Construct paths to the data files by correctly moving up one directory to 'modelling_MCDM'
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'optimized_demands.xlsx'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.xlsx'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.xlsx'))  # Corrected path

# Read the Excel files
demands = pd.read_excel(demands_path)
markets_monthly = pd.read_excel(markets_monthly_path)
markets = pd.read_excel(markets_path)


electricity_demand = demands["purchased_elec"].to_numpy() #kWh
nat_gas = demands["nat_gas"].to_numpy() #m3
credits_sold = demands["credits_held"].to_numpy()
credits_purchased = demands["credits_needed"].to_numpy()

electricity_market = markets["elec"].to_numpy()
electricity_market_sold = markets["elec_sold"].to_numpy()

NG_market = markets["nat_gas"].to_numpy()
NG_market_monthly = markets_monthly["nat_gas"].to_numpy()

heat_market_sold = markets["nat_gas_sold"].to_numpy()

H2_market = markets["hydrogen"].to_numpy()
BM_market = markets["biomass"].to_numpy()

CHP_capacity = 2500

total_hours = 24
total_times = 60
no_contract = 15
time_limit = 120
bound_duration = 60

risk_appetite_threshold = 10000

from pyomo.environ import *

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # Parameters and sets remain the same as in your original code.
    total_time = total_times
    HOURS = np.arange(total_hours)
    model.HOURS = Set(initialize=HOURS)

    MONTHS = np.arange(total_time)
    model.MONTHS = Set(initialize=MONTHS)

    no_intervals = 4
    intervals_time = int(total_hours / no_intervals)
    INTERVALS = np.arange(no_intervals)  
    model.INTERVALS = Set(initialize=INTERVALS)

    model.CONTRACTS = Set(initialize=range(no_contract))
    model.ContractTypes = Set(initialize=['Fixed', 'Indexed', 'TakeOrPay'])

    # Define Variables
    model.co2_emissions = Var(model.HOURS, within=NonNegativeReals)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)

    # Binary and continuous variables for contracts
    model.ContractStart = Var(model.MONTHS, model.CONTRACTS, within=Binary)
    model.ContractActive = Var(model.MONTHS, model.CONTRACTS, within=Binary)
    model.ContractType = Var(model.CONTRACTS, model.ContractTypes, within=Binary)
    model.ContractAmount = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(0, 2000))
    model.ContractPrice = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(0, 1))
    model.ContractStartPrice = Var(model.CONTRACTS, within=NonNegativeReals, bounds=(0, 1))
    model.ContractDuration = Var(model.CONTRACTS, within=NonNegativeIntegers, bounds=(0, bound_duration))

    model.risk_score = Var(within=NonNegativeReals, bounds=(0, risk_appetite_threshold))

    # Constraints

    # Duration constraint for each contract
    def contract_duration_rule(model, c):
        # Sum of active months should equal the contract duration
        return sum(model.ContractActive[m, c] for m in model.MONTHS) == model.ContractDuration[c]
    model.ContractDurationCon = Constraint(model.CONTRACTS, rule=contract_duration_rule)

    # Ensure minimum duration for each contract after it starts
    def min_contract_duration_rule(model, c):
        # Minimum contract duration of 3 months if the contract is active
        return sum(model.ContractActive[m, c] for m in model.MONTHS) >= 3 * sum(model.ContractStart[m, c] for m in model.MONTHS)
    model.MinContractDurationCon = Constraint(model.CONTRACTS, rule=min_contract_duration_rule)

    # Ensure only one contract type is selected per contract
    def contract_type_rule(model, c):
        return sum(model.ContractType[c, t] for t in model.ContractTypes) == 1
    model.ContractTypeConstraint = Constraint(model.CONTRACTS, rule=contract_type_rule)

    # Linking contract start to active months
    def contract_active_rule(model, m, c):
        if m == 0:
            return model.ContractActive[m, c] == model.ContractStart[m, c]
        else:
            return model.ContractActive[m, c] >= model.ContractActive[m - 1, c] + model.ContractStart[m, c] - 1
    model.ContractActiveCon = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_active_rule)

    # Ensure only one contract start per contract
    def contract_single_start_rule(model, c):
        return sum(model.ContractStart[m, c] for m in model.MONTHS) == 1
    model.ContractSingleStartCon = Constraint(model.CONTRACTS, rule=contract_single_start_rule)

    # Set start price based on contract's start month
    def set_contract_start_price_rule(model, c):
        return model.ContractStartPrice[c] == sum(NG_market_monthly[m] * model.ContractStart[m, c] for m in model.MONTHS)
    model.SetContractStartPrice = Constraint(model.CONTRACTS, rule=set_contract_start_price_rule)

    # Contract price setting rule
    def contract_price_setting_rule(model, m, c):
        return model.ContractPrice[m, c] == (
            model.ContractType[c, 'Fixed'] * model.ContractStartPrice[c] * 1.5 +
            model.ContractType[c, 'Indexed'] * NG_market_monthly[m] +
            model.ContractType[c, 'TakeOrPay'] * model.ContractStartPrice[c] * 1.1
        )
    model.ContractPriceSetting = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_price_setting_rule)

    # Minimum contract fulfillment amount
    def min_fulfillment_rule(model, m, c):
        return model.ContractAmount[m, c] >= model.ContractActive[m, c] * (
            500 * model.ContractType[c, 'TakeOrPay'] +
            250 * (model.ContractType[c, 'Fixed'] + model.ContractType[c, 'Indexed'])
        )
    model.min_fulfillment_con = Constraint(model.MONTHS, model.CONTRACTS, rule=min_fulfillment_rule)

    # Ensure demand fulfillment for natural gas
    def demand_fulfillment_rule(model, m):
        return sum(model.ContractAmount[m, c] * model.ContractActive[m, c] for c in model.CONTRACTS) >= nat_gas[m]
    model.demand_fulfillment_con = Constraint(model.MONTHS, rule=demand_fulfillment_rule)

    # CO2 emissions calculation per interval
    def total_emissions_per_interval_rule(model, i):
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[h] for h in range(i * intervals_time, (i+1) * intervals_time))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    # Risk score calculation
    def risk_score_rule(model):
        return model.risk_score == sum(
            (1 * model.ContractType[c, 'Fixed'] +
             3 * model.ContractType[c, 'Indexed'] +
             2 * model.ContractType[c, 'TakeOrPay']) * 
             model.ContractDuration[c] 
            for c in model.CONTRACTS
        )
    model.RiskScoreCon = Constraint(rule=risk_score_rule)

    # Objective function
    def objective_rule(model):
        fuel_cost_NG = sum(model.ContractAmount[m, c] * model.ContractPrice[m, c] for m in model.MONTHS for c in model.CONTRACTS)
        return fuel_cost_NG
    model.objective = Objective(rule=objective_rule, sense=minimize)

    solver = get_solver(time_limit)  # Use the imported solver configuration
    solver.solve(model, tee=True, symbolic_solver_labels=False)

    return model


if __name__ == '__main__':
    model = pyomomodel()
    for m in range(total_times):
        print(f"Month: {m}")
        for c in range(no_contract):
            # Determine the selected contract type for the current contract
            selected_contract_type = None
            for t in model.ContractTypes:
                if value(model.ContractType[c, t]) >= 0.99:  # Assuming model.ContractTypes is your set of contract types
                    selected_contract_type = t
                    break

            # Extract values for amount, price, and duration
            amount = value(model.ContractAmount[m, c])
            price = value(model.ContractPrice[m, c])
            duration = value(model.ContractDuration[c])  # Assuming this is a parameter and constant for each contract
            
            # Print the details including the selected contract type and fuel type
            print(f"  {selected_contract_type} Contract {c}: Amount: {amount:.2f}, Price: {price:.2f}, Duration: {duration}")
        print("\n")



