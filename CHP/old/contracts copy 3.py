# Import necessary modules
from pyomo.environ import *
import pandas as pd
import os
import numpy as np
import sys

# If necessary, append configuration path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))

# Import the solver options (adjust this if you have a specific solver configuration)
# from solver_options import get_solver

current_dir = os.path.dirname(__file__)

# Adjust these paths according to your directory structure
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'optimized_demands.xlsx'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.xlsx'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.xlsx'))

# Read the Excel files (ensure the files exist at the specified paths)
demands = pd.read_excel(demands_path)
markets_monthly = pd.read_excel(markets_monthly_path)
markets = pd.read_excel(markets_path)

# Extract demand and market data
nat_gas = demands["nat_gas"].to_numpy()  # m3
NG_market_monthly = markets_monthly["nat_gas"].to_numpy()

# Parameters for the model
CHP_capacity = 2500

# Set up time periods: Aggregate months into quarters
total_months = len(NG_market_monthly)
quarters = total_months // 3  # Assuming total_months is a multiple of 3
if total_months % 3 != 0:
    quarters += 1  # Include the last incomplete quarter

# Aggregate nat_gas and NG_market_monthly into quarterly data
nat_gas_quarterly = []
NG_market_quarterly = []

for i in range(quarters):
    start_idx = i * 3
    end_idx = min((i + 1) * 3, total_months)
    nat_gas_quarterly.append(np.sum(nat_gas[start_idx:end_idx]))
    NG_market_quarterly.append(np.mean(NG_market_monthly[start_idx:end_idx]))

# Define the number of contracts (reduced for performance)
no_contract = 5

# Define the maximum contract duration (adjust as needed)
max_contract_duration = quarters  # Contracts can last up to the total number of quarters

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # Sets
    model.CONTRACTS = RangeSet(1, no_contract)
    model.PERIODS = RangeSet(1, quarters)

    # Parameters

    # Predefine contract types for each contract to reduce variables
    contract_types = {
        1: 'Fixed',
        2: 'Indexed',
        3: 'Fixed',
        4: 'Indexed',
        5: 'Fixed'
    }
    model.ContractType = Param(model.CONTRACTS, initialize=contract_types, within=Any)

    # Precompute contract prices based on type
    contract_prices = {}
    for c in model.CONTRACTS:
        c_type = model.ContractType[c]
        if c_type == 'Fixed':
            # Fixed price is based on the price at the start (assume first quarter)
            fixed_price = NG_market_quarterly[0] * 1.5  # Adjust multiplier as needed
            for p in model.PERIODS:
                contract_prices[(c, p)] = fixed_price
        elif c_type == 'Indexed':
            # Indexed price varies with market price
            for p in model.PERIODS:
                contract_prices[(c, p)] = NG_market_quarterly[p - 1]  # p-1 because list indices start at 0
        else:
            # Default to market price if contract type is unrecognized
            for p in model.PERIODS:
                contract_prices[(c, p)] = NG_market_quarterly[p - 1]
    model.ContractPrice = Param(model.CONTRACTS, model.PERIODS, initialize=contract_prices)

    # Parameters for minimum contract amounts
    min_contract_amounts = {}
    for c in model.CONTRACTS:
        c_type = model.ContractType[c]
        if c_type == 'Fixed' or c_type == 'Indexed':
            min_amount = 250  # Adjust as needed
        elif c_type == 'TakeOrPay':
            min_amount = 500  # Adjust as needed
        else:
            min_amount = 0
        for p in model.PERIODS:
            min_contract_amounts[(c, p)] = min_amount
    model.MinContractAmount = Param(model.CONTRACTS, model.PERIODS, initialize=min_contract_amounts)

    # Variables
    # Contract duration (integer, up to max_contract_duration)
    model.ContractDuration = Var(model.CONTRACTS, within=NonNegativeIntegers, bounds=(1, max_contract_duration))

    # Contract amount in each period
    model.ContractAmount = Var(model.CONTRACTS, model.PERIODS, within=NonNegativeReals)

    # Constraints

    # Contract duration constraints: Assume all contracts start at the first period
    def contract_active_rule(model, c, p):
        return model.ContractAmount[c, p] <= 2000 * (p <= value(model.ContractDuration[c]))
    model.ContractActiveCon = Constraint(model.CONTRACTS, model.PERIODS, rule=contract_active_rule)

    # Minimum contract fulfillment amount
    def min_fulfillment_rule(model, c, p):
        return model.ContractAmount[c, p] >= model.MinContractAmount[c, p] * (p <= value(model.ContractDuration[c]))
    model.MinFulfillmentCon = Constraint(model.CONTRACTS, model.PERIODS, rule=min_fulfillment_rule)

    # Demand fulfillment constraint
    def demand_fulfillment_rule(model, p):
        return sum(model.ContractAmount[c, p] for c in model.CONTRACTS) >= nat_gas_quarterly[p - 1]
    model.DemandFulfillmentCon = Constraint(model.PERIODS, rule=demand_fulfillment_rule)

    # Objective function: Minimize total cost
    def objective_rule(model):
        return sum(
            model.ContractAmount[c, p] * model.ContractPrice[c, p]
            for c in model.CONTRACTS for p in model.PERIODS
        )
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Solver configuration
    solver = SolverFactory('gurobi')  # Use Gurobi solver (adjust if using a different solver)
    solver.options['TimeLimit'] = 120  # Set time limit in seconds
    solver.options['MIPGap'] = 0.02    # Set acceptable optimality gap (2%)
    solver.options['Threads'] = 4      # Use multiple CPU threads

    # Solve the model
    results = solver.solve(model, tee=True)

    return model

if __name__ == '__main__':
    model = pyomomodel()

    # Display results
    for c in model.CONTRACTS:
        c_type = model.ContractType[c]
        duration = value(model.ContractDuration[c])
        print(f"Contract {c}: Type = {c_type}, Duration = {duration} periods")
        for p in model.PERIODS:
            amount = value(model.ContractAmount[c, p])
            price = model.ContractPrice[c, p]
            if amount > 0:
                print(f"  Period {p}: Amount = {amount:.2f}, Price = {price:.2f}")
        print("\n")

    # Total cost
    total_cost = value(model.objective)
    print(f"Total Cost: {total_cost:.2f}")
