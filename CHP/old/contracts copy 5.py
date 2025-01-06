from pyomo.environ import *
import pandas as pd
import numpy as np
import os
import sys

# Path configurations for solver and data files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from solver_options import get_solver

current_dir = os.path.dirname(__file__)

# Construct paths to the data files
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'optimized_demands.xlsx'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.xlsx'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.xlsx'))

# Read the Excel files
demands = pd.read_excel(demands_path, nrows=1000)
markets_monthly = pd.read_excel(markets_monthly_path, nrows=1000)
markets = pd.read_excel(markets_path, nrows=1000)

# Extract demand and market data
nat_gas = demands["nat_gas"].to_numpy()  # m3
NG_market_monthly = markets_monthly["nat_gas"].to_numpy()

CHP_capacity = 2500
total_times = 180  # Total months
no_contract = 12
time_limit = 60
bound_duration = 36
risk_appetite_threshold = 700

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # Sets
    model.MONTHS = RangeSet(0, total_times - 1)  # Total time horizon in months
    model.CONTRACTS = RangeSet(0, no_contract - 1)  # Number of contracts
    model.ContractTypes = Set(initialize=['Fixed', 'Indexed', 'TakeOrPay'])  # Contract types

    # Parameters
    min_price = min(NG_market_monthly) * 0.2
    max_price = max(NG_market_monthly) * 5

    # Variables
    model.ContractStart = Var(model.MONTHS, model.CONTRACTS, within=Binary)
    model.ContractActive = Var(model.MONTHS, model.CONTRACTS, within=Binary)
    model.ContractAmount = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(0, 2000))
    model.ContractPrice = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(min_price, max_price))
    model.ContractType = Var(model.CONTRACTS, model.ContractTypes, within=Binary)
    model.ContractDuration = Var(model.CONTRACTS, within=NonNegativeIntegers, bounds=(1, bound_duration))

    model.risk_score = Var(within=NonNegativeReals, bounds=(0, risk_appetite_threshold))

    # Constraints

    # Ensure only one start month is allowed per contract
    def single_start_rule(model, c):
        return sum(model.ContractStart[m, c] for m in model.MONTHS) <= 1
    model.SingleStartCon = Constraint(model.CONTRACTS, rule=single_start_rule)

    # Looser continuity: Contracts are active once started for a number of months, not necessarily consecutive
    def contract_activation_rule(model, m, c):
        if m == 0:
            return model.ContractActive[m, c] == model.ContractStart[m, c]
        else:
            return model.ContractActive[m, c] >= model.ContractStart[m, c] + model.ContractActive[m - 1, c]
    model.ContractActivation = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_activation_rule)

    # Enforce contract duration
    def contract_duration_rule(model, c):
        return sum(model.ContractActive[m, c] for m in model.MONTHS) == model.ContractDuration[c]
    model.ContractDurationCon = Constraint(model.CONTRACTS, rule=contract_duration_rule)

    # Ensure each contract selects only one type
    def contract_type_rule(model, c):
        return sum(model.ContractType[c, t] for t in model.ContractTypes) == 1
    model.ContractTypeCon = Constraint(model.CONTRACTS, rule=contract_type_rule)

    # Contract price setting rule
    def contract_price_rule(model, m, c):
        # Fixed price contracts
        fixed_price = model.ContractType[c, 'Fixed'] * model.ContractPrice[m, c] * 0.7
        # Indexed contracts
        indexed_price = model.ContractType[c, 'Indexed'] * NG_market_monthly[m]
        # Take-or-pay contracts
        take_or_pay_price = model.ContractType[c, 'TakeOrPay'] * model.ContractPrice[m, c] * 0.5

        return model.ContractPrice[m, c] == fixed_price + indexed_price + take_or_pay_price
    model.ContractPriceCon = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_price_rule)

    # Minimum contract fulfillment amount
    def min_fulfillment_rule(model, m, c):
        min_amount = (800 * model.ContractType[c, 'TakeOrPay'] +
                      400 * (model.ContractType[c, 'Fixed'] + model.ContractType[c, 'Indexed']))
        return model.ContractAmount[m, c] >= model.ContractActive[m, c] * min_amount
    model.MinFulfillmentCon = Constraint(model.MONTHS, model.CONTRACTS, rule=min_fulfillment_rule)

    # Maximum contract fulfillment amount
    def max_fulfillment_rule(model, m, c):
        max_amount = (1000 * model.ContractType[c, 'TakeOrPay'] +
                      500 * (model.ContractType[c, 'Fixed'] + 2000 * model.ContractType[c, 'Indexed']))
        return model.ContractAmount[m, c] <= model.ContractActive[m, c] * max_amount
    model.MaxFulfillmentCon = Constraint(model.MONTHS, model.CONTRACTS, rule=max_fulfillment_rule)

    # Ensure natural gas demand is met each month
    def demand_fulfillment_rule(model, m):
        return sum(model.ContractAmount[m, c] for c in model.CONTRACTS) >= nat_gas[m]
    model.DemandFulfillment = Constraint(model.MONTHS, rule=demand_fulfillment_rule)

    # Objective function: minimize total contract cost
    def objective_rule(model):
        return sum(
            model.ContractAmount[m, c] * model.ContractPrice[m, c] for m in model.MONTHS for c in model.CONTRACTS
        )
    model.TotalCost = Objective(rule=objective_rule, sense=minimize)

    # Solver configuration
    solver = get_solver(time_limit)
    solver.options['MIPGap'] = 0.01
    solver.options['Threads'] = 0
    solver.options['MIPFocus'] = 1
    solver.options['Presolve'] = 2
    solver.options['Heuristics'] = 0.5
    solver.options['Cuts'] = 2
    solver.options['Symmetry'] = 2
    solver.options['Method'] = 2

    # Solve the model
    results = solver.solve(model, tee=True, symbolic_solver_labels=False)

    return model
