from pyomo.environ import *
import pandas as pd
import os
import numpy as np


demands = pd.read_excel(r"C:\Users\fcp22sma\modelling_MCDM\data\optimized_demands.xlsx")
markets = pd.read_excel(r"C:\Users\fcp22sma\modelling_MCDM\data\markets.xlsx")
markets_monthly = pd.read_excel(r"C:\Users\fcp22sma\modelling_MCDM\data\markets_monthly.xlsx")


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
no_contract = 10
time_limit = 30
bound_duration = 60

risk_appetite_threshold = 100

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    # Time periods (e.g., hours in a day)
    total_time = total_times
    HOURS = np.arange(total_hours)
    model.HOURS = Set(initialize=HOURS)

    MONTHS = np.arange(total_time)
    model.MONTHS = Set(initialize=MONTHS)

    no_intervals = 4
    intervals_time = int(total_hours / no_intervals)
    INTERVALS = np.arange(no_intervals)  # Four 6-hour intervals in a day
    model.INTERVALS = Set(initialize=INTERVALS)

    model.CONTRACTS = Set(initialize=range(no_contract))

    max_co2_emissions = 5000  # kg CO2
    M = max_co2_emissions*1E3

    # -------------- Decision Variables --------------

    # Variables
    model.co2_emissions = Var(model.HOURS, within=NonNegativeReals)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)
    # Fixed-price contract

    # Indicates if a contract is initiated in a specific month
    model.ContractStart = Var(model.MONTHS, model.CONTRACTS, within=Binary)
    model.ContractDuration = Var(model.CONTRACTS, within=NonNegativeIntegers, bounds=(0, bound_duration))
    model.ContractAmount = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(0, 2000))
    model.ContractPrice = Var(model.MONTHS, model.CONTRACTS, within=NonNegativeReals, bounds=(0, 1))
    model.ContractStartPrice = Var(model.CONTRACTS, within=NonNegativeReals, bounds=(0, 1))
    model.ContractActive = Var(model.MONTHS, model.CONTRACTS, within=Binary)

    model.ContractTypes = Set(initialize=['Fixed', 'Indexed', 'TakeOrPay'])
    model.ContractType = Var(model.CONTRACTS, model.ContractTypes, within=Binary)
    model.ContractSupplyType = Var(model.CONTRACTS, within=Binary)

    model.risk_score = Var(within=NonNegativeReals)

 # -------------- Constraints --------------
    
    def contract_duration_rule(model, c):
        return sum(model.ContractActive[m, c] for m in model.MONTHS) == model.ContractDuration[c]
    model.ContractDurationCon = Constraint(model.CONTRACTS, rule=contract_duration_rule)

    def min_contract_duration_rule(model, m, c):
        # Ensure the contract, once activated, has a minimum duration of 3 months
        return model.ContractDuration[c] >= 3 * model.ContractActive[m, c]
    model.MinContractDurationCon = Constraint(model.MONTHS, model.CONTRACTS, rule=min_contract_duration_rule)

    def contract_type_rule(model, c):
        return sum(model.ContractType[c, t] for t in model.ContractTypes) == 1
    model.ContractTypeConstraint = Constraint(model.CONTRACTS, rule=contract_type_rule)

        # Constraint to link ContractActive with ContractStart
    def contract_start_rule(model, m, c):
        if m == 0:
            return model.ContractActive[m, c] <= model.ContractStart[m, c]
        else:
            return model.ContractActive[m, c] <= model.ContractActive[m-1, c] + model.ContractStart[m, c]
    model.ContractStartCon = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_start_rule)

    # Ensure that the sum of ContractStart for each contract equals 1
    def contract_single_start_rule(model, c):
        return sum(model.ContractStart[m, c] for m in model.MONTHS) == 1
    model.ContractSingleStartCon = Constraint(model.CONTRACTS, rule=contract_single_start_rule)

    # Constraint to set the start price based on the contract's start month
    def set_contract_start_price_rule(model, c):
        return model.ContractStartPrice[c] == sum(NG_market_monthly[m] * model.ContractStart[m, c] for m in model.MONTHS)
    model.SetContractStartPrice = Constraint(model.CONTRACTS, rule=set_contract_start_price_rule)

    # Then, use this start price directly in your original constraint, simplifying it
    def contract_price_setting_rule(model, m, c):
        # This needs to be adjusted according to how prices are actually determined.
        return model.ContractPrice[m, c] == (
            model.ContractType[c, 'Fixed'] * model.ContractStartPrice[c] * 1.5 +
            model.ContractType[c, 'Indexed'] * NG_market_monthly[m] +
            model.ContractType[c, 'TakeOrPay'] * model.ContractStartPrice[c] * 1.1
        )
    model.ContractPriceSetting = Constraint(model.MONTHS, model.CONTRACTS, rule=contract_price_setting_rule)

    def min_fulfillment_rule(model, m, c):
        return model.ContractAmount[m, c] >= model.ContractActive[m, c] * (

            500 * model.ContractType[c, 'TakeOrPay'] +
            250 * (model.ContractType[c, 'Fixed'] + 
            model.ContractType[c, 'Indexed'])
            # Note: This assumes the same minimum for Fixed and Indexed contracts for simplicity
        )
    model.min_fulfillment_con = Constraint(model.MONTHS, model.CONTRACTS, rule=min_fulfillment_rule)

    def demand_fulfillment_rule(model, m):
        return sum(model.ContractAmount[m, c] * model.ContractActive[m, c] for c in model.CONTRACTS) >= nat_gas[m]
    model.demand_fulfillment_con = Constraint(model.MONTHS, rule=demand_fulfillment_rule)

    # ======== CO2 Constraints ========
    def total_emissions_per_interval_rule(model, i):
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[h] for h in range(i*intervals_time, (i+1)*intervals_time))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

        # ----------- Risk Constraints --------------
    risk_fixed = 1
    risk_indexed = 3
    risk_take_or_pay = 2

    def risk_score_rule(model):
        return model.risk_score == sum(
            (risk_fixed * model.ContractType[c, 'Fixed'] +
             risk_indexed * model.ContractType[c, 'Indexed'] +
             risk_take_or_pay * model.ContractType[c, 'TakeOrPay']) * 
             model.ContractDuration[c] 
            for c in model.CONTRACTS
        )
    model.RiskScoreCon = Constraint(rule=risk_score_rule)

    def risk_appetite_rule(model):
        return model.risk_score <= risk_appetite_threshold
    model.RiskAppetiteCon = Constraint(rule=risk_appetite_rule)

    # -------------- Objective Function --------------

    def objective_rule(model):
        fuel_cost_NG = sum(model.ContractAmount[m, c] * model.ContractPrice[m, c] for m in model.MONTHS for c in model.CONTRACTS)
        #fuel_cost_elec = sum(model.ContractAmountElec[m, c] * model.ContractPriceElec[m, c] for m in model.MONTHS for c in model.CONTRACTS)
        carbon_cost = 0
        carbon_sold = 0
        return fuel_cost_NG + carbon_cost - (carbon_sold) 
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # -------------- Solver --------------
    solver = SolverFactory("gurobi")
    solver.options['NonConvex'] = 2
    solver.options['TimeLimit'] = time_limit
    solver.options["Threads"]= 32
    solver.options["LPWarmStart"] = 2
    solver.options["FuncNonlinear"] = 1
    solver.options['mipgap'] = 0.01
    solver.solve(model, tee=True)

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



