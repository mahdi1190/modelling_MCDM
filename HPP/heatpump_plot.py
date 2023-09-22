from pyomo.environ import *
from pyomo.environ import Param, NonNegativeReals
import pandas as pd
import locale
locale.setlocale( locale.LC_ALL, '' )
import os
import numpy as np
import dash_html_components as html
from dash import dcc, html, Input, Output, State
from dash.dependencies import Input, Output


demands = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\data\demands.xlsx")
markets = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\data\markets.xlsx")

electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()

electricity_market = markets["elec"].to_numpy()
electricity_market_sold = markets["elec_sold"].to_numpy()

carbon_market = markets["carbon"].to_numpy() / 1000

NG_market = markets["nat_gas"].to_numpy()
heat_market_sold = markets["nat_gas_sold"].to_numpy()

H2_market = markets["hydrogen"].to_numpy()

BM_market = markets["biomass"].to_numpy()

stream_energy = 11 #MW

steam_flow = 10000#kg/s
stream_temperature = 373 #Kelvin
steam_temp = 298
Tc = stream_temperature
Td = 20
year_index=0

# Define dummy data
time_periods = np.arange(1, 6)  # considering 5 time periods
heat_pumps = np.arange(1, 4)  # considering 3 different types of heat pumps
chemical_plants = np.arange(1, 4)  # considering 3 different chemical plants

# Randomly generate costs for electricity, gas, and carbon for each time period
electricity_cost = {t: np.random.uniform(50, 100) for t in time_periods}
gas_cost = {t: np.random.uniform(20, 50) for t in time_periods}
carbon_cost = {t: np.random.uniform(10, 30) for t in time_periods}

# Randomly generate heat recovery for each pump and plant combination
heat_recovery = {(p, c): np.random.uniform(500, 1000) for p in heat_pumps for c in chemical_plants}

# Randomly generate cost for each heat pump
pump_cost = {p: np.random.uniform(1000, 5000) for p in heat_pumps}

# Randomly generate the expected lifetime for each pump
lifetime = {p: np.random.randint(3, 6) for p in heat_pumps}  # lifetime between 3 to 5 years



df = pd.read_csv(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\HPP\heat_pump_data.csv")

cost_data = df['Cost'].to_numpy()
capacity_data = df['Capacity'].to_numpy()
waste_data = df['Waste'].to_numpy()
max_temp_lift_data = df['Lift'].to_numpy()

waste_heat = waste_data
max_waste = max(waste_data)

CP_hp = 2
CP_steam = 2

target_temp = 700

MAX_TEMP_LIFT = 90

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    # Time periods (e.g., hours in a day)
    total_time = 11
    MONTHS = list(range(total_time))
    model.MONTHS = Set(initialize=MONTHS)

    no_pumps = 3
    model.PUMPS = Set(initialize=range(no_pumps))
    a = 1000  # Example value, adjust as needed
    b = 1   # Example value, adjust as needed
    c = 2    # Example value, adjust as needed

    model.Th = Var(model.PUMPS, within=NonNegativeReals, doc='Maximum Temperature of hot stream for each pump')
    model.COP = Var(model.PUMPS, within=NonNegativeReals, doc='Coefficient of Performance for each pump')

    # -------------- New Decision Variables related to Heat Pumps --------------

    model.heat_pump_installed = Var(model.PUMPS, within=Binary)  # Indicates if a particular heat pump is installed
    model.heat_pump_capacity_installed = Var(model.PUMPS, within=NonNegativeReals)  # Capacity of each installed heat pump
    model.heat_pump_operation = Var(model.MONTHS, model.PUMPS, within=NonNegativeReals)  # Amount of waste heat recovered by each heat pump every hour
    model.heat_pump_monthly_capacity = Param(model.MONTHS, within=NonNegativeReals, mutable=True, initialize=capacity_data) 
    model.electricity_consumption = Var(model.MONTHS, model.PUMPS, within=NonNegativeReals)  # Electricity consumption of each heat pump every hour
    
    model.T_after_heat_exchange = Var(model.MONTHS, within=NonNegativeReals)
    # -------------- Constraints related to Heat Pumps --------------
    def heat_pump_operation_rule(model, m, p):
        return model.heat_pump_operation[m, p] <= model.heat_pump_installed[p] * model.heat_pump_capacity_installed[p]
    model.heat_pump_operation_con = Constraint(model.MONTHS, model.PUMPS, rule=heat_pump_operation_rule)

    # Ensuring total installed capacity does not exceed total waste heat capacity
    def total_capacity_rule(model):
        return sum(model.heat_pump_capacity_installed[p] for p in model.PUMPS) <= max_waste
    model.total_capacity_con = Constraint(rule=total_capacity_rule)

    def Th_greater_than_Tc_rule(model, p):
        return model.Th[p] >= Tc + 5
    model.Th_greater_Tc_constraint = Constraint(model.PUMPS, rule=Th_greater_than_Tc_rule)

    def max_temp_lift_rule(model, p):
        return model.Th[p] - Tc <= MAX_TEMP_LIFT
    model.max_temp_lift_con = Constraint(model.PUMPS, rule=max_temp_lift_rule)

    def max_capacity_per_pump_rule(model, m, p):
        return model.heat_pump_capacity_installed[p] <= model.heat_pump_monthly_capacity[m] * model.heat_pump_installed[p]
    model.max_capacity_per_pump_constraint = Constraint(model.MONTHS, model.PUMPS, rule=max_capacity_per_pump_rule)

    def individual_COP_rule(model, p):
        return model.Th[p] == model.COP[p] * (model.Th[p] - Tc)  # Modify this as per your requirement
    model.individual_COP_con = Constraint(model.PUMPS, rule=individual_COP_rule)

    def electricity_to_heat_output_rule(model, m, p):
        return model.heat_pump_operation[m, p] <= model.COP[p] * model.electricity_consumption[m, p]
    model.electricity_to_heat_output_con = Constraint(model.MONTHS, model.PUMPS, rule=electricity_to_heat_output_rule)

    def waste_exchange_rule(model, m):
        return sum(model.heat_pump_operation[m, p] for p in model.PUMPS) <= waste_heat[m]
    model.waste_exchange_con = Constraint(model.MONTHS, rule=waste_exchange_rule)

    electricity_price_per_kw = 0.3 # Placeholder value

    def heat_exchange_rule(model, m):
        return CP_hp*sum(model.heat_pump_operation[m, p] for p in model.PUMPS)*(((sum(model.Th[p] for p in model.PUMPS))/no_pumps) - (steam_temp+10)) == CP_steam*steam_flow*(model.T_after_heat_exchange[m] - steam_temp)
    model.heat_exchange_con = Constraint(model.MONTHS, rule=heat_exchange_rule)

    def T_less_than_steam_rule(model, m):
        return ((sum(model.Th[p] for p in model.PUMPS))/no_pumps) - model.T_after_heat_exchange[m] >= 5
    model.T_less_than_steam_rule_constraint = Constraint(model.MONTHS, rule=T_less_than_steam_rule)

    def heat_pump_objective_rule(model):
        heat_pump_installation_cost = sum(a * model.heat_pump_capacity_installed[p] + c*model.Th[p]*model.heat_pump_installed[p] for p in model.PUMPS)
        electricity_cost = sum(model.electricity_consumption[m, p] * electricity_price_per_kw for m in model.MONTHS for p in model.PUMPS)
        heat_cost = CP_hp*sum(steam_temp - model.T_after_heat_exchange[m] for m in model.MONTHS)*1E5
        return heat_pump_installation_cost + electricity_cost + heat_cost
    model.objective = Objective(rule=heat_pump_objective_rule, sense=minimize)

    # -------------- Solver --------------
    solver = SolverFactory("gurobi")
    solver.options['NonConvex'] = 2
    solver.solve(model, tee=True)

    return model

def print_selected_heat_pumps(model):
    selected_pumps = [p for p in model.PUMPS if model.heat_pump_installed[p].value > 0.1]  # Checking for values > 0.5 to account for potential floating-point inaccuracies
    print(f"Total number of heat pumps selected: {len(selected_pumps)}")
    for p in model.PUMPS:
        print(f"Heat Pump {p}: Installed = {model.heat_pump_installed[p].value}, Capacity = {model.heat_pump_capacity_installed[p].value}, Th = {(model.Th[p].value)}, COP = {model.COP[p].value}")
        for m in model.MONTHS:
            print(f"Heat Pump {model.heat_pump_operation[m, p].value} Leccy = {model.electricity_consumption[m, p].value} Temp: {model.T_after_heat_exchange[m].value}" )
    print(max_waste)

if __name__ == "__main__":
    model = pyomomodel()

    print_selected_heat_pumps(model)