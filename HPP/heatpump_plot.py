from pyomo.environ import *
from pyomo.environ import Param, NonNegativeReals
import pandas as pd
import dash
import dash_core_components as dcc
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

stream_flow = 10 #kg/s
stream_temperature = 350 #Kelvin

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

# Randomly generate available waste heat for each chemical plant
available_waste_heat = {c: np.random.uniform(2000, 5000) for c in chemical_plants}

dummy_data = {
    "time_periods": time_periods,
    "heat_pumps": heat_pumps,
    "chemical_plants": chemical_plants,
    "electricity_cost": electricity_cost,
    "gas_cost": gas_cost,
    "carbon_cost": carbon_cost,
    "heat_recovery": heat_recovery,
    "pump_cost": pump_cost,
    "lifetime": lifetime,
    "available_waste_heat": available_waste_heat
}

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    # Time periods (e.g., hours in a day)
    total_time = 360
    MONTHS = list(range(total_time))
    model.MONTHS = Set(initialize=MONTHS)

    model.PUMPS = Set(initialize=range(3))
    a = 1000  # Example value, adjust as needed
    b = 0.8   # Example value, adjust as needed
    c = 50    # Example value, adjust as needed

    MAX_CAPACITY_PER_PUMP = 1000
    MIN_CAPACITY_PER_PUMP = 200  # Define the minimum capacity per pump
    MAX_TEMP_LIFT = 90
    no_intervals = 3
    intervals_time = int(total_time / no_intervals)
    INTERVALS = list(range(no_intervals))  # Four 6-hour intervals in a day

    waste_heat = 2100
    # -------------- New Parameters related to Heat Pumps --------------
    model.heat_pump_efficiency = Param(within=NonNegativeReals, mutable=True, default=1.0),  # Efficiency of heat pump
    model.heat_pump_cost = Param(model.PUMPS, within=NonNegativeReals, mutable=True, default=1.0) # Cost of installing a heat pump
    model.waste_heat_available = Param(model.MONTHS, within=NonNegativeReals, mutable=True, default=1.0) # Waste heat available each hour
    model.heat_pump_capacity = Param(model.PUMPS, within=NonNegativeReals, mutable=True, default=1.0) # Maximum amount of waste heat a heat pump can recover

    model.Th = Var(model.PUMPS, within=Reals, doc='Maximum Temperature of hot stream for each pump')
    model.COP = Var(model.PUMPS, within=Reals, doc='Coefficient of Performance for each pump')
    model.MW = Var(within = Reals, doc = 'Energy Rate of Stream') 

    # -------------- New Decision Variables related to Heat Pumps --------------
    model.no_of_heat_pumps = Var(within=NonNegativeIntegers) # Number of heat pumps installed

    model.heat_pump_installed = Var(model.PUMPS, within=Binary)  # Indicates if a particular heat pump is installed
    model.heat_pump_capacity_installed = Var(model.PUMPS, within=NonNegativeReals)  # Capacity of each installed heat pump
    model.heat_pump_operation = Var(model.MONTHS, model.PUMPS, within=NonNegativeReals)  # Amount of waste heat recovered by each heat pump every hour

    model.electricity_consumption = Var(model.MONTHS, model.PUMPS, within=NonNegativeReals)  # Electricity consumption of each heat pump every hour

    # -------------- Constraints related to Heat Pumps --------------
    def heat_pump_operation_rule(model, m, p):
        return model.heat_pump_operation[m, p] <= model.heat_pump_installed[p] * model.heat_pump_capacity_installed[p]
    model.heat_pump_operation_con = Constraint(model.MONTHS, model.PUMPS, rule=heat_pump_operation_rule)

    # Ensuring total installed capacity does not exceed total waste heat capacity
    def total_capacity_rule(model):
        return sum(model.heat_pump_capacity_installed[p] for p in model.PUMPS) <= waste_heat
    model.total_capacity_con = Constraint(rule=total_capacity_rule)

    def max_capacity_per_pump_rule(model, p):
        return model.heat_pump_capacity_installed[p] <= MAX_CAPACITY_PER_PUMP * model.heat_pump_installed[p]
    model.max_capacity_per_pump_constraint = Constraint(model.PUMPS, rule=max_capacity_per_pump_rule)

    def individual_COP_rule(model, p):
        return model.COP[p] * (model.Th[p] - Tc) == model.Th[p]  # Modify this as per your requirement
    model.individual_COP_con = Constraint(model.PUMPS, rule=individual_COP_rule)

    def max_temp_lift_rule(model, p):
        return model.Th[p] - Tc <= MAX_TEMP_LIFT
    model.max_temp_lift_con = Constraint(model.PUMPS, rule=max_temp_lift_rule)

    def cost_curve_rule(model, p):
        return model.heat_pump_cost[p] == a * model.heat_pump_capacity[p]**(-b) + c * model.Th[p]
    model.cost_curve = Constraint(model.PUMPS, rule=cost_curve_rule)

    def electricity_to_heat_output_rule(model, m, p):
        return model.heat_pump_operation[m, p] == model.COP[p] * model.electricity_consumption[m, p]
    model.electricity_to_heat_output_con = Constraint(model.MONTHS, model.PUMPS, rule=electricity_to_heat_output_rule)

    def min_capacity_per_pump_rule(model, p):
        return model.heat_pump_capacity_installed[p] >= MIN_CAPACITY_PER_PUMP * model.heat_pump_installed[p]
    model.min_capacity_per_pump_constraint = Constraint(model.PUMPS, rule=min_capacity_per_pump_rule)

    heat_sold_price_per_kw = 37.04 # Placeholder value
    electricity_price_per_kw = 2 # Placeholder value

    def heat_pump_objective_rule(model):
        heat_pump_installation_cost = sum(model.heat_pump_installed[p] * model.heat_pump_cost[p] * model.heat_pump_capacity_installed[p] for p in model.PUMPS)
        heat_recovered_benefit = sum(model.heat_pump_operation[m, p] * heat_sold_price_per_kw for m in model.MONTHS for p in model.PUMPS)
        electricity_cost = sum(model.electricity_consumption[m, p] * electricity_price_per_kw for m in model.MONTHS for p in model.PUMPS)
        return heat_pump_installation_cost - heat_recovered_benefit + electricity_cost
    model.objective = Objective(rule=heat_pump_objective_rule, sense=minimize)

    # -------------- Solver --------------
    solver = SolverFactory("gurobi")
    solver.options['NonConvex'] = 2
    solver.solve(model, tee=True)

    return model

def print_selected_heat_pumps(model):
    selected_pumps = [p for p in model.PUMPS if model.heat_pump_installed[p].value > 0.5]  # Checking for values > 0.5 to account for potential floating-point inaccuracies
    print(f"Total number of heat pumps selected: {len(selected_pumps)}")
    for p in selected_pumps:
        print(f"Heat Pump {p}: Installed with capacity {model.heat_pump_capacity_installed[p].value} kW")

if __name__ == "__main__":
    model = pyomomodel()

    print_selected_heat_pumps(model)
