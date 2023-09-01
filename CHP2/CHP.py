from pyomo.environ import *
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# Create a simple model
model = ConcreteModel()

# Time periods (e.g., hours in a day)
HOURS = list(range(24))
model.HOURS = Set(initialize=HOURS)

df = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\CHP2\heat_demand.xlsx")

# Sample energy profile for a chemical plant
#electricity_demand = {h: 500 + 200 * (h in range(8, 17)) for h in HOURS}  # Base demand + peak during working hours
#heat_demand = {h: 1000 + 600 * (h in range(8, 17)) for h in HOURS}  # Base demand + peak during working hours

electricity_demand = df["elec"].to_numpy()
heat_demand = df["heat"].to_numpy()
refrigeration_demand = {h: 50 for h in HOURS}  # Hypothetical refrigeration demand

# Decision Variables
model.CHP_capacity = Var(within=NonNegativeReals)  # Capacity of the CHP system
model.electricity_production = Var(model.HOURS, within=NonNegativeReals)  # Hourly electricity production
model.heat_production = Var(model.HOURS, within=NonNegativeReals)  # Hourly heat production
model.fuel_consumed = Var(model.HOURS, within=NonNegativeReals)  # Hourly fuel consumption
model.ramp_rate = Var(model.HOURS, within=NonNegativeReals)  # Hourly fuel consumption
model.heat_over_production = Var(model.HOURS, within=NonNegativeReals)
model.electricity_over_production = Var(model.HOURS, within=NonNegativeReals)
model.energy_ratio = Var(within=NonNegativeReals, bounds=(0.1, 0.4))

#Heat storage rules
storage_efficiency = 0.8  # Efficiency of heat storage (0.9 means 90% efficient)
withdrawal_efficiency = 0.7  # Efficiency of heat withdrawal (0.9 means 90% efficient)

max_storage_capacity = 1500  # Maximum heat storage capacity

model.heat_stored = Var(model.HOURS, within=NonNegativeReals)  # Heat stored at each hour
model.heat_withdrawn = Var(model.HOURS, within=NonNegativeReals)  # Heat withdrawn at each hour

#refrigeration rules
COP = 0.6 
model.refrigeration_produced = Var(model.HOURS, within=NonNegativeReals)  # Refrigeration produced at each hour
model.heat_used_for_cooling = Var(model.HOURS, within=NonNegativeReals)  # Heat used for cooling at each hour


capital_cost_per_kw = 10000  # Cost per kW of CHP capacity


operating_cost_per_kwh = 0.35  # Cost per kWh of electricity produced
fuel_cost_per_unit = 0.15  # Cost per unit of fuel
co2_per_unit_fuel = 20  # kg CO2 emitted per unit of fuel
kw_per_unit_fuel = 20  # kw of energy per unit fuel
max_co2_emissions = 100000  # Maximum allowable kg CO2 per day

max_ramp_rate = 100



# Constraints

# Energy balance constraints
def energy_ratio_rule(model, h):
    return model.heat_production[h] * model.energy_ratio == model.electricity_production[h]
model.energy_ratio_constraint = Constraint(model.HOURS, rule=energy_ratio_rule)

# CHP capacity constraint5
def capacity_rule(model, h):
    return (model.heat_production[h]) <= model.CHP_capacity
model.capacity_constraint = Constraint(model.HOURS, rule=capacity_rule)

# CO2 emissions constraint
def co2_emission_rule(model):
    return sum(co2_per_unit_fuel * model.fuel_consumed[h] for h in model.HOURS) <= max_co2_emissions
model.co2_constraint = Constraint(rule=co2_emission_rule)

# Fuel consumption rule
def fuel_consumed_rule(model, h):
    return kw_per_unit_fuel * model.fuel_consumed[h] >= model.heat_production[h] 
model.fuel_consumed_rule = Constraint(model.HOURS, rule=fuel_consumed_rule)

def ramp_up_rule(model, h):
    if h == 0:
        return Constraint.Skip  # Skip the first hour
    return model.heat_production[h] - model.heat_production[h-1] <= model.ramp_rate[h]
model.ramp_up_constraint = Constraint(model.HOURS, rule=ramp_up_rule)

def ramp_down_rule(model, h):
    if h == 0:
        return Constraint.Skip  # Skip the first hour
    return model.heat_production[h-1] - model.heat_production[h] <= model.ramp_rate[h]
model.ramp_down_constraint = Constraint(model.HOURS, rule=ramp_down_rule)

def max_ramp_rule(model, h):
    return model.ramp_rate[h] <= max_ramp_rate
model.max_ramp_constraint = Constraint(model.HOURS, rule=max_ramp_rule)

def heat_over_production_rule(model, h):
    return model.heat_production[h] - heat_demand[h] == model.heat_over_production[h]
model.heat_over_production_constraint = Constraint(model.HOURS, rule=heat_over_production_rule)

def electricity_over_production_rule(model, h):
    return model.electricity_production[h] - electricity_demand[h] == model.electricity_over_production[h]
model.electricity_over_production_constraint = Constraint(model.HOURS, rule=electricity_over_production_rule)


# Modified Heat Balance Constraint
def heat_balance_rule(model, h):
    return model.heat_production[h] + model.heat_withdrawn[h]*withdrawal_efficiency <= heat_demand[h] + model.heat_stored[h] + model.heat_used_for_cooling[h]
model.heat_balance = Constraint(model.HOURS, rule=heat_balance_rule)

# Heat Storage Dynamics Constraint
def storage_dynamics_rule(model, h):
    if h == 0:
        return Constraint.Skip  # Skip for the first hour, or set to initial storage level
    return model.heat_stored[h] == storage_efficiency * (model.heat_stored[h-1] -(model.heat_withdrawn[h]/withdrawal_efficiency) - (model.heat_used_for_cooling[h] / withdrawal_efficiency) + model.heat_production[h])
model.storage_dynamics = Constraint(model.HOURS, rule=storage_dynamics_rule)

# Heat Storage Capacity Constraint
def storage_capacity_rule(model, h):
    return model.heat_stored[h] <= max_storage_capacity
model.storage_capacity = Constraint(model.HOURS, rule=storage_capacity_rule)

def refrigeration_balance_rule(model, h):
    return model.refrigeration_produced[h] == refrigeration_demand[h]
model.refrigeration_balance = Constraint(model.HOURS, rule=refrigeration_balance_rule)

# Heat to Cooling Conversion Constraint
def heat_to_cooling_rule(model, h):
    return model.heat_used_for_cooling[h] == model.refrigeration_produced[h] / COP
model.heat_to_cooling = Constraint(model.HOURS, rule=heat_to_cooling_rule)


# Objective Function: Minimize total cost (capital cost + operating cost + fuel cost)
def objective_rule(model):
    capital_cost = capital_cost_per_kw * model.CHP_capacity
    fuel_cost = sum(fuel_cost_per_unit * model.fuel_consumed[h] for h in model.HOURS)
    demand = sum(model.electricity_over_production[h] + model.heat_over_production[h] for h in model.HOURS)
    return capital_cost + fuel_cost


model.objective = Objective(rule=objective_rule, sense=minimize)
from pyomo.opt import SolverFactory
# Create a solver
solver = SolverFactory("gurobi")
solver.options['NonConvex'] = 2
# Solve the model
solver.solve(model, tee=True)

# Print the results
print("Optimal CHP Capacity:", model.CHP_capacity())
print("Energy ratio", model.energy_ratio())
for h in model.HOURS:
    print(f"Hour {h}: Electricity Production = {model.electricity_production[h]()} kWh, Heat Production = {model.heat_production[h]()} kWh, Cold Production = {model.refrigeration_produced[h]()} kWh ({model.heat_used_for_cooling[h]()}), Fuel Consumed = {model.fuel_consumed[h]()} units")
    #print(f"Hour {h}: Heat = {model.heat_over_production[h]()} kWh, Electricity = {model.electricity_over_production[h]()} kWh, Stored = {model.heat_stored[h]()} kWh")


# Extract the model output into dictionaries or lists
electricity_production = [model.electricity_production[h]() for h in model.HOURS]
heat_production = [model.heat_production[h]() for h in model.HOURS]
cold_production = [model.refrigeration_produced[h]() for h in model.HOURS]
fuel_consumed = [model.fuel_consumed[h]() for h in model.HOURS]

electricity_over_production = [model.electricity_over_production[h]() for h in model.HOURS]
heat_over_production = [model.heat_over_production[h]() for h in model.HOURS]

