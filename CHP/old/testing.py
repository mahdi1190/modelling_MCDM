# Importing the required modules from Pyomo
from pyomo.environ import *

# Initialize the model
model = ConcreteModel()

# Time horizon
T = [i for i in range(24)]
model.T = Set(initialize=T)

# Fuels (example: gas, oil, biomass)
Fuels = ['gas', 'oil', 'biomass']
model.Fuels = Set(initialize=Fuels)

# Parameters for fuels: costs and efficiencies
fuel_costs = {'gas': 5, 'oil': 6, 'biomass': 4}  # Example costs per unit
fuel_efficiencies_e = {'gas': 0.45, 'oil': 0.4, 'biomass': 0.35}  # Example electric efficiencies
fuel_efficiencies_h = {'gas': 0.5, 'oil': 0.45, 'biomass': 0.4}  # Example thermal efficiencies

# Maximum electric power
P_e_max = 1000  # MW

# Revenue rates (in $/MWh)
R_e = 50  # Electric power
R_h = 30  # Thermal power

# Operational costs (in $/MWh)
C_op_e = 10  # Electric power
C_op_h = 5   # Thermal power

# Variables
model.P_e = Var(model.T, within=NonNegativeReals)  # Electric power
model.P_h = Var(model.T, within=NonNegativeReals)  # Thermal power
model.is_maintenance = Var(model.T, within=Binary)  # 1 if under maintenance, 0 otherwise
model.is_startup = Var(model.T, within=Binary)  # 1 if starting up, 0 otherwise
model.is_shutdown = Var(model.T, within=Binary)  # 1 if shutting down, 0 otherwise
model.fuel_used = Var(model.Fuels, model.T, within=NonNegativeReals)  # Amount of each fuel used

# Objective function
def objective_rule(model):
    total_revenue = sum((R_e * model.P_e[t] + R_h * model.P_h[t]) for t in model.T)
    total_fuel_cost = sum(model.fuel_used[f, t] * fuel_costs[f] for f in model.Fuels for t in model.T)
    total_op_cost = sum((C_op_e * model.P_e[t] + C_op_h * model.P_h[t]) for t in model.T)
    return total_revenue - total_fuel_cost - total_op_cost

model.obj = Objective(rule=objective_rule, sense=minimize)

# Constraints

# Maintenance constraint
def maintenance_rule(model, t):
    return model.P_e[t] <= P_e_max * (1 - model.is_maintenance[t])

model.maintenance_constraint = Constraint(model.T, rule=maintenance_rule)

# Startup and shutdown constraint
def startup_shutdown_rule(model, t):
    if t == 0:
        return Constraint.Skip
    return model.is_startup[t] <= 1 - model.is_shutdown[t-1]

model.startup_shutdown_constraint = Constraint(model.T, rule=startup_shutdown_rule)

# Fuel-based efficiency constraint for electric power
def efficiency_electric_fuel_rule(model, t):
    return model.P_e[t] <= sum(model.fuel_used[f, t] * fuel_efficiencies_e[f] for f in model.Fuels)

model.efficiency_electric_fuel = Constraint(model.T, rule=efficiency_electric_fuel_rule)

# Fuel-based efficiency constraint for thermal power
def efficiency_thermal_fuel_rule(model, t):
    return model.P_h[t] <= sum(model.fuel_used[f, t] * fuel_efficiencies_h[f] for f in model.Fuels)

model.efficiency_thermal_fuel = Constraint(model.T, rule=efficiency_thermal_fuel_rule)

# Solve the model
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)


# Output results
output = []
for t in model.T:
    time_output = {
        "Time": t,
        "P_e": model.P_e[t].value,
        "P_h": model.P_h[t].value,
        "Fuels": {}
    }
    for f in model.Fuels:
        time_output["Fuels"][f] = model.fuel_used[f, t].value
    output.append(time_output)