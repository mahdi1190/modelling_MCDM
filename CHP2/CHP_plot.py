from pyomo.environ import *
import pandas as pd
import dash
import dash_core_components as dcc
import os
import numpy as np
import dash_html_components as html
from dash.dependencies import Input, Output
# Initialize Dash app
last_mod_time = 0

HOURS = list(range(24))


demands = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\CHP2\heat_demand.xlsx")
markets = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\markets.xlsx")

electricity_demand = demands["elec"].to_numpy()*1.2
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()

electricity_market = markets["elec"].to_numpy()
electricity_market_sold = markets["elec_sold"].to_numpy()

NG_market = markets["elec"].to_numpy()
NG_market_sold = markets["elec_sold"].to_numpy()
app = dash.Dash(__name__)

# Layout of Dash app
app.layout = html.Div([
    dcc.Graph(id='electricity-graph'),
    dcc.Graph(id='heat-graph'),
    dcc.Graph(id='cold-graph'),
    dcc.Graph(id='heat-store-graph'),
    dcc.Graph(id='purchased_elec'),
    dcc.Interval(
        id='interval-component',
        interval=10*10000,  # Refresh every 10 seconds
        n_intervals=0
    )
])

# Define callback to update graphs
@app.callback(
    [Output('electricity-graph', 'figure'),
     Output('heat-graph', 'figure'),
     Output('cold-graph', 'figure'),
     Output('heat-store-graph', 'figure'),
     Output('purchased_elec', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n_intervals):
    global last_mod_time  # Declare as global to modify it

    # Check last modification time of the model file
    current_mod_time = os.path.getmtime(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\CHP2\CHP_plot.py")

    if current_mod_time > last_mod_time:
        last_mod_time = current_mod_time  # Update last modification time

        model = pyomomodel()

        electricity_production = [model.electricity_production[h]() for h in model.HOURS]
        heat_production = [model.heat_production[h]() for h in model.HOURS]
        cold_production = [model.refrigeration_produced[h]() for h in model.HOURS]
        fuel_consumed = [model.fuel_consumed[h]() for h in model.HOURS]
        heat_stored = [model.heat_stored[h]() for h in model.HOURS]
        heat_taken = [model.heat_withdrawn[h]() for h in model.HOURS]
        heat_plant = [model.heat_to_plant[h]() for h in model.HOURS]
        normal_heat_stored = 100*(np.array(heat_stored))/1000
        over_heat = [model.heat_over_production[h]() for h in model.HOURS]
        
        purchased_electricity = [model.purchased_electricity[h]() for h in model.HOURS]
        over_electricity = [model.electricity_over_production[h]() for h in model.HOURS]
        cooling_elec = [model.elec_used_for_cooling[h]() for h in model.HOURS]
        cooling_heat = [model.heat_used_for_cooling[h]() for h in model.HOURS]

        # Create figures based on these results

        base_layout_template = {
            'template': 'plotly_dark',
            'font': {'family': "Courier New, monospace", 'size': 12, 'color': "RebeccaPurple", 'weight': 'bold'},
            'xaxis': {'title': 'Hours', 'weight': 'bold'},
            'yaxis': {'title': 'kWh', 'weight': 'bold'},
            'title_font': {'size': 24, 'family': 'Arial, sans-serif', 'weight': 'bold'},
        }
        electricity_layout = base_layout_template.copy()
        electricity_layout.update({'title': 'Electricity Demand and Production', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': 'kWh'}})

        heat_layout = base_layout_template.copy()
        heat_layout.update({'title': 'Heat Demand and Production', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': 'kWh'}})

        cold_layout = base_layout_template.copy()
        cold_layout.update({'title': 'Cold Demand and Production', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': 'kWh'}})

        
        storage_layout = base_layout_template.copy()
        storage_layout.update({'title': 'Normalised Heat Storage', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': '%'}})

        fridge_layout = base_layout_template.copy()
        fridge_layout.update({'title': 'Normalised Heat Storage', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': '%'}})

        

        electricity_figure = {
            'data': [
                {'x': list(range(24)), 'y': [electricity_demand[h] for h in list(range(24))], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(range(24)), 'y': electricity_production, 'type': 'line', 'name': 'Production'},
                {'x': list(range(24)), 'y': [purchased_electricity[h] for h in list(range(24))], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Purchased Electricity'},
                {'x': list(range(24)), 'y': [over_electricity[h] for h in list(range(24))], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Over-Produced Electricity'}
            ],
            'layout': electricity_layout,
        }

        heat_figure = {
            'data': [
                {'x': list(range(24)), 'y': [heat_demand[h] for h in list(range(24))], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(range(24)), 'y': heat_production, 'type': 'line', 'name': 'Production'},
                {'x': list(range(24)), 'y': heat_stored, 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
                {'x': list(range(24)), 'y': over_heat, 'type': 'line', 'name': 'Over-Production'},
            ],
            'layout': heat_layout,
        }

        cold_figure = {
            'data': [
                {'x': list(range(24)), 'y': [refrigeration_demand[h] for h in list(range(24))], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(range(24)), 'y': cold_production, 'type': 'line', 'name': 'Production'},
                {'x': list(range(24)), 'y': cooling_elec, 'type': 'line', 'name': 'Cooling Electricity'},
                {'x': list(range(24)), 'y': cooling_heat, 'type': 'line', 'name': 'Cooling Heat'},
            ],
            'layout': cold_layout,
        }
        heat_storage = {
            'data': [
                {'x': list(range(24)), 'y': normal_heat_stored, 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
            ],
            'layout': storage_layout,
        }

        purchased_elec = {
            'data': [
                {'x': list(range(24)), 'y': electricity_market, 'type': 'line', 'name': 'Market'},
            ],
            'layout': electricity_layout,
        }
        # Return the updated figures
        return electricity_figure, heat_figure, cold_figure, heat_storage, purchased_elec
    else:
        raise dash.exceptions.PreventUpdate
















# Create a simple model
# Create a simple model
def pyomomodel():
    model = ConcreteModel()

    # Time periods (e.g., hours in a day)
    HOURS = list(range(24))
    model.HOURS = Set(initialize=HOURS)

    # Decision Variables
    model.CHP_capacity = Var(within=NonNegativeReals)  # Capacity of the CHP system
    model.electricity_production = Var(model.HOURS, within=NonNegativeReals)  # Hourly electricity production
    model.heat_production = Var(model.HOURS, within=NonNegativeReals)  # Hourly heat production
    model.fuel_consumed = Var(model.HOURS, within=NonNegativeReals)  # Hourly fuel consumption
    model.ramp_rate = Var(model.HOURS, within=NonNegativeReals)  # Hourly fuel consumption
    model.heat_over_production = Var(model.HOURS, within=NonNegativeReals)
    model.electricity_over_production = Var(model.HOURS, within=NonNegativeReals)
    model.energy_ratio = Var(within=NonNegativeReals, bounds=(0.1, 0.4))

    model.heat_to_plant = Var(model.HOURS, within=NonNegativeReals)# Heat supplied to plant
    model.elec_to_plant = Var(model.HOURS, within=NonNegativeReals)# Elec supplied to plant


    model.useful_heat = Var(model.HOURS, within=NonNegativeReals)# Elec supplied to plant
    model.useful_elec = Var(model.HOURS, within=NonNegativeReals)# Elec supplied to plant

    model.purchased_electricity = Var(model.HOURS, within=NonNegativeReals)
    #Heat storage rules
    storage_efficiency = 0.8  # Efficiency of heat storage (0.9 means 90% efficient)
    withdrawal_efficiency = 0.8  # Efficiency of heat withdrawal (0.9 means 90% efficient)

    max_storage_capacity = 1000  # Maximum heat storage capacity

    model.heat_stored = Var(model.HOURS, within=NonNegativeReals)  # Heat stored at each hour
    model.heat_withdrawn = Var(model.HOURS, within=NonNegativeReals)  # Heat withdrawn at each hour

    #refrigeration rules
    COP_h = 2
    COP_e = 0.01
    model.refrigeration_produced = Var(model.HOURS, within=NonNegativeReals)  # Refrigeration produced at each hour
    model.heat_used_for_cooling = Var(model.HOURS, within=NonNegativeReals)  # Heat used for cooling at each hour#
    model.elec_used_for_cooling = Var(model.HOURS, within=NonNegativeReals)  # Elec used for cooling at each hour


    capital_cost_per_kw = 1000  # Cost per kW of CHP capacity


    operating_cost_per_kwh = 0.35  # Cost per kWh of electricity produced
    fuel_cost_per_unit = 0.16  # Cost per unit of fuel
    co2_per_unit_fuel = 0.001  # kg CO2 emitted per unit of fuel # kw of energy per unit fuel
    max_co2_emissions = 1000000000  # Maximum allowable kg CO2 per day
    max_heat_over_production = 1000
    max_ramp_rate = 300
    # Constraints
    # Electricity Balance Constraint
    def electricity_balance_rule(model, h):
        return model.electricity_production[h] >= model.useful_elec[h] + model.electricity_over_production[h] - model.purchased_electricity[h]
    model.electricity_balance = Constraint(model.HOURS, rule=electricity_balance_rule)

    # Modified Heat Balance Constraint
    def heat_balance_rule(model, h):
        return model.heat_production[h] >= model.heat_over_production[h] + model.useful_heat[h] - (model.heat_withdrawn[h]*withdrawal_efficiency)
    model.heat_balance = Constraint(model.HOURS, rule=heat_balance_rule)

    def heat_demand_balance(model, h):
        return model.heat_to_plant[h] == heat_demand[h]
    model.heat_demand_rule = Constraint(model.HOURS, rule=heat_demand_balance)

    def elec_demand_balance(model, h):
        return model.elec_to_plant[h] + model.purchased_electricity[h] == electricity_demand[h]
    model.elec_demand_rule = Constraint(model.HOURS, rule=elec_demand_balance)

    def useful_elec_rule(model, h):
        return model.useful_elec[h] == model.elec_to_plant[h] + (model.elec_used_for_cooling[h] / COP_e) 
    model.useful_elec_constraint = Constraint(model.HOURS, rule=useful_elec_rule)

    def elec_over_production_rule(model, h):
        return model.electricity_over_production[h] == model.electricity_production[h] - model.useful_elec[h]
    model.elec_over_production_constraint = Constraint(model.HOURS, rule=elec_over_production_rule)

    # Energy balance constraints
    def energy_ratio_rule(model, h):
        return model.electricity_production[h] == model.heat_production[h] * model.energy_ratio
    model.energy_ratio_constraint = Constraint(model.HOURS, rule=energy_ratio_rule)

    # Constraint to limit heat overproduction
    def heat_over_production_rule(model, h):
        return model.heat_over_production[h] == model.heat_production[h] - model.useful_heat[h]
    model.heat_over_production_constraint = Constraint(model.HOURS, rule=heat_over_production_rule)

    #Constraint to define useful heat
    def useful_heat_rule(model, h):
        return model.useful_heat[h] == model.heat_to_plant[h] + (model.heat_stored[h] * storage_efficiency) + (model.heat_used_for_cooling[h] / COP_h)
    model.useful_heat_constraint = Constraint(model.HOURS, rule=useful_heat_rule)


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
        return NG_market[h] * model.fuel_consumed[h] == model.heat_production[h] 
    model.fuel_consumed_rule = Constraint(model.HOURS, rule=fuel_consumed_rule)

    def initial_heat_stored_rule(model):
        return model.heat_stored[0] == 0

    model.initial_heat_stored_constraint = Constraint(rule=initial_heat_stored_rule)

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

    # Heat Storage Dynamics Constraint
    def storage_dynamics_rule(model, h):
        if h == 0:
            return Constraint.Skip  # Skip for the first hour, or set to initial storage level
        return model.heat_stored[h] == storage_efficiency * (model.heat_stored[h-1] -(model.heat_withdrawn[h]/withdrawal_efficiency) - ((model.heat_used_for_cooling[h]/COP_h) / withdrawal_efficiency) + model.heat_over_production[h])
    model.storage_dynamics = Constraint(model.HOURS, rule=storage_dynamics_rule)

    # Heat Storage Capacity Constraint
    def storage_capacity_rule(model, h):
        return model.heat_stored[h] <= max_storage_capacity
    model.storage_capacity = Constraint(model.HOURS, rule=storage_capacity_rule)

    def refrigeration_balance_rule(model, h):
        return model.refrigeration_produced[h] == (model.elec_used_for_cooling[h] * COP_e) + (model.heat_used_for_cooling[h] * COP_h)
    model.refrigeration_balance = Constraint(model.HOURS, rule=refrigeration_balance_rule)

    def refrigeration_demand_rule(model, h):
        return model.refrigeration_produced[h] == refrigeration_demand[h]
    model.refrigeration_demand_con = Constraint(model.HOURS, rule=refrigeration_demand_rule)

    # Objective Function: Minimize total cost (capital cost + operating cost + fuel cost)
    def objective_rule(model):
        capital_cost = capital_cost_per_kw * model.CHP_capacity
        fuel_cost = sum(NG_market[h] * model.fuel_consumed[h] for h in model.HOURS)
        elec_cost = sum(model.purchased_electricity[h] * electricity_market[h] for h in model.HOURS)
        elec_saved = sum(model.electricity_over_production[h] * electricity_market_sold[h] for h in model.HOURS)
        heat_saved = sum((NG_market_sold[h] * model.heat_over_production[h]) for h in model.HOURS)
        return capital_cost + fuel_cost + elec_cost - (elec_saved + heat_saved)


    model.objective = Objective(rule=objective_rule, sense=minimize)
    from pyomo.opt import SolverFactory
    # Create a solver
    solver = SolverFactory("gurobi")
    solver.options['NonConvex'] = 2
    # Solve the model
    solver.solve(model, tee=True)

    return model

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)