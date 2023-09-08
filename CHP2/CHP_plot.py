from pyomo.environ import *
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
# Initialize Dash app
last_mod_time = 0

HOURS = list(range(24))


demands = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\CHP2\heat_demand.xlsx")
markets = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\markets.xlsx")

electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()

electricity_market = markets["elec"].to_numpy()
electricity_market_sold = markets["elec_sold"].to_numpy()

carbon_market = markets["carbon"].to_numpy()

heat_market = markets["nat_gas"].to_numpy()
heat_market_sold = markets["nat_gas_sold"].to_numpy()


app = dash.Dash(__name__)

# Initial layout
app.layout = html.Div([
    # Controls for graph size
    html.Div([
        dcc.Slider(
            id='height-slider',
            min=300,
            max=800,
            step=50,
            value=400
        ),
        html.Label('Adjust Graph Height')
    ]),
    
    # Text elements at the top
    html.Div([
        html.Div(id='total-purchased-electricity', style={'fontSize': 24}),
        html.Div(id='total-heat-cost', style={'fontSize': 24}),
        html.Div(id='final-chp-capacity', style={'fontSize': 24}),
        html.Div(id='energy-ratio', style={'fontSize': 24}),
        html.Div(id='model-cost', style={'fontSize': 24}),
        html.Div(id='credits', style={'fontSize': 24}),
    ], style={'display': 'flex', 'justifyContent': 'space-around'}),
    
    # Graphs
    dcc.Graph(id='electricity-graph'),
    dcc.Graph(id='heat-graph'),
    dcc.Graph(id='cold-graph'),
    dcc.Graph(id='heat-store-graph'),
    dcc.Graph(id='purchased_elec'),
    dcc.Graph(id='eff-graph'),  # New graph for eff
    dcc.Graph(id='carbon-credits'),  # New empty graph 1
    dcc.Graph(id='empty-graph2'),  # New empty graph 2
    dcc.Graph(id='empty-graph3'),  # New empty graph 3
    dcc.Graph(id='empty-graph4'),  # New empty graph 4
    dcc.Graph(id='empty-graph5'),  # New empty graph 5
    
    # Interval component
    dcc.Interval(
        id='interval-component',
        interval=10 * 10000,  # Refresh every 10 seconds
        n_intervals=0
    )
])

# Callback for updating graph sizes
@app.callback(
    Output('electricity-graph', 'style'),
    Output('heat-graph', 'style'),
    Output('cold-graph', 'style'),
    Output('heat-store-graph', 'style'),
    Output('purchased_elec', 'style'),
    Output('eff-graph', 'style'),  # New graph for eff
    Output('carbon-credits', 'style'),  # New empty graph 1
    Output('empty-graph2', 'style'),  # New empty graph 2
    Output('empty-graph3', 'style'),  # New empty graph 3
    Output('empty-graph4', 'style'),  # New empty graph 4
    Output('empty-graph5', 'style'),  # New empty graph 5
    [Input('height-slider', 'value')]
)
def update_graph_height(height_value):
    style = {'height': f'{height_value}px'}
    return style, style, style, style, style, style, style, style, style, style, style  # Add style for new graphs


# Callback for updating graphs and figures (existing callback)
@app.callback(
    Output('electricity-graph', 'figure'),
    Output('heat-graph', 'figure'),
    Output('cold-graph', 'figure'),
    Output('heat-store-graph', 'figure'),
    Output('purchased_elec', 'figure'),
    Output('eff-graph', 'figure'),
    Output('carbon-credits', 'figure'),
    Output('total-purchased-electricity', 'children'),
    Output('total-heat-cost', 'children'),
    Output('final-chp-capacity', 'children'),
    Output('energy-ratio', 'children'),
    Output('model-cost', 'children'),
    Output('credits', 'children'),
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

        total_purchased_electricity = sum(model.purchased_electricity[h].value for h in model.HOURS)
        total_heat_cost = sum(model.fuel_consumed[h].value * heat_market[h] for h in model.HOURS)
        final_chp_capacity = model.CHP_capacity.value
        energy_ratio = model.energy_ratio.value
        model_cost = model.objective()

        eff1 = [model.electrical_efficiency[h]() for h in model.HOURS]
        eff2 = [model.thermal_efficiency[h]() for h in model.HOURS]

        carbon_buy = [model.credits_purchased[h]() for h in model.INTERVALS]
        carbon_sell = [model.credits_sold[h]() for h in model.INTERVALS]
        carbon_held = [model.credits_held[h]() for h in model.INTERVALS]
        carbon_earn = [model.credits_earned[h]() for h in model.INTERVALS]
        carbon = [model.total_emissions_per_interval[h]() for h in model.INTERVALS]
        carbon_diff = [model.emissions_difference[h]() for h in model.INTERVALS]
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

        market_layout = base_layout_template.copy()
        market_layout.update({'title': 'Energy Market', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': 'p/kWh'}})

        eff_layout = base_layout_template.copy()
        eff_layout.update({'title': 'Efficiency', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': 'Efficiency'}})

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
                {'x': list(range(24)), 'y': electricity_market, 'type': 'line', 'name': 'Purchased Electricity', 'line': {'color': 'blue'}},
                {'x': list(range(24)), 'y': heat_market, 'type': 'line', 'name': 'Purchased Fuel', 'line': {'color': 'red'}},
                {'x': list(range(24)), 'y': heat_market_sold, 'line': {'width': 3, 'dash': 'dash', 'color': 'red'}, 'name': 'Sold Heat'},
                {'x': list(range(24)), 'y': electricity_market_sold, 'line': {'width': 3, 'dash': 'dash', 'color': 'blue'}, 'name': 'Sold Electricity'},
            ],
            'layout': market_layout,
        }
        carbon_credits_figure = {
        'data': [
            {'x': list(range(24)), 'y': eff2, 'type': 'line', 'name': 'Thermal Efficiency'},
            {'x': list(range(24)), 'y': eff1, 'type': 'line', 'name': 'Electrical Efficiency'},
        ],
        'layout': eff_layout,
    }
        
        eff_figure = {
        'data': [
            {'x': list(range(4)), 'y': carbon_buy, 'type': 'line', 'name': 'Carbon Credits Purchased'},
            {'x': list(range(4)), 'y': carbon_sell, 'type': 'line', 'name': 'Carbon Credits Sold'},
            {'x': list(range(4)), 'y': carbon_held, 'type': 'line', 'name': 'Carbon Credits Held'},
            {'x': list(range(4)), 'y': carbon_earn, 'type': 'line', 'name': 'Carbon Credits Earned'},
            #{'x': list(range(4)), 'y': carbon, 'type': 'line', 'name': 'Carbon'},
            {'x': list(range(4)), 'y': carbon_diff, 'type': 'line', 'name': 'Carbon Difference'},
        ],
        'layout': eff_layout,
    }
        # Return the updated figures
        return electricity_figure, heat_figure, cold_figure, heat_storage, purchased_elec, eff_figure, carbon_credits_figure, \
           f"Total Purchased Electricity: {locale.currency(total_purchased_electricity, grouping=True)}", \
           f"Total Heat Cost: {locale.currency(total_heat_cost, grouping=True)}", \
           f"Final CHP Capacity: {round(final_chp_capacity, 2)} KW", \
           f"Energy Ratio: {round(energy_ratio, 2)}", \
           f"Model Cost: {locale.currency(model_cost, grouping=True)}", \
           f"No Credits: {model.below_cap[2]()} {model.above_cap[2]()} ",  
    else:
        raise dash.exceptions.PreventUpdate














# Create a simple model
def pyomomodel():
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    # Time periods (e.g., hours in a day)
    HOURS = list(range(24))
    model.HOURS = Set(initialize=HOURS)

    INTERVALS = list(range(4))  # Four 6-hour intervals in a day
    model.INTERVALS = Set(initialize=INTERVALS)


    # Storage
    storage_efficiency = 0.7 # %
    withdrawal_efficiency = 0.7 # %
    max_storage_capacity = 1000 #kW
    heat_storage_loss_factor = 0.95  # %/timestep

    # Refrigeration
    COP_h = 2
    COP_e = 1

    # CHP params
    capital_cost_per_kw = 1 # $/kw
    fuel_energy = 1  # kW
    max_ramp_rate = 800 #kW/timestep

    TEMP = 700
    PRES = 50
        # Efficiency coefficients


    # Co2 params
    co2_per_unit_fuel = 0.2  # kg CO2 per kW of fuel
    co2_per_unit_elec = 0.2  # kg CO2 per kW of electricity
    max_co2_emissions = 5000  # kg CO2
    

    # -------------- Decision Variables --------------

    # CHP System Variables
    model.CHP_capacity = Var(within=NonNegativeReals)
    model.electricity_production = Var(model.HOURS, within=NonNegativeReals)
    model.heat_production = Var(model.HOURS, within=NonNegativeReals)
    model.fuel_consumed = Var(model.HOURS, within=NonNegativeReals)
    model.energy_ratio = Var(within=NonNegativeReals, bounds=(0.1, 0.9))

    model.electrical_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(0, 1))
    model.thermal_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(0, 1))

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
    model.ramp_rate = Var(model.HOURS, within=NonNegativeReals)

    # Refrigeration
    model.refrigeration_produced = Var(model.HOURS, within=NonNegativeReals)
    model.heat_used_for_cooling = Var(model.HOURS, within=NonNegativeReals)
    model.elec_used_for_cooling = Var(model.HOURS, within=NonNegativeReals)


    # Variables
    model.co2_emissions = Var(model.HOURS, within=NonNegativeReals)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)
    model.carbon_credits = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_purchased = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_earned = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_sold = Var(model.INTERVALS, within=NonNegativeReals)
    model.exceeds_cap = Var(model.INTERVALS, within=Binary)
    model.credits_held = Var(model.INTERVALS, within=NonNegativeReals)
    model.emissions_difference = Var(model.INTERVALS, domain=Reals)

    model.below_cap = Var(model.INTERVALS, domain=Binary)
    model.above_cap = Var(model.INTERVALS, domain=Binary)

    carbon_credit_price = 10  # Assume a price, you can change this
    carbon_credit_sell_price = 8  # Assume a sell price, you can change this


 # -------------- Constraints --------------
    # ======== Heat-Related Constraints ========

    # Heat Balance
    def heat_balance_rule(model, h):
        return model.heat_production[h] >= model.heat_over_production[h] + model.useful_heat[h]
    model.heat_balance = Constraint(model.HOURS, rule=heat_balance_rule)

    # Heat Demand
    def heat_demand_balance(model, h):
        return model.heat_to_plant[h] == heat_demand[h]
    model.heat_demand_rule = Constraint(model.HOURS, rule=heat_demand_balance)

    # Overproduction of Heat
    def heat_over_production_rule(model, h):
        return model.heat_over_production[h] == model.heat_production[h] - model.useful_heat[h]
    model.heat_over_production_constraint = Constraint(model.HOURS, rule=heat_over_production_rule)

    # Useful Heat
    def useful_heat_rule(model, h):
        return model.useful_heat[h] == model.heat_to_plant[h] - (model.heat_withdrawn[h] * withdrawal_efficiency) + (model.heat_used_for_cooling[h] / COP_h)
    model.useful_heat_constraint = Constraint(model.HOURS, rule=useful_heat_rule)

    # ======== Electricity-Related Constraints ========

    # Electricity Demand
    def elec_demand_balance(model, h):
        return model.elec_to_plant[h] + model.purchased_electricity[h] == electricity_demand[h]
    model.elec_demand_rule = Constraint(model.HOURS, rule=elec_demand_balance)

    # Useful Electricity
    def useful_elec_rule(model, h):
        return model.useful_elec[h] == model.elec_to_plant[h] + (model.elec_used_for_cooling[h] / COP_e)
    model.useful_elec_constraint = Constraint(model.HOURS, rule=useful_elec_rule)

    # Overproduction of Electricity
    def elec_over_production_rule(model, h):
        return model.electricity_over_production[h] == model.electricity_production[h] - model.useful_elec[h]
    model.elec_over_production_constraint = Constraint(model.HOURS, rule=elec_over_production_rule)

    # Energy Ratio
    def energy_ratio_rule(model, h):
        return (model.electricity_production[h] / model.electrical_efficiency[h]) / (model.heat_production[h] * model.thermal_efficiency[h]) == model.energy_ratio
    model.energy_ratio_constraint = Constraint(model.HOURS, rule=energy_ratio_rule)

    # ======== CHP and Fuel-Related Constraints ========

    # CHP Capacity
    def capacity_rule(model, h):
        return model.heat_production[h] <= model.CHP_capacity
    model.capacity_constraint = Constraint(model.HOURS, rule=capacity_rule)

    # Fuel Consumption
    def fuel_consumed_rule(model, h):
        return fuel_energy * model.fuel_consumed[h] * (1 - model.energy_ratio) == model.heat_production[h] / model.thermal_efficiency[h]
    model.fuel_consumed_rule = Constraint(model.HOURS, rule=fuel_consumed_rule)

    # Electrical Efficiency Constraint
    def electrical_efficiency_rule(model, h):
        return model.electrical_efficiency[h] == 0.25 + 0.025 * (model.electricity_production[h] / model.CHP_capacity) + 0.001 * TEMP 
    model.electrical_efficiency_constraint = Constraint(model.HOURS, rule=electrical_efficiency_rule)

    # Thermal Efficiency Constraint
    def thermal_efficiency_rule(model, h):
        return model.thermal_efficiency[h] == 0.2 + 0.05 * (model.heat_production[h] / model.CHP_capacity) + 0.001 * TEMP 
    model.thermal_efficiency_constraint = Constraint(model.HOURS, rule=thermal_efficiency_rule)

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

    # Initial Heat Stored
    def initial_heat_stored_rule(model):
        return model.heat_stored[0] == 0
    model.initial_heat_stored_constraint = Constraint(rule=initial_heat_stored_rule)

    # Heat Storage Dynamics
    def storage_dynamics_rule(model, h):
        if h == 0:
            return Constraint.Skip  # Skip for the first hour
        return model.heat_stored[h] == heat_storage_loss_factor * (model.heat_stored[h - 1] 
                        + (model.heat_over_production[h] * storage_efficiency) 
                        - (model.heat_withdrawn[h] / withdrawal_efficiency))
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

    # ======== CO2 Constraints ========
    def co2_emissions_rule(model, h):
        if h == 0:
            return model.co2_emissions[h] == 0
        return model.co2_emissions[h] == co2_per_unit_fuel * model.fuel_consumed[h] + co2_per_unit_elec * model.purchased_electricity[h]
    model.co2_emissions_constraint = Constraint(model.HOURS, rule=co2_emissions_rule)

    def total_emissions_per_interval_rule(model, i):
        if i == 0:
            return model.total_emissions_per_interval[i] == 0
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[h] for h in range(i*6, (i+1)*6))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    def emissions_difference_rule(model, i):
        return model.emissions_difference[i] == model.total_emissions_per_interval[i] - max_co2_emissions
    model.emissions_difference_constraint = Constraint(model.INTERVALS, rule=emissions_difference_rule)

    # Heat Storage Dynamics
    def credits_dynamics_rule(model, i):
        if i == 0:
            return model.credits_held[i] == 0
        return model.credits_held[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_purchased[i] + 1E2
    model.credits_dynamics = Constraint(model.INTERVALS, rule=credits_dynamics_rule)

    # Constraint to check if emissions are below the cap
    def below_cap_rule(model, i):
        M = 1E6  # You might want to set M to a reasonably large value if max_co2_emissions isn't large enough.
        return (max_co2_emissions - model.total_emissions_per_interval[i]) <= M * model.below_cap[i]
    model.below_cap_constraint = Constraint(model.INTERVALS, rule=below_cap_rule)

    # Constraint to calculate credits earned
    def credits_earned_rule(model, i):
        if i==0:
            return Constraint.Skip
        return model.credits_earned[i] == (max_co2_emissions - model.total_emissions_per_interval[i]) * (1-)
    model.credits_earned_constraint = Constraint(model.INTERVALS, rule=credits_earned_rule)

    # Constraint to check if emissions are above the cap
    def above_cap_rule(model, i):
        M = 1E6  # Adjust as needed
        return (model.total_emissions_per_interval[i] - max_co2_emissions) <= M * model.above_cap[i]
    model.above_cap_constraint = Constraint(model.INTERVALS, rule=above_cap_rule)

    # Constraint to calculate credits purchased
    def credits_purchased_rule(model, i):
        if i==0:
            return Constraint.Skip
        return model.credits_purchased[i] >= (model.total_emissions_per_interval[i] - max_co2_emissions) * (model.above_cap[i])
    model.credits_purchased_constraint = Constraint(model.INTERVALS, rule=credits_purchased_rule)
    # -------------- Objective Function --------------

    def objective_rule(model):
        capital_cost = capital_cost_per_kw * model.CHP_capacity
        fuel_cost = sum(heat_market[h] * model.fuel_consumed[h] for h in model.HOURS)
        elec_cost = sum(model.purchased_electricity[h] * electricity_market[h] for h in model.HOURS)
        elec_sold = sum(model.electricity_over_production[h] * electricity_market_sold[h] for h in model.HOURS)
        heat_sold = sum((heat_market_sold[h] * model.heat_over_production[h]) for h in model.HOURS)
        carbon_cost = sum(model.credits_purchased[i] * 10 for i in model.INTERVALS)
        return capital_cost + fuel_cost + elec_cost + carbon_cost - (elec_sold + heat_sold)
    

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # -------------- Solver --------------
    solver = SolverFactory("ipopt")
    #solver.options['NonConvex'] = 2
    solver.solve(model, tee=True)

    return model

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)