from pyomo.environ import *
import pandas as pd
import dash
import locale
locale.setlocale(locale.LC_ALL, '')
import os
import numpy as np
from dash import dcc, html, Input, Output
import sys
# Add the config directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
# Import the solver options
from solver_options import get_solver
# Initialize Dash app
last_mod_time = 0

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct paths to the data files by correctly moving up one directory to 'modelling_MCDM'
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'demands.xlsx'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.xlsx'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.xlsx'))  # Corrected path
load_shedding_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'load_shedding.xlsx'))  # Corrected path

# Read the Excel files
demands = pd.read_excel(demands_path, nrows=10000)
markets_monthly = pd.read_excel(markets_monthly_path, nrows=10000)
markets = pd.read_excel(markets_path, nrows=10000)
load_shedding = pd.read_excel(load_shedding_path, nrows=10000)

shortfall_penalty = load_shedding["penalty"].to_numpy()
reward = load_shedding["revenue"].to_numpy()
request = load_shedding["request"].to_numpy()

electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()
electricity_market = markets["elec"].to_numpy()
electricity_market_sold = markets["elec_sold"].to_numpy() * 0.1

carbon_market = markets["carbon"].to_numpy()

NG_market = markets["nat_gas"].to_numpy()
NG_market_monthly = markets_monthly["nat_gas"].to_numpy()

heat_market_sold = markets["nat_gas_sold"].to_numpy() * 0.1

H2_market = markets["hydrogen"].to_numpy() 

BM_market = markets["biomass"].to_numpy()

CHP_capacity = 4000

energy_ratio = 0.22

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
    dcc.Graph(id='fuel_blend'),  # New empty graph 2
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
    Output('fuel_blend', 'style'),  # New empty graph 2
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
    Output('fuel_blend', 'figure'),
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
    current_mod_time = os.path.getmtime(current_dir)

    if current_mod_time > last_mod_time:
        last_mod_time = current_mod_time  # Update last modification time

        model = pyomomodel()
        hours_list = list(model.HOURS)
        intervals_list = list(model.INTERVALS)

        electricity_production = [model.electricity_production[h]() for h in hours_list]
        heat_production = [model.heat_production[h]() for h in hours_list]
        cold_production = [model.refrigeration_produced[h]() for h in hours_list]
        fuel_consumed = [model.fuel_consumed[h]() for h in hours_list]
        heat_stored = [model.heat_stored[h]() for h in hours_list]
        heat_taken = [model.heat_withdrawn[h]() for h in hours_list]
        heat_plant = [model.heat_to_plant[h]() for h in hours_list]
        normal_heat_stored = 100*(np.array(heat_stored))/1000
        over_heat = [model.heat_over_production[h]() for h in hours_list]
        
        purchased_electricity = [model.purchased_electricity[h]() for h in hours_list]
        over_electricity = [model.electricity_over_production[h]() for h in hours_list]
        cooling_elec = [model.elec_used_for_cooling[h]() for h in hours_list]
        cooling_heat = [model.heat_used_for_cooling[h]() for h in hours_list]

        total_purchased_electricity = sum(model.purchased_electricity[h].value for h in hours_list)
        total_heat_cost = sum(model.fuel_consumed[h].value * NG_market[h] for h in hours_list)
        final_chp_capacity = CHP_capacity
        model_cost = model.objective()

        eff1 = [model.electrical_efficiency[h]() for h in hours_list]
        eff2 = [model.thermal_efficiency[h]() for h in hours_list]

        carbon_buy = [model.credits_purchased[h]() for h in intervals_list]
        carbon_sell = [model.credits_sold[h]() for h in intervals_list]
        credits = [model.carbon_credits[h]() for h in intervals_list]
        carbon_held = [model.credits_held[h]() for h in intervals_list]
        carbon_earn = [model.credits_earned[h]() for h in intervals_list]
        carbon = [model.total_emissions_per_interval[h]() for h in intervals_list]
        carbon_diff = [model.emissions_difference[h]() for h in intervals_list]

        blend_H2 = [model.fuel_blend_h2[i]() for i in intervals_list]
        blend_ng = [model.fuel_blend_ng[i]() for i in intervals_list]
        blend_bm = [model.fuel_blend_biomass[i]() for i in intervals_list]
        elec_reduction = [model.elec_reduction_by_CHP[h]() for h in hours_list]
        heat_reduction = [model.heat_reduction_by_CHP[h]() for h in hours_list]
        
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

        credits_layout = base_layout_template.copy()
        credits_layout.update({'title': 'Carbon Credit Dynamics', 'xaxis': {'title': 'Time Intervals'}, 'yaxis': {'title': 'No of Credits'}})

        blend_layout = base_layout_template.copy()
        blend_layout.update({'title': 'Fuel Blending', 'xaxis': {'title': 'Time Intervals'}, 'yaxis': {'title': '% ratio of fuels'}})

        electricity_figure = {
            'data': [
                {'x': list(model.HOURS), 'y': [electricity_demand[h] for h in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand',},
                {'x': list(model.HOURS), 'y': electricity_production, 'type': 'line', 'name': 'Production', 'line': {'color': 'blue'}},
                {'x': list(model.HOURS), 'y': [purchased_electricity[h] for h in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dash', 'line': {'color': 'hydrogen'}}, 'name': 'Purchased Electricity'},
                {'x': list(model.HOURS), 'y': [over_electricity[h] for h in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dot', 'color': 'hydrogen'}, 'name': 'Over-Produced Electricity'},
                {'x': list(model.HOURS), 'y': [elec_reduction[h] for h in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dot', 'color': 'hydrogen'}, 'name': 'Load-Shedding Electricity'}
            ],
            'layout': electricity_layout,
        }

        heat_figure = {
            'data': [
                {'x': list(model.HOURS), 'y': [heat_demand[h] for h in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(model.HOURS), 'y': heat_production, 'type': 'line', 'name': 'Production'},
                {'x': list(model.HOURS), 'y': heat_stored, 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
                {'x': list(model.HOURS), 'y': over_heat, 'type': 'line', 'name': 'Over-Production'},
                {'x': list(model.HOURS), 'y': [heat_reduction[h] for h in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dot', 'color': 'hydrogen'}, 'name': 'Load-Shedding Heat'}
            ],
            'layout': heat_layout,
        }

        cold_figure = {
            'data': [
                {'x': list(model.HOURS), 'y': [refrigeration_demand[h] for h in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(model.HOURS), 'y': cold_production, 'type': 'line', 'name': 'Production'},
                {'x': list(model.HOURS), 'y': cooling_elec, 'type': 'line', 'name': 'Cooling Electricity'},
                {'x': list(model.HOURS), 'y': cooling_heat, 'type': 'line', 'name': 'Cooling Heat'},
            ],
            'layout': cold_layout,
        }
        heat_storage = {
            'data': [
                {'x': list(model.HOURS), 'y': normal_heat_stored, 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
            ],
            'layout': storage_layout,
        }

        purchased_elec = {
            'data': [
                {'x': list(model.HOURS), 'y': electricity_market, 'type': 'line', 'name': 'Purchased Electricity', 'line': {'color': 'blue'}},
                {'x': list(model.HOURS), 'y': NG_market, 'type': 'line', 'name': 'Purchased Fuel', 'line': {'color': 'red'}},
                {'x': list(model.HOURS), 'y': heat_market_sold, 'line': {'width': 3, 'dash': 'dash', 'color': 'red'}, 'name': 'Sold Heat'},
                {'x': list(model.HOURS), 'y': electricity_market_sold, 'line': {'width': 3, 'dash': 'dash', 'color': 'blue'}, 'name': 'Sold Electricity'},
            ],
            'layout': market_layout,
        }
        eff_figure = {
        'data': [
            {'x': list(model.HOURS), 'y': eff2, 'type': 'line', 'name': 'Thermal Efficiency'},
            {'x': list(model.HOURS), 'y': eff1, 'type': 'line', 'name': 'Electrical Efficiency'},
        ],
        'layout': eff_layout,
    }
        
        carbon_credits_figure = {
        'data': [
            {'x': list(model.INTERVALS), 'y': credits, 'type': 'line', 'name': 'Carbon Credits Purchased'},
            {'x': list(model.INTERVALS), 'y': carbon_sell, 'type': 'line', 'name': 'Carbon Credits Sold'},
            {'x': list(model.INTERVALS), 'y': carbon_held, 'type': 'line', 'name': 'Carbon Credits Held'},
            {'x': list(model.INTERVALS), 'y': carbon_earn, 'type': 'line', 'name': 'Carbon Credits Earned'},
        ],
        'layout': credits_layout,
    }
        
        fuel_blend_figure = {
        'data': [
            {'x': list(model.HOURS), 'y': blend_H2, 'type': 'line', 'name': 'Hydrogen'},
            {'x': list(model.HOURS), 'y': blend_bm, 'type': 'line', 'name': 'Biomass'},
            {'x': list(model.HOURS), 'y': blend_ng, 'type': 'line', 'name': 'Natural Gas'},
        ],
        'layout': blend_layout,
    }
        # Return the updated figures
        return electricity_figure, heat_figure, cold_figure, heat_storage, purchased_elec, eff_figure, carbon_credits_figure, fuel_blend_figure, \
           f"Total Purchased Electricity: {locale.currency(total_purchased_electricity, grouping=True)}", \
           f"Total Heat Cost: {locale.currency(total_heat_cost, grouping=True)}", \
           f"Final CHP Capacity: {round(final_chp_capacity, 2)} KW", \
           f"Energy Ratio: {round(energy_ratio, 2)}", \
           f"Model Cost: {locale.currency(model_cost, grouping=True)}", \
           f"Credit Cost: NaN",  
    else:
        raise dash.exceptions.PreventUpdate

total_hours = 8400
time_limit = 90
# Create a simple model
def pyomomodel(total_hours = total_hours, time_limit = time_limit, CHP_capacity=CHP_capacity, energy_ratio = energy_ratio):
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    # Time periods (e.g., hours in a day)
    HOURS = np.arange(total_hours)
    model.HOURS = Set(initialize=HOURS)

    no_intervals = 12
    intervals_time = int(total_hours / no_intervals)
    INTERVALS = np.arange(no_intervals)  # Four 6-hour intervals in a day
    model.INTERVALS = Set(initialize=INTERVALS)

    # Storage
    storage_efficiency = 0.5 # %
    withdrawal_efficiency = 0.5 # %
    max_storage_capacity = 1000 #kW
    heat_storage_loss_factor = 0.95  # %/timestep

    # Refrigeration
    COP_h = 2
    COP_e = 1

    # CHP params
    capital_cost_per_kw = 1000 # $/kw
    fuel_energy = 1  # kW
    max_ramp_rate = 500  #kW/timestep

    TEMP = 700
    PRES = 50
    # Efficiency coefficients
    ng_coeffs = {"elec": [0.25, 0.025, 0.001], "thermal": [0.2, 0.05, 0.001]}  # Placeholder
    h2_coeffs = {"elec": [0.18, 0.02, 0.0015], "thermal": [0.15, 0.04, 0.0012]}  # Placeholder


    h2_capex = 100000  # Example value, adjust based on your case


    # Co2 params
    co2_per_unit_ng = 0.37  # kg CO2 per kW of fuel
    co2_per_unit_bm = 0.1
    co2_per_unit_h2 = 0.01
    co2_per_unit_elec = 0.23  # kg CO2 per kW of electricity
    max_co2_emissions = markets["cap"]  # kg CO2
    M = 1E6
    # -------------- Decision Variables --------------

    # CHP System Variables

    model.electricity_production = Var(model.HOURS, within=NonNegativeReals)
    model.heat_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 2))
    model.fuel_consumed = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 3))
    
    model.electrical_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(0, 1))
    model.thermal_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(0, 1))

    # Fuel variables
    model.fuel_blend_ng = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_h2 = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_biomass = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))

    #Investment decision variables
    model.invest_h2 = Var(within=Binary)  # Decision to invest in H2 system
    model.invest_time = Var(model.INTERVALS, within=Binary)  # Decision for when to invest
    model.active_h2_blending = Var(model.INTERVALS, within=Binary)  # To activate H2 blending after investment

    # Plant Supply and Useful Energy
    model.heat_to_plant = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 2))
    model.elec_to_plant = Var(model.HOURS, within=NonNegativeReals)
    model.useful_heat = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 2))
    model.useful_elec = Var(model.HOURS, within=NonNegativeReals)


    # Market and Storage
    model.purchased_electricity = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.heat_stored = Var(model.HOURS, within=NonNegativeReals)
    model.heat_withdrawn = Var(model.HOURS, within=NonNegativeReals)

    # Overproduction and Ramp Rate
    model.heat_over_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)))
    model.electricity_over_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.ramp_rate = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_ramp_rate))

    # Refrigeration
    model.refrigeration_produced = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.heat_used_for_cooling = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.elec_used_for_cooling = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))

    # Variables
    model.co2_emissions = Var(model.HOURS, within=NonNegativeReals)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)
    model.carbon_credits = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_purchased = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_earned = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions)))
    model.credits_sold = Var(model.INTERVALS, within=NonNegativeReals)
    model.exceeds_cap = Var(model.INTERVALS, within=Binary)
    model.credits_held = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions)))
    model.emissions_difference = Var(model.INTERVALS, domain=Reals, bounds=(-max(max_co2_emissions),max(max_co2_emissions)))
    model.credits_used_to_offset = Var(model.INTERVALS, within=NonNegativeReals,bounds=(0, max(max_co2_emissions)))
    model.below_cap = Var(model.INTERVALS, within=Binary)

    model.production_output = Var(model.HOURS, within=NonNegativeReals)

    # Flexibility variables for ancillary market participation
    # Flexibility variables for load shedding
    model.elec_reduction_by_CHP = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.heat_reduction_by_CHP = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)))

    # Penalty variable for shortfall in grid reduction request
    model.grid_reduction_shortfall = Var(model.HOURS, within=NonNegativeReals)

 # -------------- Constraints --------------
    # Heat Balance
    def heat_balance_rule(model, h):
        return model.heat_production[h] >= (model.heat_over_production[h] + model.useful_heat[h])
    model.heat_balance = Constraint(model.HOURS, rule=heat_balance_rule)

    # Heat Balance
    def heat_reduction_rule(model, h):
        return model.heat_reduction_by_CHP[h] == (model.elec_reduction_by_CHP[h] * (1-energy_ratio))
    model.heat_balance_reduction = Constraint(model.HOURS, rule=heat_reduction_rule)

    # Heat Demand
    def heat_demand_balance(model, h):
        return model.heat_to_plant[h] == heat_demand[h] - model.heat_reduction_by_CHP[h]
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
        return model.elec_to_plant[h] + model.purchased_electricity[h] == electricity_demand[h] - model.elec_reduction_by_CHP[h]
    model.elec_demand_rule = Constraint(model.HOURS, rule=elec_demand_balance)

    # Useful Electricity
    def useful_elec_rule(model, h):
        return model.useful_elec[h] == model.elec_to_plant[h] + (model.elec_used_for_cooling[h] / COP_e)
    model.useful_elec_constraint = Constraint(model.HOURS, rule=useful_elec_rule)

    # Overproduction of Electricity
    def elec_over_production_rule(model, h):
        return model.electricity_over_production[h] == model.electricity_production[h] - model.useful_elec[h]
    model.elec_over_production_constraint = Constraint(model.HOURS, rule=elec_over_production_rule)

    # ======== CHP and Fuel-Related Constraints ========

    # CHP Capacity
    def capacity_rule(model, h):
        return CHP_capacity >= (model.heat_production[h] + model.heat_stored[h]) + model.electricity_production[h]
    model.capacity_constraint = Constraint(model.HOURS, rule=capacity_rule)

    # Fuel Consumption
    def fuel_consumed_rule(model, h):
        return fuel_energy * model.fuel_consumed[h] * (1 - energy_ratio) * 1 == model.heat_production[h]
    model.fuel_consumed_rule = Constraint(model.HOURS, rule=fuel_consumed_rule)

    def electrical_efficiency_rule(model, h):
        efficiency_adjustment_times_CHP = (
            (model.fuel_blend_ng[h] + model.fuel_blend_biomass[h]) * 
            (ng_coeffs["elec"][0] * CHP_capacity + ng_coeffs["elec"][1] * model.electricity_production[h] + ng_coeffs["elec"][2] * TEMP * CHP_capacity) +
            model.fuel_blend_h2[h] * 
            (h2_coeffs["elec"][0] * CHP_capacity + h2_coeffs["elec"][1] * model.electricity_production[h] + h2_coeffs["elec"][2] * TEMP * CHP_capacity)
        )
        return model.electrical_efficiency[h] * CHP_capacity == efficiency_adjustment_times_CHP

    #model.electrical_efficiency_constraint = Constraint(model.HOURS, rule=electrical_efficiency_rule)

    def thermal_efficiency_rule(model, h):
        efficiency_adjustment_times_CHP = (
            (model.fuel_blend_ng[h] + model.fuel_blend_biomass[h]) * 
            (ng_coeffs["thermal"][0] * CHP_capacity + ng_coeffs["thermal"][1] * model.heat_production[h] + ng_coeffs["thermal"][2] * TEMP * CHP_capacity) +
            model.fuel_blend_h2[h] * 
            (h2_coeffs["thermal"][0] * CHP_capacity + h2_coeffs["thermal"][1] * model.heat_production[h] + h2_coeffs["thermal"][2] * TEMP * CHP_capacity)
        )
        return model.thermal_efficiency[h] * CHP_capacity == efficiency_adjustment_times_CHP

    #model.thermal_efficiency_constraint = Constraint(model.HOURS, rule=thermal_efficiency_rule)

    # Constraint to ensure that the sum of the fuel blend percentages equals 1.
    def fuel_blend_rule(model, i):
        return model.fuel_blend_ng[i] + model.fuel_blend_h2[i] + model.fuel_blend_biomass[i] == 1
    model.fuel_blend_constraint = Constraint(model.INTERVALS, rule=fuel_blend_rule)

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

    # Energy Ratio
    def energy_ratio_rule(model, h):
        return model.electricity_production[h] == (model.heat_production[h] * energy_ratio) * 1
    model.energy_ratio_constraint = Constraint(model.HOURS, rule=energy_ratio_rule)

# ======== CO2 Constraints ========
    def co2_emissions_rule(model, h):
        i = h // (len(model.HOURS) // len(model.INTERVALS))  # Map month to interval
        return model.co2_emissions[i] == (
            co2_per_unit_ng * model.fuel_blend_ng[i] * model.fuel_consumed[i] +
            co2_per_unit_h2 * model.fuel_blend_h2[i] * model.fuel_consumed[i] +
            co2_per_unit_bm * model.fuel_blend_biomass[i] * model.fuel_consumed[i] +
            co2_per_unit_elec * model.purchased_electricity[i]
        )
    model.co2_emissions_constraint = Constraint(model.HOURS, rule=co2_emissions_rule)

    # Total Emissions Per Interval Rule
    def total_emissions_per_interval_rule(model, i):
        start = i * intervals_time
        end = (i + 1) * intervals_time
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[h] for h in range(start, end))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    # Carbon Credits Needed Rule
    def carbon_credits_needed_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] >= model.total_emissions_per_interval[i] - max_co2_emissions[i*no_intervals]
    model.carbon_credits_needed_constraint = Constraint(model.INTERVALS, rule=carbon_credits_needed_rule)

    # Carbon Credits Earned Rule
    def carbon_credits_earned_rule(model, i):
        if i == 0:
            return model.credits_earned[i] == 0  # For the first interval
        return model.credits_earned[i] == (max_co2_emissions[i*no_intervals] - model.total_emissions_per_interval[i] + model.credits_used_to_offset[i]) * (1 - model.below_cap[i])
    model.carbon_credits_earned_constraint = Constraint(model.INTERVALS, rule=carbon_credits_earned_rule)

    # Carbon Credits Purchased Rule
    def carbon_credits_purchased_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] <= M * model.below_cap[i]
    model.carbon_credits_purchased_con = Constraint(model.INTERVALS, rule=carbon_credits_purchased_rule)

    # Credits Unheld Limit Rule
    def credits_unheld_limit_rule(model, i):
        if i == 0:
            return model.credits_sold[i] == 0  # For the first interval
        return model.credits_sold[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_used_to_offset[i]
    model.credits_unheld_limit = Constraint(model.INTERVALS, rule=credits_unheld_limit_rule)

    # Credits Held Dynamics Rule
    def credits_held_dynamics_rule(model, i):
        if i == 0:
            return model.credits_held[i] == model.credits_earned[i]  # For the first interval
        return model.credits_held[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_sold[i] - model.credits_used_to_offset[i]
    model.credits_held_dynamics = Constraint(model.INTERVALS, rule=credits_held_dynamics_rule)

    # Below Cap Rule
    def below_cap_rule(model, i):
        return model.total_emissions_per_interval[i] - max_co2_emissions[i*no_intervals] <= M * model.below_cap[i]
    model.below_cap_con = Constraint(model.INTERVALS, rule=below_cap_rule)

    # Force Zero Rule
    def force_0_rule(model, i):
        if i == 0:
            return model.credits_used_to_offset[i] == 0
        return Constraint.Skip
    model.force_0_rule_con = Constraint(model.INTERVALS, rule=force_0_rule)


    # Production Output Constraint
    def production_output_rule(model, h):
        # Assuming a linear relationship for simplicity; you can modify this as needed
        production_output_ratio = 0.8  # Example production-output-to-energy ratio
        return model.production_output[h] == production_output_ratio * (heat_demand[h] + electricity_demand[h] - model.elec_reduction_by_CHP[h] - model.heat_reduction_by_CHP[h])
    model.production_output_constraint = Constraint(model.HOURS, rule=production_output_rule)

    # -------------- Anchillary market participation --------------

    plant_flexibility = 0.5
    # Modified constraint to allow shortfall in grid reduction
    # Modified constraint to allow shortfall in grid reduction for each hour
    def grid_call_constraint(model, h):
        return model.elec_reduction_by_CHP[h] + model.grid_reduction_shortfall[h] >= request[h]
    model.grid_call_constraint = Constraint(model.HOURS, rule=grid_call_constraint)


        # Constraint to limit the maximum reduction based on plant flexibility
    def flexibility_constraint(model, h):
        return model.elec_reduction_by_CHP[h] <= plant_flexibility * electricity_demand[h]
    model.flexibility_constraint = Constraint(model.HOURS, rule=flexibility_constraint)
    
    # -------------- Objective Function --------------

    def objective_rule(model):
        elec_cost = sum(model.purchased_electricity[h] * electricity_market[h] for h in model.HOURS)
        elec_sold = sum(model.electricity_over_production[h] * electricity_market_sold[h] for h in model.HOURS)
        heat_sold = sum((heat_market_sold[h] * model.heat_over_production[h]) for h in model.HOURS)
        fuel_cost_NG = sum(model.fuel_blend_ng[i] * NG_market[h] * model.fuel_consumed[h] for h in model.HOURS for i in model.INTERVALS if h // intervals_time == i)
        fuel_cost_H2 = sum(model.fuel_blend_h2[i] * H2_market[h] * model.fuel_consumed[h] for h in model.HOURS for i in model.INTERVALS if h // intervals_time == i)
        fuel_cost_BM = sum(model.fuel_blend_biomass[i] * BM_market[h] * model.fuel_consumed[h] for h in model.HOURS for i in model.INTERVALS if h // intervals_time == i)
        carbon_cost = sum(model.carbon_credits[i] * carbon_market[i] for i in model.INTERVALS)
        carbon_sold = sum(model.credits_sold[i] * carbon_market[i] for i in model.INTERVALS) * 0.6

        production_revenue = sum(model.production_output[h] for h in model.HOURS)
        ancillary_revenue = sum(reward[h] * (model.elec_reduction_by_CHP[h]) for h in model.HOURS) * 10
        all_shortfall_penalty = sum(shortfall_penalty[h] * model.grid_reduction_shortfall[h] for h in model.HOURS) * 0


        return (fuel_cost_NG + fuel_cost_H2 + fuel_cost_BM) + elec_cost + carbon_cost + all_shortfall_penalty - (elec_sold + heat_sold + carbon_sold + production_revenue + ancillary_revenue)
    model.objective = Objective(rule=objective_rule, sense=minimize)
    
    # -------------- Solver --------------
    solver = get_solver(time_limit)  # Use the imported solver configuration
    solver.solve(model, tee=True, symbolic_solver_labels=False)

    return model



# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

