from pyomo.environ import *
import pandas as pd
import dash
import sys
from dash import dcc, html, Input, Output, State  # Combined into a single line
import os
# Add the config directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
# Import the solver options
from solver_options import get_solver
import numpy as np
import locale
import gc
gc.disable()  # Disable garbage collection temporarily

locale.setlocale(locale.LC_ALL, '')  # Setting locale once here

# Initialize Dash app
last_mod_time = 0

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct paths to the data files by correctly moving up one directory to 'modelling_MCDM'
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'demands.xlsx'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.xlsx'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.xlsx'))  # Corrected path

capex_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'capex.xlsx'))  # Corrected path

# Read the Excel files
demands = pd.read_excel(demands_path, nrows=10000)
markets_monthly = pd.read_excel(markets_monthly_path, nrows=10000)
markets = pd.read_excel(markets_path, nrows=10000)
capex = pd.read_excel(capex_path, nrows=10000)

electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()
electricity_market = markets["elec"].to_numpy() * 1
electricity_market_sold = markets["elec_sold"].to_numpy()

carbon_market = markets["carbon"].to_numpy() * 1

NG_market = markets["nat_gas"].to_numpy() * 1
NG_market_monthly = markets_monthly["nat_gas"].to_numpy()

heat_market_sold = markets["nat_gas_sold"].to_numpy()

H2_market = markets["hydrogen"].to_numpy() * 0.44

BM_market = markets["biomass"].to_numpy()

CHP_capacity = 4000

energy_ratio = 0.27

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

        investment_made = False
        for i in model.INTERVALS:
            if value(model.invest_time[i]) > 0.5:  # Check if the binary variable is 1
                print(f"Investment in hydrogen is made at interval {i}")
                investment_made = True
                break

        if not investment_made:
            print("No investment in hydrogen was made.")
        months_list = list(model.MONTHS)
        intervals = list(model.INTERVALS)

        electricity_production = [model.electricity_production[m]() for m in months_list]
        heat_production = [model.heat_production[m]() for m in months_list]
        cold_production = [model.refrigeration_produced[m]() for m in months_list]
        fuel_consumed = [model.fuel_consumed[m]() for m in months_list]
        heat_stored = [model.heat_stored[m]() for m in months_list]
        heat_taken = [model.heat_withdrawn[m]() for m in months_list]
        heat_plant = [model.heat_to_plant[m]() for m in months_list]
        normal_heat_stored = 100*(np.array(heat_stored))/1000
        over_heat = [model.heat_over_production[m]() for m in months_list]
        heat_to_elec = [model.heat_to_elec[m]() for m in months_list]
        
        purchased_electricity = [model.purchased_electricity[m]() for m in months_list]
        over_electricity = [model.electricity_over_production[m]() for m in months_list]
        cooling_elec = [model.elec_used_for_cooling[m]() for m in months_list]
        cooling_heat = [model.heat_used_for_cooling[m]() for m in months_list]

        total_purchased_electricity = sum(model.purchased_electricity[m].value for m in months_list)
        total_heat_cost = sum(model.fuel_consumed[m].value * NG_market_monthly[m] for m in months_list)
        final_chp_capacity = CHP_capacity
        model_cost = model.objective()

        eff1 = [model.electrical_efficiency[m]() for m in months_list]
        eff2 = [model.thermal_efficiency[m]() for m in months_list]

        carbon_buy = [model.credits_purchased[i]() for i in intervals]
        carbon_sell = [model.credits_sold[i]() for i in intervals]
        credits = [model.carbon_credits[i]() for i in intervals]
        carbon_held = [model.credits_held[i]() for i in intervals]
        carbon_earn = [model.credits_earned[i]() for i in intervals]
        carbon = [model.total_emissions_per_interval[i]() for i in intervals]
        carbon_diff = [model.emissions_difference[i]() for i in intervals]

        blend_H2 = [model.fuel_blend_h2[m]() for m in months_list]
        blend_ng = [model.fuel_blend_ng[m]() for m in months_list]
        blend_bm = [model.fuel_blend_biomass[m]() for m in months_list]

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
                {'x': list(model.MONTHS), 'y': [electricity_demand[m] for m in list(model.MONTHS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(model.MONTHS), 'y': electricity_production, 'type': 'line', 'name': 'Production', 'line': {'color': 'blue'}},
                {'x': list(model.MONTHS), 'y': [purchased_electricity[m] for m in list(model.MONTHS)], 'line': {'width': 3, 'dash': 'dash', 'line': {'color': 'hydrogen'}}, 'name': 'Purchased Electricity'},
                {'x': list(model.MONTHS), 'y': [over_electricity[m] for m in list(model.MONTHS)], 'line': {'width': 3, 'dash': 'dot', 'color': 'hydrogen'}, 'name': 'Over-Produced Electricity'}
            ],
            'layout': electricity_layout,
        }

        heat_figure = {
            'data': [
                {'x': list(model.MONTHS), 'y': [heat_demand[m] for m in list(model.MONTHS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(model.MONTHS), 'y': heat_production, 'type': 'line', 'name': 'Production'},
                {'x': list(model.MONTHS), 'y': heat_stored, 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
                {'x': list(model.MONTHS), 'y': over_heat, 'type': 'line', 'name': 'Over-Production'},
                {'x': list(model.MONTHS), 'y': heat_to_elec, 'type': 'line', 'name': 'Elec to Heat'},
            ],
            'layout': heat_layout,
        }

        cold_figure = {
            'data': [
                {'x': list(model.MONTHS), 'y': [refrigeration_demand[m] for m in list(model.MONTHS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(model.MONTHS), 'y': cold_production, 'type': 'line', 'name': 'Production'},
                {'x': list(model.MONTHS), 'y': cooling_elec, 'type': 'line', 'name': 'Cooling Electricity'},
                {'x': list(model.MONTHS), 'y': cooling_heat, 'type': 'line', 'name': 'Cooling Heat'},
            ],
            'layout': cold_layout,
        }
        heat_storage = {
            'data': [
                {'x': list(model.MONTHS), 'y': normal_heat_stored, 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
            ],
            'layout': storage_layout,
        }

        purchased_elec = {
            'data': [
                {'x': list(model.MONTHS), 'y': electricity_market, 'type': 'line', 'name': 'Purchased Electricity', 'line': {'color': 'blue'}},
                {'x': list(model.MONTHS), 'y': NG_market, 'type': 'line', 'name': 'Purchased Fuel', 'line': {'color': 'red'}},
                {'x': list(model.MONTHS), 'y': heat_market_sold, 'line': {'width': 3, 'dash': 'dash', 'color': 'red'}, 'name': 'Sold Heat'},
                {'x': list(model.MONTHS), 'y': electricity_market_sold, 'line': {'width': 3, 'dash': 'dash', 'color': 'blue'}, 'name': 'Sold Electricity'},
            ],
            'layout': market_layout,
        }
        eff_figure = {
        'data': [
            {'x': list(model.MONTHS), 'y': eff2, 'type': 'line', 'name': 'Thermal Efficiency'},
            {'x': list(model.MONTHS), 'y': eff1, 'type': 'line', 'name': 'Electrical Efficiency'},
        ],
        'layout': eff_layout,
    }
        
        carbon_credits_figure = {
        'data': [
            {'x': list(model.MONTHS), 'y': credits, 'type': 'line', 'name': 'Carbon Credits Purchased'},
            {'x': list(model.MONTHS), 'y': carbon_sell, 'type': 'line', 'name': 'Carbon Credits Sold'},
            {'x': list(model.MONTHS), 'y': carbon_held, 'type': 'line', 'name': 'Carbon Credits Held'},
            {'x': list(model.MONTHS), 'y': carbon_earn, 'type': 'line', 'name': 'Carbon Credits Earned'},
        ],
        'layout': credits_layout,
    }
        
        fuel_blend_figure = {
        'data': [
            {'x': list(model.MONTHS), 'y': blend_H2, 'type': 'line', 'name': 'Hydrogen'},
            {'x': list(model.MONTHS), 'y': blend_bm, 'type': 'line', 'name': 'Biomass'},
            {'x': list(model.MONTHS), 'y': blend_ng, 'type': 'line', 'name': 'Natural Gas'},
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

total_months = 360
time_limit = 300
# Create a simple model
def pyomomodel(total_months = total_months, time_limit = time_limit, CHP_capacity=CHP_capacity, energy_ratio = energy_ratio):
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    # Time periods (e.g., months in a year)
    MONTHS = np.arange(total_months)
    model.MONTHS = Set(initialize=MONTHS)

    no_intervals = 30
    intervals_time = int(total_months / no_intervals)
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
    max_ramp_rate = 100000  #kW/timestep

    TEMP = 700
    PRES = 50
    # Efficiency coefficients
    ng_coeffs = {"elec": [0.25, 0.025, 0.001], "thermal": [0.2, 0.05, 0.001]}  # Placeholder
    h2_coeffs = {"elec": [0.18, 0.02, 0.0015], "thermal": [0.15, 0.04, 0.0012]}  # Placeholder


    h2_capex = capex["H2"].to_numpy()  # Example value, adjust based on your case
    eb_capex = capex["EB"].to_numpy() * 0.1  # Example value, adjust based on your case


    # Co2 params
    co2_per_unit_ng = 0.37  # kg CO2 per kW of fuel
    co2_per_unit_bm = 0.1
    co2_per_unit_h2 = 0
    co2_per_unit_elec = 0.23  # kg CO2 per kW of electricity
    max_co2_emissions = markets["cap"]  # kg CO2
    M = 1E8
    # -------------- Decision Variables --------------

    # CHP System Variables

    model.electricity_production = Var(model.MONTHS, within=NonNegativeReals)
    model.heat_production = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 2))
    model.fuel_consumed = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 3))

    model.heat_to_elec = Var(model.MONTHS, within=NonNegativeReals)
    
    model.electrical_efficiency = Var(model.MONTHS, within=NonNegativeReals, bounds=(1, 1))
    model.thermal_efficiency = Var(model.MONTHS, within=NonNegativeReals, bounds=(1, 1))

    # Fuel variables (change from INTERVALS to MONTHS)
    model.fuel_blend_ng = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_h2 = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_biomass = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))

    #Investment decision variables
    model.invest_h2 = Var(within=Binary)  # Decision to invest in H2 system
    model.invest_time = Var(model.INTERVALS, within=Binary)  # Decision for when to invest
    model.active_h2_blending = Var(model.INTERVALS, within=Binary)  # To activate H2 blending after investment

    #Investment decision variables
    model.invest_eb = Var(within=Binary)  # Decision to invest in H2 system
    model.invest_time_eb = Var(model.INTERVALS, within=Binary)  # Decision for when to invest
    model.active_eb= Var(model.INTERVALS, within=Binary)  # To activate H2 blending after investment

    # Plant Supply and Useful Energy
    model.heat_to_plant = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 2))
    model.elec_to_plant = Var(model.MONTHS, within=NonNegativeReals)
    model.useful_heat = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 2))
    model.useful_elec = Var(model.MONTHS, within=NonNegativeReals)

    # Market and Storage
    model.purchased_electricity = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(electricity_demand)*2))
    model.heat_stored = Var(model.MONTHS, within=NonNegativeReals)
    model.heat_withdrawn = Var(model.MONTHS, within=NonNegativeReals)

    # Overproduction and Ramp Rate
    model.heat_over_production = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(heat_demand)))
    model.electricity_over_production = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.ramp_rate = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max_ramp_rate))

    # Refrigeration
    model.refrigeration_produced = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.heat_used_for_cooling = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.elec_used_for_cooling = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))

    # Variables
    model.co2_emissions = Var(model.MONTHS, within=NonNegativeReals)
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

        # Define time-indexed CAPEX parameters (for each interval i)
    model.h2_capex = Param(model.INTERVALS, within=NonNegativeReals, initialize={i: h2_capex[i] for i in model.INTERVALS})
    model.eb_capex = Param(model.INTERVALS, within=NonNegativeReals, initialize={i: eb_capex[i] for i in model.INTERVALS})


 # -------------- Constraints --------------
    # Heat Balance
    def heat_balance_rule(model, m):
        return model.heat_production[m] >= (model.heat_over_production[m] + model.useful_heat[m])
    model.heat_balance = Constraint(model.MONTHS, rule=heat_balance_rule)

    # Heat Demand
    def heat_demand_balance(model, m):
        return (model.heat_to_plant[m]) + model.heat_to_elec[m] == heat_demand[m]
    model.heat_demand_rule = Constraint(model.MONTHS, rule=heat_demand_balance)

    # Overproduction of Heat
    def heat_over_production_rule(model, m):
        return model.heat_over_production[m] == model.heat_production[m] - model.useful_heat[m]
    model.heat_over_production_constraint = Constraint(model.MONTHS, rule=heat_over_production_rule)

    # Useful Heat
    def useful_heat_rule(model, m):
        return model.useful_heat[m] == model.heat_to_plant[m] - (model.heat_withdrawn[m] * withdrawal_efficiency) + (model.heat_used_for_cooling[m] / COP_h)
    model.useful_heat_constraint = Constraint(model.MONTHS, rule=useful_heat_rule)

    # ======== Electricity-Related Constraints ========

    # Electricity Demand
    def elec_demand_balance(model, m):
        return model.elec_to_plant[m] + model.purchased_electricity[m] == electricity_demand[m]
    model.elec_demand_rule = Constraint(model.MONTHS, rule=elec_demand_balance)

    # Useful Electricity
    def useful_elec_rule(model, m):
        return model.useful_elec[m] == model.elec_to_plant[m] + (model.elec_used_for_cooling[m] / COP_e)
    model.useful_elec_constraint = Constraint(model.MONTHS, rule=useful_elec_rule)

    # Overproduction of Electricity
    def elec_over_production_rule(model, m):
        return model.electricity_over_production[m] == model.electricity_production[m] - model.useful_elec[m]
    model.elec_over_production_constraint = Constraint(model.MONTHS, rule=elec_over_production_rule)

    # ======== CHP and Fuel-Related Constraints ========

    # CHP Capacity
    def capacity_rule(model, m):
        return CHP_capacity >= (model.heat_production[m] + model.heat_stored[m]) + model.electricity_production[m]
    model.capacity_constraint = Constraint(model.MONTHS, rule=capacity_rule)

    # Fuel Consumption
    def fuel_consumed_rule(model, m):
        interval = m // (len(model.MONTHS) // len(model.INTERVALS))  # Map month to interval
        return fuel_energy * model.fuel_consumed[m] * (1 - energy_ratio) * model.thermal_efficiency[m] == model.heat_production[m]
    model.fuel_consumed_rule = Constraint(model.MONTHS, rule=fuel_consumed_rule)

    def electrical_efficiency_rule(model, m):
        efficiency_adjustment_times_CHP = (
            (model.fuel_blend_ng[m] + model.fuel_blend_biomass[m]) * 
            (ng_coeffs["elec"][0] * CHP_capacity + ng_coeffs["elec"][1] * model.electricity_production[m] + ng_coeffs["elec"][2] * TEMP * CHP_capacity) +
            model.fuel_blend_h2[m] * 
            (h2_coeffs["elec"][0] * CHP_capacity + h2_coeffs["elec"][1] * model.electricity_production[m] + h2_coeffs["elec"][2] * TEMP * CHP_capacity)
        )
        return model.electrical_efficiency[m] * CHP_capacity == efficiency_adjustment_times_CHP

    #model.electrical_efficiency_constraint = Constraint(model.MONTHS, rule=electrical_efficiency_rule)

    def thermal_efficiency_rule(model, m):
        efficiency_adjustment_times_CHP = (
            (model.fuel_blend_ng[m] + model.fuel_blend_biomass[m]) * 
            (ng_coeffs["thermal"][0] * CHP_capacity + ng_coeffs["thermal"][1] * model.heat_production[m] + ng_coeffs["thermal"][2] * TEMP * CHP_capacity) +
            model.fuel_blend_h2[m] * 
            (h2_coeffs["thermal"][0] * CHP_capacity + h2_coeffs["thermal"][1] * model.heat_production[m] + h2_coeffs["thermal"][2] * TEMP * CHP_capacity)
        )
        return model.thermal_efficiency[m] * CHP_capacity == efficiency_adjustment_times_CHP

    #model.thermal_efficiency_constraint = Constraint(model.MONTHS, rule=thermal_efficiency_rule)

    # Constraint to ensure that the sum of the fuel blend percentages equals 1.
    # Monthly fuel blending constraint
    def fuel_blend_rule(model, m):
        return model.fuel_blend_ng[m] + model.fuel_blend_h2[m] + model.fuel_blend_biomass[m] == 1
    model.fuel_blend_constraint = Constraint(model.MONTHS, rule=fuel_blend_rule)

    # Hydrogen blending is only active after the investment time
    def h2_blending_activation_rule(model, m):
        interval = m // (len(model.MONTHS) // len(model.INTERVALS))  # Map month to interval
        return model.fuel_blend_h2[m] <= model.active_h2_blending[interval]
    model.h2_blending_activation_constraint = Constraint(model.MONTHS, rule=h2_blending_activation_rule)

    # Ensure that hydrogen blending is only active after the investment time
    def active_h2_blending_rule(model, i):
        # Hydrogen blending is allowed in interval `i` if investment has been made in any interval `j` <= `i`
        return model.active_h2_blending[i] == sum(model.invest_time[j] for j in model.INTERVALS if j <= i)
    model.active_h2_blending_constraint = Constraint(model.INTERVALS, rule=active_h2_blending_rule)

    # Limit the investment to happen only once
    def single_investment_rule(model):
        return sum(model.invest_time[i] for i in model.INTERVALS) == model.invest_h2
    model.single_investment_constraint = Constraint(rule=single_investment_rule)

    def eb_activation_rule(model, m):
        interval = m // (len(model.MONTHS) // len(model.INTERVALS))
        return model.heat_to_elec[m] <= (model.active_eb[interval] * M) 
    model.eb_activation_constraint = Constraint(model.MONTHS, rule=eb_activation_rule)

    # Ensure that hydrogen blending is only active after the investment time
    def active_eb_rule(model, i):
        # Hydrogen blending is allowed in interval `i` if investment has been made in any interval `j` <= `i`
        return model.active_eb[i] == sum(model.invest_time_eb[j] for j in model.INTERVALS if j <= i)
    model.active_eb_constraint = Constraint(model.INTERVALS, rule=active_eb_rule)

    def single_investment_rule_eb(model):
        return sum(model.invest_time_eb[i] for i in model.INTERVALS) <= model.invest_eb  # Allow for flexibility
    model.single_investment_eb_constraint = Constraint(rule=single_investment_rule_eb)

    # Initial Heat Stored
    def initial_heat_stored_rule(model):
        return model.heat_stored[0] == 0
    model.initial_heat_stored_constraint = Constraint(rule=initial_heat_stored_rule)

    # Heat Storage Dynamics
    def storage_dynamics_rule(model, m):
        if m == 0:
            return Constraint.Skip  # Skip for the first month
        return model.heat_stored[m] == heat_storage_loss_factor * (model.heat_stored[m - 1] 
                        + (model.heat_over_production[m] * storage_efficiency) 
                        - (model.heat_withdrawn[m] / withdrawal_efficiency))
    model.storage_dynamics = Constraint(model.MONTHS, rule=storage_dynamics_rule)

    # Heat Storage Capacity
    def storage_capacity_rule(model, m):
        return model.heat_stored[m] <= max_storage_capacity
    model.storage_capacity = Constraint(model.MONTHS, rule=storage_capacity_rule)

    # ======== Refrigeration Constraints ========

    # Refrigeration Balance
    def refrigeration_balance_rule(model, m):
        return model.refrigeration_produced[m] == (model.elec_used_for_cooling[m] * COP_e) + (model.heat_used_for_cooling[m] * COP_h) 
    model.refrigeration_balance = Constraint(model.MONTHS, rule=refrigeration_balance_rule)

    # Refrigeration Demand
    def refrigeration_demand_rule(model, m):
        return model.refrigeration_produced[m] == refrigeration_demand[m]
    model.refrigeration_demand_con = Constraint(model.MONTHS, rule=refrigeration_demand_rule)

    # Energy Ratio
    def energy_ratio_rule(model, m):
        return model.electricity_production[m] == (model.heat_production[m] * energy_ratio) * model.electrical_efficiency[m]
    model.energy_ratio_constraint = Constraint(model.MONTHS, rule=energy_ratio_rule)

# ======== CO2 Constraints ========
    # CO2 Emissions Rule
    # CO2 Emissions Rule - Adjusted to reflect interval-based fuel blending
    # CO2 Emissions Rule - Adjusted to reflect monthly-based fuel blending
    def co2_emissions_rule(model, m):
        interval = m // (len(model.MONTHS) // len(model.INTERVALS))  # Map month to interval
        return model.co2_emissions[m] == ( 
            co2_per_unit_ng * model.fuel_blend_ng[m] * model.fuel_consumed[m] +
            co2_per_unit_h2 * model.fuel_blend_h2[m] * model.fuel_consumed[m] +
            co2_per_unit_bm * model.fuel_blend_biomass[m] * model.fuel_consumed[m] +
            co2_per_unit_elec * (model.purchased_electricity[m] + model.heat_to_elec[m])
        )
    model.co2_emissions_constraint = Constraint(model.MONTHS, rule=co2_emissions_rule)


    # Total Emissions Per Interval Rule
    def total_emissions_per_interval_rule(model, i):
        start = i * intervals_time
        end = (i + 1) * intervals_time
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[m] for m in range(start, end))
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

    def mutual_exclusivity_rule(model):
        return model.invest_h2 + model.invest_eb <= 1
    model.mutual_exclusivity_constraint = Constraint(rule=mutual_exclusivity_rule)


    # Modify CHP Deactivation Constraints to Gradually Reduce Production
    #NOTE WE NEED to gfradually reduc e produciton, or it becomes infgeasaible ebcause the model cannot instanteously decrease production, hence why it didn't work before

    # -------------- Objective Function --------------

    def objective_rule(model):
        elec_cost = sum((model.purchased_electricity[m] + model.heat_to_elec[m]) * electricity_market[m] for m in model.MONTHS)
        elec_sold = sum(model.electricity_over_production[m] * electricity_market_sold[m] for m in model.MONTHS)
        heat_sold = sum((heat_market_sold[m] * model.heat_over_production[m]) for m in model.MONTHS)
        
        fuel_cost_NG = sum(model.fuel_blend_ng[m] * NG_market[m] * model.fuel_consumed[m] for m in model.MONTHS)
        fuel_cost_H2 = sum(model.fuel_blend_h2[m] * H2_market[m] * model.fuel_consumed[m] for m in model.MONTHS)
        fuel_cost_BM = sum(model.fuel_blend_biomass[m] * BM_market[m] * model.fuel_consumed[m] for m in model.MONTHS)
        
        carbon_cost = sum(model.carbon_credits[i] * carbon_market[i] for i in model.INTERVALS)
        carbon_sold = sum(model.credits_sold[i] * carbon_market[i] for i in model.INTERVALS) * 0.8
    # Time-dependent CAPEX for hydrogen and electrification, indexed by i
        h2_investment_cost = sum(model.invest_time[i] * model.h2_capex[i] for i in model.INTERVALS)
        eb_investment_cost = sum(model.invest_time_eb[i] * model.eb_capex[i] for i in model.INTERVALS)
    
        
        return (fuel_cost_NG + fuel_cost_H2 + fuel_cost_BM) + elec_cost + carbon_cost + h2_investment_cost + eb_investment_cost - (elec_sold + heat_sold + carbon_sold)
    model.objective = Objective(rule=objective_rule, sense=minimize)
    # -------------- Solver --------------
    solver = get_solver(time_limit)  # Use the imported solver configuration
    solver.solve(model, tee=True, symbolic_solver_labels=False)

        # Extract results after solving the monthly model
    fuel_blend_ng_results = {m: value(model.fuel_blend_ng[m]) for m in model.MONTHS}
    fuel_blend_h2_results = {m: value(model.fuel_blend_h2[m]) for m in model.MONTHS}
    fuel_blend_biomass_results = {m: value(model.fuel_blend_biomass[m]) for m in model.MONTHS}

    investment_times = {i: value(model.invest_time[i]) for i in model.INTERVALS}
    active_h2_blending = {i: value(model.active_h2_blending[i]) for i in model.INTERVALS}


    investment_times_eb = {i: value(model.invest_time_eb[i]) for i in model.INTERVALS}
    active_eb_blending = {i: value(model.active_eb[i]) for i in model.INTERVALS}

    # Example: Save results to a CSV file for use in a more granular model
    results_df = pd.DataFrame({
        'Month': list(model.MONTHS),
        'Fuel Blend NG': list(fuel_blend_ng_results.values()),
        'Fuel Blend H2': list(fuel_blend_h2_results.values()),
        'Fuel Blend Biomass': list(fuel_blend_biomass_results.values())
    })
    results_df.to_csv('monthly_fuel_blend_results.csv', index=False)

    investment_df = pd.DataFrame({
        'Interval': list(model.INTERVALS),
        'Investment Time': list(investment_times.values()),
        'Active H2 Blending': list(active_h2_blending.values())
    })
    investment_df.to_csv('investment_decisions.csv', index=False)


    investment_df2 = pd.DataFrame({
        'Interval': list(model.INTERVALS),
        'Investment Time': list(investment_times_eb.values()),
        'Active EB': list(active_eb_blending.values())
    })
    investment_df2.to_csv('investment_decisions2.csv', index=False)

    return model

gc.enable()  # Re-enable garbage collection after the critical section


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
