from pyomo.environ import *
import pandas as pd
import dash
import locale
import os
import numpy as np
from dash import dcc, html, Input, Output
import sys
from datetime import datetime

# Set locale for currency formatting
locale.setlocale(locale.LC_ALL, '')

# Add the config directory to the Python path (assumes you have a solver_options.py file)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from solver_options import get_solver

# ================================
# Define File Paths and Read Data
# ================================

current_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
results_dir = os.path.abspath(os.path.join(current_dir, '..', 'results'))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
demands_path = os.path.join(data_dir, 'demands_hourly_year.csv')
markets_path = os.path.join(data_dir, 'markets.csv')
investment_log_path = os.path.abspath(os.path.join(current_dir, '..', 'investment_log', "eb_investment.csv"))
results_filepath = os.path.join(results_dir, 'model_results.csv')

# Read input CSV files (here we read only the first 8761 rows as in your example)
demands = pd.read_csv(demands_path, nrows=8761)
markets = pd.read_csv(markets_path, nrows=8761)
investment_data = pd.read_csv(investment_log_path)

# Set up demand, penalty/reward and market parameters (adjust as needed)
shortfall_penalty = demands["penalty"].to_numpy() * 0
reward = demands["revenue"].to_numpy() * 0
request = demands["request"].to_numpy() * 0

electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()

unit_conv = 1E3
electricity_market = markets["Electricity Price ($/kWh)"].to_numpy() * unit_conv 
electricity_market_sold = electricity_market * 1E-3
carbon_market = markets["Carbon Credit Price ($/tonne CO2)"].to_numpy()
NG_market = markets["Natural Gas Price ($/kWh)"].to_numpy()  * unit_conv
heat_market_sold = NG_market * 1E-3
H2_market = markets["Hydrogen Price ($/kWh)"].to_numpy()  * unit_conv
BM_market = markets["Biomass Price ($/kWh)"].to_numpy() * unit_conv

em_bm = markets["Biomass Carbon Intensity (kg CO2/kWh)"].to_numpy()
em_h2 = markets["Hydrogen Carbon Intensity (kg CO2/kWh)"].to_numpy()
em_ng = markets["Natural Gas Carbon Intensity (kg CO2/kWh)"].to_numpy()
em_elec = markets["Grid Carbon Intensity (kg CO2/kWh)"].to_numpy()

# Other model parameters
CHP_capacity = 18
energy_ratio = 0.25

# Global variable to track last CSV file modification time.
last_csv_mod_time = 0

# ================================
# Dash Application Layout
# ================================

app = dash.Dash(__name__)

app.layout = html.Div([
    # Graph height control
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
    
    # Top text elements
    html.Div([
        html.Div(id='total-purchased-electricity', style={'fontSize': 24}),
        html.Div(id='total-heat-cost', style={'fontSize': 24}),
        html.Div(id='final-chp-capacity', style={'fontSize': 24}),
        html.Div(id='energy-ratio', style={'fontSize': 24}),
        html.Div(id='model-cost', style={'fontSize': 24}),
        html.Div(id='credits', style={'fontSize': 24}),
    ], style={'display': 'flex', 'justifyContent': 'space-around'}),
    
    # Graph components
    dcc.Graph(id='electricity-graph'),
    dcc.Graph(id='heat-graph'),
    dcc.Graph(id='cold-graph'),
    dcc.Graph(id='heat-store-graph'),
    dcc.Graph(id='purchased_elec'),
    dcc.Graph(id='eff-graph'),
    dcc.Graph(id='carbon-credits'),
    dcc.Graph(id='fuel_blend'),
    dcc.Graph(id='empty-graph3'),
    dcc.Graph(id='empty-graph4'),
    dcc.Graph(id='empty-graph5'),
    
    # Interval for updating the graphs (e.g. every 10 seconds)
    dcc.Interval(
        id='interval-component',
        interval=10 * 1000,  # in milliseconds
        n_intervals=0
    ),
    
    # Hidden interval to update/re-solve the model (e.g. every 60 seconds)
    dcc.Interval(
        id='interval-model-update',
        interval=360 * 1000,  # in milliseconds
        n_intervals=0,
        disabled=False
    ),
    
    # Hidden div to hold model update output (for triggering the model update)
    html.Div(id='model-update-output', style={'display': 'none'}),
])

# Global variable to track the CSV file’s last modification time
last_csv_mod_time = 0
results_filepath = os.path.abspath(os.path.join(current_dir, '..', 'results', 'model_results.csv'))

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
    global last_csv_mod_time

    # Check if the CSV file exists and has been updated
    try:
        current_mod_time = os.path.getmtime(results_filepath)
    except OSError:
        # If the file does not exist yet, do not update
        raise dash.exceptions.PreventUpdate

    if current_mod_time > last_csv_mod_time:
        last_csv_mod_time = current_mod_time
        df = pd.read_csv(results_filepath)

        # Get the hourly list from the CSV
        hours_list = df['hour'].tolist()
        
        # Extract model results from the CSV (the names below must match those saved by your model)
        electricity_production = df['electricity_production'].tolist()
        heat_production = df['heat_production'].tolist()
        cold_production = df['refrigeration_produced'].tolist()
        fuel_consumed = df['fuel_consumed'].tolist()
        heat_stored = df['heat_stored'].tolist()
        heat_taken = df['heat_withdrawn'].tolist()
        heat_plant = df['heat_to_plant'].tolist()
        # Compute normalised heat storage exactly as before:
        normal_heat_stored = 100 * (np.array(heat_stored)) / 1000
        over_heat = df['over_heat'].tolist()      # Overproduction of heat
        heat_to_elec = df['heat_to_elec'].tolist()
        purchased_electricity = df['purchased_electricity'].tolist()
        over_electricity = df['over_electricity'].tolist()
        cooling_elec = df['elec_used_for_cooling'].tolist()
        cooling_heat = df['heat_used_for_cooling'].tolist()
        eff1 = df['electrical_efficiency'].tolist()
        eff2 = df['thermal_efficiency'].tolist()
        elec_reduction = df['elec_reduction_by_CHP'].tolist()
        heat_reduction = df['heat_reduction_by_CHP'].tolist()
        
        # For interval data (carbon credits, fuel blend, etc.) we assume a single interval.
        # We take the first (or average) value from the CSV.
        credits = [df['carbon_credits'].iloc[0]]
        carbon_sell = [df['credits_sold'].iloc[0]]
        carbon_held = [df['credits_held'].iloc[0]]
        carbon_earn = [df['credits_earned'].iloc[0]]
        blend_H2 = [df['fuel_blend_h2'].iloc[0]]
        blend_ng = [df['fuel_blend_ng'].iloc[0]]
        blend_bm = [df['fuel_blend_biomass'].iloc[0]]
        
        # Use your global input arrays for demands and markets (unchanged)
        # (electricity_demand, heat_demand, refrigeration_demand, electricity_market, NG_market, 
        # heat_market_sold, electricity_market_sold, etc. were defined earlier)
        
        # Define base layout templates (unchanged from your code)
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
    
        market_layout = base_layout_template.copy()
        market_layout.update({'title': 'Energy Market', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': 'p/kWh'}})
    
        eff_layout = base_layout_template.copy()
        eff_layout.update({'title': 'Efficiency', 'xaxis': {'title': 'Hours'}, 'yaxis': {'title': 'Efficiency'}})
    
        credits_layout = base_layout_template.copy()
        credits_layout.update({'title': 'Carbon Credit Dynamics', 'xaxis': {'title': 'Time Intervals'}, 'yaxis': {'title': 'No of Credits'}})
    
        blend_layout = base_layout_template.copy()
        blend_layout.update({'title': 'Fuel Blending', 'xaxis': {'title': 'Time Intervals'}, 'yaxis': {'title': '% ratio of fuels'}})
    
        # Build figures exactly as in your original callback:
        electricity_figure = {
            'data': [
                {'x': hours_list, 'y': [electricity_demand[h] for h in hours_list],
                 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': hours_list, 'y': electricity_production,
                 'type': 'line', 'name': 'Production', 'line': {'color': 'blue'}},
                {'x': hours_list, 'y': purchased_electricity,
                 'line': {'width': 3, 'dash': 'dash', 'line': {'color': 'hydrogen'}},
                 'name': 'Purchased Electricity'},
                {'x': hours_list, 'y': over_electricity,
                 'line': {'width': 3, 'dash': 'dot', 'color': 'hydrogen'},
                 'name': 'Over-Produced Electricity'},
                {'x': hours_list, 'y': elec_reduction,
                 'line': {'width': 3, 'dash': 'dot', 'color': 'hydrogen'},
                 'name': 'Load-Shedding Electricity'}
            ],
            'layout': electricity_layout,
        }
    
        heat_figure = {
            'data': [
                {'x': hours_list, 'y': [heat_demand[h] for h in hours_list],
                 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': hours_list, 'y': heat_production,
                 'type': 'line', 'name': 'Production'},
                {'x': hours_list, 'y': heat_stored,
                 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
                {'x': hours_list, 'y': over_heat,
                 'type': 'line', 'name': 'Over-Production'},
                {'x': hours_list, 'y': heat_reduction,
                 'line': {'width': 3, 'dash': 'dot', 'color': 'hydrogen'},
                 'name': 'Load-Shedding Heat'},
                {'x': hours_list, 'y': heat_to_elec,
                 'type': 'line', 'name': 'Elec to Heat'},
            ],
            'layout': heat_layout,
        }
    
        cold_figure = {
            'data': [
                {'x': hours_list, 'y': [refrigeration_demand[h] for h in hours_list],
                 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': hours_list, 'y': cold_production,
                 'type': 'line', 'name': 'Production'},
                {'x': hours_list, 'y': cooling_elec,
                 'type': 'line', 'name': 'Cooling Electricity'},
                {'x': hours_list, 'y': cooling_heat,
                 'type': 'line', 'name': 'Cooling Heat'},
            ],
            'layout': cold_layout,
        }
    
        heat_storage_figure = {
            'data': [
                {'x': hours_list, 'y': normal_heat_stored.tolist(),
                 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
            ],
            'layout': storage_layout,
        }
    
        purchased_elec_figure = {
            'data': [
                {'x': hours_list, 'y': electricity_market.tolist(),
                 'type': 'line', 'name': 'Purchased Electricity', 'line': {'color': 'blue'}},
                {'x': hours_list, 'y': NG_market.tolist(),
                 'type': 'line', 'name': 'Purchased Fuel', 'line': {'color': 'red'}},
                {'x': hours_list, 'y': heat_market_sold.tolist(),
                 'line': {'width': 3, 'dash': 'dash', 'color': 'red'}, 'name': 'Sold Heat'},
                {'x': hours_list, 'y': electricity_market_sold.tolist(),
                 'line': {'width': 3, 'dash': 'dash', 'color': 'blue'}, 'name': 'Sold Electricity'},
            ],
            'layout': market_layout,
        }
    
        eff_figure = {
            'data': [
                {'x': hours_list, 'y': eff2,
                 'type': 'line', 'name': 'Thermal Efficiency'},
                {'x': hours_list, 'y': eff1,
                 'type': 'line', 'name': 'Electrical Efficiency'},
            ],
            'layout': eff_layout,
        }
    
        carbon_credits_figure = {
            'data': [
                {'x': [0], 'y': credits,
                 'type': 'line', 'name': 'Carbon Credits Purchased'},
                {'x': [0], 'y': carbon_sell,
                 'type': 'line', 'name': 'Carbon Credits Sold'},
                {'x': [0], 'y': carbon_held,
                 'type': 'line', 'name': 'Carbon Credits Held'},
                {'x': [0], 'y': carbon_earn,
                 'type': 'line', 'name': 'Carbon Credits Earned'},
            ],
            'layout': credits_layout,
        }
    
        fuel_blend_figure = {
            'data': [
                {'x': hours_list, 'y': [blend_H2[0]] * len(hours_list),
                 'type': 'line', 'name': 'Hydrogen'},
                {'x': hours_list, 'y': [blend_bm[0]] * len(hours_list),
                 'type': 'line', 'name': 'Biomass'},
                {'x': hours_list, 'y': [blend_ng[0]] * len(hours_list),
                 'type': 'line', 'name': 'Natural Gas'},
            ],
            'layout': blend_layout,
        }
    
        # Compute summary values as before
        total_purchased_electricity = sum([p + t for p, t in zip(purchased_electricity, heat_to_elec)])
        total_heat_cost = sum([f * ng for f, ng in zip(fuel_consumed, NG_market.tolist())])
        final_chp_capacity = CHP_capacity
        model_cost = df['model_cost'].iloc[0] if 'model_cost' in df.columns else 0
    
        return (electricity_figure, heat_figure, cold_figure, heat_storage_figure,
                purchased_elec_figure, eff_figure, carbon_credits_figure, fuel_blend_figure,
                f"Total Purchased Electricity: {locale.currency(total_purchased_electricity, grouping=True)}",
                f"Total Heat Cost: {locale.currency(total_heat_cost, grouping=True)}",
                f"Final CHP Capacity: {round(final_chp_capacity, 2)} KW",
                f"Energy Ratio: {round(energy_ratio, 2)}",
                f"Model Cost: {locale.currency(model_cost, grouping=True)}",
                f"Credit Cost: NaN")
    else:
        raise dash.exceptions.PreventUpdate


# ================================
# Functions for the Pyomo Model
# ================================

def save_model_results(model, results_filepath):
    """
    Extracts selected results from the model and saves them as a CSV file.
    Adjust the list of variables to save as needed.
    """
    hours = list(model.HOURS)
    results_data = {
        'hour': hours,
        'electricity_production': [model.electricity_production[h]() for h in hours],
        'heat_production': [model.heat_production[h]() for h in hours],
        'refrigeration_produced': [model.refrigeration_produced[h]() for h in hours],
        'purchased_electricity': [model.purchased_electricity[h]() for h in hours],
        'heat_stored': [model.heat_stored[h]() for h in hours],
        'electrical_efficiency': [model.electrical_efficiency[h]() for h in hours],
        'thermal_efficiency': [model.thermal_efficiency[h]() for h in hours],
        # Dummy columns; update these if your model produces these values.
        'carbon_credits': [0]*len(hours),
        'fuel_blend_ng': [0]*len(hours),
        'fuel_blend_h2': [0]*len(hours),
        'fuel_blend_biomass': [0]*len(hours),
        # Save the model cost (same value repeated for all hours)
        'model_cost': [model.objective()] * len(hours)
    }
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(results_filepath, index=False)
    print(f"Model results saved to {results_filepath}")

# Example helper functions for investment decisions (adjust as needed)
def check_eb_active(year):
    investment_row = investment_data[investment_data['Interval'] == year]
    if not investment_row.empty:
        print("EB Active")
        return investment_row.iloc[0]['Active EB'] == 1.0
    print("EB In-active")
    return False

def check_h2_active(year):
    investment_row = investment_data[investment_data['Interval'] == year]
    if not investment_row.empty:
        print("H2 Active")
        return investment_row.iloc[0]['Active H2'] == 1.0
    print("H2 In-active")
    return False

# Global parameters for the model.
total_hours = 8760
time_limit = 300
plant_flexibility = 0.5

def pyomomodel(total_hours=total_hours, time_limit=time_limit, CHP_capacity=CHP_capacity, 
                energy_ratio=energy_ratio, eb_allowed=False):
    model = ConcreteModel()
    
    # -----------------------
    # Time Sets and Parameters
    # -----------------------
    HOURS = np.arange(total_hours)
    model.HOURS = Set(initialize=HOURS)
    no_intervals = 1
    intervals_time = int(total_hours / no_intervals)
    INTERVALS = np.arange(no_intervals)
    model.INTERVALS = Set(initialize=INTERVALS)
    
    # -----------------------
    # Other Parameters
    # -----------------------
    storage_efficiency = 0.5
    withdrawal_efficiency = 0.5
    max_storage_capacity = 1
    heat_storage_loss_factor = 0.95
    COP_h = 2
    COP_e = 1
    max_ramp_rate = 10
    fuel_energy = 1
    max_co2_emissions = (markets["Effective Carbon Credit Cap"]) * 0
    M = 1E6

    # -----------------------
    # Decision Variables
    # -----------------------
    max_heat_capacity = CHP_capacity
    max_electricity_capacity = CHP_capacity * energy_ratio
    model.heat_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_heat_capacity))
    model.electricity_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_electricity_capacity))
    model.fuel_consumed = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 3))
    model.electrical_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(1, 1))
    model.thermal_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(1, 1))
    model.fuel_blend_ng = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_h2 = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_biomass = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.invest_h2 = Var(within=Binary)
    model.invest_time = Var(model.INTERVALS, within=Binary)
    model.active_h2_blending = Var(model.INTERVALS, within=Binary)
    model.heat_to_plant = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)))
    model.elec_to_plant = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.useful_heat = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)))
    model.useful_elec = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.purchased_electricity = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.heat_stored = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_storage_capacity))
    model.heat_withdrawn = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_storage_capacity))
    model.heat_over_production = Var(model.HOURS, within=NonNegativeReals)
    model.electricity_over_production = Var(model.HOURS, within=NonNegativeReals)
    model.ramp_rate = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_ramp_rate))
    model.refrigeration_produced = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.heat_used_for_cooling = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.elec_used_for_cooling = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.co2_emissions = Var(model.HOURS, within=NonNegativeReals)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)
    model.carbon_credits = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_purchased = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_earned = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions)))
    model.credits_sold = Var(model.INTERVALS, within=NonNegativeReals)
    model.exceeds_cap = Var(model.INTERVALS, within=Binary)
    model.credits_held = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions)))
    model.emissions_difference = Var(model.INTERVALS, domain=Reals, bounds=(-max(max_co2_emissions), max(max_co2_emissions)))
    model.credits_used_to_offset = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions)))
    model.below_cap = Var(model.INTERVALS, within=Binary)
    model.production_output = Var(model.HOURS, within=NonNegativeReals)
    model.elec_reduction_by_CHP = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.heat_reduction_by_CHP = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)))
    model.grid_reduction_shortfall = Var(model.HOURS, within=NonNegativeReals)
    model.heat_to_elec = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)))
    model.heat_to_elec_allowed = Var(within=Binary)
    
    # -----------------------
    # Constraints (a representative subset)
    # -----------------------
    def heat_over_production_rule(model, h):
        return model.heat_over_production[h] == model.heat_production[h] - model.useful_heat[h]
    model.heat_over_production_constraint = Constraint(model.HOURS, rule=heat_over_production_rule)

    def elec_over_production_rule(model, h):
        return model.electricity_over_production[h] == model.electricity_production[h] - model.useful_elec[h]
    model.elec_over_production_constraint = Constraint(model.HOURS, rule=elec_over_production_rule)

    def useful_heat_rule(model, h):
        return model.useful_heat[h] == model.heat_to_plant[h] + (model.heat_withdrawn[h] * withdrawal_efficiency) + (model.heat_used_for_cooling[h] / COP_h)
    model.useful_heat_constraint = Constraint(model.HOURS, rule=useful_heat_rule)

    def useful_elec_rule(model, h):
        return model.useful_elec[h] == model.elec_to_plant[h] + (model.elec_used_for_cooling[h] / COP_e)
    model.useful_elec_constraint = Constraint(model.HOURS, rule=useful_elec_rule)

    def capacity_rule(model, h):
        return model.heat_production[h] + model.electricity_production[h] <= CHP_capacity
    model.capacity_constraint = Constraint(model.HOURS, rule=capacity_rule)

    def fuel_consumed_rule(model, h):
        return fuel_energy * model.fuel_consumed[h] * (1 - energy_ratio) == model.heat_production[h]
    model.fuel_consumed_constraint = Constraint(model.HOURS, rule=fuel_consumed_rule)

    def energy_ratio_rule(model, h):
        return model.electricity_production[h] == energy_ratio * model.heat_production[h]
    model.energy_ratio_constraint = Constraint(model.HOURS, rule=energy_ratio_rule)

    def fuel_blend_rule(model, i):
        return model.fuel_blend_ng[i] + model.fuel_blend_h2[i] + model.fuel_blend_biomass[i] == 1
    model.fuel_blend_constraint = Constraint(model.INTERVALS, rule=fuel_blend_rule)

    def ramp_up_rule(model, h):
        if h == 0:
            return Constraint.Skip
        return model.heat_production[h] - model.heat_production[h - 1] <= model.ramp_rate[h]
    model.ramp_up_constraint = Constraint(model.HOURS, rule=ramp_up_rule)

    def ramp_down_rule(model, h):
        if h == 0:
            return Constraint.Skip
        return model.heat_production[h - 1] - model.heat_production[h] <= model.ramp_rate[h]
    model.ramp_down_constraint = Constraint(model.HOURS, rule=ramp_down_rule)

    def max_ramp_rule(model, h):
        return model.ramp_rate[h] <= max_ramp_rate
    model.max_ramp_constraint = Constraint(model.HOURS, rule=max_ramp_rule)

    def initial_heat_stored_rule(model):
        return model.heat_stored[0] == 0
    model.initial_heat_stored_constraint = Constraint(rule=initial_heat_stored_rule)

    def storage_dynamics_rule(model, h):
        if h == 0:
            return Constraint.Skip
        return model.heat_stored[h] == heat_storage_loss_factor * (model.heat_stored[h - 1] + (model.heat_over_production[h] * storage_efficiency) - (model.heat_withdrawn[h] / withdrawal_efficiency))
    model.storage_dynamics = Constraint(model.HOURS, rule=storage_dynamics_rule)

    def storage_capacity_rule(model, h):
        return model.heat_stored[h] <= max_storage_capacity
    model.storage_capacity = Constraint(model.HOURS, rule=storage_capacity_rule)

    def refrigeration_balance_rule(model, h):
        return model.refrigeration_produced[h] == (model.elec_used_for_cooling[h] * COP_e) + (model.heat_used_for_cooling[h] * COP_h)
    model.refrigeration_balance = Constraint(model.HOURS, rule=refrigeration_balance_rule)

    def refrigeration_demand_rule(model, h):
        return model.refrigeration_produced[h] == refrigeration_demand[h]
    model.refrigeration_demand_constraint = Constraint(model.HOURS, rule=refrigeration_demand_rule)

    def co2_emissions_rule(model, h):
        i = min(h // intervals_time, no_intervals - 1)
        return model.co2_emissions[h] == (
            em_ng[i] * model.fuel_blend_ng[i] * model.fuel_consumed[h] +
            em_h2[i] * model.fuel_blend_h2[i] * model.fuel_consumed[h] +
            em_bm[i] * model.fuel_blend_biomass[i] * model.fuel_consumed[h] +
            em_elec[i] * model.purchased_electricity[h]
        )
    model.co2_emissions_constraint = Constraint(model.HOURS, rule=co2_emissions_rule)

    def total_emissions_per_interval_rule(model, i):
        start = i * intervals_time
        end = (i + 1) * intervals_time
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[h] for h in range(start, end))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    def carbon_credits_needed_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] >= model.total_emissions_per_interval[i] - max_co2_emissions[i]
    model.carbon_credits_needed_constraint = Constraint(model.INTERVALS, rule=carbon_credits_needed_rule)

    def carbon_credits_earned_rule(model, i):
        return model.credits_earned[i] == (max_co2_emissions[i] - model.total_emissions_per_interval[i] + model.credits_used_to_offset[i]) * (1 - model.below_cap[i])
    model.carbon_credits_earned_constraint = Constraint(model.INTERVALS, rule=carbon_credits_earned_rule)

    def carbon_credits_purchased_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] <= M * model.below_cap[i]
    model.carbon_credits_purchased_constraint = Constraint(model.INTERVALS, rule=carbon_credits_purchased_rule)

    def credits_unheld_limit_rule(model, i):
        if i == 0:
            return model.credits_sold[i] == 0
        return model.credits_sold[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_used_to_offset[i]
    model.credits_unheld_limit = Constraint(model.INTERVALS, rule=credits_unheld_limit_rule)

    def credits_held_dynamics_rule(model, i):
        if i == 0:
            return model.credits_held[i] == model.credits_earned[i]
        return model.credits_held[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_sold[i] - model.credits_used_to_offset[i]
    model.credits_held_dynamics = Constraint(model.INTERVALS, rule=credits_held_dynamics_rule)

    def below_cap_rule(model, i):
        return model.total_emissions_per_interval[i] - max_co2_emissions[i] <= M * model.below_cap[i]
    model.below_cap_constraint = Constraint(model.INTERVALS, rule=below_cap_rule)

    def force_0_rule(model, i):
        if i == 0:
            return model.credits_used_to_offset[i] == 0
        return Constraint.Skip
    model.force_0_constraint = Constraint(model.INTERVALS, rule=force_0_rule)

    def production_output_rule(model, h):
        production_output_ratio = 0.8
        return model.production_output[h] == production_output_ratio * (heat_demand[h] - model.elec_reduction_by_CHP[h])
    model.production_output_constraint = Constraint(model.HOURS, rule=production_output_rule)

    def grid_call_constraint(model, h):
        return model.elec_reduction_by_CHP[h] + model.grid_reduction_shortfall[h] <= request[h]
    model.grid_call_constraint = Constraint(model.HOURS, rule=grid_call_constraint)

    def flexibility_constraint(model, h):
        return model.elec_reduction_by_CHP[h] <= plant_flexibility * electricity_demand[h]
    model.flexibility_constraint = Constraint(model.HOURS, rule=flexibility_constraint)

    def updated_elec_demand_balance_rule(model, h):
        return model.elec_to_plant[h] + model.purchased_electricity[h] == electricity_demand[h] - model.elec_reduction_by_CHP[h]
    model.updated_elec_demand_constraint = Constraint(model.HOURS, rule=updated_elec_demand_balance_rule)

    def updated_heat_demand_balance_rule(model, h):
        return model.heat_to_plant[h] + model.heat_to_elec[h] == heat_demand[h] - model.heat_reduction_by_CHP[h]
    model.updated_heat_demand_constraint = Constraint(model.HOURS, rule=updated_heat_demand_balance_rule)

    def eb_activation_rule(model, h):
        M_large = M
        eb_bin = 0  # Set to 0 to disable electricity-to-heat conversion
        return model.heat_to_elec[h] <= (eb_bin * M_large)
    model.eb_activation_constraint = Constraint(model.HOURS, rule=eb_activation_rule)

    def objective_rule(model):
        def get_interval(h):
            return min(h // intervals_time, no_intervals - 1)
        elec_cost = sum((model.purchased_electricity[h] + model.heat_to_elec[h]) * electricity_market[h] for h in model.HOURS)
        elec_sold = sum(model.electricity_over_production[h] * electricity_market_sold[h] for h in model.HOURS)
        heat_sold = sum(heat_market_sold[h] * model.heat_over_production[h] for h in model.HOURS)
        fuel_cost_NG = sum(model.fuel_blend_ng[get_interval(h)] * NG_market[h] * model.fuel_consumed[h] for h in model.HOURS)
        fuel_cost_H2 = sum(model.fuel_blend_h2[get_interval(h)] * H2_market[h] * model.fuel_consumed[h] for h in model.HOURS)
        fuel_cost_BM = sum(model.fuel_blend_biomass[get_interval(h)] * BM_market[h] * model.fuel_consumed[h] for h in model.HOURS)
        carbon_cost = sum(model.carbon_credits[i] * carbon_market[i] for i in model.INTERVALS)
        carbon_sold = 0.6 * sum(model.credits_sold[i] * carbon_market[i] for i in model.INTERVALS)
        production_revenue = sum(model.production_output[h] for h in model.HOURS)
        ancillary_revenue = sum(reward[h] * model.elec_reduction_by_CHP[h] for h in model.HOURS)
        shortfall_penalty_total = sum(shortfall_penalty[h] * model.grid_reduction_shortfall[h] for h in model.HOURS)
        total_cost = (fuel_cost_NG + fuel_cost_H2 + fuel_cost_BM) + elec_cost + carbon_cost + shortfall_penalty_total - (elec_sold + heat_sold + carbon_sold + production_revenue + ancillary_revenue)
        return total_cost

    model.objective = Objective(rule=objective_rule, sense=minimize)
    
    # -----------------------
    # Solve the Model
    # -----------------------
    solver = get_solver(time_limit)
    solver.options['mipgap'] = 0.01
    try:
        solver.solve(model, tee=True, symbolic_solver_labels=False)
        save_model_results(model, results_filepath)
    except:
    # Save the model results to the CSV file.
        save_model_results(model, results_filepath)
    return model

# ------------------------------
# Callback to Re-Solve the Model Periodically
# ------------------------------
@app.callback(
    Output('model-update-output', 'children'),
    [Input('interval-model-update', 'n_intervals')]
)
def update_model(n):
    # Re-run the Pyomo model and save the results.
    pyomomodel()
    return f"Model updated at {datetime.now()}"

# ================================
# Run the Dash App
# ================================
if __name__ == '__main__':
    app.run_server(debug=True)
