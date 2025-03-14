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
# ================================
# Define File Paths and Read Data
# ================================
eta_h2 = 0.85
eta_ng = 0.7
current_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
results_dir = os.path.abspath(os.path.join(current_dir, '..', 'results'))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
demands_path = os.path.join(data_dir, 'demands_hourly_year.csv')
markets_path = os.path.join(data_dir, 'markets.csv')

# Read input CSV files (here we read only the first 8761 rows as in your example) 
demands = pd.read_csv(demands_path, nrows=8761)
markets = pd.read_csv(markets_path, nrows=8761)

# Set up demand, penalty/reward and market parameters (adjust as needed)
shortfall_penalty = demands["penalty"].to_numpy()
reward = demands["revenue"].to_numpy() 
request = demands["request"].to_numpy() 

electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()

resin_per_tonne = 1000

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

max_co2_emissions = markets["Effective Carbon Credit Cap"] / 12 # tonnes CO2
# Other model parameters
CHP_capacity = 15
energy_ratio = 0.25

margin = markets["Input Margin ($/tonne) PV"]
labour = markets["Fixed Cost ($/tonne)"]
app = dash.Dash(__name__)

# ------------------------------
# Updated Layout: add two new graphs
# ------------------------------
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
    dcc.Graph(id='eff-graph'),
    dcc.Graph(id='carbon-credits'),
    dcc.Graph(id='fuel_blend'),
    # New graphs:
    dcc.Graph(id='ancillary-revenue-graph'),
    dcc.Graph(id='heat-production-graph'),
    
    dcc.Graph(id='empty-graph3'),
    dcc.Graph(id='empty-graph4'),
    dcc.Graph(id='empty-graph5'),
    
    # Interval component
    dcc.Interval(
        id='interval-component',
        interval=10 * 10000,  # Refresh every 10 seconds
        n_intervals=0
    )
])

# ------------------------------
# Update graph height callback: add outputs for new graphs
# ------------------------------
@app.callback(
    Output('electricity-graph', 'style'),
    Output('heat-graph', 'style'),
    Output('cold-graph', 'style'),
    Output('heat-store-graph', 'style'),
    Output('purchased_elec', 'style'),
    Output('eff-graph', 'style'),
    Output('carbon-credits', 'style'),
    Output('fuel_blend', 'style'),
    Output('ancillary-revenue-graph', 'style'),  # New
    Output('heat-production-graph', 'style'),     # New
    Output('empty-graph3', 'style'),
    Output('empty-graph4', 'style'),
    Output('empty-graph5', 'style'),
    [Input('height-slider', 'value')]
)
def update_graph_height(height_value):
    style = {'height': f'{height_value}px'}
    # Return style for each graph (13 outputs in total now)
    return style, style, style, style, style, style, style, style, style, style, style, style, style

# ------------------------------
# Update graphs callback: add new figures for ancillary revenue and heat production
# ------------------------------
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
    Output('ancillary-revenue-graph', 'figure'),  # New output
    Output('heat-production-graph', 'figure'),     # New output
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n_intervals):
    global last_mod_time  # Declare as global to modify it

    # Check last modification time of the model file
    current_mod_time = os.path.getmtime(current_dir)

    if current_mod_time > last_mod_time:
        last_mod_time = current_mod_time  # Update last modification time

        model = pyomomodel()

        months_list = list(model.HOURS)
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
        total_heat_cost = sum(model.fuel_consumed[m].value * NG_market[m] for m in months_list)
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

        blend_H2 = [model.fuel_blend_h2[m]() for m in intervals]
        blend_ng = [model.fuel_blend_ng[m]() for m in intervals]
        blend_bm = [model.fuel_blend_biomass[m]() for m in intervals]

        # New: grid call variable (grid reduction shortfall)
        grid_call = [model.elec_reduction[m]() for m in months_list]

        production_amount = [model.production_output[m]() for m in months_list]
        # Create figures based on these results

        base_layout_template = {
            'template': 'plotly_dark',
            'font': {'family': "Courier New, monospace", 'size': 12, 'color': "RebeccaPurple", 'weight': 'bold'},
            'xaxis': {'title': 'Hours', 'weight': 'bold'},
            'yaxis': {'title': 'kWh', 'weight': 'bold'},
            'title_font': {'size': 24, 'family': 'Arial, sans-serif', 'weight': 'bold'},
        }
        electricity_layout = base_layout_template.copy()
        electricity_layout.update({'title': 'Electricity Demand and Production', 'xaxis': {'title': 'Months'}, 'yaxis': {'title': 'MWh'}})

        heat_layout = base_layout_template.copy()
        heat_layout.update({'title': 'Heat Demand and Production', 'xaxis': {'title': 'Months'}, 'yaxis': {'title': 'MWh'}})

        cold_layout = base_layout_template.copy()
        cold_layout.update({'title': 'Cold Demand and Production', 'xaxis': {'title': 'Months'}, 'yaxis': {'title': 'MWh'}})

        
        storage_layout = base_layout_template.copy()
        storage_layout.update({'title': 'Normalised Heat Storage', 'xaxis': {'title': 'Months'}, 'yaxis': {'title': '%'}})

        fridge_layout = base_layout_template.copy()
        fridge_layout.update({'title': 'Normalised Heat Storage', 'xaxis': {'title': 'Months'}, 'yaxis': {'title': '%'}})

        market_layout = base_layout_template.copy()
        market_layout.update({'title': 'Energy Market', 'xaxis': {'title': 'Months'}, 'yaxis': {'title': '$/MWh'}})

        eff_layout = base_layout_template.copy()
        eff_layout.update({'title': 'Efficiency', 'xaxis': {'title': 'Months'}, 'yaxis': {'title': 'Efficiency'}})

        credits_layout = base_layout_template.copy()
        credits_layout.update({'title': 'Carbon Credit Dynamics', 'xaxis': {'title': 'Years'}, 'yaxis': {'title': 'No of Credits'}})

        blend_layout = base_layout_template.copy()
        blend_layout.update({'title': 'Fuel Blending', 'xaxis': {'title': 'Years'}, 'yaxis': {'title': '% ratio of fuels'}})

        # ------------------------------
        # New layouts for added graphs:
        ancillary_layout = {
            'title': 'Ancillary Revenue Overview',
            'xaxis': {'title': 'Hours'},
            'yaxis': {'title': 'Value'},
            'template': 'plotly_dark'
        }
        heat_prod_layout = {
            'title': 'Heat Production vs Demand',
            'xaxis': {'title': 'Hours'},
            'yaxis': {'title': 'MWh'},
            'template': 'plotly_dark'
        }
        # ------------------------------

        electricity_figure = {
            'data': [
                {'x': list(model.HOURS), 'y': [electricity_demand[m] for m in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(model.HOURS), 'y': electricity_production, 'type': 'line', 'name': 'Production', 'line': {'color': 'blue'}},
                {'x': list(model.HOURS), 'y': [purchased_electricity[m] for m in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dash', 'line': {'color': 'hydrogen'}}, 'name': 'Purchased Electricity'},
                {'x': list(model.HOURS), 'y': [over_electricity[m] for m in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dot', 'color': 'hydrogen'}, 'name': 'Over-Produced Electricity'}
            ],
            'layout': electricity_layout,
        }

        heat_figure = {
            'data': [
                {'x': list(model.HOURS), 'y': [heat_demand[m] for m in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
                {'x': list(model.HOURS), 'y': heat_production, 'type': 'line', 'name': 'Production'},
                {'x': list(model.HOURS), 'y': heat_stored, 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored'},
                {'x': list(model.HOURS), 'y': over_heat, 'type': 'line', 'name': 'Over-Production'},
                {'x': list(model.HOURS), 'y': heat_to_elec, 'type': 'line', 'name': 'Elec to Heat'},
            ],
            'layout': heat_layout,
        }

        cold_figure = {
            'data': [
                {'x': list(model.HOURS), 'y': [refrigeration_demand[m] for m in list(model.HOURS)], 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
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
                {'x': list(model.HOURS), 'y': credits, 'type': 'line', 'name': 'Carbon Credits Purchased'},
                {'x': list(model.HOURS), 'y': carbon_sell, 'type': 'line', 'name': 'Carbon Credits Sold'},
                {'x': list(model.HOURS), 'y': carbon_held, 'type': 'line', 'name': 'Carbon Credits Held'},
                {'x': list(model.HOURS), 'y': carbon_earn, 'type': 'line', 'name': 'Carbon Credits Earned'},
            ],
            'layout': credits_layout,
        }
        
        fuel_blend_figure = {
            'data': [
                {'x': list(intervals), 'y': blend_H2, 'type': 'line', 'name': 'Hydrogen'},
                {'x': list(intervals), 'y': blend_bm, 'type': 'line', 'name': 'Biomass'},
                {'x': list(intervals), 'y': blend_ng, 'type': 'line', 'name': 'Natural Gas'},
            ],
            'layout': blend_layout,
        }

        # ------------------------------
        # New Figures:
        # Ancillary Revenue Overview: Purchased Electricity vs Grid Call
        ancillary_revenue_figure = {
            'data': [
                {'x': list(model.HOURS), 'y': [model.purchased_electricity[h]() for h in list(model.HOURS)], 'type': 'line', 'name': 'Purchased Electricity'},
                {'x': list(model.HOURS), 'y': grid_call, 'type': 'line', 'name': 'Grid Call'},
            ],
            'layout': ancillary_layout,
        }

        # Heat Production vs Demand graph
        heat_production_figure = {
            'data': [
                {'x': list(model.HOURS), 'y': production_amount, 'type': 'line', 'name': 'Heat Production'},
            ],
            'layout': heat_prod_layout,
        }
        # ------------------------------

        # Return the updated figures (note the new figures added at the end)
        return (electricity_figure, heat_figure, cold_figure, heat_storage, purchased_elec, 
                eff_figure, carbon_credits_figure, fuel_blend_figure, 
                f"Total Purchased Electricity: {locale.currency(total_purchased_electricity, grouping=True)}", 
                f"Total Heat Cost: {locale.currency(total_heat_cost, grouping=True)}", 
                f"Final CHP Capacity: {round(final_chp_capacity, 2)} KW", 
                f"Energy Ratio: {round(energy_ratio, 2)}", 
                f"Model Cost: {locale.currency(model_cost, grouping=True)}", 
                f"Credit Cost: NaN",
                ancillary_revenue_figure,  # New
                heat_production_figure)     # New
    else:
        raise dash.exceptions.PreventUpdate

total_hours = 8760
time_limit = 300
def pyomomodel(total_hours=8760, time_limit=300, CHP_capacity=15, energy_ratio=0.25,
               eb_allowed=0, h2_allowed=0, ccs_allowed=0, demand_data=None, market_data=None,
               warm_start_values=None):    
    # Determine scaling factor locally (not a model variable)
    scaling_factor = 3.0 if eb_allowed else 1.0
    """
    Hourly Pyomo model that:
      - Uses hourly data (indexed by model.HOURS).
      - Uses INTERVALS (e.g., months) for fuel blending decisions and carbon credits.
      - Keeps fuel blending decisions as optimization variables (on an interval basis).
      - Externally fixes the investment decisions (for CCS, H2, and Electrification) via parameters.
    
    Parameters:
      total_hours  : total number of hours (e.g., 8760 for one year)
      time_limit   : solver time limit (in seconds)
      CHP_capacity : CHP capacity (kW)
      energy_ratio : ratio linking heat production to electricity production
    """
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
    results_dir = os.path.abspath(os.path.join(current_dir, '..', 'results'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # -------------------- Demand Data --------------------
    unit_conv = 1E3
    if demand_data is None:
        demands_path = os.path.join(data_dir, 'demands_hourly_year.csv')
        # For backwards compatibility, read first year
        demands = pd.read_csv(demands_path, nrows=8761)
        shortfall_penalty = demands["penalty"].to_numpy()* unit_conv
        reward = demands["revenue"].to_numpy() * unit_conv
        request = demands["request"].to_numpy()
        electricity_demand = demands["elec"].to_numpy()
        heat_demand = demands["heat"].to_numpy()
        refrigeration_demand = demands["cool"].to_numpy()
    else:
        shortfall_penalty = demand_data["penalty"]
        reward = demand_data["revenue"]
        request = demand_data["request"]
        electricity_demand = demand_data["elec"]
        heat_demand = demand_data["heat"]
        refrigeration_demand = demand_data["cool"]
    
    # -------------------- Market Data --------------------
    if market_data is None:
        markets_path = os.path.join(data_dir, 'markets.csv')
        markets = pd.read_csv(markets_path, nrows=8761)
        electricity_market = markets["Electricity Price ($/kWh)"].to_numpy() * unit_conv 
        electricity_market_sold = electricity_market * 1E-3
        carbon_market = markets["Carbon Credit Price ($/tonne CO2)"].to_numpy()
        NG_market = markets["Natural Gas Price ($/kWh)"].to_numpy() * unit_conv
        heat_market_sold = NG_market * 1E-3
        H2_market = markets["Hydrogen Price ($/kWh)"].to_numpy() * unit_conv
        BM_market = markets["Biomass Price ($/kWh)"].to_numpy() * unit_conv
        em_bm = markets["Biomass Carbon Intensity (kg CO2/kWh)"].to_numpy()
        em_h2 = markets["Hydrogen Carbon Intensity (kg CO2/kWh)"].to_numpy()
        em_ng = markets["Natural Gas Carbon Intensity (kg CO2/kWh)"].to_numpy()
        em_elec = markets["Grid Carbon Intensity (kg CO2/kWh)"].to_numpy()
        max_co2_emissions = markets["Effective Carbon Credit Cap"].to_numpy() / 12
        margin = markets["Input Margin ($/tonne) PV"]
        labour = markets["Fixed Cost ($/tonne)"]
    else:
        electricity_market = market_data["electricity_market"]
        electricity_market_sold = market_data["electricity_market_sold"]
        carbon_market = market_data["carbon_market"]
        NG_market = market_data["NG_market"]
        heat_market_sold = market_data["heat_market_sold"]
        H2_market = market_data["H2_market"]
        BM_market = market_data["BM_market"]
        em_bm = market_data["em_bm"]
        em_h2 = market_data["em_h2"]
        em_ng = market_data["em_ng"]
        em_elec = market_data["em_elec"]
        max_co2_emissions = market_data["max_co2_emissions"]
        margin = market_data["margin"]
        labour = market_data["labour"]

    active_eb = eb_allowed
    active_h2 = h2_allowed
    active_ccs = ccs_allowed
    model = ConcreteModel()

    # -------------- Sets --------------
    # Define HOURS (0, 1, ..., total_hours-1)
    HOURS = np.arange(total_hours)
    model.HOURS = Set(initialize=HOURS)

    # Define INTERVALS to represent months (e.g., 12 intervals if total_hours=8760)
    no_intervals = 1
    intervals_time = int(total_hours / no_intervals)
    INTERVALS = np.arange(no_intervals)
    model.INTERVALS = Set(initialize=INTERVALS)

    # -------------- Parameters (Fixed Data) --------------
    storage_efficiency = 0.5
    withdrawal_efficiency = 0.5
    max_storage_capacity = 0      # kW
    heat_storage_loss_factor = 0.95

    COP_h = 2
    COP_e = 1
    
    max_ramp_rate = 1        # kW per timestep


    co2_stream_temp = 400
    co2_stream_pressure = 10
    co2_concentration = 0.2

    capex_factor = ((1 + 0.001 * (co2_stream_temp - 300)) *
                    (1 - 0.02 * (co2_stream_pressure - 10)))
    ccs_capex_dict = {i: 1E8 * capex_factor for i in INTERVALS}

    # CCS transport and storage costs
    transport_cost_per_kg_co2 = 0.05   # $ per kg CO₂ transported
    storage_cost_per_kg_co2 = 0.03     # $ per kg CO₂ stored
    res_emissions = 0.55
    ccs_energy_penalty_factor = 0.02
    M = 1E6

    # Adjusted capture efficiency (computed outside constraints)
    base_capture_efficiency = 0.9
    stream_factor = ((1 - 0.0001 * (co2_stream_temp - 300)) *
                     (1 + 0.05 * (co2_stream_pressure - 10)) *
                     (1 + 0.2 * (co2_concentration - 0.2)))
    adjusted_capture_efficiency_value = base_capture_efficiency * stream_factor

    # ----------------- Investment Parameters (Fixed Externally) -----------------
    # These binary parameters indicate whether a technology is active in an interval.
    # (They are not decision variables here.)
    model.active_h2_blending = Param(model.INTERVALS, within=Binary, mutable=True, default=0)
    model.active_eb = Param(model.INTERVALS, within=Binary, mutable=True, default=1)
    model.active_ccs = Param(model.INTERVALS, within=Binary, mutable=True, default=0)

    # ----------------- Decision Variables -----------------
    # Hourly operational variables
    model.electricity_production = Var(model.HOURS, within=NonNegativeReals)
    model.heat_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)*2))
    model.fuel_consumed = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)*3))
    model.heat_to_elec = Var(model.HOURS, within=NonNegativeReals)
    model.electrical_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(1, 1), initialize=1)
    model.thermal_efficiency = Var(model.HOURS, within=NonNegativeReals, bounds=(1, 1), initialize=1)
    
    model.heat_to_plant = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)*2), initialize=max(heat_demand))
    model.elec_to_plant = Var(model.HOURS, within=NonNegativeReals)
    model.useful_heat = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)*2))
    model.useful_elec = Var(model.HOURS, within=NonNegativeReals)

    model.purchased_electricity = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)*2))
    model.heat_stored = Var(model.HOURS, within=NonNegativeReals)
    model.heat_withdrawn = Var(model.HOURS, within=NonNegativeReals)

    model.heat_over_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(heat_demand)))
    model.electricity_over_production = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(electricity_demand)))
    model.ramp_rate = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max_ramp_rate),initialize=max_ramp_rate)

    model.refrigeration_produced = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.heat_used_for_cooling = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.elec_used_for_cooling = Var(model.HOURS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))

    model.co2_emissions = Var(model.HOURS, within=NonNegativeReals, initialize=0)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals, initialize=0)
    model.carbon_credits = Var(model.INTERVALS, within=NonNegativeReals, initialize=0)
    model.credits_purchased = Var(model.INTERVALS, within=NonNegativeReals, initialize=0)
    model.credits_earned = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions*10)), initialize=0)
    model.credits_sold = Var(model.INTERVALS, within=NonNegativeReals, initialize=0)
    model.exceeds_cap = Var(model.INTERVALS, within=Binary, initialize=0)
    model.credits_held = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions*10)))
    model.emissions_difference = Var(model.INTERVALS, domain=Reals,
                                     bounds=(-max(max_co2_emissions*10), max(max_co2_emissions*10)), initialize=0)
    model.credits_used_to_offset = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions*10)))
    model.below_cap = Var(model.INTERVALS, within=Binary, initialize=1)

    model.captured_co2 = Var(model.HOURS, within=NonNegativeReals)
    model.uncaptured_co2 = Var(model.HOURS, within=NonNegativeReals)
    model.ccs_energy_penalty = Var(model.HOURS, within=NonNegativeReals)
    model.transport_cost = Var(model.INTERVALS, within=NonNegativeReals)

    # Auxiliary variable for linking fuel consumption and CO₂ with active CCS
    fuel_consumed_ub = model.fuel_consumed[0].ub
    total_fuel_co2_ub = fuel_consumed_ub
    model.total_fuel_co2_active_ccs = Var(model.HOURS, within=NonNegativeReals, initialize=0)

    # Fuel blending decision variables (defined on INTERVALS)
    model.fuel_blend_ng = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_h2 = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_biomass = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, 1))

    model.elec_reduction = Var(model.HOURS, within=NonNegativeReals)
    model.grid_reduction_shortfall = Var(model.HOURS, within=NonNegativeReals)
    model.production_output = Var(model.HOURS, within=NonNegativeReals)
    # ----------------- Expressions -----------------
    def total_fuel_co2_rule(model, h):
        interval = h // intervals_time
        return ((model.fuel_blend_ng[interval] * em_ng[h] / eta_ng) +
                (model.fuel_blend_h2[interval] * em_h2[h]/ eta_h2) +
                (model.fuel_blend_biomass[interval] * em_bm[h]) / eta_ng) * model.fuel_consumed[h]
    model.total_fuel_co2 = Expression(model.HOURS, rule=total_fuel_co2_rule)

    # ----------------- Constraints -----------------
    # (1) Fuel blending in each interval must sum to 1.
    def fuel_blend_constraint_rule(model, i):
        return model.fuel_blend_ng[i] + model.fuel_blend_h2[i] + model.fuel_blend_biomass[i] == 1
    model.fuel_blend_constraint = Constraint(model.INTERVALS, rule=fuel_blend_constraint_rule)

    # (2) Heat balance
    def heat_balance_rule(model, h):
        return model.heat_production[h] >= (model.heat_over_production[h] + model.useful_heat[h])
    model.heat_balance = Constraint(model.HOURS, rule=heat_balance_rule)

    # (3) Heat demand (including any CCS energy penalty)
    def heat_demand_balance(model, h):
        prod_elec_heat = 2 # <-- if we invest in an electric boiler, or 1 if we don't
        return (model.heat_to_plant[h]* (1-active_eb)) + model.heat_to_elec[h] >= heat_demand[h] + model.ccs_energy_penalty[h] - ((1 - active_eb)*prod_elec_heat + (active_eb))*model.elec_reduction[h]
    model.heat_demand_rule = Constraint(model.HOURS, rule=heat_demand_balance)

    # (4) Overproduction of heat
    def heat_over_production_rule(model, h):
        return model.heat_over_production[h] == model.heat_production[h] - model.useful_heat[h]
    model.heat_over_production_constraint = Constraint(model.HOURS, rule=heat_over_production_rule)

    # (5) Useful heat definition
    def useful_heat_rule(model, h):
        return model.useful_heat[h] == (model.heat_to_plant[h] * (1-active_eb))- (model.heat_withdrawn[h] * withdrawal_efficiency) + (model.heat_used_for_cooling[h] / COP_h)
    model.useful_heat_constraint = Constraint(model.HOURS, rule=useful_heat_rule)

    # (6) Electricity demand
    def elec_demand_balance(model, h):
        return model.elec_to_plant[h] + model.purchased_electricity[h] >= electricity_demand[h] - model.elec_reduction[h]
    model.elec_demand_rule = Constraint(model.HOURS, rule=elec_demand_balance)

    # (7) Useful electricity
    def useful_elec_rule(model, h):
        return model.useful_elec[h] == model.elec_to_plant[h] + (model.elec_used_for_cooling[h] / COP_e)
    model.useful_elec_constraint = Constraint(model.HOURS, rule=useful_elec_rule)

    # (8) Overproduction of electricity
    def elec_over_production_rule(model, h):
        return model.electricity_over_production[h] == model.electricity_production[h] - model.useful_elec[h]
    model.elec_over_production_constraint = Constraint(model.HOURS, rule=elec_over_production_rule)

    # (9) CHP capacity constraint
    def capacity_rule(model, h):
        return CHP_capacity >= (model.heat_production[h] + model.heat_stored[h]) + model.electricity_production[h]
    model.capacity_constraint = Constraint(model.HOURS, rule=capacity_rule)

    # (10) Fuel consumption relationship
    def fuel_consumed_rule(model, h):
        return model.fuel_consumed[h] * (1 - energy_ratio) == model.heat_production[h]
    model.fuel_consumed_rule_con = Constraint(model.HOURS, rule=fuel_consumed_rule)

    # (11) H2 blending activation:
    #     The fuel blend for H2 (a decision variable on the interval) may be >0 only if H2 is active.
    def h2_blending_activation_rule(model, h):
        interval = h // intervals_time
        return model.fuel_blend_h2[interval] <= active_h2
    model.h2_blending_activation_constraint = Constraint(model.HOURS, rule=h2_blending_activation_rule)

    # (12) Electrification activation: heat_to_elec is limited if Electrification is not active.
    def eb_activation_rule(model, h):
        interval = h // intervals_time
        return model.heat_to_elec[h] <= active_eb * M
    model.eb_activation_constraint = Constraint(model.HOURS, rule=eb_activation_rule)

    # (13) Initial heat stored is zero.
    def initial_heat_stored_rule(model):
        return model.heat_stored[0] == 0
    model.initial_heat_stored_constraint = Constraint(rule=initial_heat_stored_rule)

    # (14) Heat storage dynamics (for h > 0)
    def storage_dynamics_rule(model, h):
        if h == 0:
            return Constraint.Skip
        return model.heat_stored[h] == heat_storage_loss_factor * (model.heat_stored[h-1] +
               (model.heat_over_production[h] * storage_efficiency) -
               (model.heat_withdrawn[h] / withdrawal_efficiency))
    model.storage_dynamics = Constraint(model.HOURS, rule=storage_dynamics_rule)

    # (15) Heat storage capacity constraint
    def storage_capacity_rule(model, h):
        return model.heat_stored[h] <= max_storage_capacity
    model.storage_capacity = Constraint(model.HOURS, rule=storage_capacity_rule)

    # (16) Refrigeration balance
    def refrigeration_balance_rule(model, h):
        return model.refrigeration_produced[h] == (model.elec_used_for_cooling[h] * COP_e) + (model.heat_used_for_cooling[h] * COP_h)
    model.refrigeration_balance = Constraint(model.HOURS, rule=refrigeration_balance_rule)

    # (17) Refrigeration demand
    def refrigeration_demand_rule(model, h):
        return model.refrigeration_produced[h] == refrigeration_demand[h]
    model.refrigeration_demand_con = Constraint(model.HOURS, rule=refrigeration_demand_rule)

    # Total Emissions Per Interval Rule
    def total_emissions_per_interval_rule(model, i):
        start = i * intervals_time
        end = start + intervals_time
        # Sum both uncaptured fuel emissions and electricity-related emissions
        return model.total_emissions_per_interval[i] == sum(
            model.uncaptured_co2[h] + em_elec[h] * (model.purchased_electricity[h] + model.heat_to_elec[h])
            for h in range(start, end)
        )
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    def carbon_credits_needed_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] >= \
               model.total_emissions_per_interval[i] - max_co2_emissions[i * intervals_time]
    model.carbon_credits_needed_constraint = Constraint(model.INTERVALS, rule=carbon_credits_needed_rule)

    def carbon_credits_earned_rule(model, i):
        if i == 0:
            return model.credits_earned[i] == 0
        return model.credits_earned[i] == (max_co2_emissions[i * intervals_time] - model.total_emissions_per_interval[i] \
                                          + model.credits_used_to_offset[i]) * (1 - model.below_cap[i])
    model.carbon_credits_earned_constraint = Constraint(model.INTERVALS, rule=carbon_credits_earned_rule)

    def carbon_credits_purchased_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] <= M * model.below_cap[i]
    model.carbon_credits_purchased_con = Constraint(model.INTERVALS, rule=carbon_credits_purchased_rule)

    def credits_unheld_limit_rule(model, i):
        if i == 0:
            return model.credits_sold[i] == 0
        return model.credits_sold[i] <= model.credits_held[i-1] + model.credits_earned[i] - model.credits_used_to_offset[i]
    model.credits_unheld_limit = Constraint(model.INTERVALS, rule=credits_unheld_limit_rule)

    def credits_held_dynamics_rule(model, i):
        if i == 0:
            return model.credits_held[i] == model.credits_earned[i]
        return model.credits_held[i] <= model.credits_held[i-1] + model.credits_earned[i] - model.credits_sold[i] - model.credits_used_to_offset[i]
    model.credits_held_dynamics = Constraint(model.INTERVALS, rule=credits_held_dynamics_rule)

    def below_cap_rule(model, i):
        return model.total_emissions_per_interval[i] - max_co2_emissions[i * intervals_time] <= M * model.below_cap[i]
    model.below_cap_con = Constraint(model.INTERVALS, rule=below_cap_rule)

    def force_0_rule(model, i):
        if i == 0:
            return model.credits_used_to_offset[i] == 0
        return Constraint.Skip
    model.force_0_rule_con = Constraint(model.INTERVALS, rule=force_0_rule)

    # Captured CO2 constraint (hourly)
    def captured_co2_constraint(model, h):
        # In the monthly version, the captured CO2 was linked to an auxiliary variable that
        # linearized active_ccs * total_fuel_co2. Now we simply use the fixed activation.
        return model.captured_co2[h] <= active_ccs * adjusted_capture_efficiency_value * (model.total_fuel_co2[h] + res_emissions)
    model.captured_co2_constraint = Constraint(model.HOURS, rule=captured_co2_constraint)

    # Uncaptured CO2 constraint (hourly)
    def uncaptured_co2_constraint(model, h):
        # The idea is that if CCS is active, a portion of the fuel-related CO₂ is captured;
        # otherwise, it is left uncaptured (plus a residual emission term).
        return model.uncaptured_co2[h] == model.total_fuel_co2[h] + res_emissions * (1 - active_ccs) - model.captured_co2[h]
    model.uncaptured_co2_constraint = Constraint(model.HOURS, rule=uncaptured_co2_constraint)

    # CCS Energy Penalty (hourly)
    def ccs_energy_penalty_rule(model, h):
        # The energy penalty is proportional to the captured CO₂. If CCS is off (active_ccs = 0), no penalty applies.
        return model.ccs_energy_penalty[h] == ccs_energy_penalty_factor * model.captured_co2[h]
    model.ccs_energy_penalty_constraint = Constraint(model.HOURS, rule=ccs_energy_penalty_rule)

    # Electricity Production Constraint remains unchanged (hourly)
    def electricity_production_constraint(model, h):
        return model.electricity_production[h] == model.heat_production[h] * energy_ratio
    model.electricity_production_constraint = Constraint(model.HOURS, rule=electricity_production_constraint)

    # CCS Transport and Storage Costs (applied on intervals)
    def transport_and_storage_cost_rule(model, i):
        start = i * intervals_time
        end = start + intervals_time
        # Total captured CO₂ over the interval (e.g. one month)
        total_captured_co2 = sum(model.captured_co2[h] for h in range(start, end))
        return model.transport_cost[i] == total_captured_co2 * (transport_cost_per_kg_co2 + storage_cost_per_kg_co2)
    model.transport_storage_cost_constraint = Constraint(model.INTERVALS, rule=transport_and_storage_cost_rule)

#ANCHILLARY MARKETS
    #produciton output based off thermal demand
    def production_output_rule(model, h):
        return model.production_output[h] == 1.1 * ((heat_demand[h] + electricity_demand[h]) - model.elec_reduction[h])
    model.production_output_constraint = Constraint(model.HOURS, rule=production_output_rule)

    # Flexibility constraint: the maximum allowable reduction scales with the electricity demand.
    def flexibility_constraint(model, h):
        # Define plant flexibility based on electrification decision.
        if eb_allowed:
            plant_flexibility = 0.5
        else:
            plant_flexibility = 0.2
        return model.elec_reduction[h] <= plant_flexibility * electricity_demand[h] * scaling_factor
    model.flexibility_constraint = Constraint(model.HOURS, rule=flexibility_constraint)

    # Grid call constraint: the sum of reduction and shortfall is bounded by the grid request scaled appropriately.
    def grid_call_constraint(model, h):
        return model.elec_reduction[h] + model.grid_reduction_shortfall[h] <= request[h] * scaling_factor * 10
    model.grid_call_constraint = Constraint(model.HOURS, rule=grid_call_constraint)

    # ------------------ Cost Expressions ------------------
    # Electricity cost: cost of purchased electricity and converting heat to electricity.
    model.elec_cost = Expression(expr=sum(
        (model.purchased_electricity[h] + model.heat_to_elec[h]) * electricity_market[h]
        for h in model.HOURS))

    # Natural Gas fuel cost.
    model.fuel_cost_NG = Expression(expr=sum(
        (model.fuel_blend_ng[h // intervals_time] * NG_market[h] * model.fuel_consumed[h]) / eta_ng
        for h in model.HOURS))

    # Hydrogen fuel cost.
    model.fuel_cost_H2 = Expression(expr=sum(
        (model.fuel_blend_h2[h // intervals_time] * H2_market[h] * model.fuel_consumed[h]) / eta_h2
        for h in model.HOURS))

    # Biomass fuel cost.
    model.fuel_cost_BM = Expression(expr=sum(
        (model.fuel_blend_biomass[h // intervals_time] * BM_market[h] * model.fuel_consumed[h]) / eta_ng
        for h in model.HOURS))

    # Carbon cost from purchasing credits (indexed over intervals).
    model.carbon_cost = Expression(expr=sum(
        model.carbon_credits[i] * carbon_market[i]
        for i in model.INTERVALS))

    # Transport and storage cost for captured CO₂ (indexed over intervals).
    model.transport_storage_cost = Expression(expr=sum(
        model.transport_cost[i] for i in model.INTERVALS))

    # Shortfall penalty cost from grid reduction.
    model.shortfall_penalty_total = Expression(expr=sum(
        shortfall_penalty[h] * model.grid_reduction_shortfall[h] * scaling_factor
        for h in model.HOURS))


    model.labour = Expression(expr=sum(
        labour[i] * 100000
        for i in model.INTERVALS) / len(model.INTERVALS))
    # ------------------ Revenue Expressions ------------------
    # Revenue from selling excess electricity.
    model.elec_sold_expr = Expression(expr=sum(
        model.electricity_over_production[h] * electricity_market_sold[h]
        for h in model.HOURS))

    # Revenue from selling excess heat.
    model.heat_sold_expr = Expression(expr=sum(
        heat_market_sold[h] * model.heat_over_production[h]
        for h in model.HOURS))

    # Revenue from selling carbon credits.
    model.carbon_sold_expr = Expression(expr=sum(
        model.credits_sold[i] * carbon_market[i]
        for i in model.INTERVALS))

    # Production revenue (e.g., based on production output and a price per tonne).
    model.production_revenue = Expression(expr=sum(
        model.production_output[h] * margin[h] for h in model.HOURS) )

    model.ancillary_revenue = Expression(expr=sum(
        reward[h] * model.elec_reduction[h] * scaling_factor
        for h in model.HOURS))

    # ------------------ Aggregated Total Expressions ------------------
    # Total costs: Sum of all cost components.
    model.total_costs = Expression(expr=
        model.elec_cost +
        model.fuel_cost_NG +
        model.fuel_cost_H2 +
        model.fuel_cost_BM +
        model.carbon_cost +
        model.labour +
        model.transport_storage_cost +
        model.shortfall_penalty_total)

    # Total revenues: Sum of all revenue components.
    model.total_revenues = Expression(expr=
        model.elec_sold_expr +
        model.heat_sold_expr +
        model.carbon_sold_expr +
        model.ancillary_revenue +
        model.production_revenue)

    # ------------------ Objective Expression ------------------
    # Objective: Minimize the difference between total costs and total revenues.
    model.objective_expr = Expression(expr=model.total_costs - model.total_revenues)

    # Set the model objective using the expression.
    model.objective = Objective(expr=model.objective_expr, sense=minimize)

    # ----------------- Solve the Model -----------------
    solver = get_solver(time_limit)
    solver.solve(model, tee=True, symbolic_solver_labels=False)

    # ----------------- Extract and Save Results -----------------
    # Fuel blending results by interval (as these are decision variables)
    fuel_blend_ng_results = {i: value(model.fuel_blend_ng[i]) for i in model.INTERVALS}
    fuel_blend_h2_results = {i: value(model.fuel_blend_h2[i]) for i in model.INTERVALS}
    fuel_blend_biomass_results = {i: value(model.fuel_blend_biomass[i]) for i in model.INTERVALS}

    results_df = pd.DataFrame({
        'Interval': list(model.INTERVALS),
        'Fuel Blend NG': [fuel_blend_ng_results[i] for i in model.INTERVALS],
        'Fuel Blend H2': [fuel_blend_h2_results[i] for i in model.INTERVALS],
        'Fuel Blend Biomass': [fuel_blend_biomass_results[i] for i in model.INTERVALS]
    })
    #results_df.to_csv('hourly_fuel_blend_results.csv', index=False)

    # Save the externally set investment (technology activation) information
    investment_info = pd.DataFrame({
        'Interval': list(model.INTERVALS),
        'Active H2': [value(model.active_h2_blending[i]) for i in model.INTERVALS],
        'Active EB': [value(model.active_eb[i]) for i in model.INTERVALS],
        'Active CCS': [value(model.active_ccs[i]) for i in model.INTERVALS]
    })
    #investment_info.to_csv('investment_info.csv', index=False)

    return model


gc.enable()  # Re-enable garbage collection after the critical section


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
