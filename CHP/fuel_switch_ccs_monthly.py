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
import datetime
import numpy as np
import locale
import gc

BASE_NAME = "base_case"

gc.disable()  # Disable garbage collection temporarily

locale.setlocale(locale.LC_ALL, '')  # Setting locale once here

# Initialize Dash app
last_mod_time = 0

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct paths to the data files by correctly moving up one directory to 'modelling_MCDM'
demands_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'demands_monthly_30.csv'))
markets_monthly_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets_monthly.csv'))
markets_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'markets.csv'))  # Corrected path

capex_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'capex_costs_over_time.csv'))  # Corrected path
unit_conv = 1E3
# Read the Excel files
demands = pd.read_csv(demands_path, nrows=361)
markets_monthly = pd.read_csv(markets_monthly_path, nrows=361)
capex = pd.read_csv(capex_path, nrows=361)

electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()

electricity_market = markets_monthly["Electricity Price ($/kWh)"].to_numpy() * unit_conv
electricity_market_sold = electricity_market * 1E-3

carbon_market = markets_monthly["Carbon Credit Price ($/tonne CO2)"].to_numpy() 

NG_market = markets_monthly["Natural Gas Price ($/kWh)"].to_numpy()  * unit_conv
heat_market_sold = NG_market * 1E-3
H2_market = markets_monthly["Hydrogen Price ($/kWh)"].to_numpy()  * unit_conv 
BM_market = markets_monthly["Biomass Price ($/kWh)"].to_numpy() * unit_conv

em_bm = markets_monthly["Biomass Carbon Intensity (kg CO2/kWh)"].to_numpy()
em_h2 = markets_monthly["Hydrogen Carbon Intensity (kg CO2/kWh)"].to_numpy()
em_ng = markets_monthly["Natural Gas Carbon Intensity (kg CO2/kWh)"].to_numpy()
em_elec = markets_monthly["Grid Carbon Intensity (kg CO2/kWh)"].to_numpy()
res_emissions = 450
CHP_capacity = 10000


energy_ratio = 0.25

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


def save_all_model_variables(model, total_months, no_intervals, results_folder, timestamp, simulation_name):
    """
    Save model variables into one CSV file.
    Rows correspond to months [0, total_months-1].
    For variables indexed by MONTHS, the value is taken directly.
    For variables indexed by INTERVALS, we map each month to an interval using:
        interval = m // (total_months // no_intervals)
    Scalar variables are replicated for each month.
    Variables that are indexed over other sets are skipped.
    """
    # Create a list of months (time index)
    months = list(range(total_months))
    
    # Define the mapping function from month to interval.
    # (Assumes total_months is exactly divisible by no_intervals.)
    def month_to_interval(m):
        return m // (total_months // no_intervals)
    
    # Prepare a dictionary to build the DataFrame.
    data = {'Time': months}
    
    # Try to access your model sets, if defined.
    try:
        model_months = set(model.MONTHS)
    except AttributeError:
        model_months = None

    try:
        model_intervals = set(model.INTERVALS)
    except AttributeError:
        model_intervals = None

    # Iterate over all Var components in the model.
    for var_component in model.component_objects(Var, active=True):
        var_name = var_component.name
        col = []  # values for each month
        
        # If the variable is scalar, replicate its value.
        if not var_component.is_indexed():
            scalar_val = value(var_component)
            col = [scalar_val] * total_months
        else:
            # Get the variable's keys and turn them into a set.
            keys = list(var_component.keys())
            keys_set = set(keys)
            
            # If the keys match the MONTHS set, extract the value per month.
            if model_months is not None and keys_set == model_months:
                for m in months:
                    col.append(value(var_component[m]))
            # If the keys match the INTERVALS set, map each month to its interval.
            elif model_intervals is not None and keys_set == model_intervals:
                for m in months:
                    interval = month_to_interval(m)
                    col.append(value(var_component[interval]))
            else:
                # Variable is indexed by something else (e.g., multiple indices) – skip it.
                print(f"Skipping variable '{var_name}' because its indexing does not match MONTHS or INTERVALS exclusively.")
                continue
        data[var_name] = col

    # Create the DataFrame and save to CSV.
    df = pd.DataFrame(data)
    filepath = os.path.join(results_folder, f"{BASE_NAME}_monthly.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved mapped model variables to {filepath}")


def create_results_folder():
    # Change here: use BASE_NAME only once
    base_dir = os.path.join(os.path.dirname(__file__), "results", BASE_NAME)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print("Created results folder:", base_dir)
    else:
        print("Results folder already exists:", base_dir)
    return base_dir

total_months = 360
time_limit = 300
# Create a simple model
def pyomomodel(total_months = total_months, time_limit = time_limit, CHP_capacity=CHP_capacity, energy_ratio = energy_ratio):
    # Create model
    model = ConcreteModel()
    ccs_energy_penalty_factor = 2  # mW thermal per tonne CO2 captured (adjust as needed)

    # -------------- Parameters --------------
    # Time periods (e.g., months in a year)
    MONTHS = np.arange(total_months)
    model.MONTHS = Set(initialize=MONTHS)

    no_intervals = 30
    intervals_time = int(total_months / no_intervals)
    INTERVALS = np.arange(no_intervals)  # Four 6-hour intervals in a day
    model.INTERVALS = Set(initialize=INTERVALS)

    # Storage
    storage_efficiency = 0.5 # 
    withdrawal_efficiency = 0.5 # 
    max_storage_capacity = 0 #kW
    heat_storage_loss_factor = 0.95  # %/timestep

    # Refrigeration
    COP_h = 2
    COP_e = 1

    # CHP params
    capital_cost_per_kw = 1000 # $/kw
    fuel_energy = 1  # kW
    max_ramp_rate = 5000  #kW/timestep

    TEMP = 700
    PRES = 50
    # Efficiency coefficients
    ng_coeffs = {"elec": [0.25, 0.025, 0.001], "thermal": [0.2, 0.05, 0.001]}  # Placeholder
    h2_coeffs = {"elec": [0.18, 0.02, 0.0015], "thermal": [0.15, 0.04, 0.0012]}  # Placeholder


    h2_capex = capex["Hydrogen CHP Retrofit CAPEX ($/kW)"].to_numpy() * CHP_capacity # Example value, adjust based on your case
    eb_capex = capex["Electric Boiler CAPEX ($/kW)"].to_numpy() * CHP_capacity # Example value, adjust based on your case
    ccs_capex = capex["CCS System CAPEX ($/tonne CO2/year)"] * 14000

    max_co2_emissions = markets_monthly["Effective Carbon Credit Cap"]   # tonnes CO2
    M = 1E6

    # CO2 stream properties (can be varied)
    co2_stream_temp = 400  # CO2 stream temperature (in Kelvin)
    co2_stream_pressure = 10  # CO2 stream pressure (in bar)
    co2_concentration = 0.2  # CO2 concentration in flue gas

    # --- Compute adjusted_capture_efficiency outside constraints ---
    base_capture_efficiency = 0.8  # Base capture efficiency
    stream_factor = (
        (1 - 0.0001 * (co2_stream_temp - 300)) *
        (1 + 0.05 * (co2_stream_pressure - 10)) *
        (1 + 0.2 * (co2_concentration - 0.2))
    )
    adjusted_capture_efficiency_value = base_capture_efficiency * stream_factor

    # CCS Transport and Storage Costs
    transport_cost_per_kg_co2 = 0.05*1E3  # $ per kg of CO2 transported
    storage_cost_per_kg_co2 = 0.03*1E3    # $ per kg of CO2 stored

    # -------------- Decision Variables --------------

    # CHP System Variables

    model.electricity_production = Var(model.MONTHS, within=NonNegativeReals)
    model.heat_production = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 2))
    model.fuel_consumed = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(heat_demand) * 3))

    model.heat_to_elec = Var(model.MONTHS, within=NonNegativeReals)
    
    model.electrical_efficiency = Var(model.MONTHS, within=NonNegativeReals, bounds=(1, 1),  initialize=1)
    model.thermal_efficiency = Var(model.MONTHS, within=NonNegativeReals, bounds=(1, 1), initialize=1)

    # Fuel variables (change from INTERVALS to MONTHS)
    model.fuel_blend_ng = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_h2 = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_biomass = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_energy = Var(model.MONTHS, within=NonNegativeReals)
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
    model.ramp_rate = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max_ramp_rate), initialize=max_ramp_rate)

    # Refrigeration
    model.refrigeration_produced = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.heat_used_for_cooling = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    model.elec_used_for_cooling = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, max(refrigeration_demand)))
    # Variables
    model.co2_emissions = Var(model.MONTHS, within=NonNegativeReals, initialize=0)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)
    model.carbon_credits = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_purchased = Var(model.INTERVALS, within=NonNegativeReals, initialize=0)
    model.credits_earned = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions*10)), initialize=0)
    model.credits_sold = Var(model.INTERVALS, within=NonNegativeReals, initialize=0)
    model.exceeds_cap = Var(model.INTERVALS, within=Binary,  initialize=0)
    model.credits_held = Var(model.INTERVALS, within=NonNegativeReals, bounds=(0, max(max_co2_emissions*10)),  initialize=0)
    model.emissions_difference = Var(model.INTERVALS, domain=Reals, bounds=(-max(max_co2_emissions*10),max(max_co2_emissions*10)),  initialize=0)
    model.credits_used_to_offset = Var(model.INTERVALS, within=NonNegativeReals,bounds=(0, max(max_co2_emissions*10)))
    model.below_cap = Var(model.INTERVALS, within=Binary,  initialize=1)

        # Define time-indexed CAPEX parameters (for each interval i)
    model.h2_capex = Param(model.INTERVALS, within=NonNegativeReals, initialize={i: h2_capex[i] for i in model.INTERVALS})
    model.eb_capex = Param(model.INTERVALS, within=NonNegativeReals, initialize={i: eb_capex[i] for i in model.INTERVALS})
    model.ccs_capex = Param(model.INTERVALS, within=NonNegativeReals, initialize={i: ccs_capex[i] for i in model.INTERVALS})

    # Add CCS investment decision variables
    model.invest_ccs = Var(within=Binary)  # Decision to invest in CCS
    model.invest_time_ccs = Var(model.INTERVALS, within=Binary)  # Decision for when to invest in CCS
    model.active_ccs = Var(model.INTERVALS, within=Binary)  # CCS activation after investment

    # Add transport cost variable for CCS
    model.transport_cost = Var(model.INTERVALS, within=NonNegativeReals)

    # Auxiliary variables for CCS calculations
    model.captured_co2 = Var(model.MONTHS, within=NonNegativeReals)
    model.uncaptured_co2 = Var(model.MONTHS, within=NonNegativeReals)
    model.ccs_energy_penalty = Var(model.MONTHS, within=NonNegativeReals)

    # --- Parameters ---
    # Calculate total_fuel_co2_ub
    fuel_consumed_ub = model.fuel_consumed[0].ub  # Assuming all m have the same upper bound
    total_fuel_co2_ub = fuel_consumed_ub

    # --- Define Variables ---
    # Auxiliary variable for product
    model.total_fuel_co2_active_ccs = Var(model.MONTHS, within=NonNegativeReals)

    model.total_fuel_co2 = Expression(model.MONTHS, rule=lambda model, m: (
        model.fuel_blend_ng[m] * em_ng[m] * model.fuel_consumed[m] +
        model.fuel_blend_h2[m] * em_h2[m] * model.fuel_consumed[m] +
        model.fuel_blend_biomass[m] * em_bm[m] * model.fuel_consumed[m]
    ))


 # -------------- Constraints --------------
    # Heat Balance
    def heat_balance_rule(model, m):
        return model.heat_production[m] >= (model.heat_over_production[m] + model.useful_heat[m])
    model.heat_balance = Constraint(model.MONTHS, rule=heat_balance_rule)

    # Heat Demand
    def heat_demand_balance(model, m):
        return (model.heat_to_plant[m]) + model.heat_to_elec[m] >= (heat_demand[m])
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
        return model.elec_to_plant[m] + model.purchased_electricity[m] >= (electricity_demand[m])
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

    def fuel_energy_rule(model, m):
        return model.fuel_energy[m] == model.fuel_blend_ng[m] + model.fuel_blend_h2[m] + model.fuel_blend_biomass[m]
    model.fuel_energy_rule_con = Constraint(model.MONTHS, rule=fuel_energy_rule)
    
    def fuel_consumed_rule(model, m):
        return model.fuel_energy[m] * model.fuel_consumed[m] * (1 - energy_ratio) == model.heat_production[m] + model.ccs_energy_penalty[m]
    model.fuel_consumed_rule_con = Constraint(model.MONTHS, rule=fuel_consumed_rule)


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
        return sum(model.invest_time_eb[i] for i in model.INTERVALS) == model.invest_eb  # Allow for flexibility
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

# ======== CO2 Constraints ========
    # CO2 Emissions Rule
    # Total Emissions Per Interval Rule
    def total_emissions_per_interval_rule(model, i):
        start = i * intervals_time
        end = start + intervals_time
        # Sum both uncaptured fuel emissions and electricity-related emissions
        return model.total_emissions_per_interval[i] == sum(
            model.uncaptured_co2[m] + em_elec[m] * (model.purchased_electricity[m] + model.heat_to_elec[m])
            for m in range(start, end)
        )
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    # Carbon Credits Needed Rule
    def carbon_credits_needed_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] >= model.total_emissions_per_interval[i] - max_co2_emissions[i*intervals_time]
    model.carbon_credits_needed_constraint = Constraint(model.INTERVALS, rule=carbon_credits_needed_rule)

    # Carbon Credits Earned Rule
    def carbon_credits_earned_rule(model, i):
        if i == 0:
            return model.credits_earned[i] == 0  # For the first interval
        return model.credits_earned[i] == (max_co2_emissions[i*intervals_time] - model.total_emissions_per_interval[i] + model.credits_used_to_offset[i]) * (1 - model.below_cap[i])
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
        return model.total_emissions_per_interval[i] - max_co2_emissions[i*intervals_time] <= M * model.below_cap[i]
    model.below_cap_con = Constraint(model.INTERVALS, rule=below_cap_rule)

    # Force Zero Rule
    def force_0_rule(model, i):
        if i == 0:
            return model.credits_used_to_offset[i] == 0
        return Constraint.Skip
    model.force_0_rule_con = Constraint(model.INTERVALS, rule=force_0_rule)

    # CCS Investment Activation Constraints
    # Ensure CCS is only active after investment
    def active_ccs_rule(model, i):
        # CCS is active in interval `i` if investment has been made in any interval `j` <= `i`
        return model.active_ccs[i] == sum(model.invest_time_ccs[j] for j in model.INTERVALS if j <= i)
    model.active_ccs_constraint = Constraint(model.INTERVALS, rule=active_ccs_rule)

    # Limit CCS activation to investment status
    def ccs_activation_limit_rule(model, i):
        return model.active_ccs[i] <= model.invest_ccs
    model.ccs_activation_limit_constraint = Constraint(model.INTERVALS, rule=ccs_activation_limit_rule)

    def single_investment_ccs_rule(model):
        return sum(model.invest_time_ccs[i] for i in model.INTERVALS) == model.invest_ccs
    model.single_investment_ccs_constraint = Constraint(rule=single_investment_ccs_rule)

    # Mutual Exclusivity Constraint for Investments
    def mutual_exclusivity_rule(model):
        return model.invest_h2 + model.invest_eb + model.invest_ccs <= 1
    model.mutual_exclusivity_constraint = Constraint(rule=mutual_exclusivity_rule)

    # --- Captured CO2 Constraint ---
# --- Constraints ---
    # Linearize the product of total_fuel_co2 and active_ccs[interval]
    def total_fuel_co2_active_ccs_upper(model, m):
        interval = m // (len(model.MONTHS) // len(model.INTERVALS))
        return model.total_fuel_co2_active_ccs[m] <= total_fuel_co2_ub * model.active_ccs[interval]
    model.total_fuel_co2_active_ccs_upper_con = Constraint(model.MONTHS, rule=total_fuel_co2_active_ccs_upper)

    def total_fuel_co2_active_ccs_lower(model, m):
        return model.total_fuel_co2_active_ccs[m] >= 0
    model.total_fuel_co2_active_ccs_lower_con = Constraint(model.MONTHS, rule=total_fuel_co2_active_ccs_lower)

    def total_fuel_co2_active_ccs_link1(model, m):
        return model.total_fuel_co2_active_ccs[m] <= model.total_fuel_co2[m]
    model.total_fuel_co2_active_ccs_link1_con = Constraint(model.MONTHS, rule=total_fuel_co2_active_ccs_link1)

    def total_fuel_co2_active_ccs_link2(model, m):
        interval = m // (len(model.MONTHS) // len(model.INTERVALS))
        return model.total_fuel_co2_active_ccs[m] >= model.total_fuel_co2[m] - total_fuel_co2_ub * (1 - model.active_ccs[interval])
    model.total_fuel_co2_active_ccs_link2_con = Constraint(model.MONTHS, rule=total_fuel_co2_active_ccs_link2)

    # Captured CO2 constraint
    def captured_co2_constraint(model, m):
        interval = m // (len(model.MONTHS) // len(model.INTERVALS))
        return model.captured_co2[m] <= adjusted_capture_efficiency_value * (model.total_fuel_co2_active_ccs[m] + res_emissions*model.active_ccs[interval])
    model.captured_co2_constraint = Constraint(model.MONTHS, rule=captured_co2_constraint)

    # Uncaptured CO2 constraint
    def uncaptured_co2_constraint(model, m):
        interval = m // (len(model.MONTHS) // len(model.INTERVALS))
        return model.uncaptured_co2[m] == model.total_fuel_co2[m] + res_emissions*(1-model.active_ccs[interval]) - model.captured_co2[m]
    model.uncaptured_co2_constraint = Constraint(model.MONTHS, rule=uncaptured_co2_constraint)

        # Already declared earlier: model.ccs_energy_penalty = Var(model.MONTHS, within=NonNegativeReals)
    def ccs_energy_penalty_rule(model, m):
        return model.ccs_energy_penalty[m] == ccs_energy_penalty_factor * model.captured_co2[m]
    model.ccs_energy_penalty_constraint = Constraint(model.MONTHS, rule=ccs_energy_penalty_rule)

    # Adjust Electricity Production Constraint to include CCS energy penalty
    def electricity_production_constraint(model, m):
        return model.electricity_production[m] == (model.heat_production[m] * energy_ratio)
    model.electricity_production_constraint = Constraint(model.MONTHS, rule=electricity_production_constraint)

    # CCS Transport and Storage Costs
    def transport_and_storage_cost_rule(model, i):
        start = i * intervals_time
        end = start + intervals_time

        # Total CO2 captured in interval i
        total_captured_co2 = sum(model.captured_co2[m] for m in range(start, end))

        # Total transport and storage cost
        return model.transport_cost[i] == total_captured_co2 * (transport_cost_per_kg_co2 + storage_cost_per_kg_co2)
    model.transport_storage_cost_constraint = Constraint(model.INTERVALS, rule=transport_and_storage_cost_rule)

    # -------------- Objective Function --------------

    # Modify the objective function to include CCS-related costs
    def objective_rule(model):
        # Existing cost components
        elec_cost = sum(
            (model.purchased_electricity[m] + model.heat_to_elec[m]) * electricity_market[m]
            for m in model.MONTHS
        )
        elec_sold = sum(
            model.electricity_over_production[m] * electricity_market_sold[m]
            for m in model.MONTHS
        )
        heat_sold = sum(
            heat_market_sold[m] * model.heat_over_production[m]
            for m in model.MONTHS
        )

        fuel_cost_NG = sum(
            model.fuel_blend_ng[m] * NG_market[m] * model.fuel_consumed[m]
            for m in model.MONTHS
        )
        fuel_cost_H2 = sum(
            model.fuel_blend_h2[m] * H2_market[m] * model.fuel_consumed[m]
            for m in model.MONTHS
        )
        fuel_cost_BM = sum(
            model.fuel_blend_biomass[m] * BM_market[m] * model.fuel_consumed[m]
            for m in model.MONTHS
        )

        carbon_cost = sum(
            model.carbon_credits[i] * carbon_market[i]
            for i in model.INTERVALS
        )
        carbon_sold = sum(
            model.credits_sold[i] * carbon_market[i]
            for i in model.INTERVALS
        )

        # CCS-related costs
        transport_storage_cost = sum(
            model.transport_cost[i]
            for i in model.INTERVALS
        )
        ccs_investment_cost = sum(
            model.invest_time_ccs[i] * model.ccs_capex[i]
            for i in model.INTERVALS
        )

        # Time-dependent CAPEX for hydrogen and electrification
        h2_investment_cost = sum(
            model.invest_time[i] * model.h2_capex[i]
            for i in model.INTERVALS
        )
        eb_investment_cost = sum(
            model.invest_time_eb[i] * model.eb_capex[i]
            for i in model.INTERVALS
        )

        # Total costs
        total_costs = (
            fuel_cost_NG + fuel_cost_H2 + fuel_cost_BM +
            elec_cost + carbon_cost +
            h2_investment_cost + eb_investment_cost + ccs_investment_cost +
            transport_storage_cost
        )

        # Total revenues
        total_revenues = elec_sold + heat_sold + carbon_sold

        return total_costs - total_revenues 
    model.objective = Objective(rule=objective_rule, sense=minimize)

    # -------------- Solver --------------

    # Solve the model
    solver = get_solver(time_limit)
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
    save_path = 'investment_log/'  # Update 'filename.csv' with your desired file name
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
    investment_df.to_csv(save_path+'h2_investment.csv', index=False)

 
    investment_df2 = pd.DataFrame({
        'Interval': list(model.INTERVALS),
        'Investment Time': list(investment_times_eb.values()),
        'Active EB': list(active_eb_blending.values())
    })
    investment_df2.to_csv(save_path+'eb_investment.csv', index=False)
    # Extract investment decision variables
    # Hydrogen investment
    h2_investment_made = value(model.invest_h2)
    h2_investment_times = [i for i in model.INTERVALS if value(model.invest_time[i]) > 0.5]

    # Electrification (EB) investment
    eb_investment_made = value(model.invest_eb)
    eb_investment_times = [i for i in model.INTERVALS if value(model.invest_time_eb[i]) > 0.5]

    # CCS investment
    ccs_investment_made = value(model.invest_ccs)
    ccs_investment_times = [i for i in model.INTERVALS if value(model.invest_time_ccs[i]) > 0.5]

    # Print investment information
    print("Investment Decisions:")
    if h2_investment_made:
        print(f"- Hydrogen investment is made at intervals: {h2_investment_times}")
    else:
        print("- No hydrogen investment is made.")

    if eb_investment_made:
        print(f"- Electrification (EB) investment is made at intervals: {eb_investment_times}")
    else:
        print("- No electrification (EB) investment is made.")

    if ccs_investment_made:
        print(f"- CCS investment is made at intervals: {ccs_investment_times}")
    else:
        print("- No CCS investment is made.")

    # Map intervals to months
    h2_investment_months = [i * intervals_time for i in h2_investment_times]
    eb_investment_months = [i * intervals_time for i in eb_investment_times]
    ccs_investment_months = [i * intervals_time for i in ccs_investment_times]

    print("Investment Timing in Months:")
    if h2_investment_made:
        print(f"- Hydrogen investment months: {h2_investment_months}")
    if eb_investment_made:
        print(f"- Electrification (EB) investment months: {eb_investment_months}")
    if ccs_investment_made:
        print(f"- CCS investment months: {ccs_investment_months}")

    # Save investment information to a DataFrame
    investment_info = pd.DataFrame({
        'Investment Type': ['H2', 'EB', 'CCS'],
        'Investment Made': [h2_investment_made, eb_investment_made, ccs_investment_made],
        'Investment Intervals': [h2_investment_times, eb_investment_times, ccs_investment_times],
        'Investment Months': [h2_investment_months, eb_investment_months, ccs_investment_months]
    })

    # Save to CSV
    investment_info.to_csv('investment_info.csv', index=False)

    # ---------------------------
    # Save every model variable mapped by time (month)
    
    # Create a timestamp string (e.g., "20250218_153045")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Define your results folder (e.g., "results/monthly_results" relative to current_dir)
    results_folder = os.path.join(current_dir, 'results', BASE_NAME)
    os.makedirs(results_folder, exist_ok=True)
    
    # Save the model variables to one CSV file.
    save_all_model_variables(model, total_months, 30, results_folder, timestamp, "simulation_name")
    # ---------------------------

    return model

gc.enable()  # Re-enable garbage collection after the critical section


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
