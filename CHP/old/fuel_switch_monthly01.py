import os
import sys
import locale
import logging
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
from pyomo.environ import (ConcreteModel, Var, Set, NonNegativeReals, Reals, Binary,
                           Constraint, Objective, minimize, value)
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set locale for currency formatting
locale.setlocale(locale.LC_ALL, '')

# Add the config directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from solver_options import get_solver  # Import the solver options

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct paths to the data files
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
demands_path = os.path.join(data_dir, 'demands.xlsx')
markets_monthly_path = os.path.join(data_dir, 'markets_monthly.xlsx')
markets_path = os.path.join(data_dir, 'markets.xlsx')
capex_path = os.path.join(data_dir, 'capex.xlsx')

# Read the Excel files
demands = pd.read_excel(demands_path, nrows=10000)
markets_monthly = pd.read_excel(markets_monthly_path, nrows=10000)
markets = pd.read_excel(markets_path, nrows=10000)
capex = pd.read_excel(capex_path, nrows=10000) 

# Convert data to numpy arrays
electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()
electricity_market = markets["elec"].to_numpy()
electricity_market_sold = markets["elec_sold"].to_numpy()
carbon_market = markets["carbon"].to_numpy()
NG_market = markets["nat_gas"].to_numpy()
NG_market_monthly = markets_monthly["nat_gas"].to_numpy()
heat_market_sold = markets["nat_gas_sold"].to_numpy()
H2_market = markets["hydrogen"].to_numpy()
BM_market = markets["biomass"].to_numpy()
max_co2_emissions = markets["cap"].to_numpy()

# Initialize Dash app
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
    dcc.Graph(id='eff-graph'),
    dcc.Graph(id='carbon-credits'),
    dcc.Graph(id='fuel_blend'),

    # Interval component
    dcc.Interval(
        id='interval-component',
        interval=10 * 1000,  # Refresh every 10 seconds
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
    Output('eff-graph', 'style'),
    Output('carbon-credits', 'style'),
    Output('fuel_blend', 'style'),
    Input('height-slider', 'value')
)
def update_graph_height(height_value):
    style = {'height': f'{height_value}px'}
    return [style] * 8  # Apply the same style to all graphs

# Main function to run the Pyomo model with caching
@lru_cache(maxsize=1)
def run_pyomo_model():
    total_months = len(electricity_demand)
    no_intervals = 20
    time_limit = 300
    model = initialize_model(total_months, no_intervals)
    define_parameters(model)
    define_variables(model)
    define_constraints(model)
    define_objective(model)
    solve_model(model, time_limit)
    results = extract_results(model)
    return results

def initialize_model(total_months, no_intervals):
    model = ConcreteModel()
    model.MONTHS = Set(initialize=np.arange(total_months))
    model.INTERVALS = Set(initialize=np.arange(no_intervals))
    model.total_months = total_months
    model.no_intervals = no_intervals
    model.intervals_time = int(total_months / no_intervals)
    return model

def define_parameters(model):
    # Storage parameters
    model.storage_efficiency = 0.5  # %
    model.withdrawal_efficiency = 0.5  # %
    model.max_storage_capacity = 1000  # kW
    model.heat_storage_loss_factor = 0.95  # %/timestep

    # Refrigeration
    model.COP_h = 2
    model.COP_e = 1

    # CHP parameters
    model.capital_cost_per_kw = 1000  # $/kw
    model.fuel_energy = 1  # kW
    model.max_ramp_rate = 100000  # kW/timestep

    model.TEMP = 700
    model.PRES = 50
    # Efficiency coefficients
    model.ng_coeffs = {"elec": [0.25, 0.025, 0.001], "thermal": [0.2, 0.05, 0.001]}  # Placeholder
    model.h2_coeffs = {"elec": [0.18, 0.02, 0.0015], "thermal": [0.15, 0.04, 0.0012]}  # Placeholder

    # Capex
    model.h2_capex = {i: capex["H2"].to_numpy()[i] for i in model.INTERVALS}
    model.eb_capex = {i: capex["EB"].to_numpy()[i] * 0.1 for i in model.INTERVALS}

    # CO2 parameters
    model.co2_per_unit_ng = 0.37  # kg CO2 per kW of fuel
    model.co2_per_unit_bm = 0.1
    model.co2_per_unit_h2 = 0
    model.co2_per_unit_elec = 0.23  # kg CO2 per kW of electricity
    model.max_co2_emissions = max_co2_emissions  # kg CO2
    model.M = 1E8

    # Other parameters
    model.CHP_capacity = 4000
    model.energy_ratio = 0.27

def define_variables(model):
    # CHP System Variables
    model.electricity_production = Var(model.MONTHS, within=NonNegativeReals)
    model.heat_production = Var(model.MONTHS, within=NonNegativeReals)
    model.fuel_consumed = Var(model.MONTHS, within=NonNegativeReals)

    model.heat_to_elec = Var(model.MONTHS, within=NonNegativeReals)

    model.electrical_efficiency = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))
    model.thermal_efficiency = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))

    # Fuel variables
    model.fuel_blend_ng = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_h2 = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))
    model.fuel_blend_biomass = Var(model.MONTHS, within=NonNegativeReals, bounds=(0, 1))

    # Investment decision variables
    model.invest_h2 = Var(within=Binary)
    model.invest_time = Var(model.INTERVALS, within=Binary)
    model.active_h2_blending = Var(model.INTERVALS, within=Binary)

    model.invest_eb = Var(within=Binary)
    model.invest_time_eb = Var(model.INTERVALS, within=Binary)
    model.active_eb = Var(model.INTERVALS, within=Binary)

    # Plant Supply and Useful Energy
    model.heat_to_plant = Var(model.MONTHS, within=NonNegativeReals)
    model.elec_to_plant = Var(model.MONTHS, within=NonNegativeReals)
    model.useful_heat = Var(model.MONTHS, within=NonNegativeReals)
    model.useful_elec = Var(model.MONTHS, within=NonNegativeReals)

    # Market and Storage
    model.purchased_electricity = Var(model.MONTHS, within=NonNegativeReals)
    model.heat_stored = Var(model.MONTHS, within=NonNegativeReals)
    model.heat_withdrawn = Var(model.MONTHS, within=NonNegativeReals)

    # Overproduction and Ramp Rate
    model.heat_over_production = Var(model.MONTHS, within=NonNegativeReals)
    model.electricity_over_production = Var(model.MONTHS, within=NonNegativeReals)
    model.ramp_rate = Var(model.MONTHS, within=NonNegativeReals)

    # Refrigeration
    model.refrigeration_produced = Var(model.MONTHS, within=NonNegativeReals)
    model.heat_used_for_cooling = Var(model.MONTHS, within=NonNegativeReals)
    model.elec_used_for_cooling = Var(model.MONTHS, within=NonNegativeReals)

    # CO2 Emissions and Credits
    model.co2_emissions = Var(model.MONTHS, within=NonNegativeReals)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)
    model.carbon_credits = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_purchased = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_earned = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_sold = Var(model.INTERVALS, within=NonNegativeReals)
    model.exceeds_cap = Var(model.INTERVALS, within=Binary)
    model.credits_held = Var(model.INTERVALS, within=NonNegativeReals)
    model.emissions_difference = Var(model.INTERVALS, domain=Reals)
    model.credits_used_to_offset = Var(model.INTERVALS, within=NonNegativeReals)
    model.below_cap = Var(model.INTERVALS, within=Binary)

def define_constraints(model):
    # Heat Balance
    def heat_balance_rule(model, m):
        return model.heat_production[m] >= model.heat_over_production[m] + model.useful_heat[m]
    model.heat_balance = Constraint(model.MONTHS, rule=heat_balance_rule)

    # Heat Demand
    def heat_demand_balance(model, m):
        return model.heat_to_plant[m] + model.heat_to_elec[m] == heat_demand[m]
    model.heat_demand_rule = Constraint(model.MONTHS, rule=heat_demand_balance)

    # Overproduction of Heat
    def heat_over_production_rule(model, m):
        return model.heat_over_production[m] == model.heat_production[m] - model.useful_heat[m]
    model.heat_over_production_constraint = Constraint(model.MONTHS, rule=heat_over_production_rule)

    # Useful Heat
    def useful_heat_rule(model, m):
        return model.useful_heat[m] == model.heat_to_plant[m] - model.heat_withdrawn[m] * model.withdrawal_efficiency + model.heat_used_for_cooling[m] / model.COP_h
    model.useful_heat_constraint = Constraint(model.MONTHS, rule=useful_heat_rule)

    # Electricity Demand
    def elec_demand_balance(model, m):
        return model.elec_to_plant[m] + model.purchased_electricity[m] == electricity_demand[m]
    model.elec_demand_rule = Constraint(model.MONTHS, rule=elec_demand_balance)

    # Useful Electricity
    def useful_elec_rule(model, m):
        return model.useful_elec[m] == model.elec_to_plant[m] + model.elec_used_for_cooling[m] / model.COP_e
    model.useful_elec_constraint = Constraint(model.MONTHS, rule=useful_elec_rule)

    # Overproduction of Electricity
    def elec_over_production_rule(model, m):
        return model.electricity_over_production[m] == model.electricity_production[m] - model.useful_elec[m]
    model.elec_over_production_constraint = Constraint(model.MONTHS, rule=elec_over_production_rule)

    # CHP Capacity
    def capacity_rule(model, m):
        return model.CHP_capacity >= model.heat_production[m] + model.heat_stored[m] + model.electricity_production[m]
    model.capacity_constraint = Constraint(model.MONTHS, rule=capacity_rule)

    # Fuel Consumption
    def fuel_consumed_rule(model, m):
        return model.fuel_energy * model.fuel_consumed[m] * (1 - model.energy_ratio) * model.thermal_efficiency[m] == model.heat_production[m]
    model.fuel_consumed_rule = Constraint(model.MONTHS, rule=fuel_consumed_rule)

    # Fuel Blend Constraint
    def fuel_blend_rule(model, m):
        return model.fuel_blend_ng[m] + model.fuel_blend_h2[m] + model.fuel_blend_biomass[m] == 1
    model.fuel_blend_constraint = Constraint(model.MONTHS, rule=fuel_blend_rule)

    # H2 Blending Activation
    def h2_blending_activation_rule(model, m):
        interval = m // model.intervals_time
        return model.fuel_blend_h2[m] <= model.active_h2_blending[interval]
    model.h2_blending_activation_constraint = Constraint(model.MONTHS, rule=h2_blending_activation_rule)

    # Active H2 Blending
    def active_h2_blending_rule(model, i):
        return model.active_h2_blending[i] == sum(model.invest_time[j] for j in model.INTERVALS if j <= i)
    model.active_h2_blending_constraint = Constraint(model.INTERVALS, rule=active_h2_blending_rule)

    # Single Investment
    def single_investment_rule(model):
        return sum(model.invest_time[i] for i in model.INTERVALS) == model.invest_h2
    model.single_investment_constraint = Constraint(rule=single_investment_rule)

    # EB Activation
    def eb_activation_rule(model, m):
        interval = m // model.intervals_time
        return model.heat_to_elec[m] <= model.active_eb[interval] * model.M
    model.eb_activation_constraint = Constraint(model.MONTHS, rule=eb_activation_rule)

    # Active EB
    def active_eb_rule(model, i):
        return model.active_eb[i] == sum(model.invest_time_eb[j] for j in model.INTERVALS if j <= i)
    model.active_eb_constraint = Constraint(model.INTERVALS, rule=active_eb_rule)

    # Single Investment EB
    def single_investment_rule_eb(model):
        return sum(model.invest_time_eb[i] for i in model.INTERVALS) <= model.invest_eb
    model.single_investment_eb_constraint = Constraint(rule=single_investment_rule_eb)

    # Initial Heat Stored
    def initial_heat_stored_rule(model):
        return model.heat_stored[0] == 0
    model.initial_heat_stored_constraint = Constraint(rule=initial_heat_stored_rule)

    # Heat Storage Dynamics
    def storage_dynamics_rule(model, m):
        if m == 0:
            return Constraint.Skip
        return model.heat_stored[m] == model.heat_storage_loss_factor * (model.heat_stored[m - 1] + model.heat_over_production[m] * model.storage_efficiency - model.heat_withdrawn[m] / model.withdrawal_efficiency)
    model.storage_dynamics = Constraint(model.MONTHS, rule=storage_dynamics_rule)

    # Storage Capacity
    def storage_capacity_rule(model, m):
        return model.heat_stored[m] <= model.max_storage_capacity
    model.storage_capacity = Constraint(model.MONTHS, rule=storage_capacity_rule)

    # Refrigeration Balance
    def refrigeration_balance_rule(model, m):
        return model.refrigeration_produced[m] == model.elec_used_for_cooling[m] * model.COP_e + model.heat_used_for_cooling[m] * model.COP_h
    model.refrigeration_balance = Constraint(model.MONTHS, rule=refrigeration_balance_rule)

    # Refrigeration Demand
    def refrigeration_demand_rule(model, m):
        return model.refrigeration_produced[m] == refrigeration_demand[m]
    model.refrigeration_demand_con = Constraint(model.MONTHS, rule=refrigeration_demand_rule)

    # Energy Ratio
    def energy_ratio_rule(model, m):
        return model.electricity_production[m] == model.heat_production[m] * model.energy_ratio * model.electrical_efficiency[m]
    model.energy_ratio_constraint = Constraint(model.MONTHS, rule=energy_ratio_rule)

    # CO2 Emissions Rule
    def co2_emissions_rule(model, m):
        return model.co2_emissions[m] == (
            model.co2_per_unit_ng * model.fuel_blend_ng[m] * model.fuel_consumed[m] +
            model.co2_per_unit_h2 * model.fuel_blend_h2[m] * model.fuel_consumed[m] +
            model.co2_per_unit_bm * model.fuel_blend_biomass[m] * model.fuel_consumed[m] +
            model.co2_per_unit_elec * (model.purchased_electricity[m] + model.heat_to_elec[m])
        )
    model.co2_emissions_constraint = Constraint(model.MONTHS, rule=co2_emissions_rule)

    # Total Emissions Per Interval
    def total_emissions_per_interval_rule(model, i):
        start = i * model.intervals_time
        end = (i + 1) * model.intervals_time
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[m] for m in range(start, end))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    # Carbon Credits Needed
    def carbon_credits_needed_rule(model, i):
        idx = i * model.intervals_time
        return model.carbon_credits[i] + model.credits_used_to_offset[i] >= model.total_emissions_per_interval[i] - model.max_co2_emissions[idx]
    model.carbon_credits_needed_constraint = Constraint(model.INTERVALS, rule=carbon_credits_needed_rule)

    # Carbon Credits Earned
    def carbon_credits_earned_rule(model, i):
        idx = i * model.intervals_time
        if i == 0:
            return model.credits_earned[i] == 0
        return model.credits_earned[i] == (model.max_co2_emissions[idx] - model.total_emissions_per_interval[i] + model.credits_used_to_offset[i]) * (1 - model.below_cap[i])
    model.carbon_credits_earned_constraint = Constraint(model.INTERVALS, rule=carbon_credits_earned_rule)

    # Carbon Credits Purchased
    def carbon_credits_purchased_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] <= model.M * model.below_cap[i]
    model.carbon_credits_purchased_con = Constraint(model.INTERVALS, rule=carbon_credits_purchased_rule)

    # Credits Unheld Limit
    def credits_unheld_limit_rule(model, i):
        if i == 0:
            return model.credits_sold[i] == 0
        return model.credits_sold[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_used_to_offset[i]
    model.credits_unheld_limit = Constraint(model.INTERVALS, rule=credits_unheld_limit_rule)

    # Credits Held Dynamics
    def credits_held_dynamics_rule(model, i):
        if i == 0:
            return model.credits_held[i] == model.credits_earned[i]
        return model.credits_held[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_sold[i] - model.credits_used_to_offset[i]
    model.credits_held_dynamics = Constraint(model.INTERVALS, rule=credits_held_dynamics_rule)

    # Below Cap Rule
    def below_cap_rule(model, i):
        idx = i * model.intervals_time
        return model.total_emissions_per_interval[i] - model.max_co2_emissions[idx] <= model.M * model.below_cap[i]
    model.below_cap_con = Constraint(model.INTERVALS, rule=below_cap_rule)

    # Force Zero Rule
    def force_0_rule(model, i):
        if i == 0:
            return model.credits_used_to_offset[i] == 0
        return Constraint.Skip
    model.force_0_rule_con = Constraint(model.INTERVALS, rule=force_0_rule)

    # Mutual Exclusivity
    def mutual_exclusivity_rule(model):
        return model.invest_h2 + model.invest_eb <= 1
    model.mutual_exclusivity_constraint = Constraint(rule=mutual_exclusivity_rule)

def define_objective(model):
    def objective_rule(model):
        elec_cost = sum((model.purchased_electricity[m] + model.heat_to_elec[m]) * electricity_market[m] for m in model.MONTHS)
        elec_sold = sum(model.electricity_over_production[m] * electricity_market_sold[m] for m in model.MONTHS)
        heat_sold = sum(heat_market_sold[m] * model.heat_over_production[m] for m in model.MONTHS)

        fuel_cost_NG = sum(model.fuel_blend_ng[m] * NG_market[m] * model.fuel_consumed[m] for m in model.MONTHS)
        fuel_cost_H2 = sum(model.fuel_blend_h2[m] * H2_market[m] * model.fuel_consumed[m] for m in model.MONTHS)
        fuel_cost_BM = sum(model.fuel_blend_biomass[m] * BM_market[m] * model.fuel_consumed[m] for m in model.MONTHS)

        carbon_cost = sum(model.carbon_credits[i] * carbon_market[i] for i in model.INTERVALS)
        carbon_sold = sum(model.credits_sold[i] * carbon_market[i] for i in model.INTERVALS) * 0.8

        h2_investment_cost = sum(model.invest_time[i] * model.h2_capex[i] for i in model.INTERVALS)
        eb_investment_cost = sum(model.invest_time_eb[i] * model.eb_capex[i] for i in model.INTERVALS)

        total_cost = (fuel_cost_NG + fuel_cost_H2 + fuel_cost_BM) + elec_cost + carbon_cost + h2_investment_cost + eb_investment_cost - (elec_sold + heat_sold + carbon_sold)
        return total_cost
    model.objective = Objective(rule=objective_rule, sense=minimize)

def solve_model(model, time_limit):
    solver = get_solver(time_limit)
    solver.solve(model, tee=False)

def extract_results(model):
    results = {}
    months_list = list(model.MONTHS)
    intervals = list(model.INTERVALS)

    results['electricity_production'] = [value(model.electricity_production[m]) for m in months_list]
    results['heat_production'] = [value(model.heat_production[m]) for m in months_list]
    results['refrigeration_produced'] = [value(model.refrigeration_produced[m]) for m in months_list]
    results['heat_stored'] = [value(model.heat_stored[m]) for m in months_list]
    results['purchased_electricity'] = [value(model.purchased_electricity[m]) for m in months_list]
    results['electrical_efficiency'] = [value(model.electrical_efficiency[m]) for m in months_list]
    results['thermal_efficiency'] = [value(model.thermal_efficiency[m]) for m in months_list]
    results['carbon_credits'] = [value(model.carbon_credits[i]) for i in intervals]
    results['credits_sold'] = [value(model.credits_sold[i]) for i in intervals]
    results['credits_held'] = [value(model.credits_held[i]) for i in intervals]
    results['credits_earned'] = [value(model.credits_earned[i]) for i in intervals]
    results['fuel_blend_h2'] = [value(model.fuel_blend_h2[m]) for m in months_list]
    results['fuel_blend_ng'] = [value(model.fuel_blend_ng[m]) for m in months_list]
    results['fuel_blend_biomass'] = [value(model.fuel_blend_biomass[m]) for m in months_list]
    results['total_purchased_electricity'] = sum(results['purchased_electricity'])
    results['total_heat_cost'] = sum(value(model.fuel_consumed[m]) * NG_market_monthly[m] for m in months_list)
    results['final_chp_capacity'] = model.CHP_capacity
    results['model_cost'] = value(model.objective)
    results['energy_ratio'] = model.energy_ratio

    return results

# Callback for updating graphs and figures
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
    Input('interval-component', 'n_intervals')
)
def update_graphs(n_intervals):
    results = run_pyomo_model()

    # Extract the data
    months_list = list(range(len(electricity_demand)))
    intervals = list(range(len(results['carbon_credits'])))

    # Prepare figures
    base_layout_template = {
        'template': 'plotly_dark',
        'font': {'family': "Courier New, monospace", 'size': 12, 'color': "RebeccaPurple", 'weight': 'bold'},
        'xaxis': {'title': 'Time'},
        'yaxis': {'title': 'Value'},
        'title_font': {'size': 24, 'family': 'Arial, sans-serif', 'weight': 'bold'},
    }

    # Electricity Figure
    electricity_layout = base_layout_template.copy()
    electricity_layout.update({'title': 'Electricity Demand and Production'})
    electricity_figure = {
        'data': [
            {'x': months_list, 'y': electricity_demand, 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
            {'x': months_list, 'y': results['electricity_production'], 'type': 'line', 'name': 'Production', 'line': {'color': 'blue'}},
            {'x': months_list, 'y': results['purchased_electricity'], 'line': {'width': 3, 'dash': 'dot', 'color': 'green'}, 'name': 'Purchased Electricity'},
            {'x': months_list, 'y': results['electricity_over_production'], 'line': {'width': 3, 'dash': 'dot', 'color': 'orange'}, 'name': 'Over-Produced Electricity'}
        ],
        'layout': electricity_layout,
    }

    # Heat Figure
    heat_layout = base_layout_template.copy()
    heat_layout.update({'title': 'Heat Demand and Production'})
    heat_figure = {
        'data': [
            {'x': months_list, 'y': heat_demand, 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
            {'x': months_list, 'y': results['heat_production'], 'type': 'line', 'name': 'Production'},
            {'x': months_list, 'y': results['heat_stored'], 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored Heat'}
        ],
        'layout': heat_layout,
    }

    # Cold Figure
    cold_layout = base_layout_template.copy()
    cold_layout.update({'title': 'Cold Demand and Production'})
    cold_figure = {
        'data': [
            {'x': months_list, 'y': refrigeration_demand, 'line': {'width': 3, 'dash': 'dash'}, 'name': 'Demand'},
            {'x': months_list, 'y': results['refrigeration_produced'], 'type': 'line', 'name': 'Production'}
        ],
        'layout': cold_layout,
    }

    # Heat Storage Figure
    storage_layout = base_layout_template.copy()
    storage_layout.update({'title': 'Heat Storage'})
    heat_storage_figure = {
        'data': [
            {'x': months_list, 'y': results['heat_stored'], 'line': {'width': 3, 'dash': 'dot'}, 'name': 'Stored Heat'},
        ],
        'layout': storage_layout,
    }

    # Purchased Electricity Figure
    market_layout = base_layout_template.copy()
    market_layout.update({'title': 'Energy Market'})
    purchased_elec_figure = {
        'data': [
            {'x': months_list, 'y': electricity_market, 'type': 'line', 'name': 'Electricity Market Price', 'line': {'color': 'blue'}},
            {'x': months_list, 'y': NG_market, 'type': 'line', 'name': 'Fuel Market Price', 'line': {'color': 'red'}}
        ],
        'layout': market_layout,
    }

    # Efficiency Figure
    eff_layout = base_layout_template.copy()
    eff_layout.update({'title': 'Efficiency'})
    eff_figure = {
        'data': [
            {'x': months_list, 'y': results['thermal_efficiency'], 'type': 'line', 'name': 'Thermal Efficiency'},
            {'x': months_list, 'y': results['electrical_efficiency'], 'type': 'line', 'name': 'Electrical Efficiency'},
        ],
        'layout': eff_layout,
    }

    # Carbon Credits Figure
    credits_layout = base_layout_template.copy()
    credits_layout.update({'title': 'Carbon Credit Dynamics'})
    carbon_credits_figure = {
        'data': [
            {'x': intervals, 'y': results['carbon_credits'], 'type': 'line', 'name': 'Carbon Credits Purchased'},
            {'x': intervals, 'y': results['credits_sold'], 'type': 'line', 'name': 'Carbon Credits Sold'},
            {'x': intervals, 'y': results['credits_held'], 'type': 'line', 'name': 'Carbon Credits Held'},
            {'x': intervals, 'y': results['credits_earned'], 'type': 'line', 'name': 'Carbon Credits Earned'},
        ],
        'layout': credits_layout,
    }

    # Fuel Blend Figure
    blend_layout = base_layout_template.copy()
    blend_layout.update({'title': 'Fuel Blending'})
    fuel_blend_figure = {
        'data': [
            {'x': months_list, 'y': results['fuel_blend_h2'], 'type': 'line', 'name': 'Hydrogen'},
            {'x': months_list, 'y': results['fuel_blend_biomass'], 'type': 'line', 'name': 'Biomass'},
            {'x': months_list, 'y': results['fuel_blend_ng'], 'type': 'line', 'name': 'Natural Gas'},
        ],
        'layout': blend_layout,
    }

    # Return the updated figures and text
    return (
        electricity_figure,
        heat_figure,
        cold_figure,
        heat_storage_figure,
        purchased_elec_figure,
        eff_figure,
        carbon_credits_figure,
        fuel_blend_figure,
        f"Total Purchased Electricity: {locale.currency(results['total_purchased_electricity'], grouping=True)}",
        f"Total Heat Cost: {locale.currency(results['total_heat_cost'], grouping=True)}",
        f"Final CHP Capacity: {round(results['final_chp_capacity'], 2)} KW",
        f"Energy Ratio: {round(results['energy_ratio'], 2)}",
        f"Model Cost: {locale.currency(results['model_cost'], grouping=True)}",
        f"Credit Cost: NaN"
    )

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
