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

last_mod_time = 0
df = pd.read_csv(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\CCS\CCS_data.csv")
markets = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\data\markets.xlsx")

carbon_market = markets["carbon"].to_numpy()

max_co2_emissions = [150000, 140000, 13000, 120000, 110000]

co2_data = df['co2'].to_numpy()
production_data = df['co2'].to_numpy()


co2_flow = max(co2_data)

app = dash.Dash(__name__)

# Initial layout
app.layout = html.Div([    
    # Text elements at the top
    html.Div([
        html.Div(id='credits', style={'fontSize': 24}),
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    
    # Carbon Credits Graph
    dcc.Graph(id='carbon-credits'),
    
    # Interval component
    dcc.Interval(
        id='interval-component',
        interval=10 * 10000,  # Refresh every 10 seconds
        n_intervals=0
    )
])

# Callback for updating the carbon credits graph
@app.callback(
    Output('carbon-credits', 'figure'),
    Output('credits', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n_intervals):
    global last_mod_time  # Declare as global to modify it

    # Check last modification time of the model file
    current_mod_time = os.path.getmtime(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\CCS\CCS_plot.py")

    if current_mod_time > last_mod_time:
        last_mod_time = current_mod_time  # Update last modification time

        model = pyomomodel()

        carbon_offset = [model.credits_used_to_offset[h]() for h in model.INTERVALS]
        carbon_sell = [model.credits_sold[h]() for h in model.INTERVALS]
        credits = [model.carbon_credits[h]() for h in model.INTERVALS]
        carbon_held = [model.credits_held[h]() for h in model.INTERVALS]
        carbon_earn = [model.credits_earned[h]() for h in model.INTERVALS]

        credits_layout = {
            'template': 'plotly_dark',
            'font': {'family': "Courier New, monospace", 'size': 12, 'color': "RebeccaPurple", 'weight': 'bold'},
            'xaxis': {'title': 'Time Intervals'},
            'yaxis': {'title': 'No of Credits'},
            'title': 'Carbon Credit Dynamics',
            'title_font': {'size': 24, 'family': 'Arial, sans-serif', 'weight': 'bold'},
            'showlegend': False,
        }

        carbon_credits_figure = {
            'data': [
                {'x': list(model.INTERVALS), 'y': credits, 'type': 'line', 'name': 'Carbon Credits Purchased'},
                {'x': list(model.INTERVALS), 'y': carbon_sell, 'type': 'line', 'name': 'Carbon Credits Sold'},
                {'x': list(model.INTERVALS), 'y': carbon_held, 'type': 'line', 'name': 'Carbon Credits Held'},
                {'x': list(model.INTERVALS), 'y': carbon_earn, 'type': 'line', 'name': 'Carbon Credits Earned'},
                {'x': list(model.INTERVALS), 'y': carbon_offset, 'type': 'line', 'name': 'Carbon Credits Used To Offset'},
            ],
            'layout': credits_layout,
        }

        return carbon_credits_figure, f"Credit Cost: NaN"
    else:
        raise dash.exceptions.PreventUpdate

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    total_time = 360
    TIME_PERIODS = list(range(total_time))
    model.TIME_PERIODS = Set(initialize=TIME_PERIODS)

    M = 1E12
    no_intervals = 5
    intervals_time = int(total_time / no_intervals)
    INTERVALS = list(range(no_intervals))  # Four 6-hour intervals in a day
    model.INTERVALS = Set(initialize=INTERVALS)
    # Other parameters like cost function parameters, carbon market prices, etc.

    # -------------- Decision Variables --------------
    model.ccs_investment_decision = Var(model.TIME_PERIODS, within=Binary)
    model.CCS_capacity = Var(within=NonNegativeReals)
    model.carbon_captured = Var(model.TIME_PERIODS, within=NonNegativeReals)
    model.ccs_availability = Var(model.TIME_PERIODS, within=Binary)
    model.total_emissions_per_interval = Var(model.INTERVALS, within=NonNegativeReals)
    model.co2_emissions = Var(model.TIME_PERIODS, within=NonNegativeReals)
    # Additional decision variables as needed.
    model.carbon_credits = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_earned = Var(model.INTERVALS, domain=Reals)
    model.credits_sold = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_held = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_unheld = Var(model.INTERVALS, within=NonNegativeReals)
    model.credits_used_to_offset = Var(model.INTERVALS, within=NonNegativeReals)
    model.below_cap = Var(model.INTERVALS, within=Binary)

    # -------------- Constraints --------------
    # Define constraints related to CCS size, carbon capture, investment decision, etc.

    def system_purchased(model, t):
        if t == 0:
            return model.ccs_availability[t] == model.ccs_investment_decision[t]
        else:
            return model.ccs_availability[t] == model.ccs_availability[t-1] + model.ccs_investment_decision[t]
    model.system_purchased_con = Constraint(model.TIME_PERIODS, rule=system_purchased)

    def co2_emissions_rule(model, t):
        return model.co2_emissions[t] == (co2_data[t])
    model.co2_emissions_constraint = Constraint(model.TIME_PERIODS, rule=co2_emissions_rule)

    def total_emissions_per_interval_rule(model, i):
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[h] for h in range(i*intervals_time, (i+1)*intervals_time))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    def carbon_credits_needed_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] >= model.total_emissions_per_interval[i] - max_co2_emissions[i]
    model.carbon_credits_needed_constraint = Constraint(model.INTERVALS, rule=carbon_credits_needed_rule)

    def carbon_credits_earned_rule(model, i):
        if i == 0:
            return model.credits_earned[i] == 0  # For the first interval
        return model.credits_earned[i] == (max_co2_emissions[i] - model.total_emissions_per_interval[i] + model.credits_used_to_offset[i]) * (1 - model.below_cap[i])
    model.carbon_credits_earned_constraint = Constraint(model.INTERVALS, rule=carbon_credits_earned_rule)

    def carbon_credits_purchased_rule(model, i):
        return model.carbon_credits[i] + model.credits_used_to_offset[i] <= M * model.below_cap[i]
    model.carbon_credits_purchased_con = Constraint(model.INTERVALS, rule=carbon_credits_purchased_rule)

    # Ensure credits_unheld doesn't exceed credits earned and previously held
    def credits_unheld_limit_rule(model, i):
        if i == 0:
            return model.credits_sold[i] == 0  # For the first interval
        return model.credits_sold[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_used_to_offset[i]
    model.credits_unheld_limit = Constraint(model.INTERVALS, rule=credits_unheld_limit_rule)

    def credits_held_dynamics_rule(model, i):
        if i == 0:
            return model.credits_held[i] == model.credits_earned[i] # For the first interval
        return model.credits_held[i] <= model.credits_held[i - 1] + model.credits_earned[i] - model.credits_sold[i] - model.credits_used_to_offset[i]
    model.credits_held_dynamics = Constraint(model.INTERVALS, rule=credits_held_dynamics_rule)

    def below_cap_rule(model, i):
        return model.total_emissions_per_interval[i] - max_co2_emissions[i] <= M * model.below_cap[i]
    model.below_cap_con = Constraint(model.INTERVALS, rule=below_cap_rule)

    def force_0_rule(model, i):
        if i == 0:
            return model.credits_used_to_offset[i] == 0
        return Constraint.Skip
    model.force_0_rule_con = Constraint(model.INTERVALS, rule=force_0_rule)
    
    # -------------- Objective Function --------------
    def ccs_objective_rule(model):
        capital_cost = (3000 + 20 * model.CCS_capacity) * sum(model.ccs_investment_decision[t] for t in model.TIME_PERIODS)
        operating_cost = sum(model.ccs_investment_decision[t] * model.co2_emissions[t] * 1000 for t in model.TIME_PERIODS)
        carbon_cost = sum((model.carbon_credits[i]) * carbon_market[i] for i in model.INTERVALS)
        carbon_sold = sum(model.credits_sold[i] * carbon_market[i] for i in model.INTERVALS) * 0.5
        return (operating_cost + capital_cost + carbon_cost) - carbon_sold 
    model.objective = Objective(rule=ccs_objective_rule, sense=minimize)


    # -------------- Solver --------------
    solvere="gurobi"
    solver = SolverFactory(solvere)
    if solvere == "gurobi":
        solver.options['NonConvex'] = 2
        solver.options['TimeLimit'] = 60
        solver.options["Threads"]= 16
        solver.options["LPWarmStart "] = 2
    solver.solve(model, tee=True)

    return model

def print_selected_CCS(model):
    for t in model.TIME_PERIODS:
        print(f"Month {t} - Bought? - {model.ccs_investment_decision[t].value}, {model.ccs_availability[t].value}")
    for i in model.INTERVALS:
        print(f"Interval {i} - Credits - {model.carbon_credits[i].value}, {model.credits_sold[i].value}")

if __name__ == "__main__":
    app.run_server(debug=True)
    model = pyomomodel()
    print_selected_CCS(model)
