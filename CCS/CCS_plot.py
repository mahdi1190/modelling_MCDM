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


df = pd.read_csv(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\CCS\CCSdata.csv")
markets = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\data\markets.xlsx")

carbon_market = markets["carbon"].to_numpy()

co2_data = df['co2'].to_numpy()
production_data = df['co2'].to_numpy()

def pyomomodel():
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    total_time = 360
    TIME_PERIODS = list(range(total_time))
    model.TIME_PERIODS = Set(initialize=TIME_PERIODS)


    no_intervals = 3
    intervals_time = int(total_time / no_intervals)
    INTERVALS = list(range(no_intervals))  # Four 6-hour intervals in a day
    model.INTERVALS = Set(initialize=INTERVALS)
    # Other parameters like cost function parameters, carbon market prices, etc.

    # -------------- Decision Variables --------------
    model.ccs_investment_decision = Var(model.TIME_PERIODS, within=Binary)
    model.ccs_size = Var(within=NonNegativeReals)
    model.carbon_captured = Var(model.TIME_PERIODS, within=NonNegativeReals)
    model.ccs_availability = Var(model.TIME_PERIODS, within=Binary)
    # Additional decision variables as needed.

    # -------------- Constraints --------------
    # Define constraints related to CCS size, carbon capture, investment decision, etc.

    def system_purchased(model, t):
        if t == 0:
            return model.ccs_availability[t] == model.ccs_investment_decision[t]
        else:
            return model.ccs_availability[t] == model.ccs_availability[t-1] + model.ccs_investment_decision[t]
    model.system_purchased_con = Constraint(model.TIME_PERIODS, rule=system_purchased)

    def total_emissions_per_interval_rule(model, i):
        return model.total_emissions_per_interval[i] == sum(model.co2_emissions[t] for t in range(i*intervals_time, (i+1)*intervals_time))
    model.total_emissions_per_interval_constraint = Constraint(model.INTERVALS, rule=total_emissions_per_interval_rule)

    def co2_emissions_rule(model, h):
        return model.co2_emissions[h] == (
            co2_data 
        )
    model.co2_emissions_constraint = Constraint(model.HOURS, rule=co2_emissions_rule)


    def capacity_rule(model, h):
        return model.CCS_capacity >= (model.heat_production[h] + model.heat_stored[h]) + model.electricity_production[h]
    model.capacity_constraint = Constraint(model.HOURS, rule=capacity_rule)
    # -------------- Objective Function --------------
    def ccs_objective_rule(model):
        capital_cost = 3000 + 20 * model.CCS_capacity
        cost = sum(model.ccs_investment_decision[t] * (ccs_cost_function + operating_costs) for t in model.TIME_PERIODS)
        revenue = sum(model.carbon_captured[t] * carbon_market_price[t] for t in model.TIME_PERIODS)
        return cost - revenue
    model.objective = Objective(rule=ccs_objective_rule, sense=minimize)


    # -------------- Solver --------------
    solvere="gurobi"
    solver = SolverFactory(solvere)
    if solvere == "gurobi":
        solver.options['NonConvex'] = 2
        solver.options['TimeLimit'] = 60
        solver.options["Threads"]= 16
        solver.options["LPWarmStart "]= 2
    solver.solve(model, tee=True)

    return model


if __name__ == "__main__":
    model = pyomomodel()