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

demands = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\data\demands.xlsx")
markets = pd.read_excel(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\data\markets.xlsx")

electricity_demand = demands["elec"].to_numpy()
heat_demand = demands["heat"].to_numpy()
refrigeration_demand = demands["cool"].to_numpy()
electricity_market = markets["elec"].to_numpy()
electricity_market_sold = markets["elec_sold"].to_numpy()

carbon_market = markets["carbon"].to_numpy() / 1000

NG_market = markets["nat_gas"].to_numpy()
heat_market_sold = markets["nat_gas_sold"].to_numpy()

H2_market = markets["hydrogen"].to_numpy()

BM_market = markets["biomass"].to_numpy()



def bidding_optimization_model():
    # Create a new model
    model = ConcreteModel()

    # Define sets, parameters, and decision variables
    # ...

    # Define the objective function
    # ...

    # Define constraints
    # ...


    # -------------- Solver --------------
    solvere="gurobi"
    solver = SolverFactory(solvere)
    if solvere == "gurobi":
        solver.options['NonConvex'] = 2
        solver.options['TimeLimit'] = 10
        solver.options["Threads"]= 16
        solver.options["LPWarmStart "]= 2
    solver.solve(model, tee=True)

    return model


if __name__ == "__main__":
    model = bidding_optimization_model()