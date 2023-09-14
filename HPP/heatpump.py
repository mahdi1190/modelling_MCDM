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

stream_energy = 11 #MW

stream_flow = 10 #kg/s
stream_temperature = 350 #Kelvin

Tc = stream_temperature
Td = 20
year_index=0


def run_model(inital_data, sheet_name):
    HP_data = pd.read_excel(inital_data, sheet_name=sheet_name)


    capacity = np.array(HP_data["capacity"]) #HP maximum capacity in MW
    max_temp_lift = np.array(HP_data["lift_temp"]) #HP maximum temperature to raise to in degrees C#
    max_temp = np.array(HP_data["working_temp"]) #HP maximum temperature to raise to in degrees C
    cap_cost = np.array(HP_data["cap_cost"]) #HP maximum temperature to raise to in degrees C




    model = ConcreteModel('HP Optimization')

    model.Th = Var(within = Reals, doc = 'Maximum Temperature of hot stream') 
    model.COP = Var(within = Reals, doc = 'Coefficient of Performance') 
    model.MW = Var(within = Reals, doc = 'Energy Rate of Stream') 

    def Constraint1_Rule(model):
        return model.Th - Tc <= max_temp_lift[year_index]
    model.Constraint1 = Constraint(rule = Constraint1_Rule, doc = 'Maximum Temperature Lift')

    def Constraint2_Rule(model):
        return model.Th >= Tc+Td
    model.Constraint2 = Constraint(rule = Constraint2_Rule, doc = 'Minimum Temperature Difference')

    def Constraint3_Rule(model):
        return model.COP * (model.Th-Tc) == ((model.Th))
    model.Constraint3 = Constraint(rule = Constraint3_Rule, doc = 'Coefficient of Performance')

    def Constraint4_Rule(model):
        return model.Th <= max_temp[year_index]
    model.Constraint4 = Constraint(rule = Constraint4_Rule, doc = 'Maximum Working Temperature')
    
    def obj_rule(model):
        return model.COP
    model.Objective_Function = Objective(rule = obj_rule, sense = maximize, doc = 'Maximised Balance')

    from pyomo.opt import SolverFactory
    solver = SolverFactory('gurobi')
    solver.options['NonConvex'] = 2

    Solution = solver.solve(model, tee = True)


    print(model.COP.value, model.Th.value)

def lorenz(Th, Tc):
    """Find the COP_lorenz of a heat pump based on hot and cold stream temperatures in Kelvin"""

    COP_lorenz = ((Th)/(Th-Tc))

    return COP_lorenz

if __name__ == "__main__":
    run_model("HPP/HPP.xlsx", "HPperformance")