import pandas as pd
import math
import numpy as np
from pyomo.environ import *

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
        return model.COP == ((model.Th)/(model.Th-Tc))
    model.Constraint3 = Constraint(rule = Constraint3_Rule, doc = 'Coefficient of Performance')

    def Constraint4_Rule(model):
        return model.Th <= max_temp[year_index]
    model.Constraint4 = Constraint(rule = Constraint4_Rule, doc = 'Maximum Working Temperature')
    
    def obj_rule(model):
        return model.COP
    model.Objective_Function = Objective(rule = obj_rule, sense = maximize, doc = 'Maximised Balance')

    from pyomo.opt import SolverFactory
    solver = SolverFactory('baron')

    Solution = solver.solve(model, tee = True)


    print(model.COP.value, model.Th.value)

def lorenz(Th, Tc):
    """Find the COP_lorenz of a heat pump based on hot and cold stream temperatures in Kelvin"""

    COP_lorenz = ((Th)/(Th-Tc))

    return COP_lorenz

if __name__ == "__main__":
    run_model("HPP/HPP.xlsx", "HPperformance")