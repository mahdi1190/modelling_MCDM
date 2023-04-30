import pandas as pd
import math
import numpy as np
from pyomo.environ import *


stream_energy = 11 #MW
stream_temperature = 350 #Kelvin


def run_model(inital_data, sheet_name):
    HP_data = pd.read_excel(inital_data, sheet_name=sheet_name)


    capacity = np.array(HP_data["capacity"]) #HP maximum capacity in MW
    max_temp = np.array(HP_data["temp"]) #HP maximum temperature to raise to in degrees C
    cap_cost = np.array(HP_data["cap_cost"]) #HP maximum temperature to raise to in degrees C




    model = ConcreteModel('HP Optimization')

    model.T_lift = Var(within = Reals, doc = 'Temperature Lift') 



def lorenz(Th, Tc):
    """Find the COP_lorenz of a heat pump based on hot and cold stream temperatures in Kelvin"""

    COP_lorenz = ((Th)/(Th-Tc))

    return COP_lorenz

if __name__ == "__main__":
    run_model("HPP/HPP.xlsx", "HPperformance")