from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import locale
locale.setlocale( locale.LC_ALL, '' )
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
# Import the solver options
from solver_options import get_solver
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

steam_flow = 10000#kg/s
stream_temperature = 373 #Kelvin
steam_temp = 298
Tc = stream_temperature
Td = 20
year_index=0

df = pd.read_csv(r"C:\Users\Sheikh M Ahmed\modelling_MCDM\HPP\heat_pump_data.csv")

cost_data = df['Cost'].to_numpy()
capacity_data = df['Capacity'].to_numpy()
waste_data = df['Waste'].to_numpy()
max_temp_lift_data = df['Lift'].to_numpy()


waste_heat = waste_data
max_waste = max(waste_data)

CP_hp = 2
CP_steam = 2

target_temp = 700

MAX_TEMP_LIFT = 90

print("hi2")
def pyomomodel():
    # Create model
    model = ConcreteModel()

    # -------------- Parameters --------------
    # Time periods (e.g., hours in a day)
    total_time = 360
    
    MONTHS = list(range(total_time))
    model.MONTHS = Set(initialize=MONTHS)

    no_pumps = 10
    model.PUMPS = Set(initialize=range(no_pumps))
    a = 1  # Example value, adjust as needed
    b = 1   # Example value, adjust as needed
    c = 0    # Example value, adjust as needed

    model.Th = Var(model.PUMPS, within=NonNegativeReals, doc='Maximum Temperature of hot stream for each pump')
    model.COP = Var(model.PUMPS, within=NonNegativeReals, doc='Coefficient of Performance for each pump')
    model.heat_pump_purchase_decision = Var(model.MONTHS, within=Binary)

    # -------------- New Decision Variables related to Heat Pumps --------------
    model.no_of_heat_pumps = Var(within=NonNegativeIntegers, initialize=0) # Number of heat pumps installed

    model.heat_pump_installed = Var(model.PUMPS, within=Binary, initialize=0)  # Indicates if a particular heat pump is installed
    model.heat_pump_capacity_installed = Var(model.PUMPS, within=NonNegativeReals)  # Capacity of each installed heat pump
    model.heat_pump_operation = Var(model.MONTHS, model.PUMPS, within=NonNegativeReals)  # Amount of waste heat recovered by each heat pump every hour 
    model.electricity_consumption = Var(model.MONTHS, model.PUMPS, within=NonNegativeReals)  # Electricity consumption of each heat pump every hour
    
    model.T_after_heat_exchange = Var(model.MONTHS, within=NonNegativeReals)

    model.pump_purchase_decision = Var(model.MONTHS, model.PUMPS, within=Binary, initialize=0)
    model.pump_availability = Var(model.MONTHS, model.PUMPS, within=Binary, initialize=0)

    # -------------- Constraints related to Heat Pumps --------------

    def pump_availability_rule(model, m, p):
        if m == 0:
            return model.pump_availability[m, p] == model.pump_purchase_decision[m, p]
        else:
            return model.pump_availability[m, p] == model.pump_availability[m-1, p] + model.pump_purchase_decision[m, p]
    model.pump_availability_con = Constraint(model.MONTHS, model.PUMPS, rule=pump_availability_rule)

    def pump_installed_rule(model, p):
        return model.heat_pump_installed[p] == sum(model.pump_purchase_decision[m, p] for m in model.MONTHS)
    model.pump_installed_con = Constraint(model.PUMPS, rule=pump_installed_rule)

    def heat_pump_operation_rule(model, m, p):
        return model.heat_pump_operation[m, p] <= model.heat_pump_capacity_installed[p] * model.pump_availability[m, p]
    model.heat_pump_operation_con = Constraint(model.MONTHS, model.PUMPS, rule=heat_pump_operation_rule)

    def total_pumps_rule(model):
        return model.no_of_heat_pumps == sum(model.pump_purchase_decision[m, p] for p in model.PUMPS for m in model.MONTHS)
    model.total_pumps_con = Constraint(rule=total_pumps_rule)

    # Ensuring total installed capacity does not exceed total waste heat capacity
    def total_capacity_rule(model):
        return sum(model.heat_pump_capacity_installed[p] for p in model.PUMPS) <= max_waste
    model.total_capacity_con = Constraint(rule=total_capacity_rule)

    def Th_greater_than_Tc_rule(model, p):
        return model.Th[p] >= Tc + 5
    model.Th_greater_Tc_constraint = Constraint(model.PUMPS, rule=Th_greater_than_Tc_rule)

    def max_temp_lift_rule(model, p):
        return model.Th[p] - Tc <= MAX_TEMP_LIFT
    model.max_temp_lift_con = Constraint(model.PUMPS, rule=max_temp_lift_rule)

    def max_capacity_per_pump_rule(model, m, p):
        return model.heat_pump_capacity_installed[p] * model.pump_availability[m, p] <= capacity_data[m]
    model.max_capacity_per_pump_constraint = Constraint(model.MONTHS, model.PUMPS, rule=max_capacity_per_pump_rule)

    def individual_COP_rule(model, p):
        return model.Th[p] == model.COP[p] * (model.Th[p] - Tc)  # Modify this as per your requirement
    model.individual_COP_con = Constraint(model.PUMPS, rule=individual_COP_rule)

    def electricity_to_heat_output_rule(model, m, p):
        return model.heat_pump_operation[m, p] <= model.COP[p] * model.electricity_consumption[m, p] 
    model.electricity_to_heat_output_con = Constraint(model.MONTHS, model.PUMPS, rule=electricity_to_heat_output_rule)

    def waste_exchange_rule(model, m):
        return sum(model.heat_pump_operation[m, p] for p in model.PUMPS) <= waste_heat[m]
    model.waste_exchange_con = Constraint(model.MONTHS, rule=waste_exchange_rule)

    def heat_exchange_rule(model, m):
        return CP_hp*sum(model.heat_pump_operation[m, p] for p in model.PUMPS)*(((sum(model.Th[p] for p in model.PUMPS))/no_pumps) - (steam_temp+10)) == CP_steam*steam_flow*(model.T_after_heat_exchange[m] - steam_temp)
    model.heat_exchange_con = Constraint(model.MONTHS, rule=heat_exchange_rule)

    def T_less_than_steam_rule(model, m):
        return ((sum(model.Th[p] for p in model.PUMPS))/no_pumps) - model.T_after_heat_exchange[m] >= 5
    model.T_less_than_steam_rule_constraint = Constraint(model.MONTHS, rule=T_less_than_steam_rule)

    def heat_pump_objective_rule(model):
        heat_pump_installation_cost = sum(a * model.heat_pump_capacity_installed[p] for p in model.PUMPS for m in model.MONTHS) + model.no_of_heat_pumps * 1E2
        electricity_cost = sum(model.electricity_consumption[m, p] * electricity_market[m*12] for m in model.MONTHS for p in model.PUMPS)
        heat_cost = sum((target_temp - model.T_after_heat_exchange[m]) * NG_market[m*12] for m in model.MONTHS)*1000
        return heat_pump_installation_cost + electricity_cost + heat_cost
    model.objective = Objective(rule=heat_pump_objective_rule, sense=minimize)


    # -------------- Solver --------------
    solver = get_solver(time_limit)  # Use the imported solver configuration
    solver.solve(model, tee=True, symbolic_solver_labels=False)

    return model

def print_selected_heat_pumps(model):
    selected_pumps = model.no_of_heat_pumps.value 
    print(f"Total number of heat pumps selected: {(selected_pumps)}")
    for p in model.PUMPS:
        print(f"Heat Pump {p}: Installed = {model.heat_pump_installed[p].value}, Capacity = {model.heat_pump_capacity_installed[p].value}, Th = {(model.Th[p].value)}, COP = {model.COP[p].value}")
        for m in model.MONTHS:
            print(f"Month {m} - Cap {model.heat_pump_operation[m, p].value} Lec = {round(model.electricity_consumption[m, p].value,2)} Temp: {model.T_after_heat_exchange[m].value} MA: {model.pump_availability[m,p].value} MP: {model.pump_purchase_decision[m, p].value}" )
    print(max_waste)
if __name__ == "__main__":
    model = pyomomodel()

    print_selected_heat_pumps(model)