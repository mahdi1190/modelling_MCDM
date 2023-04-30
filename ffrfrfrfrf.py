import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:\\Users\\james\\Desktop\\Uni\\CPE460\\Assignment 2\Data.csv")

hydrogen= np.arange(7.7, 7.9, 0.01)
from pyomo.environ import *

consumption = []
generation = []
spot = []

final=[]

for i in range(len(df)):
    consumption.append(float(df['Electricity Consumption (kWh)'][i]))
    generation.append(float(df['Wind Generation (kW)'][i]))
    spot.append(float(0.001 * df['Electricity Spot Price (GBP/MWh)'][i]))

for hyd in hydrogen:
    moneyx = []
    a = []
    for i in range(len(df)):
        model = ConcreteModel('Grid Optimisation')

        model.a = Var(within = Reals, doc = 'coeff')
        model.money = Var(within = Reals, doc = 'Money')
        def Constraint0_Rule(model):
            return model.a >= 0
        model.Constraint0 = Constraint(rule = Constraint0_Rule, doc = 'a Min')

        def Constraint1_Rule(model):
            return model.a <= 1
        model.Constraint1 = Constraint(rule = Constraint1_Rule, doc = 'a Max')

        def Constraint2_Rule(model):
            return model.money == model.a * (generation[i] - consumption[i]) * (spot[i]) + (1 - model.a) * (hyd / 53.3) * (generation[i] - consumption[i])
        model.Constraint2 = Constraint(rule = Constraint2_Rule)

        def Constraint3_Rule3(model):
            return (1 - model.a) * (hyd / 53.3) * (generation[i] - consumption[i]) >= 0
        model.Constraint3 = Constraint(rule = Constraint3_Rule3)

        def obj_rule(model):
            return model.a * (generation[i] - consumption[i]) * (spot[i]) + (1 - model.a) * (hyd/ 53.3) * (generation[i] - consumption[i])
        model.Objective_Function = Objective(rule = obj_rule, sense = maximize, doc = 'Maximised Balance')

        from pyomo.opt import SolverFactory

        solver = SolverFactory('cbc')

        Solution = solver.solve(model, tee = True)

        a.append(model.a.value)
        moneyx.append(model.money.value)
    final.append(sum(moneyx))

df2 = pd.DataFrame(final, columns = ['final'])
print(final)
plt.show()
df2.to_csv("C:\\Users\\james\\Desktop\\Uni\\CPE460\\Assignment 2\Iteration 7.7 to 7.9.csv")