from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Define the Structure of the Bayesian Network
model = BayesianModel([
    ('NaturalGasCrisis', 'ElectricityPrice'),
    ('NaturalGasCrisis', 'HydrogenProduction'),
    ('PoliticalStability', 'NaturalGasCrisis'),
    ('ElectricityPrice', 'HydrogenProduction')
])

# Step 2: Define Conditional Probability Distributions (CPDs)
cpd_ng = TabularCPD(variable='NaturalGasCrisis', variable_card=2,
                    evidence=['PoliticalStability'], evidence_card=[2],
                    values=[[0.95, 0.05],  # Probabilities of NoCrisis given Stable and Unstable
                            [0.05, 0.95]],  # Probabilities of Crisis given Stable and Unstable
                    state_names={'NaturalGasCrisis': ['NoCrisis', 'Crisis'],
                                 'PoliticalStability': ['Stable', 'Unstable']})


cpd_pol = TabularCPD(variable='PoliticalStability', variable_card=2,
                     values=[[0.95], [0.05]], 
                     state_names={'PoliticalStability': ['Stable', 'Unstable']})

cpd_elec = TabularCPD(variable='ElectricityPrice', variable_card=2, 
                      evidence=['NaturalGasCrisis'], evidence_card=[2],
                      values=[[0.8, 0.4], [0.2, 0.6]],
                      state_names={'ElectricityPrice': ['Low', 'High'], 'NaturalGasCrisis': ['NoCrisis', 'Crisis']})

cpd_hydro = TabularCPD(variable='HydrogenProduction', variable_card=2,
                       evidence=['NaturalGasCrisis', 'ElectricityPrice'], evidence_card=[2, 2],
                       values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
                       state_names={'HydrogenProduction': ['High', 'Low'], 'NaturalGasCrisis': ['NoCrisis', 'Crisis'], 'ElectricityPrice': ['Low', 'High']})

# Step 3: Attach CPDs to the model
model.add_cpds(cpd_ng, cpd_pol, cpd_elec, cpd_hydro)

# Step 4: Check model validity
print("Model Check: ", model.check_model())
