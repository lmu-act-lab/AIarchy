from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx

# from pgmpy.inference.causal_inference import CausalInference
import matplotlib.pyplot as plt

model = BayesianNetwork(
    [
        ("SES", "Tutoring"),
        ("SES", "ECs"),
        ("SES", "Time studying"),
        ("Motivation", "Time studying"),
        ("SES", "Exercise"),
        ("Tutoring", "Grades"),
        ("Time studying", "Grades"),
        ("Time studying", "Social"),
        ("Exercise", "Health"),
        ("Hours slept", "Health"),
        ("ECs", "Grades"),
        ("ECs", "Social"),
    ]
)

ses_cpt = TabularCPD(variable="SES", variable_card=3, values=[[0.29], [0.52], [0.19]])
motivation_cpt = TabularCPD(
    variable="Motivation", variable_card=2, values=[[0.5], [0.5]]
)
tutoring_cpt = TabularCPD(
    variable="Tutoring",
    variable_card=2,
    values=[[0.9, 0.6, 0.2], [0.1, 0.4, 0.8]],
    evidence=["SES"],
    evidence_card=[3],
)
hours_slept_cpt = TabularCPD(
    variable="Hours slept", variable_card=3, values=[[0.2], [0.7], [0.1]]
)

model.add_cpds(ses_cpt)
model.add_cpds(motivation_cpt)
model.add_cpds(tutoring_cpt)
model.add_cpds(hours_slept_cpt)

# Print the CPDs
for cpd in model.get_cpds():
    print(f"CPD of {cpd.variable}:")
    print(cpd)
    print("\n")


plt.figure(figsize=(12, 8))
G = nx.DiGraph()

for edge in model.edges():
    G.add_edge(*edge)

pos = nx.spring_layout(G, k=2)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2000,
    node_color="skyblue",
    font_size=15,
    font_weight="bold",
    arrowsize=20,
)
plt.title("Bayesian Network Structure")
plt.show()