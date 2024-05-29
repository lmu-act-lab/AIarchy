from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from pgmpy.inference import VariableElimination
from collections import Counter
import networkx as nx
import pandas as pd
import util
import random
import numpy as np

# from pgmpy.inference.causal_inference import CausalInference
import matplotlib.pyplot as plt

MODEL_0 = BayesianNetwork(
    [
        ("SES", "Tutoring"),
        ("SES", "ECs"),
        ("SES", "Time studying"),
        ("Motivation", "Time studying"),
        ("SES", "Exercise"),
        ("ECs", "Time studying"),
        ("Time studying", "Sleep"),
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
    variable="Sleep",
    variable_card=3,
    values=[[0.1, 0.25, 0.55], [0.3, 0.4, 0.25], [0.6, 0.35, 0.2]],
    evidence=["Time studying"],
    evidence_card=[3],
)


def normalize_cpt(values):
    values = np.array(values)
    column_sums = values.sum(axis=0)
    return (values / column_sums).tolist()


# Example to normalize all CPTs
time_studying_cpt = TabularCPD(
    variable="Time studying",
    variable_card=3,
    values=normalize_cpt(
        [
            [0.6, 0.7, 0.3, 0.4, 0.55, 0.65, 0.1, 0.2, 0.4, 0.5, 0.1, 0.2],
            [0.25, 0.2, 0.5, 0.45, 0.3, 0.25, 0.4, 0.35, 0.4, 0.35, 0.3, 0.25],
            [0.15, 0.1, 0.2, 0.15, 0.15, 0.1, 0.5, 0.45, 0.2, 0.15, 0.6, 0.55],
        ]
    ),
    evidence=["SES", "Motivation", "ECs"],
    evidence_card=[3, 2, 2],
)

exercise_cpt = TabularCPD(
    variable="Exercise",
    variable_card=2,
    values=[[0.6, 0.4, 0.2], [0.4, 0.6, 0.8]],
    evidence=["SES"],
    evidence_card=[3],
)

extracurricular_cpt = TabularCPD(
    variable="ECs",
    variable_card=2,
    values=[[0.7, 0.5, 0.1], [0.3, 0.5, 0.9]],
    evidence=["SES"],
    evidence_card=[3],
)

MODEL_0.add_cpds(ses_cpt)
MODEL_0.add_cpds(motivation_cpt)
MODEL_0.add_cpds(tutoring_cpt)
MODEL_0.add_cpds(time_studying_cpt)
MODEL_0.add_cpds(exercise_cpt)
MODEL_0.add_cpds(extracurricular_cpt)
MODEL_0.add_cpds(hours_slept_cpt)

# # Print the CPDs
# for cpd in model.get_cpds():
#     print(f"CPD of {cpd.variable}:")
#     print(cpd)
#     print("\n")


# plt.figure(figsize=(12, 8))
# G = nx.DiGraph()

# for edge in model.edges():
#     G.add_edge(*edge)

# pos = nx.spring_layout(G, k=2)

# nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     node_size=2000,
#     node_color="skyblue",
#     font_size=15,
#     font_weight="bold",
#     arrowsize=20,
# )
# plt.title("Bayesian Network Structure")
# plt.show()

reflective_vars = {"Time studying", "Exercise", "Sleep", "ECs"}
fixed_vars = {"SES"}
periodic_vars = {"Tutoring", "Motivation"}

sampler = BayesianModelSampling(MODEL_0)
samples = sampler.forward_sample(size=50)

N_SAMPLES = 30


def reward(sample: list[dict[str, int]]):
    rewards = util.Counter()
    rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
    rewards["social"] += sample["ECs"] - sample["Time studying"]
    rewards["health"] += sample["Sleep"] + sample["Exercise"]
    return rewards


sampler = BayesianModelSampling(MODEL_0)
samples = sampler.likelihood_weighted_sample(size=1, evidence=[State("SES", 0)])
samples = samples.drop(columns=["_weight"], axis=1)
sample = samples.iloc[0].to_dict()

weights = util.Counter({"grades": 0.4, "social": 0.4, "health": 0.2})
rewards = reward(sample)
weighted_reward = {}
for key in weights.keys():
    weighted_reward[key] = rewards[key] * weights[key]

print(sample)
print(weighted_reward)


def calculate_expected_reward(
    sample: dict[str, int],
    rewards: dict[str, int],
    model: BayesianNetwork,
):
    # Associate with model later
    inference = VariableElimination(model)
    grade_query = ["Time studying", "Tutoring"]
    health_query = ["Sleep", "Exercise"]
    social_query = ["ECs", "Time studying"]
    grade_prob = inference.query(
        variables=grade_query,
        evidence={
            key: value for key, value in sample.items() if key not in grade_query
        },
    )

    health_prob = inference.query(
        variables=health_query,
        evidence={
            key: value for key, value in sample.items() if key not in health_query
        },
    )
    social_prob = inference.query(
        variables=social_query,
        evidence={
            key: value for key, value in sample.items() if key not in social_query
        },
    )

    expected_reward: dict[str, float] = util.Counter()
    for category, reward in rewards.items():
        match category:
            case "grades":
                assignment = {
                    "Time studying": sample["Time studying"],
                    "Tutoring": sample["Tutoring"],
                }
                expected_reward["grades"] += reward * grade_prob.get_value(**assignment)
            case "health":
                assignment = {
                    "Sleep": sample["Sleep"],
                    "Exercise": sample["Exercise"],
                }
                expected_reward["health"] += reward * health_prob.get_value(
                    **assignment
                )
            case "social":
                assignment = {
                    "ECs": sample["ECs"],
                    "Time studying": sample["Time studying"],
                }
                expected_reward["social"] += reward * social_prob.get_value(
                    **assignment
                )
    return expected_reward


cumulative_expected: float = 0
iter: int = 0
sample_rewards = util.Counter()
while iter < N_SAMPLES:
    iter += 1
    evidence = [State("SES", 0)]

    sampler = BayesianModelSampling(MODEL_0)
    samples = sampler.likelihood_weighted_sample(size=1, evidence=[State("SES", 0)])
    samples = samples.drop(columns=["_weight"], axis=1)
    sample = samples.iloc[0].to_dict()

    weights = util.Counter({"grades": 0.4, "social": 0.4, "health": 0.2})
    rewards = reward(sample)
    weighted_reward = {}
    for key in weights.keys():
        weighted_reward[key] = rewards[key] * weights[key]

    cumulative_expected += sum(
        calculate_expected_reward(sample, weighted_reward, MODEL_0).values()
    )
print(cumulative_expected / N_SAMPLES)
