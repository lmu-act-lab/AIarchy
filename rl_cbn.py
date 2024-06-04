from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from pgmpy.inference import VariableElimination
from pgmpy.inference.CausalInference import CausalInference
from collections import Counter
import networkx as nx
import pandas as pd
import util
import random
import numpy as np
import itertools

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

HELLO_WORLD_MODEL = BayesianNetwork([("A", "B")])
a_cpt = TabularCPD(variable="A", variable_card=2, values=[[0.5], [0.5]])


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


# def normalize_cpt(values):
#     values = np.array(values)
#     column_sums = values.sum(axis=0)
#     return (values / column_sums).tolist()


# Example to normalize all CPTs
time_studying_cpt = TabularCPD(
    variable="Time studying",
    variable_card=3,
    values=(
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

# Print the CPDs
print("Original CPDs of the model:")
for cpd in MODEL_0.get_cpds():
    print(f"CPD of {cpd.variable}:")
    print(cpd)
    print("\n")


# plt.figure(figsize=(12, 8))
# G = nx.DiGraph()

# for edge in HELLO_WORLD_MODEL.edges():
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

grades_query = ["Time studying", "Tutoring"]
social_query = ["ECs", "Time studying"]
health_query = ["Sleep", "Exercise"]

N_SAMPLES = 30

def reward(sample: dict[str, int]):
    rewards = util.Counter()
    rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
    rewards["social"] += sample["ECs"] - sample["Time studying"]
    rewards["health"] += sample["Sleep"] + sample["Exercise"]
    return rewards


def hello_world_utility(sample: dict[str, int]):
    return sample["A"]


weights = util.Counter({"grades": 0.4, "social": 0.4, "health": 0.2})


def calculate_expected_reward(
    sample: dict[str, int],
    rewards: dict[str, int],
    model: BayesianNetwork,
):
    # Associate with model later
    inference = VariableElimination(model)
    grades_query = ["Time studying", "Tutoring"]
    health_query = ["Sleep", "Exercise"]
    social_query = ["ECs", "Time studying"]
    grade_prob = inference.query(
        variables=grades_query,
        evidence={
            key: value for key, value in sample.items() if key not in grades_query
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


def time_step(fixed_evidence, weights, samples, do=False, intervention={}):
    cumulative_expected: float = 0
    iter: int = 0
    while iter < samples:
        iter += 1
        if do:
            sample = MODEL_0.simulate(
                n_samples=1, evidence=fixed_evidence, do=intervention
            )
        else:
            sample = MODEL_0.simulate(n_samples=1, evidence=fixed_evidence)

        sample = sample.iloc[0].to_dict()
        weights = util.Counter(weights)
        rewards = reward(sample)
        weighted_reward = {}
        for key in weights.keys():
            weighted_reward[key] = rewards[key] * weights[key]

        cumulative_expected += sum(
            calculate_expected_reward(sample, weighted_reward, MODEL_0).values()
        )
    return cumulative_expected / samples


card_dict = {key: MODEL_0.get_cardinality(key) for key in reflective_vars}
cumulative_dict = {}
cumulative_dict["no intervention"] = time_step(
    {"SES": 0},
    weights,
    30,
)

for var, card in card_dict.items():
    for val in range(card):
        cumulative_dict[f"{var}: {val}"] = time_step(
            fixed_evidence={"SES": 0},
            weights=weights,
            samples=30,
            do=True,
            intervention={var: val},
        )

# for k, v in cumulative_dict.items():
#     print(f"{k} : {v} \n")

# Given the intervention with the highest expected reward, adjust the CPT for that variable in the model
max_key = max(cumulative_dict, key=cumulative_dict.get)
variable, value = max_key.split(": ")
# print(MODEL_0.get_cpds(variable))


def make_value_more_likely(cpd, value_index, increase_factor=1.5):
    values = cpd.values
    total_values = values.shape[0]

    # Ensure value_index is within bounds
    if value_index < 0 or value_index >= total_values:
        raise ValueError("Invalid value_index for the given CPD.")

    # print([values[value_index]])

    # Increase the probability of the specified value
    values[value_index] += np.minimum(
        values[value_index] * increase_factor * cumulative_dict[max_key], 1
    )

    # Normalize the probabilities to ensure they sum to 1
    sum_values = np.sum(values, axis=0)
    normalized_values = values / sum_values

    # Update the CPD with normalized values
    cpd.values = normalized_values
    return cpd


NEXT_MODEL = MODEL_0.copy()
NEXT_MODEL.remove_cpds(variable)
NEXT_MODEL.add_cpds(
    make_value_more_likely(MODEL_0.get_cpds(variable), int(value), increase_factor=1.05)
)

ITERATIONS = 1000

while ITERATIONS > 0:
    print("in loop")
    cumulative_dict = {}
    cumulative_dict["no intervention"] = time_step(
        {"SES": 0},
        weights,
        30,
    )

    for var, card in card_dict.items():
        for val in range(card):
            cumulative_dict[f"{var}: {val}"] = time_step(
                fixed_evidence={"SES": 0},
                weights=weights,
                samples=30,
                do=True,
                intervention={var: val},
            )

    max_key = max(cumulative_dict, key=cumulative_dict.get)
    if max_key == "no intervention":
        break
    variable, value = max_key.split(": ")

    new_cpd = make_value_more_likely(
        NEXT_MODEL.get_cpds(variable), int(value), increase_factor=1.05
    )
    NEXT_MODEL.remove_cpds(NEXT_MODEL.get_cpds(variable))
    NEXT_MODEL.add_cpds(new_cpd)

    ITERATIONS -= 1

print("Final Model \n")
for cpd in NEXT_MODEL.get_cpds():
    print(f"{cpd}\n")
