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
import copy
import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class Student:

    def __init__(self, fixed_evidence={}) -> None:
        self.model: BayesianNetwork = self.define_model(
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
        self.reflective_vars: set[str] = {"Time studying", "Exercise", "Sleep", "ECs"}
        self.fixed_vars: set[str] = {"SES"}
        self.fixed_assignment: dict[str, int] = {}

        self.periodic_vars = {"Tutoring", "Motivation"}

        self.grades_query = ["Time studying", "Tutoring"]
        self.social_query = ["ECs", "Time studying"]
        self.health_query = ["Sleep", "Exercise"]

        self.sample_num = 30

        self.weights = util.Counter({"grades": 0.4, "social": 0.4, "health": 0.2})

        self.add_cpt(
            TabularCPD(variable="SES", variable_card=3, values=[[0.29], [0.52], [0.19]])
        )
        self.add_cpt(
            TabularCPD(variable="Motivation", variable_card=2, values=[[0.5], [0.5]])
        )
        self.add_cpt(
            TabularCPD(
                variable="Tutoring",
                variable_card=2,
                values=[[0.9, 0.6, 0.2], [0.1, 0.4, 0.8]],
                evidence=["SES"],
                evidence_card=[3],
            )
        )
        self.add_cpt(
            TabularCPD(
                variable="Time studying",
                variable_card=3,
                values=(
                    [
                        [0.6, 0.7, 0.3, 0.4, 0.55, 0.65, 0.1, 0.2, 0.4, 0.5, 0.1, 0.2],
                        [
                            0.25,
                            0.2,
                            0.5,
                            0.45,
                            0.3,
                            0.25,
                            0.4,
                            0.35,
                            0.4,
                            0.35,
                            0.3,
                            0.25,
                        ],
                        [
                            0.15,
                            0.1,
                            0.2,
                            0.15,
                            0.15,
                            0.1,
                            0.5,
                            0.45,
                            0.2,
                            0.15,
                            0.6,
                            0.55,
                        ],
                    ]
                ),
                evidence=["SES", "Motivation", "ECs"],
                evidence_card=[3, 2, 2],
            )
        )
        self.add_cpt(
            TabularCPD(
                variable="Exercise",
                variable_card=2,
                values=[[0.6, 0.4, 0.2], [0.4, 0.6, 0.8]],
                evidence=["SES"],
                evidence_card=[3],
            )
        )
        self.add_cpt(
            TabularCPD(
                variable="ECs",
                variable_card=2,
                values=[[0.7, 0.5, 0.1], [0.3, 0.5, 0.9]],
                evidence=["SES"],
                evidence_card=[3],
            )
        )
        self.add_cpt(
            TabularCPD(
                variable="Sleep",
                variable_card=3,
                values=[[0.1, 0.25, 0.55], [0.3, 0.4, 0.25], [0.6, 0.35, 0.2]],
                evidence=["Time studying"],
                evidence_card=[3],
            )
        )
        if not fixed_evidence:
            for var in self.fixed_vars:
                example_sample: dict[str, int] = (
                    self.model.simulate(1).iloc[0].to_dict()
                )
                self.fixed_assignment[var] = example_sample[var]
        else:
            self.fixed_assignment = fixed_evidence
        self.card_dict: dict[str, int] = {
            key: self.model.get_cardinality(key) for key in self.reflective_vars
        }

        self.original_model = copy.deepcopy(self.model)

    def define_model(self, edges: list[tuple[str, str]]):
        return BayesianNetwork(edges)

    def add_cpt(self, cpt: TabularCPD):
        self.model.add_cpds(cpt)
        pass

    def get_cpts(self) -> list[TabularCPD]:
        return self.model.get_cpds()

    def get_original_cpts(self) -> list[TabularCPD]:
        return self.original_model.get_cpds()

    def display_cpts(self) -> None:
        for cpd in self.model.get_cpds():
            print(f"CPD of {cpd.variable}:")
            print(cpd)
            print("\n")

    def display_original_cpts(self) -> None:
        for cpd in self.original_model.get_cpds():
            print(f"CPD of {cpd.variable}:")
            print(cpd)
            print("\n")

    def draw_model(self):
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()

        for edge in self.model.edges():
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

    def reward(self, sample: dict[str, int]):
        rewards = util.Counter()
        rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
        rewards["social"] += sample["ECs"] - sample["Time studying"]
        rewards["health"] += sample["Sleep"] + sample["Exercise"]
        return rewards

    def calculate_expected_reward(
        self,
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
                    expected_reward["grades"] += reward * grade_prob.get_value(
                        **assignment
                    )
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

    def get_cpt_vals(self, variable, value):
        cpd = self.model.get_cpds(variable)
        return cpd.values[value]

    def get_original_cpt_vals(self, variable, value):
        cpd = self.original_model.get_cpds(variable)
        return cpd.values[value]

    def time_step(self, fixed_evidence, weights, samples, do={}):
        cumulative_expected: float = 0
        iter: int = 0
        while iter < samples:
            iter += 1
            if do:
                sample = self.model.simulate(
                    n_samples=1, evidence=fixed_evidence, do=do
                )
            else:
                sample = self.model.simulate(n_samples=1, evidence=fixed_evidence)

            sample = sample.iloc[0].to_dict()
            weights = util.Counter(weights)
            rewards = self.reward(sample)
            weighted_reward = {}
            for key in weights.keys():
                weighted_reward[key] = rewards[key] * weights[key]

            cumulative_expected += sum(
                self.calculate_expected_reward(
                    sample, weighted_reward, self.model
                ).values()
            )
        return cumulative_expected / samples

    # for k, v in cumulative_dict.items():
    #     print(f"{k} : {v} \n")

    # Given the intervention with the highest expected reward, adjust the CPT for that variable in the model

    # print(MODEL_0.get_cpds(variable))

    def nudge_cpt(self, cpd, value_index, increase_factor, cumulative_dict):
        values = cpd.values
        total_values = values.shape[0]

        # Ensure value_index is within bounds
        if value_index < 0 or value_index >= total_values:
            raise ValueError("Invalid value_index for the given CPD.")

        # print([values[value_index]])
        max_key = max(cumulative_dict, key=cumulative_dict.get)

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

    def train(self, iterations: int):
        while iterations > 0:
            print("in loop")
            cumulative_dict = {}
            cumulative_dict["no intervention"] = self.time_step(
                self.fixed_assignment,
                self.weights,
                30,
            )

            for var, card in self.card_dict.items():
                for val in range(card):
                    cumulative_dict[f"{var}: {val}"] = self.time_step(
                        fixed_evidence=self.fixed_assignment,
                        weights=self.weights,
                        samples=30,
                        do={var: val},
                    )

            max_key = max(cumulative_dict, key=cumulative_dict.get)
            variable, value = max_key.split(": ")

            new_cpd = self.nudge_cpt(
                self.model.get_cpds(variable), int(value), 0.05, cumulative_dict
            )
            self.model.remove_cpds(self.model.get_cpds(variable))
            self.model.add_cpds(new_cpd)

            iterations -= 1
