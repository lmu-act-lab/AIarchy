from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference.CausalInference import CausalInference
from collections import Counter
import networkx as nx
import pandas as pd
import src.util as util
import random
import copy
import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt
import openpyxl
import csv
from typing import *

# from numba import jit, cuda
# import numpy as np

warnings.filterwarnings("ignore")


class Student:

    def __init__(
        self,
        fixed_evidence: dict[str, int] = {},
        weights: dict[str, float] = {},
        threshold=0.1,
        downweigh_factor=0.8,
    ) -> None:
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

        self.memory: list[dict[str, float]] = []
        self.utility_vars: set[str] = {"grades", "social", "health"}
        self.reflective_vars: set[str] = {"Time studying", "Exercise", "Sleep", "ECs"}
        self.fixed_vars: set[str] = {"SES"}
        self.fixed_assignment: dict[str, int] = {}

        self.periodic_vars: set[str] = {"Tutoring", "Motivation"}

        self.grades_query: list[str] = ["Time studying", "Tutoring"]
        self.social_query: list[str] = ["ECs"]
        self.health_query: list[str] = ["Sleep", "Exercise"]

        self.structural_model: BayesianNetwork = BayesianNetwork(self.model.edges())
        self.structural_model.add_nodes_from(self.utility_vars)
        self.structural_model.add_edges_from(
            [
                ("Time studying", "grades"),
                ("Tutoring", "grades"),
                ("ECs", "social"),
                ("Sleep", "health"),
                ("Exercise", "health"),
            ]
        )
        self.structural_model.do(self.reflective_vars, True)

        self.sample_num: int = 100
        self.alpha: float = 0.01
        self.ema_alpha: float = 0.1
        self.threshold: float = threshold
        self.downweigh_factor: float = downweigh_factor
        self.ema: dict[str, float] = {key: 1 for key in self.utility_vars}
        if not weights:
            self.weights: dict[str, float] = util.Counter(
                {key: 1 / len(self.utility_vars) for key in self.utility_vars}
            )
        self.weight_history: dict[int, dict[str, float]] = {}

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

        self.original_model: BayesianNetwork = copy.deepcopy(self.model)

    def define_model(self, edges: list[tuple[str, str]]) -> BayesianNetwork:
        return BayesianNetwork(edges)

    def add_cpt(self, cpt: TabularCPD) -> None:
        self.model.add_cpds(cpt)

    def get_cpts(self) -> list[TabularCPD]:
        return self.model.get_cpds()

    def get_original_cpts(self) -> list[TabularCPD]:
        return self.original_model.get_cpds()

    def get_var_cpt(self, var: str) -> list[TabularCPD]:
        return self.model.get_cpds(var)

    def get_original__var_cpt(self, var: str) -> list[TabularCPD]:
        return self.original_model.get_cpds(var)

    def display_memory(self) -> None:
        for time_step in self.memory:
            print(time_step)
            print("\n")

    def display_weights(self) -> None:
        for time_step, weights in self.weight_history.items():
            print(f"Iteration {time_step}:")
            print(weights)
            print("\n")

    def display_cpts(self) -> None:
        for cpd in self.model.get_cpds():
            print(f"CPD of {cpd.variable}:")
            print(cpd)
            print("\n")

    def display_var_cpt(self, var: str) -> None:
        for cpd in self.model.get_cpds():
            if cpd.variable == var:
                print(f"CPD of {cpd.variable}:")
                print(cpd)
                print("\n")

    def display_original_cpts(self) -> None:
        for cpd in self.original_model.get_cpds():
            print(f"CPD of {cpd.variable}:")
            print(cpd)
            print("\n")

    def draw_model(self) -> None:
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()

        for edge in self.structural_model.edges():
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

    def plot_memory(self) -> None:
        plot_data: dict[str, list[float]] = {}

        for rewards in self.memory:
            for key, value in rewards.items():
                if key not in plot_data:
                    plot_data[key] = []
                plot_data[key].append(value)

        # Plot each key as a separate line
        for key, values in plot_data.items():
            plt.plot(values, label=key)

        # Add labels and a legend
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Line Graph for Each Key")
        plt.legend()

        # Show the plot
        plt.show()

    def reward(self, sample: dict[str, int]) -> dict[str, float]:
        rewards: dict[str, float] = util.Counter()
        rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
        rewards["social"] += sample["ECs"]
        rewards["health"] += sample["Sleep"] + sample["Exercise"]
        return rewards

    def calculate_expected_reward(
        self,
        sample: dict[str, int],
        rewards: dict[str, float],
        model: BayesianNetwork,
    ) -> dict[str, float]:
        inference: CausalInference = CausalInference(model)
        grade_prob: DiscreteFactor = inference.query(
            variables=self.grades_query,
            evidence={
                key: value
                for key, value in sample.items()
                if key not in self.grades_query
            },
        )

        health_prob: DiscreteFactor = inference.query(
            variables=self.health_query,
            evidence={
                key: value
                for key, value in sample.items()
                if key not in self.health_query
            },
        )
        social_prob: DiscreteFactor = inference.query(
            variables=self.social_query,
            evidence={
                key: value
                for key, value in sample.items()
                if key not in self.social_query
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
                    }
                    expected_reward["social"] += reward * social_prob.get_value(
                        **assignment
                    )
        return expected_reward

    def get_cpt_vals(self, variable: str, value: int) -> float:
        cpd: TabularCPD = self.model.get_cpds(variable)
        return cpd.values[value]

    def get_original_cpt_vals(self, variable: str, value: int) -> float:
        cpd: TabularCPD = self.original_model.get_cpds(variable)
        return cpd.values[value]

    def time_step(
        self,
        fixed_evidence: dict[str, int],
        weights: dict[str, float],
        samples: int,
        do: dict[str, int] = {},
    ) -> tuple[dict[str, int | float], dict[str, int | float]]:
        list_of_rewards: list[dict[str, float]] = []
        iter: int = 0
        list_of_samples: list[dict[str, int]] = []
        while iter < samples:
            iter += 1
            if do:
                sample_df = self.model.simulate(
                    n_samples=1, evidence=fixed_evidence, do=do
                )
            else:
                sample_df = self.model.simulate(n_samples=1, evidence=fixed_evidence)

            sample: dict[str, int] = sample_df.iloc[0].to_dict()
            list_of_samples.append(sample)
            rewards: dict[str, float] = self.reward(sample)
            weighted_reward = {}
            for key in weights.keys():
                weighted_reward[key] = rewards[key] * weights[key]

            list_of_rewards.append(
                self.calculate_expected_reward(sample, weighted_reward, self.model)
            )
        return (
            self.get_average_dict(list_of_samples, True),
            self.get_average_dict(list_of_rewards),
        )

    def get_utilities_from_reflective(self, reflective_var: str) -> list[str]:
        dependent_utility_vars: list[str] = []
        for utility in self.utility_vars:
            if self.structural_model.is_dconnected(reflective_var, utility):
                dependent_utility_vars.append(utility)
        return dependent_utility_vars

    def get_average_dict(
        self, list_of_dicts: list[dict[str, float]], round_vals: bool = False
    ) -> dict[str, int | float]:
        average_dict = sum((Counter(sample) for sample in list_of_dicts), Counter())
        for key in average_dict:
            average_dict[key] /= float(len(list_of_dicts))
            if round_vals:
                average_dict[key] = round(average_dict[key])
        return average_dict

    def nudge_cpt(self, cpd, evidence, increase_factor, reward) -> TabularCPD:
        values: list = cpd.values
        indices: list[int] = [evidence[variable] for variable in cpd.variables]
        # Convert indices to tuple to use for array indexing
        indices: tuple[int] = tuple(indices)
        # Update the value at the specified indices
        values[indices] += values[indices] * increase_factor * reward

        # Normalize the probabilities to ensure they sum to 1
        sum_values: list = np.sum(values, axis=0)
        normalized_values: list = values / sum_values

        # Update the CPD with normalized values
        cpd.values: list = normalized_values
        return cpd

    def train(self, iterations: int) -> dict[int, "Student"]:
        cpts: dict[int, Student] = {}
        reward: float = 0
        for i in range(iterations):
            print("in loop")
            # if i == 30:
            #     self.model.get_cpds("Motivation").values[1] += 0.4
            #     self.model.get_cpds("Motivation").values[0] -= 0.4
            cumulative_dict: dict[str, tuple[dict[str, int], dict[str, float]]] = {}

            utility_to_adjust: list[str] = []
            if i > 0:
                for var, ema in self.ema.items():
                    new_ema = (1 - self.ema_alpha) * (
                        ema + self.ema_alpha * self.memory[-1][var]
                    )
                    self.ema[var] = new_ema
                    if new_ema < self.threshold:
                        utility_to_adjust.append(var)

            cumulative_dict["no intervention"] = self.time_step(
                self.fixed_assignment,
                self.weights,
                self.sample_num,
            )

            # if utility_to_adjust:
            utility_to_weights: dict[str, dict[str, float]] = {}
            for var in utility_to_adjust:
                new_weights: dict[str, float] = copy.deepcopy(self.weights)
                new_weights[var] *= self.downweigh_factor
                new_weights = {
                    key: value / sum(new_weights.values())
                    for key, value in new_weights.items()
                }
                utility_to_weights[var] = new_weights
                cumulative_dict[f"{var}- weight change"] = self.time_step(
                    self.fixed_assignment,
                    new_weights,
                    self.sample_num,
                )

            for var, card in self.card_dict.items():
                for val in range(card):
                    cumulative_dict[f"{var}: {val}"] = self.time_step(
                        fixed_evidence=self.fixed_assignment,
                        weights=self.weights,
                        samples=self.sample_num,
                        do={var: val},
                    )

            max_key: str = max(
                cumulative_dict, key=lambda k: sum(cumulative_dict[k][1].values())
            )

            if max_key == "no intervention":
                continue

            if "weight change" in max_key:
                variable = max_key.split("-")[0]
                self.weights = utility_to_weights[variable]
                self.weight_history[i] = self.weights
                continue

            variable: str = max_key.split(": ")[0]
            evidence: dict[str, int] = cumulative_dict[max_key][0]
            self.memory.append(cumulative_dict["no intervention"][1])

            for reward_signal in self.get_utilities_from_reflective(variable):
                reward += cumulative_dict[max_key][1][reward_signal]
            new_cpd = self.nudge_cpt(
                self.model.get_cpds(variable),
                evidence,
                self.alpha,
                reward,
            )
            self.model.remove_cpds(self.model.get_cpds(variable))
            self.model.add_cpds(new_cpd)
            # if i == 29 or i == 31 or i == 99:
            #     cpts[i] = copy.deepcopy(self)
        return cpts

    # Function to convert CPDs to DataFrame
    def cpd_to_dataframe(self, cpd: TabularCPD) -> pd.DataFrame:
        levels: list[range] = [range(card) for card in cpd.cardinality]
        index: pd.MultiIndex = pd.MultiIndex.from_product(levels, names=cpd.variables)

        # Create the DataFrame
        df: pd.DataFrame = pd.DataFrame(
            cpd.values.flatten(), index=index, columns=["Delta"]
        )
        return df

    def compute_cpd_delta(self, var: str) -> DiscreteFactor:
        var_cpd: list[TabularCPD] = self.model.get_cpds(var)
        delta_values: list = var_cpd.values - self.original_model.get_cpds(var).values
        delta_cpd: DiscreteFactor = DiscreteFactor(
            variables=var_cpd.variables,
            cardinality=var_cpd.cardinality,
            values=delta_values,
        )
        return delta_cpd

    def write_cpds_to_csv(self, cpds: DiscreteFactor, name: str, folder: str) -> None:
        for cpd in cpds:
            if cpd.variable not in self.reflective_vars:
                continue
            df: pd.DataFrame = self.cpd_to_dataframe(cpd)
            file_name: str = f"{name}, {cpd.variable}.csv"
            df.to_csv(f"{folder}/{file_name}")

    def write_delta_cpd_to_csv(
        self, cpds: DiscreteFactor, name: str, folder: str
    ) -> None:
        for cpd in cpds:
            if cpd.variable not in self.reflective_vars:
                continue
            df: pd.DataFrame = self.cpd_to_dataframe(
                self.compute_cpd_delta(cpd.variable)
            )
            file_name: str = f"Delta for {name}, {cpd.variable}.csv"
            df.to_csv(f"{folder}/{file_name}")


# x = Student(fixed_evidence={"SES": 0}, threshold=0.1)
# y = Student(fixed_evidence={"SES": 0}, threshold=0.2)
# z = Student(fixed_evidence={"SES": 0}, threshold=0.3)

# x.train(250)
# y.train(250)
# z.train(250)

# x.plot_memory()
# y.plot_memory()
# z.plot_memory()

# print("x\n")
# x.display_weights()
# print("y\n")
# y.display_weights()
# print("z\n")
# z.display_weights()
# TO DO:
# 1. Refactor Code
# 2. More robust unit tests
# a. Test that the weights are being adjusted correctly
