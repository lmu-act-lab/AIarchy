from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State, DiscreteFactor
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
import openpyxl
import csv

from numba import jit, cuda
import numpy as np

warnings.filterwarnings("ignore")


class Student:

    def __init__(self, fixed_evidence={}, weights={}) -> None:
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

        self.sample_num = 100
        self.alpha = 0.01

        self.weights = (
            util.Counter({"grades": 0.15, "social": 0.4, "health": 0.45})
            if not weights
            else weights
        )

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

    def get_var_cpt(self, var) -> list[TabularCPD]:
        return self.model.get_cpds(var)

    def get_original__var_cpt(self, var) -> list[TabularCPD]:
        return self.original_model.get_cpds(var)

    def display_cpts(self) -> None:
        for cpd in self.model.get_cpds():
            print(f"CPD of {cpd.variable}:")
            print(cpd)
            print("\n")

    def display_var_cpt(self, var) -> None:
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
        rewards["social"] += sample["ECs"]
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
        social_query = ["ECs"]
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
        list_of_samples = []
        while iter < samples:
            iter += 1
            if do:
                sample = self.model.simulate(
                    n_samples=1, evidence=fixed_evidence, do=do
                )
            else:
                sample = self.model.simulate(n_samples=1, evidence=fixed_evidence)

            sample = sample.iloc[0].to_dict()
            list_of_samples.append(sample)
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
        return (self.get_average_sample(list_of_samples), cumulative_expected / samples)

    # for k, v in cumulative_dict.items():
    #     print(f"{k} : {v} \n")

    # Given the intervention with the highest expected reward, adjust the CPT for that variable in the model

    # print(MODEL_0.get_cpds(variable))

    # def nudge_cpt(self, cpd, value_index, increase_factor, cumulative_dict):
    #     values = cpd.values
    #     total_values = values.shape[0]

    #     # Ensure value_index is within bounds
    #     if value_index < 0 or value_index >= total_values:
    #         raise ValueError("Invalid value_index for the given CPD.")

    #     # print([values[value_index]])
    #     max_key = max(cumulative_dict, key=lambda x: cumulative_dict[x][1])

    #     # Increase the probability of the specified value
    #     values[value_index] += np.minimum(
    #         values[value_index] * increase_factor * (cumulative_dict[max_key])[1], 1
    #     )

    #     # Normalize the probabilities to ensure they sum to 1
    #     sum_values = np.sum(values, axis=0)
    #     normalized_values = values / sum_values

    #     # Update the CPD with normalized values
    #     cpd.values = normalized_values
    #     return cpd

    def get_average_sample(self, list_of_samples):
        average_sample = sum((Counter(sample) for sample in list_of_samples), Counter())
        for key in average_sample:
            average_sample[key] /= len(list_of_samples)
            average_sample[key] = round(average_sample[key])
        return average_sample

    # def nudge_cpt(self, cpd, increase_factor, var, value, reward, evidence):
    #     values = cpd.values
    #     total_values = values.size
    #     if var in evidence:
    #         del evidence[var]

    #     # Convert evidence to index
    #     evidence_index = []
    #     for var in cpd.variables:
    #         if var in evidence:
    #             evidence_index.append(cpd.state_names[var].index(evidence[var]))
    #         else:
    #             evidence_index.append(value)  # Default to first state if no evidence
    #     value_index = np.ravel_multi_index(evidence_index, values.shape)

    #     # Ensure value_index is within bounds
    #     if value_index < 0 or value_index >= total_values:
    #         raise ValueError("Invalid value_index for the given CPD.")

    #     # Increase the probability of the specified value
    #     values.ravel()[value_index] += np.minimum(
    #         values.ravel()[value_index] * increase_factor * reward,
    #         1,
    #     )

    #     # Normalize the probabilities to ensure they sum to 1
    #     sum_values = np.sum(values, axis=0)
    #     normalized_values = values / sum_values

    #     # Update the CPD with normalized values
    #     cpd.values = normalized_values
    #     return cpd

    def nudge_cpt(self, cpd, evidence, query_var, query_val, increase_factor, reward):
        values = cpd.values
        indices = [evidence[variable] for variable in cpd.variables]
        # Convert indices to tuple to use for array indexing
        indices = tuple(indices)
        # Update the value at the specified indices
        values[indices] += values[indices] * increase_factor * reward

        # Normalize the probabilities to ensure they sum to 1
        sum_values = np.sum(values, axis=0)
        normalized_values = values / sum_values

        # Update the CPD with normalized values
        cpd.values = normalized_values
        return cpd

    def train(self, iterations: int):
        cpts = {}
        for i in range(iterations):
            print("in loop")
            if i == 30:
                self.model.get_cpds("Motivation").values[1] += 0.4
                self.model.get_cpds("Motivation").values[0] -= 0.4
            cumulative_dict = {}
            cumulative_dict["no intervention"] = self.time_step(
                self.fixed_assignment,
                self.weights,
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

            max_key = max(cumulative_dict, key=lambda x: cumulative_dict[x][1])
            if max_key == "no intervention":
                continue
            variable, value = max_key.split(": ")
            new_cpd = self.nudge_cpt(
                self.model.get_cpds(variable),
                (cumulative_dict[max_key])[0],
                variable,
                int(value),
                self.alpha,
                (cumulative_dict[max_key])[1],
            )
            self.model.remove_cpds(self.model.get_cpds(variable))
            self.model.add_cpds(new_cpd)
            if i == 29 or i == 31 or i == 99:
                cpts[i] = copy.deepcopy(self)
        return cpts

    # Function to convert CPDs to DataFrame
    def cpd_to_dataframe(self, cpd):
        levels = [range(card) for card in cpd.cardinality]
        index = pd.MultiIndex.from_product(levels, names=cpd.variables)

        # Create the DataFrame
        df = pd.DataFrame(cpd.values.flatten(), index=index, columns=["Delta"])
        return df

    def compute_cpd_delta(self, var):
        var_cpd = self.model.get_cpds(var)
        delta_values = var_cpd.values - self.original_model.get_cpds(var).values
        delta_cpd = DiscreteFactor(
            variables=var_cpd.variables,
            cardinality=var_cpd.cardinality,
            values=delta_values,
        )
        return delta_cpd

    def write_cpds_to_csv(self, cpds, name):
        for cpd in cpds:
            if cpd.variable not in self.reflective_vars:
                continue
            df = self.cpd_to_dataframe(cpd)
            file_name = f"{name}, {cpd.variable}.csv"
            df.to_csv(f"data/{file_name}")

    def write_delta_cpd_to_csv(self, cpds, name):
        for cpd in cpds:
            if cpd.variable not in self.reflective_vars:
                continue
            df = self.cpd_to_dataframe(self.compute_cpd_delta(cpd.variable))
            file_name = f"Delta for {name}, {cpd.variable}.csv"
            df.to_csv(f"data/{file_name}")


broke_academic_student = Student(
    fixed_evidence={"SES": 0},
    weights=util.Counter({"grades": 0.6, "social": 0.15, "health": 0.25}),
)
rich_academic_student = Student(
    fixed_evidence={"SES": 2},
    weights=util.Counter({"grades": 0.6, "social": 0.15, "health": 0.25}),
)
broke_jock_student = Student(
    fixed_evidence={"SES": 0},
    weights=util.Counter({"grades": 0.15, "social": 0.4, "health": 0.45}),
)
rich_jock_student = Student(
    fixed_evidence={"SES": 2},
    weights=util.Counter({"grades": 0.15, "social": 0.4, "health": 0.45}),
)

# iter_model_x = x.train(1)
# iter_model_y = y.train(1)

brs = broke_academic_student.train(100)
ras = rich_academic_student.train(100)
bjs = broke_jock_student.train(100)
rjs = rich_jock_student.train(100)

for iter, student in brs.items():
    broke_academic_student.write_cpds_to_csv(
        student.get_cpts(), f"Trained for bas @ iteration {iter}"
    )
for iter, student in ras.items():
    broke_academic_student.write_cpds_to_csv(
        student.get_cpts(), f"Trained for ras @ iteration {iter}"
    )
for iter, student in bjs.items():
    broke_academic_student.write_cpds_to_csv(
        student.get_cpts(), f"Trained for bjs @ iteration {iter}"
    )
for iter, student in rjs.items():
    broke_academic_student.write_cpds_to_csv(
        student.get_cpts(), f"Trained for rjs @ iteration {iter}"
    )
# x.display_original_cpts()
# x.display_var_cpt("Time studying")
# x.display_var_cpt("ECs")
# x.display_var_cpt("Exercise")
# x.display_var_cpt("Sleep")

# y.display_original_cpts()
# y.display_var_cpt("ECs")
# y.display_var_cpt("Time studying")
# y.display_var_cpt("Exercise")
# y.display_var_cpt("Sleep")
# for iteration, student in iter_model_x.items():
#     print(f"SES 0 Student: iteration {iteration} :\n")
#     student.display_var_cpt("Time studying")
#     student.display_var_cpt("ECs")
#     student.display_var_cpt("Exercise")
#     student.display_var_cpt("Sleep")


# print("================================================")

# for iteration, student in iter_model_y.items():
#     print(f"SES 2 Student: iteration {iteration} :\n")
#     student.display_var_cpt("Time studying")
#     student.display_var_cpt("ECs")
#     student.display_var_cpt("Exercise")
#     student.display_var_cpt("Sleep")


# Extract CPDs and save to Excel file
# broke_academic_student.write_cpds_to_csv(
#     broke_academic_student.get_cpts(), "Trained for Broke Academic Student"
# )
# broke_academic_student.write_delta_cpd_to_csv(
#     broke_academic_student.get_cpts(), "Broke Academic Student"
# )

# rich_academic_student.write_cpds_to_csv(
#     rich_academic_student.get_cpts(), "Trained for Rich Academic Student"
# )
# rich_academic_student.write_delta_cpd_to_csv(
#     rich_academic_student.get_cpts(), "Trained for Rich Academic Student"
# )

# broke_academic_student.write_cpds_to_csv(
#     broke_jock_student.get_cpts(), "Trained for Broke Jock Student"
# )
# broke_academic_student.write_delta_cpd_to_csv(
#     broke_jock_student.get_cpts(), "Trained for Broke Jock Student"
# )

# rich_jock_student.write_cpds_to_csv(
#     rich_jock_student.get_cpts(), "Trained for Rich Jock Student"
# )
# rich_jock_student.write_delta_cpd_to_csv(
#     rich_jock_student.get_cpts(), "Trained for Rich Jock Student"
# )
