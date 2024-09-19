import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import copy
import random
from util import *
from typing import Callable
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork  # type: ignore
from pgmpy.inference import CausalInference  # type: ignore

warnings.filterwarnings("ignore")


class CausalLearningAgent:

    def __init__(
        self,
        sampling_edges: list[tuple[str, str]],
        utility_edges: list[tuple[str, str]],
        cpts: list[TabularCPD],
        utility_vars: set[str],
        reflective_vars: set[str],
        chance_vars: set[str],
        glue_vars: set[str],
        reward_func: Callable[[dict[str, int]], dict[str, float]],
        fixed_evidence: dict[str, int] = {},
        weights: dict[str, float] = {},
        threshold: float = 0.05,
        temperature: float = 1,
        downweigh_factor: float = 0.95,
        sample_num: int = 100,
        alpha: float = 0.01,
        ema_alpha: float = 0.1,
    ) -> None:
        """
        Initializes two Bayesian Networks, one for sampling and one to maintain structure. Also initializes a number of other variables to be used in the learning process.

        Parameters
        ----------
        sampling_edges : list[tuple[str, str]]
            Edges of the network to be used in the sampling model.
        utility_edges : list[tuple[str, str]]
            Edges for the reward signals/utility vars (don't repeat the ones in sampling_edges).
        cpts : list[TabularCPD]
            CPTs for the sampling model.
        utility_vars : set[str]
            Utility nodes/reward signals.
        reflective_vars : set[str]
            Vars agent can intervene on.
        chance_vars : set[str]
            Vars that the agent cannot control.
        glue_vars : set[str]
            Vars that will be affected by agents from greater hierarchies.
        reward_func : Callable[[dict[str, int]], dict[str, float]]
            Reward function for the agent.
        fixed_evidence : dict[str, int], optional
            Any fixed evidence for that particular agent, by default {}
        weights : dict[str, float], optional
            Any starting weights for archetype of agent (len must match how many utility vars there are), by default {}
        threshold : float, optional
            Any starting weights for archetype of agent (len must match how many utility vars there are), by default 0.1
        temperature : float, optional
            Temperature for simulated annealing, by default 1
        downweigh_factor : float, optional
            Amount to downweigh weights by, by default 0.8
        sample_num : int, optional
            Number of samples taken for one time step, by default 100
        alpha : float, optional
            Learning rate, by default 0.01
        ema_alpha : float, optional
            EMA adjustment rate, by default 0.1
        """

        self.sampling_model: BayesianNetwork = BayesianNetwork(sampling_edges)

        self.memory: list[TimeStep] = []
        self.utility_vars: set[str] = utility_vars
        self.reflective_vars: set[str] = reflective_vars
        self.chance_vars: set[str] = chance_vars
        self.glue_vars: set[str] = glue_vars
        self.non_utility_vars: set[str] = reflective_vars | chance_vars | glue_vars
        self.fixed_assignment: dict[str, int] = fixed_evidence

        self.structural_model: BayesianNetwork = BayesianNetwork(
            self.sampling_model.edges()
        )
        self.structural_model.add_nodes_from(self.utility_vars)
        self.structural_model.add_edges_from(utility_edges)

        self.reward_attr_model = self.structural_model.do(self.reflective_vars)
        self.inference_model = self.structural_model.do(self.non_utility_vars)

        self.sample_num: int = sample_num
        self.alpha: float = alpha
        self.ema_alpha: float = ema_alpha
        self.threshold: float = threshold
        self.temperature: float = temperature
        self.downweigh_factor: float = downweigh_factor
        self.ema: dict[str, float] = {key: 1 for key in self.utility_vars}
        if not weights:
            self.weights: dict[str, float] = Counter(
                {key: 1 / len(self.utility_vars) for key in self.utility_vars}
            )
        self.weight_history: dict[int, dict[str, float]] = {}
        self.reward_func = reward_func

        for cpt in cpts:
            self.sampling_model.add_cpds(cpt)

        self.card_dict: dict[str, int] = {
            key: self.sampling_model.get_cardinality(key)
            for key in self.reflective_vars
        }

        self.original_model: BayesianNetwork = copy.deepcopy(
            self.sampling_model)

    def get_cpts(self) -> list[TabularCPD]:
        """
        Returns the CPDs of the sampling model.

        Returns
        -------
        list[TabularCPD]
            List of CPDs.
        """
        return self.sampling_model.get_cpds()

    def get_original_cpts(self) -> list[TabularCPD]:
        """
        Returns the CPDs of the original model.

        Returns
        -------
        list[TabularCPD]
            List of CPDs.
        """
        return self.original_model.get_cpds()

    def get_var_cpt(self, var: str) -> list[TabularCPD]:
        """
        Returns the CPDs of a particular variable in the sampling model.

        Parameters
        ----------
        var : str
            Variable to get the CPD of.

        Returns
        -------
        list[TabularCPD]
            List of CPDs.
        """
        return self.sampling_model.get_cpds(var)

    def get_original__var_cpt(self, var: str) -> list[TabularCPD]:
        """
        Returns the CPDs of a particular variable in the original model.

        Parameters
        ----------
        var : str
            Variable to get the CPD of.

        Returns
        -------
        list[TabularCPD]
            List of CPDs.
        """
        return self.original_model.get_cpds(var)

    def display_memory(self) -> None:
        """
        Displays the memory of the agent
        """
        for time_step in self.memory:
            print(time_step)
            print("\n")

    def display_weights(self) -> None:
        """
        Displays the weights of the agent.
        """
        for time_step, weights in self.weight_history.items():
            print(f"Iteration {time_step}:")
            print(weights)
            print("\n")

    def display_cpts(self) -> None:
        """
        Displays the CPDs of the sampling model.
        """
        for cpd in self.sampling_model.get_cpds():
            print(f"CPD of {cpd.variable}:")
            print(cpd)
            print("\n")

    def display_var_cpt(self, var: str) -> None:
        """
        Displays the CPDs of a particular variable in the sampling model.

        Parameters
        ----------
        var : str
            Variable to display the CPD of.
        """
        for cpd in self.sampling_model.get_cpds():
            if cpd.variable == var:
                print(f"CPD of {cpd.variable}:")
                print(cpd)
                print("\n")

    def display_original_var_cpt(self, var: str) -> None:
        """
        Displays the CPDs of a particular variable in the sampling model.

        Parameters
        ----------
        var : str
            Variable to display the CPD of.
        """
        for cpd in self.original_model.get_cpds():
            if cpd.variable == var:
                print(f"CPD of {cpd.variable}:")
                print(cpd)
                print("\n")

    def display_original_cpts(self) -> None:
        """
        Displays the CPDs of the original model.
        """
        for cpd in self.original_model.get_cpds():
            print(f"CPD of {cpd.variable}:")
            print(cpd)
            print("\n")

    def draw_model(self):
        """
        Draws the structural model of the agent.
        """
        G = nx.DiGraph()
        G.add_edges_from(self.structural_model.edges())

        color_map = {
            "reflective": "blue",
            "glue": "green",
            "chance": "red",
            "utility": "yellow",
        }

        edge_colors = []

        for edge in G.edges():
            source, target = edge
            if target in self.reflective_vars:
                edge_colors.append(color_map["reflective"])
            elif target in self.glue_vars:
                edge_colors.append(color_map["glue"])
            elif target in self.chance_vars:
                edge_colors.append(color_map["chance"])
            elif target in self.utility_vars:
                edge_colors.append(color_map["utility"])
            else:
                edge_colors.append("black")

        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            edges=G.edges(),
            edge_color=edge_colors,
            with_labels=True,
            node_size=2000,
            node_color="lightgrey",
            font_size=12,
            font_weight="bold",
        )

        plt.show()

    def plot_memory(self) -> None:
        """
        Plots the perceived reward at each iteration.
        """
        plot_data: dict[str, list[float]] = {}

        for time_step in self.memory:
            for key, value in time_step.average_reward.items():
                if key not in plot_data:
                    plot_data[key] = []
                plot_data[key].append(value)

        for key, values in plot_data.items():
            plt.plot(values, label=key)

        plt.xlabel("Iteration")
        plt.ylabel("Perceived Reward")
        plt.title("Perceived reward at iteration t")
        plt.legend()

        plt.show()

    def plot_memory_against(self, other_cla: "CausalLearningAgent") -> None:
        """
        Plots the perceived reward at each iteration.
        """
        plot_data: dict[str, list[float]] = {}

        for time_step in self.memory:
            for key, value in time_step.average_reward.items():
                if f"original, {key}" not in plot_data:
                    plot_data[f"original, {key}"] = []
                plot_data[f"original, {key}"].append(value)

        for time_step in other_cla.memory:
            for key, value in time_step.average_reward.items():
                if f"compared, {key}" not in plot_data:
                    plot_data[f"compared, {key}"] = []
                plot_data[f"compared, {key}"].append(value)

        for key, values in plot_data.items():
            plt.plot(values, label=key)

        plt.xlabel("Iteration")
        plt.ylabel("Perceived Reward")
        plt.title("Perceived reward at iteration t")
        plt.legend()

        plt.show()

    def calculate_expected_reward(
        self,
        sample: dict[str, int],
        rewards: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculates expected reward given a sample and its associated reward.

        Parameters
        ----------
        sample : dict[str, int]
            sample associated with reward
        rewards : dict[str, float]
            weighted reward received from sample

        Returns
        -------
        dict[str, float]
            expected reward given sample & structure
        """

        inference: CausalInference = CausalInference(self.structural_model)
        rewards_queries: dict[str, list[str]] = {}
        reward_probs: dict[str, DiscreteFactor] = {}
        expected_rewards: dict[str, float] = {}

        for category in rewards.keys():
            reward_query: list[str] = []
            for var in self.non_utility_vars:
                if self.inference_model.is_dconnected(var, category):
                    reward_query.append(var)
            rewards_queries[category] = reward_query

        for category, connected_vars in rewards_queries.items():
            reward_probs[category] = inference.query(
                variables=connected_vars,
                evidence={
                    var: ass for var, ass in sample.items() if var not in connected_vars
                },
            )

        for category, reward in rewards.items():
            assignment = {var: sample[var]
                          for var in rewards_queries[category]}
            expected_rewards[category] += reward * reward_probs[category].get_value(
                **assignment
            )

        return expected_rewards

    def get_cpt_vals(self, variable: str, value: int) -> float:
        """
        Get the value of an assigned variable in the updated CPT

        Parameters
        ----------
        variable : str
            Variable to assign value to.
        value : int
            Value to assign variable.

        Returns
        -------
        float
            Prob. value associated with assignment.
        """
        cpd: TabularCPD = self.sampling_model.get_cpds(variable)
        return cpd.values[value]

    def get_original_cpt_vals(self, variable: str, value: int) -> float:
        """
        Get the value of an assigned variable in the original CPT

        Parameters
        ----------
        variable : str
            Variable to assign value to.
        value : int
            Value to assign variable.

        Returns
        -------
        float
            Prob. value associated with assignment.
        """
        cpd: TabularCPD = self.original_model.get_cpds(variable)
        return cpd.values[value]

    def time_step(
        self,
        fixed_evidence: dict[str, int],
        weights: dict[str, float],
        samples: int,
        do: dict[str, int] = {},
    ) -> "TimeStep":
        memory = pd.DataFrame(columns=list(self.structural_model.nodes()))
        for _ in range(samples):
            weighted_reward: dict[str, float] = {}
            if do:
                sample_df = self.sampling_model.simulate(
                    n_samples=1,
                    evidence=fixed_evidence,
                    do=do,
                )
            else:
                sample_df = self.sampling_model.simulate(
                    n_samples=1, evidence=fixed_evidence
                )
            sample_dict: dict[str, int] = sample_df.iloc[0].to_dict()
            rewards: dict[str, float] = self.reward_func(sample_dict)
            for var, weight in weights.items():
                weighted_reward[var] = rewards[var] * weight
            combined_dict: pd.DataFrame = pd.DataFrame(
                [{**sample_dict, **weighted_reward}]
            )
            memory = pd.concat([memory, combined_dict], ignore_index=True)
        return TimeStep(memory, self.non_utility_vars, self.utility_vars, weights, next(iter(do)) if do else None)

    def nudge_cpt(
        self,
        cpd: TabularCPD,
        evidence: dict[str, int],
        increase_factor: float,
        reward: float,
    ) -> TabularCPD:
        """
        Nudges specific value in CPT.

        Parameters
        ----------
        cpd : TabularCPD
            Structure representing the CPT.
        evidence : dict[str, int]
            Evidence of value to be nudged.
        increase_factor : float
            How much to nudge value.
        reward : float
            Reward signal to nudge value by.

        Returns
        -------
        TabularCPD
            New CPT with nudged value.
        """
        values: np.ndarray = cpd.values
        indices: list[int] = [evidence[variable] for variable in cpd.variables]

        tuple_indices: tuple[int, ...] = tuple(indices)

        values[tuple_indices] += values[tuple_indices] * \
            increase_factor * reward

        sum_values: list = np.sum(values, axis=0)
        normalized_values: list = values / sum_values

        cpd.values = normalized_values
        return cpd

    def train(self, iterations: int, style: str) -> None:
        if style == "SA":
            return self.train_SA(iterations)
        if style == "ME":
            return self.train_ME(iterations)

    def train_SA(self, iterations: int) -> None:
        """
        Train the agent for a specified number of iterations using Simulated Annealing

        Parameters
        ----------
        iterations : int
            Number of iterations to train the agent for.
        """
        while iterations > 0:
            utility_to_adjust: set[str] = set()
            if len(self.memory) > 0:
                for var, ema in self.ema.items():
                    new_ema = (1 - self.ema_alpha) * (
                        ema + self.ema_alpha *
                        self.memory[-1].average_reward[var]
                    )
                    self.ema[var] = new_ema
                    if new_ema < self.threshold:
                        utility_to_adjust.add(var)

            tweak_var: str = random.choice(
                list(utility_to_adjust | self.reflective_vars)
            )
            if tweak_var in utility_to_adjust:
                adjusted_weights: dict[str,
                                       float] = copy.deepcopy(self.weights)
                adjusted_weights[tweak_var] *= self.downweigh_factor
                adjusted_weights = normalize(adjusted_weights)

                normal_time_step: TimeStep = self.time_step(
                    self.fixed_assignment, self.weights, self.sample_num
                )
                weight_adjusted_time_step: TimeStep = self.time_step(
                    self.fixed_assignment, adjusted_weights, self.sample_num
                )

                delta: float = sum(
                    weight_adjusted_time_step.average_reward.values()
                ) - sum(normal_time_step.average_reward.values())

                if delta >= 0 or random.random() <= np.exp(
                    -abs(delta) / self.temperature
                ):
                    self.memory.append(weight_adjusted_time_step)
                    self.weights = adjusted_weights
            else:
                true_reward: float = 0
                random_assignment: int = random.randint(
                    0, self.card_dict[tweak_var] - 1
                )

                normal_time_step = self.time_step(
                    self.fixed_assignment, self.weights, self.sample_num
                )
                interventional_time_step: TimeStep = self.time_step(
                    self.fixed_assignment,
                    self.weights,
                    self.sample_num,
                    {tweak_var: random_assignment},
                )
                delta = sum(interventional_time_step.average_reward.values()) - sum(
                    normal_time_step.average_reward.values()
                )

                if delta >= 0 or random.random() <= np.exp(
                    -abs(delta) / self.temperature
                ):
                    self.memory.append(interventional_time_step)
                    filtered_df: pd.DataFrame = interventional_time_step.memory
                    for var, value in interventional_time_step.average_sample.items():
                        filtered_df = filtered_df[filtered_df[var] == value]
                    for reward_signal in self.get_utilities_from_reflective(tweak_var):
                        true_reward += filtered_df[reward_signal].sum()
                    adjusted_cpt = self.nudge_cpt(
                        self.sampling_model.get_cpds(tweak_var),
                        interventional_time_step.average_sample,
                        self.alpha,
                        true_reward,
                    )
                    self.sampling_model.remove_cpds(
                        self.sampling_model.get_cpds(tweak_var)
                    )
                    self.sampling_model.add_cpds(adjusted_cpt)

                self.temperature *= 0.99
                iterations -= 1

    def train_ME(self, iterations: int) -> None:
        """
        Trains the agent for a specified number of iterations exploiting at every step using interventions.

        Parameters
        ----------
        iterations : int
            Number of iterations to train the agent for
        """
        while iterations > 0:
            utility_to_adjust: set[str] = set()
            time_steps: set[TimeStep] = set()

            if len(self.memory) > 0:
                for var, ema in self.ema.items():
                    new_ema = (1 - self.ema_alpha) * (
                        ema + self.ema_alpha *
                        self.memory[-1].average_reward[var]
                    )
                    self.ema[var] = new_ema
                    if new_ema < self.threshold:
                        utility_to_adjust.add(var)
            time_steps.add(self.time_step(
                self.fixed_assignment, self.weights, self.sample_num))
            for tweak_var in utility_to_adjust:
                adjusted_weights: dict[str,
                                       float] = copy.deepcopy(self.weights)
                adjusted_weights[tweak_var] *= self.downweigh_factor
                adjusted_weights = normalize(adjusted_weights)

                time_steps.add(self.time_step(
                    self.fixed_assignment, adjusted_weights, self.sample_num
                ))

            for var, card in self.card_dict.items():
                for val in range(card):
                    time_steps.add(self.time_step(
                        fixed_evidence=self.fixed_assignment,
                        weights=self.weights,
                        samples=self.sample_num,
                        do={var: val},
                    ))

            best_timestep: TimeStep = max(
                time_steps, key=lambda time_step: sum(time_step.average_reward.values()))

            if best_timestep.weights != self.weights:
                self.memory.append(best_timestep)
                self.weights = best_timestep.weights
            else:
                true_reward: float = 0
                self.memory.append(best_timestep)
                filtered_df: pd.DataFrame = best_timestep.memory
                for var, value in best_timestep.average_sample.items():
                    filtered_df = filtered_df[filtered_df[var] == value]
                for reward_signal in self.get_utilities_from_reflective(best_timestep.tweak_var):
                    true_reward += filtered_df[reward_signal].sum()
                adjusted_cpt = self.nudge_cpt(
                    self.sampling_model.get_cpds(best_timestep.tweak_var),
                    best_timestep.average_sample,
                    self.alpha,
                    true_reward,
                )
                self.sampling_model.remove_cpds(
                    self.sampling_model.get_cpds(best_timestep.tweak_var)
                )
                self.sampling_model.add_cpds(adjusted_cpt)

            iterations -= 1

    def get_utilities_from_reflective(self, reflective_var: str) -> list[str]:
        """
        Get all utility variables that are d-connected to a reflective variable.

        Parameters
        ----------
        reflective_var : str
            Reflective variable to find dependent utility variables for.

        Returns
        -------
        list[str]
            Dependent utility variables.
        """
        dependent_utility_vars: list[str] = []
        for utility in self.utility_vars:
            if self.structural_model.is_dconnected(reflective_var, utility):
                dependent_utility_vars.append(utility)
        return dependent_utility_vars

    def cpd_to_dataframe(self, cpd: TabularCPD) -> pd.DataFrame:
        """
        Convert a CPD to a DataFrame.

        Parameters
        ----------
        cpd : TabularCPD
            The CPD to convert.

        Returns
        -------
        pd.DataFrame
            The CPD as a DataFrame.
        """
        levels: list[range] = [range(card) for card in cpd.cardinality]
        index: pd.MultiIndex = pd.MultiIndex.from_product(
            levels, names=cpd.variables)

        df: pd.DataFrame = pd.DataFrame(
            cpd.values.flatten(), index=index, columns=["Delta"]
        )
        return df

    def compute_cpd_delta(self, var: str) -> DiscreteFactor:
        """
        Compute the delta of a CPD compared to the original model.

        Parameters
        ----------
        var : str
            Variable to compute the delta for.

        Returns
        -------
        DiscreteFactor
            The CPD with the delta values.
        """
        var_cpd: TabularCPD = self.sampling_model.get_cpds(var)
        delta_values: np.ndarray = (
            var_cpd.values - self.original_model.get_cpds(var).values
        )
        delta_cpd: DiscreteFactor = DiscreteFactor(
            variables=var_cpd.variables,
            cardinality=var_cpd.cardinality,
            values=delta_values,
        )
        return delta_cpd

    def write_cpds_to_csv(self, cpds: DiscreteFactor, name: str, folder: str) -> None:
        """
        Write CPDs to a CSV file.

        Parameters
        ----------
        cpds : DiscreteFactor
            CPDs to write to a CSV file
        name : str
            Name of the file.
        folder : str
            Folder to save the file in.
        """
        for cpd in cpds:
            if cpd.variable not in self.reflective_vars:
                continue
            df: pd.DataFrame = self.cpd_to_dataframe(cpd)
            file_name: str = f"{name}, {cpd.variable}.csv"
            df.to_csv(f"{folder}/{file_name}")

    def write_delta_cpd_to_csv(
        self, cpds: DiscreteFactor, name: str, folder: str
    ) -> None:
        """
        Write the delta of CPDs to a CSV file.

        Parameters
        ----------
        cpds : DiscreteFactor
            CPDs to write to a CSV file
        name : str
            Name of the file.
        folder : str
            The folder to save the file in.
        """
        for cpd in cpds:
            if cpd.variable not in self.reflective_vars:
                continue
            df: pd.DataFrame = self.cpd_to_dataframe(
                self.compute_cpd_delta(cpd.variable)
            )
            file_name: str = f"Delta for {name}, {cpd.variable}.csv"
            df.to_csv(f"{folder}/{file_name}")


class TimeStep:
    def __init__(
        self, memory: pd.DataFrame, sample_vars: set[str], reward_vars: set[str], weights: dict[str, float], tweak_var: str
    ):
        """
        Custom Timestep class that holds memory, average sample values,
        and average reward values for that timestep.

        Parameters
        ----------
        memory : pd.DataFrame
            All samples and rewards for that timestep.
        sample_vars : set[str]
            Sample variables.
        reward_vars : set[str]
            Utility/reward variables.
        weights : dict[str, float]
            Weights associated with rewards
        """
        self.tweak_var = tweak_var
        self.memory: pd.DataFrame = memory
        self.average_sample: dict[str, int] = {}
        self.average_reward: dict[str, float] = {}
        self.weights = weights

        for var in sample_vars:
            self.average_sample[var] = self.get_rounded_column_average(var)

        for var in reward_vars:
            self.average_reward[var] = self.get_column_average(var)

    def get_rounded_column_average(self, column_name: str) -> int:
        """
        Calculate and round the average of a specified column in the memory DataFrame.

        Parameters
        ----------
        column_name : str
            The name of the column to calculate the average for.
        decimal_places : int, optional
            The number of decimal places to round the average to, by default 2.

        Returns
        -------
        float
            The rounded average value of the specified column.
        """
        average = self.memory[column_name].mean()
        return round(average)

    def get_column_average(self, column_name: str) -> float:
        """
        Calculate and round the average of a specified column in the memory DataFrame.

        Parameters
        ----------
        column_name : str
            The name of the column to calculate the average for.

        Returns
        -------
        float
            The average value of the specified column.
        """
        average = self.memory[column_name].mean()
        return average

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TimeStep):
            return self.memory.all().all() == other.memory.all().all() and self.average_sample == other.average_sample and self.average_reward == other.average_reward
        return False

    def __hash__(self):
        # Hash sample_vars, reward_vars as sorted tuples to maintain order consistency
        sample_vars_hash = hash(tuple(sorted(self.average_sample.keys())))
        reward_vars_hash = hash(tuple(sorted(self.average_reward.keys())))
        weights_hash = hash(tuple(sorted(self.weights.items())))
        tweak_var_hash = hash(self.tweak_var)

        # Combine the hashes using XOR or a tuple for a unique combination
        return hash((sample_vars_hash, reward_vars_hash, weights_hash, tweak_var_hash))


#! future research
# * look into bayesian structure learning from data
# *
def reward(sample: dict[str, int]) -> dict[str, float]:
    rewards: dict[str, float] = Counter()
    rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
    rewards["social"] += sample["ECs"]
    rewards["health"] += sample["Sleep"] + sample["Exercise"]
    return rewards


list_of_cpts = [
    TabularCPD(variable="SES", variable_card=3,
               values=[[0.29], [0.52], [0.19]]),
    TabularCPD(variable="Motivation", variable_card=2, values=[[0.5], [0.5]]),
    TabularCPD(
        variable="Tutoring",
        variable_card=2,
        values=[[0.9, 0.6, 0.2], [0.1, 0.4, 0.8]],
        evidence=["SES"],
        evidence_card=[3],
    ),
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
    ),
    TabularCPD(
        variable="Exercise",
        variable_card=2,
        values=[[0.6, 0.4, 0.2], [0.4, 0.6, 0.8]],
        evidence=["SES"],
        evidence_card=[3],
    ),
    TabularCPD(
        variable="ECs",
        variable_card=2,
        values=[[0.7, 0.5, 0.1], [0.3, 0.5, 0.9]],
        evidence=["SES"],
        evidence_card=[3],
    ),
    TabularCPD(
        variable="Sleep",
        variable_card=3,
        values=[[0.1, 0.25, 0.55], [0.3, 0.4, 0.25], [0.6, 0.35, 0.2]],
        evidence=["Time studying"],
        evidence_card=[3],
    ),
]

x = CausalLearningAgent(
    sampling_edges=[
        ("SES", "Tutoring"),
        ("SES", "ECs"),
        ("SES", "Time studying"),
        ("Motivation", "Time studying"),
        ("SES", "Exercise"),
        ("ECs", "Time studying"),
        ("Time studying", "Sleep"),
    ],
    utility_edges=[
        ("Time studying", "grades"),
        ("Tutoring", "grades"),
        ("ECs", "social"),
        ("Sleep", "health"),
        ("Exercise", "health"),
    ],
    cpts=list_of_cpts,
    utility_vars={"grades", "social", "health"},
    reflective_vars={"Time studying", "Exercise", "Sleep", "ECs"},
    chance_vars={"Tutoring", "Motivation", "SES"},
    glue_vars=set(),
    reward_func=reward,
    fixed_evidence={"SES": 1},
)

y = CausalLearningAgent(
    sampling_edges=[
        ("SES", "Tutoring"),
        ("SES", "ECs"),
        ("SES", "Time studying"),
        ("Motivation", "Time studying"),
        ("SES", "Exercise"),
        ("ECs", "Time studying"),
        ("Time studying", "Sleep"),
    ],
    utility_edges=[
        ("Time studying", "grades"),
        ("Tutoring", "grades"),
        ("ECs", "social"),
        ("Sleep", "health"),
        ("Exercise", "health"),
    ],
    cpts=list_of_cpts,
    utility_vars={"grades", "social", "health"},
    reflective_vars={"Time studying", "Exercise", "Sleep", "ECs"},
    chance_vars={"Tutoring", "Motivation", "SES"},
    glue_vars=set(),
    reward_func=reward,
    fixed_evidence={"SES": 0},
)

x.train(10, "ME")
y.train(10, "SA")
for var in x.reflective_vars:
    x.display_original_var_cpt(var)
    print('=====================')
    x.display_var_cpt(var)

x.plot_memory()
x.plot_memory_against(y)
