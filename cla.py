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
from types import UnionType
from cla_neural_network import CLANeuralNetwork as CLANN
from itertools import product

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
        ema_utility: float = 1.0,
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
        self.sampling_model.add_nodes_from(chance_vars | reflective_vars | glue_vars)
        self.utility_edges: list[tuple[str, str]] = utility_edges
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

        self.reward_attr_model = copy.deepcopy(self.structural_model)
        self.reward_attr_model.do(self.reflective_vars, True)

        self.sample_num: int = sample_num
        self.alpha: float = alpha
        self.ema_alpha: float = ema_alpha
        self.threshold: float = threshold
        self.temperature: float = temperature
        self.downweigh_factor: float = downweigh_factor
        self.ema: dict[str, float] = {key: ema_utility for key in self.utility_vars}
        if not weights:
            self.weights: dict[str, float] = Counter(
                {key: 1 / len(self.utility_vars) for key in self.utility_vars}
            )
        self.reward_func = reward_func
        self.cum_memory: pd.DataFrame
        self.u_hat_models: dict[str, CLANN] = {}

        for utility in self.utility_vars:
            self.u_hat_models[utility] = CLANN(
                self.structural_model.get_parents(utility), [utility]
            )
        self.inference = CausalInference(self.sampling_model)

        for cpt in cpts:
            self.sampling_model.add_cpds(cpt)

        self.card_dict: dict[str, int] = {
            key: self.sampling_model.get_cardinality(key)
            for key in (self.reflective_vars | self.chance_vars)
        }

        self.original_model: BayesianNetwork = copy.deepcopy(self.sampling_model)

        self.parameters: dict[str, UnionType[float, int]] = {
            "sample_num": self.sample_num,
            "alpha": self.alpha,
            "ema_alpha": self.ema_alpha,
            "threshold": self.threshold,
            "temperature": self.temperature,
            "downweigh_factor": self.downweigh_factor,
            "ema": self.ema,
        }

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
        for normal_time_step in self.memory:
            print(normal_time_step)
            print("\n")

    # def display_weights(self) -> None:
    #     """
    #     Displays the weights of the agent.
    #     """
    #     for time_step, weights in self.weight_history.items():
    #         print(f"Iteration {time_step}:")
    #         print(weights)
    #         print("\n")
    #
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

        # Adjust bottom space: less space between parameters, enough for plot
        num_params = len(self.parameters)
        # Adjust to ensure enough space for the plot and params
        bottom_margin = 0.15 + 0.04 * num_params

        plt.subplots_adjust(bottom=bottom_margin)

        # Dynamically place text under the graph using figure coordinates
        for i, (param, param_val) in enumerate(self.parameters.items()):
            # If the parameter is a dictionary, we format it for readability
            if isinstance(param_val, dict):
                param_val_str = ", ".join([f"{k}: {v}" for k, v in param_val.items()])
            else:
                param_val_str = str(param_val)

            # Reduced vertical space between parameters
            plt.figtext(
                0.5,
                bottom_margin - (0.04 * i) - 0.1,
                f"{param}: {param_val_str}",
                ha="center",
                va="top",
                wrap=True,
                fontsize=10,
                transform=plt.gcf().transFigure,
            )

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

    def plot_weights(self) -> None:
        """
        Plots the weights at each iteration.
        """
        plot_data: dict[str, list[float]] = {}

        for time_step in self.memory:
            for key, value in time_step.weights.items():
                if key not in plot_data:
                    plot_data[key] = []
                plot_data[key].append(value)

        for key, values in plot_data.items():
            plt.plot(values, label=key)

        plt.xlabel("Iteration")
        plt.ylabel("Weight")
        plt.title("Weight at iteration t")
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
        do : dict[str, int]
            any do-assignments

        Returns
        -------
        dict[str, float]
            expected reward given sample & structure
        """
        inference: CausalInference = CausalInference(self.sampling_model)
        rewards_queries: dict[str, list[str]] = {}
        reward_probs: dict[str, DiscreteFactor] = {}
        expected_rewards: dict[str, float] = Counter()

        for category in rewards.keys():
            rewards_queries[category] = self.structural_model.get_parents(category)

        for category, connected_vars in rewards_queries.items():
            reward_probs[category] = inference.query(
                variables=connected_vars,
                evidence={
                    var: ass
                    for var, ass in sample.items()
                    if var not in (connected_vars)
                },
                show_progress=False,
            )

        for category, reward in rewards.items():
            assignment = {var: sample[var] for var in rewards_queries[category]}
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
                    n_samples=1, evidence=fixed_evidence, do=do, show_progress=False
                )
            else:
                sample_df = self.sampling_model.simulate(
                    n_samples=1, evidence=fixed_evidence, show_progress=False
                )
            sample_dict: dict[str, int] = sample_df.iloc[0].to_dict()
            rewards: dict[str, float] = self.reward_func(
                sample_dict, self.utility_edges
            )
            for var, weight in weights.items():
                weighted_reward[var] = rewards[var] * weight
            combined_dict: pd.DataFrame = pd.DataFrame(
                [{**sample_dict, **weighted_reward}]
            )
            memory = pd.concat([memory, combined_dict], ignore_index=True)
        return TimeStep(
            memory,
            self.non_utility_vars,
            self.utility_vars,
            weights,
            next(iter(do)) if do else None,
        )

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

        values[tuple_indices] += values[tuple_indices] * increase_factor * reward

        sum_values: list = np.sum(values, axis=0)
        normalized_values: list = values / sum_values

        cpd.values = normalized_values
        return cpd

    def nudge_cpt_new(
        self,
        cpd: TabularCPD,
        tweak_dict: dict[str, int],
        increase_factor: float,
        reward: float,
    ) -> TabularCPD:
        """
        Nudges multiple values in CPT.

        Parameters
        ----------
        cpd : TabularCPD
            Structure representing the CPT.
        tweak_dict : dict[str, int]
            Specific reflective variable value to be nudged (all conditions).
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
        tweak_var = next(iter(tweak_dict))
        parents = self.structural_model.get_parents(tweak_var)

        # Convert parents to a tuple before using it as a key
        parents_tuple = tuple(parents)

        conditions: list[dict[str, int]] = [
            {variable: value for variable, value in zip(parents, condition_values)}
            for condition_values in product(
                *[range(self.card_dict[parent]) for parent in parents_tuple]
            )
        ]

        for condition in conditions:
            condition[tweak_var] = tweak_dict[tweak_var]
            indices: list[int] = [condition[variable] for variable in cpd.variables]

            tuple_indices: tuple[int, ...] = tuple(indices)

            values[tuple_indices] += values[tuple_indices] * increase_factor * reward

        # Corrected the indentation to ensure normalization happens after the loop
        sum_values: np.ndarray = np.sum(values, axis=0)
        normalized_values: np.ndarray = values / sum_values

        cpd.values = normalized_values
        return cpd

    def train_SA(self, iterations: int) -> None:
        """
        Train the agent for a specified number of iterations using Simulated Annealing

        Parameters
        ----------
        iterations : int
            Number of iterations to train the agent for.
        """
        while iterations > 0:
            print(iterations)
            utility_to_adjust: set[str] = set()
            if len(self.memory) > 0:
                for var, ema in self.ema.items():
                    new_ema = (1 - self.ema_alpha) * (ema) + (
                        self.ema_alpha * self.memory[-1].average_reward[var]
                    )
                    self.ema[var] = new_ema
                    if new_ema < self.threshold:
                        utility_to_adjust.add(var)

            tweak_var: str = random.choice(
                list(utility_to_adjust | self.reflective_vars)
            )
            normal_time_step: TimeStep = self.time_step(
                self.fixed_assignment, self.weights, self.sample_num
            )
            for util, model in self.u_hat_models.items():
                numeric = normal_time_step.memory.apply(pd.to_numeric)
                model.train(numeric)

            if tweak_var in utility_to_adjust:
                adjusted_weights: dict[str, float] = copy.deepcopy(self.weights)
                adjusted_weights[tweak_var] *= self.downweigh_factor
                adjusted_weights = normalize(adjusted_weights)

                weight_adjusted_time_step: TimeStep = self.time_step(
                    self.fixed_assignment, adjusted_weights, self.sample_num
                )

                delta: float = sum(
                    weight_adjusted_time_step.average_reward.values()
                ) - sum(normal_time_step.average_reward.values())

                if delta >= 0 or random.random() <= np.exp(
                    (-1 * delta) / self.temperature
                ):
                    self.weights = adjusted_weights
            else:
                tweak_val_comparison: list[dict[str, float]] = []
                # random_assignment: int = random.randint(
                #     0, self.card_dict[tweak_var] - 1
                # )

                # interventional_time_step: TimeStep = self.time_step(
                #     self.fixed_assignment,
                #     self.weights,
                #     self.sample_num,
                #     {tweak_var: random_assignment},
                # )

                # Compare every possible value for our random tweak variable
                for tweak_dir in range(self.card_dict[tweak_var]):
                    reward: dict[str, float] = Counter()
                    for utility in self.utility_vars:
                        # Get all possible parent combinations for the utility
                        parent_combinations = list(
                            product(
                                *[
                                    range(self.card_dict[parent])
                                    for parent in self.structural_model.get_parents(
                                        utility
                                    )
                                ]
                            )
                        )
                        for parent_values in parent_combinations:
                            # Create dictionary of parents and their values
                            parent_dict = {
                                parent: value
                                for parent, value in zip(
                                    self.structural_model.get_parents(utility),
                                    parent_values,
                                    strict=True,
                                )
                            }
                            # Tweak var takes precedence over combination of parent values
                            parent_dict[tweak_var] = tweak_dir
                            reward[utility] += (
                                # Get probability of parent values given tweak direction
                                self.inference.query(
                                    variables=[
                                        key
                                        for key in parent_dict.keys()
                                        if key != tweak_var
                                    ],
                                    do={tweak_var: tweak_dir},
                                    adjustment_set=[],
                                    show_progress=False,
                                ).get_value(
                                    **{
                                        key: value
                                        for key, value in parent_dict.items()
                                        if key != tweak_var
                                    }
                                )
                                # Multiply by u hat (predicted utility) given current combo of parent values
                                * self.u_hat_models[utility].predict(
                                    pd.DataFrame([parent_dict])
                                )[0, 0]
                            )
                    # Add expected utility to comparison
                    tweak_val_comparison.append(reward)
                tweak_val = tweak_val_comparison.index(
                    max(tweak_val_comparison, key=lambda x: sum(x.values()))
                )

                interventional_reward = sum(tweak_val_comparison[tweak_val].values())

                normal_rewards = self.calculate_expected_reward(
                    normal_time_step.average_sample, normal_time_step.average_reward
                )
                delta = interventional_reward - sum(normal_rewards.values())

                if delta >= 0 or random.random() <= np.exp(
                    (-1 * delta) / self.temperature
                ):
                    # filtered_df: pd.DataFrame = copy.deepcopy(
                    #     interventional_time_step.memory
                    # )
                    # for var, value in interventional_time_step.average_sample.items():
                    #     filtered_df = filtered_df[filtered_df[var] == value]
                    # for reward_signal in self.get_utilities_from_reflective(tweak_var):
                    #     true_reward += filtered_df[reward_signal].sum()

                    # adjusted_cpt = self.nudge_cpt(
                    #     self.sampling_model.get_cpds(tweak_var),
                    #     interventional_time_step.average_sample,
                    #     self.alpha,
                    #     interventional_reward,
                    # )
                    adjusted_cpt = self.nudge_cpt_new(
                        self.sampling_model.get_cpds(tweak_var),
                        {tweak_var: tweak_val},
                        self.alpha,
                        interventional_reward,
                    )
                    self.sampling_model.remove_cpds(
                        self.sampling_model.get_cpds(tweak_var)
                    )
                    self.sampling_model.add_cpds(adjusted_cpt)

            self.memory.append(normal_time_step)
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
                        ema + self.ema_alpha * self.memory[-1][1].average_reward[var]
                    )
                    self.ema[var] = new_ema
                    if new_ema < self.threshold:
                        utility_to_adjust.add(var)

            for tweak_var in utility_to_adjust:
                adjusted_weights: dict[str, float] = copy.deepcopy(self.weights)
                adjusted_weights[tweak_var] *= self.downweigh_factor
                adjusted_weights = normalize(adjusted_weights)

                time_steps.add(
                    self.time_step(
                        self.fixed_assignment, adjusted_weights, self.sample_num
                    )
                )

            for var, card in self.card_dict.items():
                for val in range(card):
                    time_steps.add(
                        self.time_step(
                            fixed_evidence=self.fixed_assignment,
                            weights=self.weights,
                            samples=self.sample_num,
                            do={var: val},
                        )
                    )

            best_timestep: TimeStep = max(
                time_steps, key=lambda time_step: sum(time_step.average_reward.values())
            )

            if best_timestep.weights != self.weights:
                self.memory.append((best_timestep, best_timestep))
                self.weights = best_timestep.weights
            else:
                true_reward: float = 0
                filtered_df: pd.DataFrame = best_timestep.memory
                for var, value in best_timestep.average_sample.items():
                    filtered_df = filtered_df[filtered_df[var] == value]
                for reward_signal in self.get_utilities_from_reflective(
                    best_timestep.tweak_var
                ):
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
                self.memory.append(
                    (
                        best_timestep,
                        self.time_step(
                            self.fixed_assignment, self.weights, self.sample_num
                        ),
                    )
                )

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
            if self.reward_attr_model.is_dconnected(reflective_var, utility):
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
        index: pd.MultiIndex = pd.MultiIndex.from_product(levels, names=cpd.variables)

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
        self,
        memory: pd.DataFrame,
        sample_vars: set[str],
        reward_vars: set[str],
        weights: dict[str, float],
        tweak_var: str,
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
            return (
                self.memory.all().all() == other.memory.all().all()
                and self.average_sample == other.average_sample
                and self.average_reward == other.average_reward
            )
        return False

    def __hash__(self):
        sample_vars_hash = hash(tuple(sorted(self.average_sample.keys())))
        reward_vars_hash = hash(tuple(sorted(self.average_reward.keys())))
        weights_hash = hash(tuple(sorted(self.weights.items())))
        tweak_var_hash = hash(self.tweak_var)

        return hash((sample_vars_hash, reward_vars_hash, weights_hash, tweak_var_hash))


def train(
    mc_rep: int, style: str, agents: list[CausalLearningAgent]
) -> list[CausalLearningAgent]:
    """
    Trains n agents for a specified number of monte carlo repetitions using a specified style.

    Parameters
    ----------
    mc_rep : int
        Monte Carlo Repetitions.
    style : str
        Simulated Annealing/Maximum Exploitation.
    agents : list[CausalLearningAgent]
        List of agents to train.
    """

    match style:
        case "SA":
            for agent in agents:
                agent.train_SA(mc_rep)
        case "ME":
            for agent in agents:
                agent.train_ME(mc_rep)
    return agents


def plot_monte_carlo(agents: list[CausalLearningAgent]) -> None:
    """
    Plots the average reward across agents.

    Parameters
    ----------
    agents : list[CausalLearningAgent]
        List of agents to plot.
    """
    average_rewards: list[float] = []
    for iteration in range(len(agents[0].memory)):
        total_reward = 0
        for agent in agents:
            total_reward += sum(agent.memory[iteration].average_reward.values())
        average_rewards.append(total_reward / len(agents))
    plt.plot(average_rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward Across {len(agents)} Agents")
    plt.show()


#! future research
# * look into bayesian structure learning from data
# *
def reward(
    sample: dict[str, int], utility_edges: list[tuple[str, str]]
) -> dict[str, float]:
    rewards: dict[str, float] = Counter()
    rewards["grades"] += sample["Time studying"] + sample["Tutoring"]
    rewards["social"] += sample["ECs"]
    rewards["health"] += sample["Sleep"] + sample["Exercise"]
    return rewards


def default_reward(
    sample: dict[str, int], utility_edges: list[tuple[str, str]]
) -> dict[str, float]:
    rewards: dict[str, float] = Counter()
    for var, utility in utility_edges:
        rewards[utility] += sample[var]
    return rewards


list_of_cpts = [
    TabularCPD(variable="SES", variable_card=3, values=[[0.29], [0.52], [0.19]]),
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

struct_1: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[("refl_2", "refl_1")],
    utility_edges=[("refl_2", "util_2"), ("refl_1", "util_1")],
    cpts=[
        TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_2"],
            evidence_card=[2],
        ),
    ],
    utility_vars={"util_1", "util_2"},
    reflective_vars={"refl_1", "refl_2"},
    chance_vars=set(),
    glue_vars=set(),
    reward_func=default_reward,
    fixed_evidence={},
)

struct_2: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[("chance_1", "refl_1"), ("chance_2", "refl_1")],
    utility_edges=[("refl_1", "util_1")],
    cpts=[
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_2", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
            evidence=["chance_1", "chance_2"],
            evidence_card=[2, 2],
        ),
    ],
    utility_vars={"util_1"},
    reflective_vars={"refl_1"},
    chance_vars={"chance_1", "chance_2"},
    glue_vars=set(),
    reward_func=default_reward,
    fixed_evidence={},
)

struct_3: CausalLearningAgent = CausalLearningAgent(
    sampling_edges=[("refl_1", "chance_1"), ("refl_2", "chance_1")],
    utility_edges=[("chance_1", "util_1")],
    cpts=[
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="chance_1",
            variable_card=2,
            values=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
            evidence=["refl_1", "refl_2"],
            evidence_card=[2, 2],
        ),
    ],
    utility_vars={"util_1"},
    reflective_vars={"refl_1", "refl_2"},
    chance_vars={"chance_1"},
    glue_vars=set(),
    reward_func=default_reward,
    fixed_evidence={},
)

struct_4: CausalLearningAgent = CausalLearningAgent(
    [("chance_1", "refl_1")],
    [("refl_1", "util_1")],
    [
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["chance_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1"},
    set(),
    default_reward,
    {},
)

struct_5: CausalLearningAgent = CausalLearningAgent(
    [("refl_1", "chance_1")],
    [("chance_1", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="chance_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1"},
    set(),
    default_reward,
    {},
)

struct_6 = CausalLearningAgent(
    [("chance_1", "refl_1"), ("chance_1", "refl_2")],
    [("refl_1", "util_1"), ("refl_2", "util_1")],
    [
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["chance_1"],
            evidence_card=[2],
        ),
        TabularCPD(
            variable="refl_2",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["chance_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1", "refl_2"},
    {"chance_1"},
    set(),
    default_reward,
    {},
)

struct_7 = CausalLearningAgent(
    [("refl_1", "chance_1"), ("refl_1", "chance_2")],
    [("chance_1", "util_1"), ("chance_2", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="chance_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_1"],
            evidence_card=[2],
        ),
        TabularCPD(
            variable="chance_2",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1", "chance_2"},
    set(),
    default_reward,
    {},
)

struct_8: CausalLearningAgent = CausalLearningAgent(
    [],
    [("refl_1", "util_1"), ("chance_1", "util_1"), ("chance_2", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_2", variable_card=2, values=[[0.5], [0.5]]),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1", "chance_2"},
    set(),
    default_reward,
    {},
)

struct_9: CausalLearningAgent = CausalLearningAgent(
    [],
    [("refl_1", "util_1"), ("chance_1", "util_1"), ("refl_2", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
    ],
    {"util_1"},
    {"refl_1", "refl_2"},
    {"chance_1"},
    set(),
    default_reward,
    {},
)

struct_10: CausalLearningAgent = CausalLearningAgent(
    [("chance_1", "refl_1"), ("refl_1", "chance_2")],
    [("chance_2", "util_1")],
    [
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(
            variable="refl_1",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["chance_1"],
            evidence_card=[2],
        ),
        TabularCPD(
            variable="chance_2",
            variable_card=2,
            values=[[0.5, 0.5], [0.5, 0.5]],
            evidence=["refl_1"],
            evidence_card=[2],
        ),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1", "chance_2"},
    set(),
    default_reward,
    {},
)

struct_11: CausalLearningAgent = CausalLearningAgent(
    [],
    [("refl_1", "util_1"), ("chance_1", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="chance_1", variable_card=2, values=[[0.5], [0.5]]),
    ],
    {"util_1"},
    {"refl_1"},
    {"chance_1"},
    set(),
    default_reward,
    {},
)

struct_13: CausalLearningAgent = CausalLearningAgent(
    [],
    [("refl_1", "util_1"), ("refl_2", "util_1")],
    [
        TabularCPD(variable="refl_1", variable_card=2, values=[[0.5], [0.5]]),
        TabularCPD(variable="refl_2", variable_card=2, values=[[0.5], [0.5]]),
    ],
    {"util_1"},
    {"refl_1", "refl_2"},
    set(),
    set(),
    default_reward,
    {},
)

# print("hello")

# print("training x now")
# x.train(150, "SA")
# print('training y now')
# y.train(150, "SA")
# print('training s1 now')
# struct_1.train(50, "SA")
# print('training s2 now')
# struct_2.train(50, "SA")
# print('training s3 now')
# struct_3.train(50, "SA")
# print('training s4 now')
# struct_4.train(50, "SA")
# print('training s5 now')
# struct_5.train(50, "SA")
# print('training s6 now')
# struct_6.train(50, "SA")
# print('training s7 now')
# struct_7.train(50, "SA")
# print('training s8 now')
# struct_8.train(50, "SA")
# print('training s9 now')
# struct_9.train(50, "SA")
# print('training s10 now')
# struct_10.train(50, "SA")
# print('training s11 now')
# struct_11.train(50, "SA")
# print('training s13 now')
# struct_13.train(50, "SA")


# x.plot_memory()
# y.plot_memory()
# struct_1.plot_memory()
# struct_2.plot_memory()
# struct_3.plot_memory()
# struct_4.plot_memory()
# struct_5.plot_memory()
# struct_6.plot_memory()
# struct_7.plot_memory()
# struct_8.plot_memory()
# struct_9.plot_memory()
# struct_10.plot_memory()
# struct_11.plot_memory()
# struct_13.plot_memory()

# x.plot_weights()
# y.plot_weights()
# struct_1.plot_weights()
# struct_2.plot_weights()
# struct_3.plot_weights()
# struct_4.plot_weights()
# struct_5.plot_weights()
# struct_6.plot_weights()
# struct_7.plot_weights()
# struct_8.plot_weights()
# struct_9.plot_weights()
# struct_10.plot_weights()
# struct_11.plot_weights()
# struct_13.plot_weights()

# x.plot_memory_against(y)
# struct_2.plot_memory_against(struct_3)
# struct_4.plot_memory_against(struct_5)
# struct_6.plot_memory_against(struct_7)
# struct_8.plot_memory_against(struct_9)
# struct_11.plot_memory_against(struct_13)

# x.write_cpds_to_csv(x.sampling_model.get_cpds(), "x", "data")
# y.write_cpds_to_csv(y.sampling_model.get_cpds(), "y", "data")
# struct_1.write_cpds_to_csv(
#     struct_1.sampling_model.get_cpds(), "struct_1", "data")
# struct_2.write_cpds_to_csv(
#     struct_2.sampling_model.get_cpds(), "struct_2", "data")
# struct_3.write_cpds_to_csv(
#     struct_3.sampling_model.get_cpds(), "struct_3", "data")
# struct_4.write_cpds_to_csv(
#     struct_4.sampling_model.get_cpds(), "struct_4", "data")
# struct_5.write_cpds_to_csv(
#     struct_5.sampling_model.get_cpds(), "struct_5", "data")
# struct_6.write_cpds_to_csv(
#     struct_6.sampling_model.get_cpds(), "struct_6", "data")
# struct_7.write_cpds_to_csv(
#     struct_7.sampling_model.get_cpds(), "struct_7", "data")
# struct_8.write_cpds_to_csv(
#     struct_8.sampling_model.get_cpds(), "struct_8", "data")
# struct_9.write_cpds_to_csv(
#     struct_9.sampling_model.get_cpds(), "struct_9", "data")
# struct_10.write_cpds_to_csv(
#     struct_10.sampling_model.get_cpds(), "struct_10", "data")

# x.write_delta_cpd_to_csv(x.sampling_model.get_cpds(), "x", "data")
# y.write_delta_cpd_to_csv(y.sampling_model.get_cpds(), "y", "data")
# struct_1.write_delta_cpd_to_csv(
#     struct_1.sampling_model.get_cpds(), "struct_1", "data")
# struct_2.write_delta_cpd_to_csv(
#     struct_2.sampling_model.get_cpds(), "struct_2", "data")
# struct_3.write_delta_cpd_to_csv(
#     struct_3.sampling_model.get_cpds(), "struct_3", "data")
# struct_4.write_delta_cpd_to_csv(
#     struct_4.sampling_model.get_cpds(), "struct_4", "data")
# struct_5.write_delta_cpd_to_csv(
#     struct_5.sampling_model.get_cpds(), "struct_5", "data")
# struct_6.write_delta_cpd_to_csv(
#     struct_6.sampling_model.get_cpds(), "struct_6", "data")
# struct_7.write_delta_cpd_to_csv(
#     struct_7.sampling_model.get_cpds(), "struct_7", "data")
# struct_8.write_delta_cpd_to_csv(
#     struct_8.sampling_model.get_cpds(), "struct_8", "data")
# struct_9.write_delta_cpd_to_csv(
#     struct_9.sampling_model.get_cpds(), "struct_9", "data")
# struct_10.write_delta_cpd_to_csv(
#     struct_10.sampling_model.get_cpds(), "struct_10", "data")

# Monte carlo stuff
x_agents = [copy.deepcopy(x) for _ in range(10)]
train(100, "SA", x_agents)


y_agents = [copy.deepcopy(y) for _ in range(10)]
train(100, "SA", y_agents)


struct_1_agents = [copy.deepcopy(struct_1) for _ in range(10)]
train(50, "SA", struct_1_agents)


struct_2_agents = [copy.deepcopy(struct_2) for _ in range(10)]
train(50, "SA", struct_2_agents)


struct_3_agents = [copy.deepcopy(struct_3) for _ in range(10)]
train(50, "SA", struct_3_agents)


struct_4_agents = [copy.deepcopy(struct_4) for _ in range(10)]
train(50, "SA", struct_4_agents)


struct_5_agents = [copy.deepcopy(struct_5) for _ in range(10)]
train(50, "SA", struct_5_agents)


struct_6_agents = [copy.deepcopy(struct_6) for _ in range(10)]
train(50, "SA", struct_6_agents)


struct_7_agents = [copy.deepcopy(struct_7) for _ in range(10)]
train(50, "SA", struct_7_agents)


struct_8_agents = [copy.deepcopy(struct_8) for _ in range(10)]
train(50, "SA", struct_8_agents)


struct_9_agents = [copy.deepcopy(struct_9) for _ in range(10)]
train(50, "SA", struct_9_agents)


struct_10_agents = [copy.deepcopy(struct_10) for _ in range(10)]
train(50, "SA", struct_10_agents)


struct_11_agents = [copy.deepcopy(struct_11) for _ in range(10)]
train(50, "SA", struct_11_agents)


struct_13_agents = [copy.deepcopy(struct_13) for _ in range(10)]
train(50, "SA", struct_13_agents)


print("x done")
plot_monte_carlo(x_agents)
print("y done")
plot_monte_carlo(y_agents)
print("struct_1 done")
plot_monte_carlo(struct_1_agents)
print("struct_2 done")
plot_monte_carlo(struct_2_agents)
print("struct_3 done")
plot_monte_carlo(struct_3_agents)
print("struct_4 done")
plot_monte_carlo(struct_4_agents)
print("struct_5 done")
plot_monte_carlo(struct_5_agents)
print("struct_6 done")
plot_monte_carlo(struct_6_agents)
print("struct_7 done")
plot_monte_carlo(struct_7_agents)
print("struct_8 done")
plot_monte_carlo(struct_8_agents)
print("struct_9 done")
plot_monte_carlo(struct_9_agents)
print("struct_10 done")
plot_monte_carlo(struct_10_agents)
print("struct_11 done")
plot_monte_carlo(struct_11_agents)
print("struct_13 done")
plot_monte_carlo(struct_13_agents)