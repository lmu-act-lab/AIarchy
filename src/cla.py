import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import copy
import random
from src.util import *
from typing import Callable
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from pgmpy.factors.discrete import DiscreteFactor
from src.ModifiedBayesianNetwork import BayesianNetwork  # type: ignore
from pgmpy.inference import CausalInference  # type: ignore
from types import UnionType
from src.cla_neural_network import CLANeuralNetwork as CLANN
from itertools import product
from src.timestep import TimeStep

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
        hidden_vars: list[str] = [],
        weights: dict[str, float] = {},
        temperature: float = 1,
        downweigh_factor: float = 0.975,
        sample_num: int = 7,
        cpt_increase_factor: float = 0.01,
        downweigh_alpha: float = 0.01,
        cooling_factor: float = 0.99,
        u_hat_epochs: int = 1,
        reward_noise: float = 0.0,
        frustration_threshold: float = 0.25,
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
            Frustration value -- threshold for how low EMA of recent rewards for each util needs to be to consider downweighing that utility. (high = makes changes to values faster.)
        temperature : float, optional
            Temperature for simulated annealing, by default 1. The willingness to test new lifestyles (high = open-mind, low = stubborn)
        downweigh_factor : float, optional
            Amount to downweigh weights by, by default 0.8
        sample_num : int, optional
            Number of samples taken for one time step, by default 7. Each sample is one day in the sim-student's school life, at the end of which they get .
        cpt_increase_factor : float, optional
            Learning rate, by default 0.01
        downweigh_alpha : float, optional
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
        self.hidden_vars: list[str] = hidden_vars

        self.structural_model: BayesianNetwork = BayesianNetwork(
            self.sampling_model.edges()
        )
        self.structural_model.add_nodes_from(self.utility_vars)
        self.structural_model.add_edges_from(utility_edges)

        self.reward_attr_model = copy.deepcopy(self.structural_model)
        self.reward_attr_model.do(self.reflective_vars, True)

        self.sample_num: int = sample_num
        self.cpt_increase_factor: float = cpt_increase_factor
        self.downweigh_alpha: float = downweigh_alpha
        self.temperature: float = temperature
        self.cooling_factor: float = cooling_factor
        self.downweigh_factor: float = downweigh_factor
        self.ema: dict[str, float] = {key: float(1.0) for key in self.utility_vars}
        if not weights:
            self.weights: dict[str, float] = Counter(
                {key: 1 / len(self.utility_vars) for key in self.utility_vars}
            )
        else:
            self.weights = weights
        self.objective_weights = Counter(
            {key: 1 / len(self.utility_vars) for key in self.utility_vars}
        )
        self.reward_func = reward_func
        self.reward_noise = reward_noise
        self.cum_memory: pd.DataFrame
        self.u_hat_models: dict[str, CLANN] = {}

        for utility in self.utility_vars:
            network_inputs = list(
                    set(self.structural_model.get_parents(utility))
                    - set(self.hidden_vars)
                )
            if len(network_inputs) == 0:
                self.u_hat_models[utility] = None
            else:
                self.u_hat_models[utility] = CLANN(
                    network_inputs,
                    [utility],
                )
        self.inference = CausalInference(self.sampling_model)

        for cpt in cpts:
            self.sampling_model.add_cpds(cpt)

        self.card_dict: dict[str, int] = {
            key: self.sampling_model.get_cardinality(key)
            for key in (self.reflective_vars | self.chance_vars)
        }
        self.downweigh_threshold = {}
        # for utility in self.utility_vars:
        #     max_reward = sum(
        #         [
        #             (self.sampling_model.get_cardinality(parent) - 1)
        #             for parent in self.structural_model.get_parents(utility)
        #         ]
        #     )
        #     self.downweigh_threshold[utility] = (
        #         max_reward / len(self.utility_vars)
        #     ) / 2
        for utility in self.utility_vars:
            self.downweigh_threshold[utility] = frustration_threshold
        self.original_model: BayesianNetwork = copy.deepcopy(self.sampling_model)

        self.parameters: dict[str, UnionType[float, int]] = {
            "cpt_increase_factor": self.cpt_increase_factor,
            "downweigh_alpha": self.downweigh_alpha,
            "threshold": self.downweigh_threshold,
            "temperature": self.temperature,
            "downweigh_factor": self.downweigh_factor,
            "ema": self.ema,
            "agent_weights" : self.weights,
            "reward_noise" : reward_noise
        }
        self.ema_history: list[dict[str, float]] = []
        self.parent_combinations = self.par_dict()
        self.u_hat_epochs = u_hat_epochs
        self.query_history: dict[
            tuple[BayesianNetwork, dict[str, int]], DiscreteFactor
        ] = {}
        self.lower_tier_pooled_reward = None

    def par_dict(self):
        utility_to_parent_combos = {}
        for utility in self.utility_vars:
            # Get all possible parent combinations for the utility
            parent_combinations = list(
                product(
                    *[
                        range(self.card_dict[parent])
                        for parent in self.structural_model.get_parents(utility)
                    ]
                )
            )
            all_combos = []
            # print(f"parent combos: {parent_combinations}")
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
                all_combos.append(parent_dict)
            utility_to_parent_combos[utility] = all_combos
        return utility_to_parent_combos

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

    def cdn_query(self, query_vars, do_evidence):
        shared_vars = [var for var in query_vars if var in do_evidence.keys()]
        if len(shared_vars) > 0:
            model_copy = self.sampling_model.copy()
            for var in shared_vars:
                cpt = model_copy.get_cpds(node=var)
                for idx, value in enumerate(cpt.values):
                    cpt.values[idx] = 0 if idx != do_evidence[var] else 1
            model_copy.do(nodes=shared_vars)
            inference = CausalInference(model_copy)
            query = inference.query(variables=query_vars, show_progress=False)
            return query

        updated_model = self.sampling_model.do(nodes=list(do_evidence.keys()))
        inference = CausalInference(updated_model)
        query = inference.query(
            variables=query_vars, evidence=do_evidence, show_progress=False
        )
        return query

    # @profile
    def time_step(
        self,
        fixed_evidence: dict[str, int],
        weights: dict[str, float],
        samples: int,
        do: dict[str, int] = {},
    ) -> "TimeStep":
        # Generate all samples at once
        if do:
            sample_df = self.sampling_model.simulate(
                n_samples=samples, evidence=fixed_evidence, do=do, show_progress=False
            )
        else:
            sample_df = self.sampling_model.simulate(
                n_samples=samples, evidence=fixed_evidence, show_progress=False
            )

        # Compute weighted rewards for each row
        def compute_weighted_rewards(row):
            sample_dict = row.to_dict()
            rewards = self.reward_func(sample_dict, self.utility_edges, self.reward_noise, self.lower_tier_pooled_reward)
            # Multiply each reward by its corresponding weight
            weighted_reward = {var: rewards[var] * weights[var] for var in weights}
            return pd.Series(weighted_reward)

        weighted_rewards_df = sample_df.apply(compute_weighted_rewards, axis=1)

        # Combine the original samples with the weighted rewards
        combined_df = pd.concat([sample_df, weighted_rewards_df], axis=1)

        return TimeStep(
            combined_df,
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

        values[tuple_indices] += np.round(
            values[tuple_indices] * increase_factor * reward, decimals=3
        )

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

            values[tuple_indices] += np.round(
                values[tuple_indices] * increase_factor * reward, decimals=3
            )

        # Corrected the indentation to ensure normalization happens after the loop
        sum_values: np.ndarray = np.sum(values, axis=0)
        normalized_values: np.ndarray = np.round(values / sum_values, decimals=4)

        cpd.values = normalized_values
        return cpd

    # @profile
    def train_SA(self, iterations: int) -> None:
        """
        Train the agent for a specified number of iterations using Simulated Annealing

        Parameters
        ----------
        iterations : int
            Number of iterations to train the agent for.
        """
        while iterations > 0:
            self.ema_history.append(copy.deepcopy(self.ema))
            utility_to_adjust: set[str] = set()
            if len(self.memory) > 0:
                for var, ema in self.ema.items():
                    new_ema = (1 - self.downweigh_alpha) * (ema) + (
                        self.downweigh_alpha * self.memory[-1].average_reward[var]
                    )
                    self.ema[var] = new_ema
                    if new_ema < self.downweigh_threshold[var]:
                        utility_to_adjust.add(var)

            tweak_var: str = random.choice(
                list(utility_to_adjust | self.reflective_vars)
            )
            normal_time_step: TimeStep = self.time_step(
                self.fixed_assignment, self.weights, self.sample_num
            )
            for util, model in self.u_hat_models.items():
                if model is None:
                    continue
                hidden_removed = normal_time_step.memory.drop(
                    columns=[col for col in self.hidden_vars]
                )
                numeric = hidden_removed.apply(pd.to_numeric)
                model.train(numeric, epochs=self.u_hat_epochs)

            if tweak_var in utility_to_adjust:
                adjusted_weights: dict[str, float] = Counter(copy.deepcopy(self.weights))
                adjusted_weights[tweak_var] *= self.downweigh_factor
                adjusted_weights = normalize(adjusted_weights)
                weight_adjusted_time_step: TimeStep = self.time_step(
                    self.fixed_assignment, adjusted_weights, self.sample_num
                )

                delta: float = sum(
                    weight_adjusted_time_step.average_reward.values()
                ) - sum(normal_time_step.average_reward.values())

                if delta >= 0 or random.random() <= np.exp(
                    (-1 * np.tanh(delta)) / self.temperature
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
                    # print(f"tweak_var {tweak_var}")
                    # print(f"tweak_dir {tweak_dir}")
                    reward: dict[str, float] = Counter()
                    for utility in self.utility_vars:
                        if self.u_hat_models[utility] is None:
                            reward[utility] = 0.0
                            continue
                        # Get all possible parent combinations for the utility
                        # parent_combinations = list(
                        #     product(
                        #         *[
                        #             range(self.card_dict[parent])
                        #             for parent in self.structural_model.get_parents(
                        #                 utility
                        #             )
                        #         ]
                        #     )
                        # )
                        # print(f"parent combos: {parent_combinations}")
                        # for parent_values in parent_combinations:
                        #     # Create dictionary of parents and their values
                        #     parent_dict = {
                        #         parent: value
                        #         for parent, value in zip(
                        #             self.structural_model.get_parents(utility),
                        #             parent_values,
                        #             strict=True,
                        #         )
                        #     }
                        # print(f"parent_dict: {parent_dict}")
                        # # Tweak var takes precedence over combination of parent values
                        # if tweak_var in parent_dict.keys():
                        #     parent_dict[tweak_var] = tweak_dir
                        # print(f"parent_dict after: {parent_dict}")
                        for parent_dict in self.parent_combinations[utility]:
                            parent_prob = None
                            reward_attribution: float = 0.0
                            query = frozenset(
                                    {
                                        key: value for key, value in parent_dict.items()
                                    }.items()
                                )
                            # if (
                            #     self.sampling_model,
                            #     query,
                            # ) in self.query_history.keys():
                            #     parent_prob = self.query_history[
                            #         (
                            #             self.sampling_model,
                            #             query,
                            #         )
                            #     ]
                            # else:
                            parent_prob = (
                                # Get probability of parent values given tweak direction
                                self.cdn_query(
                                    [key for key in parent_dict.keys()],
                                    {tweak_var: tweak_dir},
                                ).get_value(
                                    **{
                                        key: value
                                        for key, value in parent_dict.items()
                                    }
                                )
                            )
                            # Multiply by u hat (predicted utility) given current combo of parent values
                            reward_attribution = (
                                parent_prob
                                * self.u_hat_models[utility].predict(
                                    pd.DataFrame([parent_dict])
                                )[0, 0]
                            )
                            reward[utility] += reward_attribution
                            # self.query_history[
                            #     (
                            #         copy.deepcopy(self.sampling_model),
                            #         query,
                            #     )
                            # ] = parent_prob
                    # Add expected utility to comparison
                    tweak_val_comparison.append(reward)
                for rewards in tweak_val_comparison:
                    for utility in rewards.keys():
                        rewards[utility] *= self.weights[utility]
                tweak_val = tweak_val_comparison.index(
                    max(tweak_val_comparison, key=lambda x: sum(x.values()))
                )

                total_rewards_per_dir = [sum(d.values()) for d in tweak_val_comparison]
                highest_reward = max(total_rewards_per_dir)

                # Gather all tied indices
                candidates = [i for i, s in enumerate(total_rewards_per_dir) if s == highest_reward]

                # Pick one at random
                tweak_val = random.choice(candidates)

                interventional_reward = sum(tweak_val_comparison[tweak_val].values())
                # normal_rewards = self.calculate_expected_reward(
                #     normal_time_step.average_sample, normal_time_step.average_reward
                # )
                normal_rewards = normal_time_step.average_reward

                delta = interventional_reward - sum(normal_rewards.values())
                if delta >= 0 or random.random() <= np.exp(
                    (-1 * np.tanh(delta)) / self.temperature
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
                    #     self.cpt_increase_factor,
                    #     interventional_reward,
                    # )
                    # pass
                    adjusted_cpt = self.nudge_cpt_new(
                        self.sampling_model.get_cpds(tweak_var),
                        {tweak_var: tweak_val},
                        self.cpt_increase_factor,
                        interventional_reward,
                    )
                    self.sampling_model.remove_cpds(
                        self.sampling_model.get_cpds(tweak_var)
                    )
                    self.sampling_model.add_cpds(adjusted_cpt)

            self.memory.append(normal_time_step)
            self.temperature *= self.cooling_factor
            iterations -= 1

    # def train_ME(self, iterations: int) -> None:
    #     """
    #     Trains the agent for a specified number of iterations exploiting at every step using interventions.

    #     Parameters
    #     ----------
    #     iterations : int
    #         Number of iterations to train the agent for
    #     """
    #     while iterations > 0:
    #         utility_to_adjust: set[str] = set()
    #         time_steps: set[TimeStep] = set()

    #         if len(self.memory) > 0:
    #             for var, ema in self.ema.items():
    #                 new_ema = (1 - self.downweigh_alpha) * (
    #                     ema + self.downweigh_alpha * self.memory[-1][1].average_reward[var]
    #                 )
    #                 self.ema[var] = new_ema
    #                 if new_ema < self.threshold:
    #                     utility_to_adjust.add(var)

    #         for tweak_var in utility_to_adjust:
    #             adjusted_weights: dict[str, float] = copy.deepcopy(self.weights)
    #             adjusted_weights[tweak_var] *= self.downweigh_factor
    #             adjusted_weights = normalize(adjusted_weights)

    #             time_steps.add(
    #                 self.time_step(
    #                     self.fixed_assignment, adjusted_weights, self.sample_num
    #                 )
    #             )

    #         for var, card in self.card_dict.items():
    #             for val in range(card):
    #                 time_steps.add(
    #                     self.time_step(
    #                         fixed_evidence=self.fixed_assignment,
    #                         weights=self.weights,
    #                         samples=self.sample_num,
    #                         do={var: val},
    #                     )
    #                 )

    #         best_timestep: TimeStep = max(
    #             time_steps, key=lambda time_step: sum(time_step.average_reward.values())
    #         )

    #         if best_timestep.weights != self.weights:
    #             self.memory.append((best_timestep, best_timestep))
    #             self.weights = best_timestep.weights
    #         else:
    #             true_reward: float = 0
    #             filtered_df: pd.DataFrame = best_timestep.memory
    #             for var, value in best_timestep.average_sample.items():
    #                 filtered_df = filtered_df[filtered_df[var] == value]
    #             for reward_signal in self.get_utilities_from_reflective(
    #                 best_timestep.tweak_var
    #             ):
    #                 true_reward += filtered_df[reward_signal].sum()
    #             adjusted_cpt = self.nudge_cpt(
    #                 self.sampling_model.get_cpds(best_timestep.tweak_var),
    #                 best_timestep.average_sample,
    #                 self.cpt_increase_factor,
    #                 true_reward,
    #             )
    #             self.sampling_model.remove_cpds(
    #                 self.sampling_model.get_cpds(best_timestep.tweak_var)
    #             )
    #             self.sampling_model.add_cpds(adjusted_cpt)
    #             self.memory.append(
    #                 (
    #                     best_timestep,
    #                     self.time_step(
    #                         self.fixed_assignment, self.weights, self.sample_num
    #                     ),
    #                 )
    #             )

    #         iterations -= 1

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
