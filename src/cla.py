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
from src.ModifiedCausalInference import FastCausalInference  # type: ignore
from pgmpy.inference import CausalInference  # type: ignore
from types import UnionType
from src.cla_neural_network import CLANeuralNetwork as CLANN
from itertools import product
from src.timestep import TimeStep

warnings.filterwarnings("ignore")


class CausalLearningAgent:
    # Class-level shared cache accessible to all Monte Carlo agents
    # Key: (query_vars, do_evidence, model_hash) -> DiscreteFactor
    _shared_cdn_cache: dict[
        tuple[tuple[str, ...], tuple[tuple[str, int], ...], int], DiscreteFactor
    ] = {}

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
        self.u_hat_models: dict[str, CLANN | None] = {}
        # Cache for cdn_query to avoid repeated inference on identical queries
        # Key: (query_vars, do_evidence, model_hash) -> DiscreteFactor
        self._cdn_cache: dict[
            tuple[tuple[str, ...], tuple[tuple[str, int], ...], int], DiscreteFactor
        ] = {}
        # Cache model hash to avoid recomputing on every query
        self._model_hash: int | None = None

        for utility in self.utility_vars:
            # Use sorted list to ensure deterministic ordering that matches par_dict()
            parents = sorted(self.structural_model.get_parents(utility))
            network_inputs = [p for p in parents if p not in self.hidden_vars]
            if len(network_inputs) == 0:
                self.u_hat_models[utility] = None
            else:
                self.u_hat_models[utility] = CLANN(
                    network_inputs,
                    [utility],
                )
        # Model will be validated after add_cpds() in the loop below
        # Initialize inference after all CPDs are added
        self.inference = None  # Will be set after add_cpds validates

        # Add all CPDs without validating after each (validate=False)
        for cpt in cpts:
            self.sampling_model.add_cpds(cpt, validate=False)
        # Validate once after all CPDs are added, but skip utility variables
        # Utility variables don't have CPDs since they're computed from reward functions
        self.sampling_model.check_model_without_utility_vars(utility_vars=self.utility_vars)
        # All CPDs validated - now create inference
        self.inference = FastCausalInference(self.sampling_model)

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

    def update_u_hat_models_for_pooling(self) -> None:
        """
        Update u_hat models to include pooling variables as context when available.
        This should be called after lower_tier_pooled_reward is set.
        """
        if (
            not hasattr(self, "lower_tier_pooled_reward")
            or not self.lower_tier_pooled_reward
        ):
            return

        for utility in self.utility_vars:
            if (
                utility in self.lower_tier_pooled_reward
                and utility not in self.hidden_vars
            ):
                # Recreate the model with pooling variable as input
                network_inputs = list(
                    set(self.structural_model.get_parents(utility))
                    - set(self.hidden_vars)
                )
                network_inputs.append(utility)  # Add the pooling variable itself

                self.u_hat_models[utility] = CLANN(
                    network_inputs,
                    [utility],
                )

    def par_dict(self):
        """
        Build a mapping from each utility variable to all combinations of its parents' values.

        Returns
        -------
        dict[str, list[dict[str, int]]]
            For each utility, a list of dictionaries where each dictionary assigns a value
            to every parent of that utility.
        """
        utility_to_parent_combos = {}
        for utility in self.utility_vars:
            # Get parents in sorted order to ensure deterministic ordering
            parents = sorted(self.structural_model.get_parents(utility))
            # Get all possible parent combinations for the utility
            parent_combinations = list(
                product(
                    *[range(self.card_dict[parent]) for parent in parents]
                )
            )
            all_combos = []
            for parent_values in parent_combinations:
                # Create dictionary of parents and their values
                parent_dict = {
                    parent: value
                    for parent, value in zip(
                        parents,
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
        inference: CausalInference = self.inference
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

    def cdn_query(self, query_vars: list[str], do_evidence: dict[str, int]):
        """
        Query the sampling model under a do-intervention, handling any overlap between
        queried variables and intervention variables.

        Parameters
        ----------
        query_vars : list[str]
            Variables to query.
        do_evidence : dict[str, int]
            Interventional assignments applied via do().

        Returns
        -------
        DiscreteFactor
            Factor over the queried variables under the intervention.
        """
        # Use cached hash if available, otherwise compute and cache it
        if self._model_hash is None:
            self._model_hash = hash(self.sampling_model)
        
        # Create cache key that includes the model state
        cache_key = (
            tuple(sorted(query_vars)),
            tuple(sorted(do_evidence.items())),
            self._model_hash  # Use cached hash
        )
        
        # Check instance cache first
        if cache_key in self._cdn_cache:
            return self._cdn_cache[cache_key]
        
        # Check shared cache if not found in instance cache
        if cache_key in self._shared_cdn_cache:
            # Copy to instance cache for faster future access
            self._cdn_cache[cache_key] = self._shared_cdn_cache[cache_key]
            return self._cdn_cache[cache_key]

        shared_vars = [var for var in query_vars if var in do_evidence.keys()]
        if len(shared_vars) > 0:
            model_copy = self.sampling_model.copy()
            for var in shared_vars:
                cpt = model_copy.get_cpds(node=var)
                for idx, value in enumerate(cpt.values):
                    cpt.values[idx] = 0 if idx != do_evidence[var] else 1
            model_copy.do(nodes=shared_vars, inplace=True)
            inference = FastCausalInference(model_copy)
            query = inference.query(variables=query_vars, show_progress=False)
            # Store in both instance and shared caches
            self._cdn_cache[cache_key] = query
            self._shared_cdn_cache[cache_key] = query
            return query

        updated_model = self.sampling_model.do(nodes=list(do_evidence.keys()))
        inference = FastCausalInference(updated_model)
        query = inference.query(
            variables=query_vars, evidence=do_evidence, show_progress=False
        )
        # Store in both instance and shared caches
        self._cdn_cache[cache_key] = query
        self._shared_cdn_cache[cache_key] = query
        return query

    # @profile
    def time_step(
        self,
        fixed_evidence: dict[str, int],
        weights: dict[str, float],
        samples: int,
        do: dict[str, int] = {},
    ) -> "TimeStep":
        """
        Simulate one time step by generating samples, computing per-utility rewards,
        and returning a summary `TimeStep`.

        Parameters
        ----------
        fixed_evidence : dict[str, int]
            Evidence to condition the simulation on.
        weights : dict[str, float]
            Current utility weights used for weighted rewards.
        samples : int
            Number of samples to simulate.
        do : dict[str, int], optional
            Optional do-intervention assignments for reflective variables, by default {}.

        Returns
        -------
        TimeStep
            Record of samples, weighted rewards, and summary statistics.
        """
        # Generate all samples at once
        if do:
            sample_df = self.sampling_model.simulate(
                n_samples=samples, evidence=fixed_evidence, do=do, show_progress=False
            )
        else:
            sample_df = self.sampling_model.simulate(
                n_samples=samples, evidence=fixed_evidence, show_progress=False
            )

        # Vectorized reward computation - much faster than pandas apply()
        sample_dicts = sample_df.to_dict('records')
        all_rewards = [
            self.reward_func(sample_dict, self.utility_edges, self.reward_noise, self.lower_tier_pooled_reward)
            for sample_dict in sample_dicts
        ]
        
        # Create weighted rewards DataFrame efficiently
        weighted_rewards_data = {
            var: [rewards[var] * weights[var] for rewards in all_rewards]
            for var in weights
        }
        weighted_rewards_df = pd.DataFrame(weighted_rewards_data, index=sample_df.index)

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
    def train_SA(
        self,
        iterations: int,
        checkpoint_callback: Callable[[int], None] | None = None,
        checkpoint_interval: int | None = None,
        start_iteration: int = 0,
    ) -> None:
        """
        Train the agent using simulated annealing for a number of iterations.

        Parameters
        ----------
        iterations : int
            Number of iterations to train the agent for.
        checkpoint_callback : Callable[[int], None] | None, optional
            Callback function to call when saving checkpoints. Called with absolute iteration number.
        checkpoint_interval : int | None, optional
            Save checkpoint every N iterations. Must be provided if checkpoint_callback is set.
        start_iteration : int, optional
            Starting iteration number (for continuing training). Defaults to 0.
        """
        current_iteration = start_iteration
        while iterations > 0:
            # Track EMA history for analysis and plotting
            self.ema_history.append(copy.deepcopy(self.ema))

            # Step 1: update EMA and identify which utilities to downweigh
            utilities_to_downweigh: set[str] = self._update_ema_and_collect_adjustments()

            # Step 2: pick a candidate variable to tweak
            candidate_var: str = random.choice(list(utilities_to_downweigh | self.reflective_vars))

            # Step 3: simulate current policy and train u-hat models
            normal_time_step: TimeStep = self.time_step(self.fixed_assignment, self.weights, self.sample_num)
            self._train_u_hat_models(normal_time_step)

            # Step 4: apply either a weight change or a reflective intervention
            if candidate_var in utilities_to_downweigh:
                self._maybe_apply_weight_adjustment(candidate_var, normal_time_step)
            else:
                chosen_value, interventional_reward = self._evaluate_reflective_assignment(candidate_var, normal_time_step)
                delta: float = interventional_reward - sum(normal_time_step.average_reward.values())
                if self._accept_metropolis(delta):
                    adjusted_cpt = self.nudge_cpt_new(
                        self.sampling_model.get_cpds(candidate_var),
                        {candidate_var: chosen_value},
                        self.cpt_increase_factor,
                        interventional_reward,
                    )
                    self.sampling_model.remove_cpds(self.sampling_model.get_cpds(candidate_var))
                    self.sampling_model.add_cpds(adjusted_cpt)
                    # add_cpds() now validates the model, so we can use FastCausalInference
                    # Refresh persistent inference after CPD change
                    self.inference = FastCausalInference(self.sampling_model)
                    # Invalidate cached interventional queries due to model change
                    self._cdn_cache.clear()
                    # Invalidate hash cache - will recompute on next query
                    self._model_hash = None

            # Step 5: record and cool
            self.memory.append(normal_time_step)
            self.temperature *= self.cooling_factor
            iterations -= 1
            current_iteration += 1
            
            # Save checkpoint if callback provided and interval reached
            if checkpoint_callback and checkpoint_interval:
                if current_iteration % checkpoint_interval == 0:
                    checkpoint_callback(current_iteration)

    def _update_ema_and_collect_adjustments(self) -> set[str]:
        """
        Update EMA of utility rewards and return the set of utilities whose EMA
        falls below the frustration threshold.

        Returns
        -------
        set[str]
            Utilities that should be considered for downweighing.
        """
        utilities_to_downweigh: set[str] = set()
        if len(self.memory) > 0:
            for util_name, ema_value in self.ema.items():
                new_ema = (1 - self.downweigh_alpha) * ema_value + (
                    self.downweigh_alpha * self.memory[-1].average_reward[util_name]
                )
                self.ema[util_name] = new_ema
                if new_ema < self.downweigh_threshold[util_name]:
                    utilities_to_downweigh.add(util_name)
        return utilities_to_downweigh

    def _train_u_hat_models(self, normal_time_step: TimeStep) -> None:
        """
        Train per-utility predictive models on the latest simulated samples.
        Optimized to avoid repeated DataFrame operations.
        """
        # Preprocess data once for all models
        hidden_cols_to_drop = [col for col in self.hidden_vars if col in normal_time_step.memory.columns]
        if hidden_cols_to_drop:
            base_data = normal_time_step.memory.drop(columns=hidden_cols_to_drop)
        else:
            base_data = normal_time_step.memory
        
        # Data should already be numeric from simulation, but ensure it if needed
        # Skip apply(pd.to_numeric) as it's expensive and data is already numeric from simulate()
        
        for util_name, model in self.u_hat_models.items():
            if model is None:
                continue
            
            # Use base data and only modify if pooling is needed
            if (
                hasattr(self, "lower_tier_pooled_reward")
                and self.lower_tier_pooled_reward
                and util_name in self.lower_tier_pooled_reward
                and util_name not in self.hidden_vars
            ):
                # Create a copy only when needed
                numeric = base_data.copy()
                numeric[util_name] = self.lower_tier_pooled_reward[util_name]
            else:
                numeric = base_data

            model.train(numeric, epochs=self.u_hat_epochs)

    def _maybe_apply_weight_adjustment(
        self, candidate_var: str, normal_time_step: TimeStep
    ) -> None:
        """
        Try down-weighing the specified utility and accept the change with
        simulated annealing.
        """
        adjusted_weights: dict[str, float] = Counter(copy.deepcopy(self.weights))
        adjusted_weights[candidate_var] *= self.downweigh_factor
        adjusted_weights = normalize(adjusted_weights)

        weight_adjusted_time_step: TimeStep = self.time_step(
            self.fixed_assignment, adjusted_weights, self.sample_num
        )

        delta: float = sum(weight_adjusted_time_step.average_reward.values()) - sum(
            normal_time_step.average_reward.values()
        )

        if self._accept_metropolis(delta):
            self.weights = adjusted_weights

    def _accept_metropolis(self, delta: float) -> bool:
        """
        Metropolis acceptance criterion used by simulated annealing.
        """
        return bool(
            delta >= 0 or random.random() <= np.exp((-1 * np.tanh(delta)) / self.temperature)
        )

    def _evaluate_reflective_assignment(
        self, reflective_var: str, normal_time_step: TimeStep
    ) -> tuple[int, float]:
        """
        Evaluate each possible assignment for a reflective variable by combining
        (a) the probability of parents under a do-intervention and (b) the
        predicted utility from the u-hat model. Returns the best assignment and
        its total weighted reward (post-weights).
        
        Optimized to batch neural network predictions and reduce redundant CDN queries.
        """
        num_candidate_values = self.card_dict[reflective_var]
        
        # Group utilities by their parent sets (sorted) to batch CDN queries
        # Key: sorted tuple of parent vars, Value: list of utilities with those parents
        parent_set_to_utilities: dict[tuple[str, ...], list[str]] = {}
        utility_to_parent_set: dict[str, tuple[str, ...]] = {}
        
        for utility in self.utility_vars:
            if self.u_hat_models[utility] is None:
                continue
            # Use sorted tuple to ensure consistent grouping regardless of nn_input_cols order
            parent_set = tuple(sorted(self.u_hat_models[utility].input_cols))
            if parent_set not in parent_set_to_utilities:
                parent_set_to_utilities[parent_set] = []
            parent_set_to_utilities[parent_set].append(utility)
            utility_to_parent_set[utility] = parent_set
        
        # Pre-compute all CDN queries once per (candidate_value, parent_set) pair
        cdn_results_cache: dict[tuple[int, tuple[str, ...]], DiscreteFactor] = {}
        for candidate_value in range(num_candidate_values):
            for parent_set in parent_set_to_utilities.keys():
                cache_key = (candidate_value, parent_set)
                if cache_key not in cdn_results_cache:
                    # Query with sorted variables (matches cache key logic in cdn_query)
                    cdn_results_cache[cache_key] = self.cdn_query(
                        list(parent_set), 
                        {reflective_var: candidate_value}
                    )
        
        # Now compute rewards using cached CDN results (same computation order as before)
        per_value_rewards: list[dict[str, float]] = []
        for candidate_value in range(num_candidate_values):
            utility_reward_by_name: dict[str, float] = Counter()
            
            # Process utilities in original order to preserve behavior
            for utility in self.utility_vars:
                if self.u_hat_models[utility] is None:
                    utility_reward_by_name[utility] = 0.0
                    continue
                
                parent_combos = self.parent_combinations[utility]
                if len(parent_combos) == 0:
                    utility_reward_by_name[utility] = 0.0
                    continue
                
                # Get the expected input column order from the neural network
                nn_input_cols = self.u_hat_models[utility].input_cols
                parent_set = utility_to_parent_set[utility]
                
                # Get cached CDN result
                cdn_result = cdn_results_cache[(candidate_value, parent_set)]
                
                # Extract probabilities - same logic as before
                if list(cdn_result.variables) == nn_input_cols:
                    # Order matches, can use fast flatten
                    parent_probs = cdn_result.values.flatten()
                else:
                    # Fallback: extract probabilities respecting the actual parent_combos order
                    parent_probs = np.array([
                        cdn_result.get_value(**{col: parent_dict[col] for col in nn_input_cols})
                        for parent_dict in parent_combos
                    ])
                
                # Build parent array ensuring column order matches neural network expectations
                parent_array = np.array(
                    [[parent_dict[col] for col in nn_input_cols] for parent_dict in parent_combos],
                    dtype=np.float32
                )
                
                # Batch predict
                predictions = self.u_hat_models[utility].predict(parent_array)[:, 0]
                
                # Compute expected utility as dot product
                utility_reward_by_name[utility] = float(np.dot(parent_probs, predictions))
            
            per_value_rewards.append(utility_reward_by_name)

        # Apply current objective weights (vectorized) - same as before
        for rewards in per_value_rewards:
            for utility in rewards.keys():
                rewards[utility] *= self.weights[utility]

        total_by_value = [sum(d.values()) for d in per_value_rewards]
        best_total = max(total_by_value)
        candidate_indices = [i for i, s in enumerate(total_by_value) if s == best_total]
        chosen_value = random.choice(candidate_indices)
        interventional_reward = sum(per_value_rewards[chosen_value].values())
        return chosen_value, interventional_reward

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

    def write_cpds_to_csv(self, cpds: list[TabularCPD], name: str, folder: str) -> None:
        """
        Write CPDs to a CSV file.

        Parameters
        ----------
        cpds : list[TabularCPD]
            CPDs to write to a CSV file.
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
        self, cpds: list[TabularCPD], name: str, folder: str
    ) -> None:
        """
        Write the delta of CPDs to a CSV file.

        Parameters
        ----------
        cpds : list[TabularCPD]
            CPDs to write to a CSV file.
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

    @classmethod
    def clear_shared_cache(cls) -> None:
        """Clear the shared cache across all agents."""
        cls._shared_cdn_cache.clear()

    @classmethod
    def get_shared_cache_size(cls) -> int:
        """Get the size of the shared cache."""
        return len(cls._shared_cdn_cache)

    def populate_from_shared_cache(self) -> None:
        """Populate this agent's cache from the shared cache."""
        self._cdn_cache.update(self._shared_cdn_cache)

if __name__ == "__main__":
    def default_reward(
        sample: dict[str, int],
        utility_edges: list[tuple[str, str]],
        noise: float = 0.0,
        lower_tier_pooled_reward: dict[str, float] = None,
    ) -> dict[str, float]:
        rewards: dict[str, float] = Counter()
        for var, utility in utility_edges:
            noise_term = random.gauss(0, noise) if noise > 0 else 0.0
            if utility == "teacher_regulation":
                rewards[utility] -= sample["grade_leniency"]
            else:
                rewards[utility] += sample[var] + noise_term

        if lower_tier_pooled_reward:
            for util, reward in lower_tier_pooled_reward.items():
                rewards[util] += reward

        return rewards

    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    cla = CausalLearningAgent(
        sampling_edges=[("A", "B"), ("B", "C")],
        utility_edges=[("A", "U1"), ("B", "U2"), ("C", "U3")],
        cpts=[TabularCPD("A", 2, [[0.5], [0.5]]), TabularCPD("B", 2, [[0.5, 0.5], [0.5, 0.5]], evidence=["A"], evidence_card=[2]), TabularCPD("C", 2, [[0.5, 0.5], [0.5, 0.5]], evidence=["B"], evidence_card=[2])],
        utility_vars={"U1", "U2", "U3"},
        reflective_vars={"A", "B", "C"},
        chance_vars=set(),
        glue_vars=set(),
        reward_func=default_reward,
    )
    cla.train_SA(100)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(10)
    stats.dump_stats('cla_profile.prof')