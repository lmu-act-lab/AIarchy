import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import copy
from typing import Callable
from util import *
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork  # type: ignore
from pgmpy.inference import CausalInference  # type: ignore


class CausalLearningAgent:

    def __init__(
        self,
        sampling_edges: list[tuple[str, str]],
        structural_edges: list[tuple[str, str]],
        cpts: list[TabularCPD],
        utility_vars: set[str],
        reflective_vars: set[str],
        chance_vars: set[str],
        glue_vars: set[str],
        reward_func: Callable[[dict[str, int]], dict[str, float]],
        fixed_evidence: dict[str, int] = {},
        weights: dict[str, float] = {},
        threshold: float = 0.1,
        downweigh_factor: float = 0.8,
        sample_num: int = 100,
        alpha: float = 0.01,
        ema_alpha: float = 0.1,
    ) -> None:
        """
        Initializes two Bayesian Networks, one for sampling and one to maintain structure. Also initializes a number of other variables to be used in the learning process.

        Parameters
        ----------
        sampling_edges : list[tuple[str, str]]
            edges of the network to be used in the sampling model.
        structural_edges : list[tuple[str, str]]
            edges for the reward signals/utility vars (don't repeat the ones in sampling_edges).
        cpts : list[TabularCPD]
            CPTs for the sampling model.
        utility_vars : set[str]
            utility nodes/reward signals.
        reflective_vars : set[str]
            vars agent can intervene on.
        chance_vars : set[str]
            vars that the agent cannot control.
        glue_vars : set[str]
            vars that will be affected by agents from greater hierarchies.
        reward_func : Callable[[dict[str, int]], dict[str, float]]
            reward function for the agent.
        fixed_evidence : dict[str, int], optional
             any fixed evidence for that particular agent, by default {}
        weights : dict[str, float], optional
            any starting weights for archetype of agent (len must match how many utility vars there are), by default {}
        threshold : float, optional
            any starting weights for archetype of agent (len must match how many utility vars there are), by default 0.1
        downweigh_factor : float, optional
            amount to downweigh weights by, by default 0.8
        sample_num : int, optional
            number of samples taken for one time step, by default 100
        alpha : float, optional
            learning rate, by default 0.01
        ema_alpha : float, optional
            ema adjustment rate, by default 0.1
        """

        self.sampling_model: BayesianNetwork = BayesianNetwork(sampling_edges)

        self.memory: list[dict[str, float]] = []
        self.utility_vars: set[str] = utility_vars
        self.reflective_vars: set[str] = reflective_vars
        self.chance_vars: set[str] = chance_vars
        self.glue_vars: set[str] = glue_vars
        self.non_utility_vars: set[str] = reflective_vars | chance_vars | glue_vars
        self.fixed_assignment: dict[str, int] = {}

        self.structural_model: BayesianNetwork = BayesianNetwork(
            self.sampling_model.edges()
        )
        self.structural_model.add_nodes_from(self.utility_vars)
        self.structural_model.add_edges_from(structural_edges)

        self.reward_attr_model = self.structural_model.do(self.reflective_vars)
        self.inference_model = self.structural_model.do(self.non_utility_vars)

        self.sample_num: int = sample_num
        self.alpha: float = alpha
        self.ema_alpha: float = ema_alpha
        self.threshold: float = threshold
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

        self.fixed_assignment = fixed_evidence

        self.card_dict: dict[str, int] = {
            key: self.sampling_model.get_cardinality(key)
            for key in self.reflective_vars
        }

        self.original_model: BayesianNetwork = copy.deepcopy(self.sampling_model)

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
        # Initialize the graph
        G = nx.DiGraph()
        G.add_edges_from(self.structural_model.edges())

        # Define colors for each type of variable
        color_map = {
            "reflective": "blue",
            "glue": "green",
            "chance": "red",
            "utility": "yellow",
        }

        # Initialize a list to hold edge colors
        edge_colors = []

        # Determine the color of each edge based on the variable type
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
                edge_colors.append("black")  # Default color

        # Draw the graph
        pos = nx.spring_layout(G)  # Positions for all nodes
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

        # Display the plot
        plt.show()

    def plot_memory(self) -> None:
        """
        Plots the perceived reward at each iteration.
        """
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
        plt.xlabel("Iteration")
        plt.ylabel("Perceived Reward")
        plt.title("Perceived reward at iteration t")
        plt.legend()

        # Show the plot
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
        cpd: TabularCPD = self.model.get_cpds(variable)
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
        memory: pd.DataFrame = pd.DataFrame(
            columns=list(self.utility_vars | self.non_utility_vars)
        )
        # todo: add custom object as return type for time_step
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
            memory = memory.append(
                {
                    **sample_dict,
                    **self.calculate_expected_reward(sample_dict, weighted_reward),
                },
                ignore_index=True,
            )
        return TimeStep(memory, self.non_utility_vars)

    def nudge_cpt(self, cpd, evidence, increase_factor, reward) -> TabularCPD:
        values: list = cpd.values
        indices: list[int] = [evidence[variable] for variable in cpd.variables]
        # Convert indices to tuple to use for array indexing
        tuple_indices: tuple[int] = tuple(indices)
        # Update the value at the specified indices
        values[tuple_indices] += values[tuple_indices] * increase_factor * reward

        # Normalize the probabilities to ensure they sum to 1
        sum_values: list = np.sum(values, axis=0)
        normalized_values: list = values / sum_values

        # Update the CPD with normalized values
        cpd.values = normalized_values
        return cpd

    def train(self, iterations: int) -> None:
        for _ in range(iterations):
            utility_to_adjust: set[str] = set()
            if _ > 0:
                for var, ema in self.ema.items():
                    new_ema = (1 - self.ema_alpha) * (
                        ema + self.ema_alpha * self.memory[-1][var]
                    )
                    self.ema[var] = new_ema
                    if new_ema < self.threshold:
                        utility_to_adjust.add(var)

            tweak: str = random.choice(list(utility_to_adjust | self.reflective_vars))
            if tweak in utility_to_adjust:
                # perform weight change intervention
                # compare reward with no intervention
                pass
            else:
                # perform reflective var intervention
                # compare reward with no intervention
                pass
        # 1. do ema check to see eligible weights if any
        # 2. pick a reflective var / eligible weight @ random
        #   A. pick a condition e.g. x_2 = 0, x_3 = 1
        #   B. pick a direction to tweak e.g. p(x_1 = 0| x_2 = 0, x_3 = 1) +- alpha
        #   AB_viv. pick a reflective var direction to perform intervention e.g. do(x_1 = 0)
        #   C. If 2B is a sideways/upwards move then commit the tweak by
        #      getting the average sample and only attributing the rewards
        #      that adhere to the average sample
        #      Else if downwards move then only commit if some coinflip:
        #      RNG <= e^(-|delta|/T) where T is the temperature
        #      where delta is the difference between the new and old reward
        # 3. cooling schedule reduce T by some factor
        # 4. repeat until iterations reached


class TimeStep:
    def __init__(
        self,
        memory: pd.DataFrame,
        sample_vars: set[str],
    ):
        self.memory: pd.DataFrame = memory
        self.average_sample: dict[str, float] = {}

        for var in sample_vars:
            self.average_sample[var] = self.get_column_average(var)

    def get_column_average(self, column_name: str) -> float:
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


#! future research
# * look into bayesian structure learning from data
# *
