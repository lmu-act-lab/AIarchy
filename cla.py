import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import copy
from typing import Callable
from util import *
from pgmpy.factors.discrete import TabularCPD  # type: ignore
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

        Args:
            sampling_edges (list[tuple[str, str]]): edges of the network to be used in the sampling model.
            structural_edges (list[tuple[str, str]]): edges for the reward signals/utility vars (don't repeat the ones in sampling_edges).
            cpts (list[TabularCPD]): cpts for the sampling model.
            utility_vars (set[str]): utility nodes/reward signals.
            reflective_vars (set[str]): vars agent can intervene on.
            chance_vars (set[str]): vars that the agent cannot control.
            glue_vars (set[str]): vars that will change from greater hierarchies.
            fixed_evidence (dict[str, int], optional): any fixed evidence for that particular agent. Defaults to {}.
            weights (dict[str, float], optional): any starting weights for archetype of agent (len must match how many utility vars there are). Defaults to {}.
            threshold (float, optional): threshold for EMA used in downweighing weights over time. Defaults to 0.1.
            downweigh_factor (float, optional): amount to downweigh weights by. Defaults to 0.8.
            sample_num (int, optional): number of samples taken for one time step. Defaults to 100.
            alpha (float, optional): learning rate. Defaults to 0.01.
            ema_alpha (float, optional): ema adjustment rate. Defaults to 0.1.
        """
        self.sampling_model: BayesianNetwork = BayesianNetwork(sampling_edges)

        self.memory: list[dict[str, float]] = []
        self.utility_vars: set[str] = utility_vars
        self.reflective_vars: set[str] = reflective_vars
        self.chance_vars: set[str] = chance_vars
        self.glue_vars: set[str] = glue_vars
        self.fixed_assignment: dict[str, int] = {}

        self.structural_model: BayesianNetwork = BayesianNetwork(
            self.sampling_model.edges()
        )
        self.structural_model.add_nodes_from(self.utility_vars)
        self.structural_model.add_edges_from(structural_edges)
        self.structural_model.do(self.reflective_vars, True)

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

        Returns:
            list[TabularCPD]: List of CPDs.
        """
        return self.sampling_model.get_cpds()

    def get_original_cpts(self) -> list[TabularCPD]:
        """
        Returns the CPDs of the original model.

        Returns:
            list[TabularCPD]: List of CPDs.
        """
        return self.original_model.get_cpds()

    def get_var_cpt(self, var: str) -> list[TabularCPD]:
        """
        Returns the CPDs of a particular variable.

        Args:
            var (str): The variable to get the CPD of.

        Returns:
            list[TabularCPD]: List of CPDs.
        """
        return self.sampling_model.get_cpds(var)

    def get_original__var_cpt(self, var: str) -> list[TabularCPD]:
        """
        Returns the CPDs of a particular variable in the original model.

        Args:
            var (str): The variable to get the CPD of.

        Returns:
            list[TabularCPD]: List of CPDs.
        """
        return self.original_model.get_cpds(var)

    def display_memory(self) -> None:
        """
        Displays the memory of the agent.
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
        Displays the CPD of a particular variable.

        Args:
            var (str): The variable to display the CPD of.
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
        
        pass