import pandas as pd
import numpy as np
import warnings
import matplotlib as plt
import copy
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

        for cpt in cpts:
            self.sampling_model.add_cpds(cpt)

        self.fixed_assignment = fixed_evidence

        self.card_dict: dict[str, int] = {
            key: self.sampling_model.get_cardinality(key)
            for key in self.reflective_vars
        }

        self.original_model: BayesianNetwork = copy.deepcopy(self.sampling_model)

    def get_cpts(
        self, vars: set[str] = set(), original: bool = False
    ) -> list[TabularCPD]:
        """
        Returns the cpts of the model

        Args:
            vars (set[str], optional): specific vars you'd want the cpts for. Defaults to {}.
            original (bool, optional): whether you want to see the original cpts. Defaults to False.

        Returns:
            list[TabularCPD]: a list of cpts pertaining to the sampling model.
        """
        return (
            (
                self.original_model.get_cpds()
                if original
                else self.sampling_model.get_cpds()
            )
            if not vars
            else [
                (
                    self.original_model.get_cpds(var)
                    if original
                    else self.sampling_model.get_cpds(var)
                )
                for var in vars
            ]
        )
    
    def display_memory(self) -> None:
        for time_step in self.memory:
            print(time_step)
            print("\n")

    def display_weights(self) -> None:
        for time_step, weights in self.weight_history.items():
            print(f"Iteration {time_step}:")
            print(weights)
            print("\n")

    # def display_cpt(self, var: str = "") -> None:
    #     for cpd in self.model.get_cpds():
    #         if cpd.variable == var:
    #             print(f"CPD of {cpd.variable}:")
    #             print(cpd)
    #             print("\n")
