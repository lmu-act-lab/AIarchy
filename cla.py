import pandas as pd
import numpy as np
import warnings
import matplotlib as plt
import copy
from util import *
from pgmpy.factors.discrete import TabularCPD # type: ignore
from pgmpy.models import BayesianNetwork # type: ignore
from pgmpy.inference import CausalInference # type: ignore


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
