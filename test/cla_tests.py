import unittest
import numpy as np
import pandas as pd
import copy
import warnings

from pgmpy.factors.discrete import TabularCPD
from src.cla import CausalLearningAgent
from src.cla_neural_network import CLANeuralNetwork as CLANN
from src.util import Counter

warnings.filterwarnings("ignore")

def default_reward(
    sample: dict[str, int], utility_edges: list[tuple[str, str]]
) -> dict[str, float]:
    rewards: dict[str, float] = Counter()
    for var, utility in utility_edges:
        rewards[utility] += sample[var]
    return rewards

x = CausalLearningAgent(
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

refl_1 = np.random.randint(0, 2, 1000)
refl_2 = np.random.randint(0, 2, 1000)

df = pd.DataFrame({
    "refl_1": refl_1,
    "refl_2": refl_2,
    "util_1": refl_1,
    "util_2": refl_2
})

class TestInitialize(unittest.TestCase):
    def test_class(self):
        self.assertIsInstance(x, CausalLearningAgent)

    def test_weight_init(self):
        self.assertEqual(x.weights, {"util_1" : 0.5, "util_2" : 0.5})

    def test_calculate_expected_utility(self):
        self.assertEqual(x.calculate_expected_reward({"refl_1" : 1, "refl_2" : 0}, {"util_1" : 1, "util_2" : 0}), {'util_1': np.float64(0.5), 'util_2': np.float64(0.0)})

    def test_u_hat_init(self):
        self.assertEqual(CLANN(["refl_1"], ["util_1"]).input_cols, x.u_hat_models["util_1"].input_cols)
        self.assertEqual(CLANN(["refl_1"], ["util_1"]).output_cols, x.u_hat_models["util_1"].output_cols)
        self.assertEqual(CLANN(["refl_2"], ["util_2"]).input_cols, x.u_hat_models["util_2"].input_cols)
        self.assertEqual(CLANN(["refl_2"], ["util_2"]).output_cols, x.u_hat_models["util_2"].output_cols)

    def test_u_hat_training(self):
        for model in x.u_hat_models.values():
            model.train(df)
        self.assertAlmostEqual(x.u_hat_models["util_1"].predict(pd.DataFrame([{"refl_1": 0}]))[0, 0], 0, 5)
        self.assertAlmostEqual(x.u_hat_models["util_2"].predict(pd.DataFrame([{"refl_2": 1}]))[0, 0], 1, 5)

    def test_time_step(self):
        a_time_step = x.time_step(x.fixed_assignment, x.weights, 1000)
        self.assertSetEqual(set(a_time_step.memory.columns), set(df.columns))
        self.assertEqual(a_time_step.memory.shape, df.shape)


class TestTraining(unittest.TestCase):
        def test_CPT_nudged(self):
            pass

        def test_weight_changed(self):
            pass
        
        def test_monte_carlo_average(self):
            pass

if __name__ == "__main__":
    unittest.main()
