import unittest
import numpy as np
import pandas as pd
import copy
import warnings
import os
import tempfile

from pgmpy.factors.discrete import TabularCPD
from src.cla import CausalLearningAgent
from src.cla_neural_network import CLANeuralNetwork as CLANN
from src.util import Counter

warnings.filterwarnings("ignore")

def default_reward(
    sample: dict[str, int], utility_edges: list[tuple[str, str]], reward_noise: float = 0.0, lower_tier_pooled_reward=None
) -> dict[str, float]:
    rewards: dict[str, float] = Counter()
    for var, utility in utility_edges:
        rewards[utility] += sample[var]
    return rewards


def build_agent() -> CausalLearningAgent:
    return CausalLearningAgent(
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


# Training data for u-hat tests
refl_1 = np.random.randint(0, 2, 1000)
refl_2 = np.random.randint(0, 2, 1000)

df = pd.DataFrame({
    "refl_1": refl_1,
    "refl_2": refl_2,
    "util_1": refl_1,
    "util_2": refl_2
})


class TestInitialize(unittest.TestCase):
    def setUp(self):
        self.agent = build_agent()

    def test_class(self):
        self.assertIsInstance(self.agent, CausalLearningAgent)

    def test_weight_init(self):
        self.assertEqual(self.agent.weights, {"util_1": 0.5, "util_2": 0.5})
        self.assertAlmostEqual(sum(self.agent.weights.values()), 1.0, 6)

    def test_calculate_expected_utility(self):
        res = self.agent.calculate_expected_reward({"refl_1": 1, "refl_2": 0}, {"util_1": 1, "util_2": 0})
        self.assertEqual(res, {'util_1': np.float64(0.5), 'util_2': np.float64(0.0)})

    def test_u_hat_init(self):
        self.assertEqual(CLANN(["refl_1"], ["util_1"]).input_cols, self.agent.u_hat_models["util_1"].input_cols)
        self.assertEqual(CLANN(["refl_1"], ["util_1"]).output_cols, self.agent.u_hat_models["util_1"].output_cols)
        self.assertEqual(CLANN(["refl_2"], ["util_2"]).input_cols, self.agent.u_hat_models["util_2"].input_cols)
        self.assertEqual(CLANN(["refl_2"], ["util_2"]).output_cols, self.agent.u_hat_models["util_2"].output_cols)

    def test_u_hat_training(self):
        for model in self.agent.u_hat_models.values():
            model.train(df)
        self.assertAlmostEqual(self.agent.u_hat_models["util_1"].predict(pd.DataFrame([{"refl_1": 0}]))[0, 0], 0, 5)
        self.assertAlmostEqual(self.agent.u_hat_models["util_2"].predict(pd.DataFrame([{"refl_2": 1}]))[0, 0], 1, 5)

    def test_time_step(self):
        a_time_step = self.agent.time_step(self.agent.fixed_assignment, self.agent.weights, 200)
        self.assertTrue(set({"refl_1", "refl_2", "util_1", "util_2"}).issubset(set(a_time_step.memory.columns)))
        self.assertEqual(a_time_step.memory.shape[0], 200)

    def test_par_dict(self):
        par = self.agent.par_dict()
        # Each utility has 1 reflective parent with cardinality 2
        self.assertEqual(len(par["util_1"]), 2)
        self.assertEqual(len(par["util_2"]), 2)
        self.assertIn({"refl_1": 0}, par["util_1"])
        self.assertIn({"refl_1": 1}, par["util_1"])

    def test_cdn_query_with_and_without_overlap(self):
        # Without overlap
        factor_no_overlap = self.agent.cdn_query(["refl_1"], {"refl_2": 1})
        self.assertIsNotNone(factor_no_overlap)
        # With overlap (query var in do evidence)
        factor_overlap = self.agent.cdn_query(["refl_2"], {"refl_2": 1})
        self.assertIsNotNone(factor_overlap)


class TestLearningOps(unittest.TestCase):
    def setUp(self):
        self.agent = build_agent()

    def test_nudge_cpt_changes_value_and_normalizes(self):
        cpd = copy.deepcopy(self.agent.get_var_cpt("refl_2"))
        evidence = {var: 0 for var in cpd.variables}
        before = cpd.values.copy()
        updated = self.agent.nudge_cpt(cpd, evidence, increase_factor=0.1, reward=1.0)
        self.assertFalse(np.allclose(before, updated.values))
        # Columns should remain normalized
        col_sums = np.sum(updated.values, axis=0)
        self.assertTrue(np.allclose(col_sums, np.ones_like(col_sums)))

    def test_nudge_cpt_new_changes_values_and_normalizes(self):
        cpd = copy.deepcopy(self.agent.get_var_cpt("refl_2"))
        before = cpd.values.copy()
        updated = self.agent.nudge_cpt_new(cpd, {"refl_2": 1}, increase_factor=0.05, reward=1.0)
        self.assertFalse(np.allclose(before, updated.values))
        col_sums = np.sum(updated.values, axis=0)
        self.assertTrue(np.allclose(col_sums, np.ones_like(col_sums)))

    def test_compute_cpd_delta_initially_zero(self):
        # Original model is a deep copy at init
        delta_factor = self.agent.compute_cpd_delta("refl_2")
        self.assertTrue(np.allclose(delta_factor.values, np.zeros_like(delta_factor.values)))

    def test_cpt_helpers_and_value_accessors(self):
        # get_cpts returns a list of CPDs
        cpds = self.agent.get_cpts()
        self.assertTrue(any(c.variable == "refl_1" for c in cpds))
        # get_original_cpts returns a list on the original model
        orig_cpds = self.agent.get_original_cpts()
        self.assertTrue(any(c.variable == "refl_2" for c in orig_cpds))
        # var-specific helpers
        cpd_refl1 = self.agent.get_var_cpt("refl_1")
        cpd_orig_refl1 = self.agent.get_original__var_cpt("refl_1")
        self.assertEqual(cpd_refl1.variable, cpd_orig_refl1.variable)
        # value accessors
        val_updated = self.agent.get_cpt_vals("refl_2", 0)
        val_original = self.agent.get_original_cpt_vals("refl_2", 0)
        self.assertAlmostEqual(float(val_updated), float(val_original), 6)

    def test_write_cpds_and_delta_to_csv(self):
        temp_dir = tempfile.mkdtemp()
        try:
            cpds = self.agent.get_cpts()
            self.agent.write_cpds_to_csv(cpds, name="TestAgent", folder=temp_dir)
            self.agent.write_delta_cpd_to_csv(cpds, name="TestAgent", folder=temp_dir)
            # Files should be created for reflective vars
            expected_one = os.path.join(temp_dir, "TestAgent, refl_1.csv")
            expected_two = os.path.join(temp_dir, "TestAgent, refl_2.csv")
            delta_one = os.path.join(temp_dir, "Delta for TestAgent, refl_1.csv")
            delta_two = os.path.join(temp_dir, "Delta for TestAgent, refl_2.csv")
            self.assertTrue(os.path.exists(expected_one))
            self.assertTrue(os.path.exists(expected_two))
            self.assertTrue(os.path.exists(delta_one))
            self.assertTrue(os.path.exists(delta_two))
        finally:
            # Clean up temp files
            for fname in [
                "TestAgent, refl_1.csv",
                "TestAgent, refl_2.csv",
                "Delta for TestAgent, refl_1.csv",
                "Delta for TestAgent, refl_2.csv",
            ]:
                path = os.path.join(temp_dir, fname)
                if os.path.exists(path):
                    os.remove(path)
            os.rmdir(temp_dir)


if __name__ == "__main__":
    unittest.main()
