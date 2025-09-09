import unittest
import os
import shutil
import tempfile
import numpy as np
from pgmpy.factors.discrete import TabularCPD

from src.training_environment import TrainingEnvironment
from src.cla import CausalLearningAgent
from src.util import Counter


def default_reward(sample: dict[str, int], utility_edges: list[tuple[str, str]], noise: float = 0.0, lower_tier_pooled_reward=None) -> dict[str, float]:
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


class TestTrainingEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = TrainingEnvironment()
        self.agent = build_agent()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_SA_runs(self):
        agents = [self.agent]
        trained = self.env.train(mc_rep=2, style="SA", agents=agents)
        self.assertEqual(len(trained[0].memory), 2)

    def test_pre_training_visualization_saves(self):
        self.env.pre_training_visualization(self.agent, name=self.temp_dir, save=True)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "pre_training_visualization.png")))

    def test_post_training_visualization_saves(self):
        # run a tiny train to populate memory & ema_history
        self.env.train(mc_rep=2, style="SA", agents=[self.agent])
        self.env.post_training_visualization([self.agent], name=self.temp_dir, save=True)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "post_training_visualization.png")))

    def test_show_cpt_changes_saves(self):
        # no training needed; initial delta is zero but file should still save
        self.env.show_cpt_changes([self.agent], name=self.temp_dir, save=True)
        # Expect files for reflective vars
        self.assertTrue(
            any(fname.startswith("cpt_changes_refl_") for fname in os.listdir(self.temp_dir))
        )

    def test_plot_cpt_comparison_saves(self):
        old_cpds = self.agent.get_cpts()
        # tweak one CPT a bit
        cpd = self.agent.get_var_cpt("refl_2")
        cpd.values[0][0] = 0.6
        cpd.values[1][0] = 0.4
        new_cpds = [cpd] + [c for c in old_cpds if c.variable != "refl_2"]
        self.env.plot_cpt_comparison(self.agent, old_cpds, new_cpds, name=self.temp_dir, save=True)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "cpt_comparison.png")))

    def test_plot_monte_carlo_saves(self):
        # run a tiny train to populate memory
        self.env.train(mc_rep=2, style="SA", agents=[self.agent])
        self.env.plot_monte_carlo([self.agent], show_params=False, name=self.temp_dir, save=True)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "monte_carlo.png")))

    def test_plot_weighted_rewards_saves(self):
        self.env.train(mc_rep=2, style="SA", agents=[self.agent])
        self.env.plot_weighted_rewards([self.agent], show_params=True, name=self.temp_dir, save=True)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "weighted_rewards.png")))

    def test_plot_u_hat_model_losses_saves(self):
        # train a model a bit to create a loss history
        normal_ts = self.agent.time_step(self.agent.fixed_assignment, self.agent.weights, 40)
        for util, model in self.agent.u_hat_models.items():
            if model is not None:
                model.train(normal_ts.memory, epochs=2)
        self.env.plot_u_hat_model_losses([self.agent], name=self.temp_dir, save=True)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "u_hat_model_losses.png")))

    def test_save_reward_csv(self):
        self.env.train(mc_rep=2, style="SA", agents=[self.agent])
        self.env.save_reward_csv([self.agent], name=self.temp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "rewards.csv")))


if __name__ == "__main__":
    unittest.main()
