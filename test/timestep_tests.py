import unittest
import pandas as pd
from src.timestep import TimeStep


class TestTimeStep(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "a": [0, 1, 1, 0],
            "b": [1, 1, 0, 0],
            "reward_x": [0.0, 1.0, 0.0, 1.0],
        })
        self.ts = TimeStep(
            memory=self.df,
            sample_vars={"a", "b"},
            reward_vars={"reward_x"},
            weights={"reward_x": 1.0},
            tweak_var=None,
        )

    def test_average_sample_is_integer(self):
        # a average is 0.5 -> rounds to 0 or 1 depending on Python's rounding, check in set {0,1}
        self.assertIn(self.ts.average_sample["a"], {0, 1})
        self.assertIn(self.ts.average_sample["b"], {0, 1})

    def test_average_reward_is_mean(self):
        self.assertAlmostEqual(self.ts.average_reward["reward_x"], self.df["reward_x"].mean(), 6)

    def test_get_rounded_column_average(self):
        val = self.ts.get_rounded_column_average("a")
        self.assertIn(val, {0, 1})

    def test_get_column_average(self):
        self.assertEqual(self.ts.get_column_average("reward_x"), self.df["reward_x"].mean())

    def test_equality_and_hash(self):
        ts2 = TimeStep(
            memory=self.df.copy(),
            sample_vars={"a", "b"},
            reward_vars={"reward_x"},
            weights={"reward_x": 1.0},
            tweak_var=None,
        )
        self.assertEqual(self.ts, ts2)
        # Put into a set to test hashability
        s = set([self.ts, ts2])
        self.assertEqual(len(s), 1)


if __name__ == "__main__":
    unittest.main()
