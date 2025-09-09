import unittest
import numpy as np
import pandas as pd

from src.cla_neural_network import CLANeuralNetwork as CLANN


class TestCLANeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Simple linear dataset: y = x1 + x2
        rng = np.random.default_rng(42)
        x1 = rng.integers(0, 2, size=200)
        x2 = rng.integers(0, 2, size=200)
        y = x1 + x2
        self.df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        self.model = CLANN(["x1", "x2"], ["y"], hidden_units=8, learning_rate=0.01)

    def test_train_appends_losses(self):
        self.model.train(self.df, epochs=10, batch_size=32, validation_split=0.1)
        self.assertGreaterEqual(len(self.model.losses), 1)
        self.assertTrue(np.isfinite(self.model.losses[-1]))

    def test_predict_shape(self):
        self.model.train(self.df, epochs=5, batch_size=32, validation_split=0.1)
        preds = self.model.predict(pd.DataFrame([{"x1": 1, "x2": 0}]))
        self.assertEqual(preds.shape, (1, 1))

    def test_evaluate_returns_float(self):
        self.model.train(self.df, epochs=5, batch_size=32, validation_split=0.1)
        loss = self.model.evaluate(self.df)
        self.assertTrue(isinstance(loss, float))
        self.assertTrue(np.isfinite(loss))

    def test_second_train_not_worse(self):
        self.model.train(self.df, epochs=5, batch_size=32, validation_split=0.1)
        first = self.model.losses[-1]
        self.model.train(self.df, epochs=5, batch_size=32, validation_split=0.1)
        second = self.model.losses[-1]
        self.assertLessEqual(second, first + 1e-6)


if __name__ == "__main__":
    unittest.main()
