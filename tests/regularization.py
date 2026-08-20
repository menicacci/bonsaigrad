import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import Dropout, Module, Sequential


class TestDropout(unittest.TestCase):

    def test_drops_values_and_scales_the_remainder_during_training(self):
        layer = Dropout(0.5, rng=np.random.default_rng(0))
        inputs = np.ones((2, 3))

        outputs = layer(inputs)

        mask = np.random.default_rng(0).random(inputs.shape) >= 0.5
        np.testing.assert_array_equal(outputs.data, 2.0 * mask)
        self.assertIsInstance(layer, Module)
        self.assertEqual(layer.parameters(), ())

    def test_backpropagates_through_the_same_scaled_mask(self):
        layer = Dropout(0.5, rng=np.random.default_rng(0))
        inputs = Leaf(np.ones((2, 3)))

        layer(inputs).sum().wire()

        mask = np.random.default_rng(0).random(inputs.data.shape) >= 0.5
        np.testing.assert_array_equal(inputs.grad, 2.0 * mask)

    def test_is_an_identity_during_evaluation(self):
        layer = Dropout(0.5, rng=np.random.default_rng(0))
        inputs = Leaf([1.0, 2.0, 3.0])

        self.assertIs(layer.eval(), layer)
        self.assertIs(layer(inputs), inputs)

    def test_sequential_propagates_evaluation_mode(self):
        dropout = Dropout(0.5, rng=np.random.default_rng(0))
        model = Sequential(dropout)

        outputs = model.eval()([1.0, 2.0, 3.0])

        np.testing.assert_array_equal(outputs.data, [1.0, 2.0, 3.0])
        self.assertFalse(dropout.training)

    def test_rejects_invalid_drop_probabilities(self):
        for p in (-0.1, 1.0, 1.1):
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    Dropout(p)
