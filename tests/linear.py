import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import Linear, Module


class TestLinear(unittest.TestCase):

    def test_initializes_trainable_weight_and_bias(self):
        default = Linear(2, 3)
        first = Linear(2, 3, np.random.default_rng(7))
        second = Linear(2, 3, np.random.default_rng(7))

        self.assertEqual(default.weight.data.shape, (2, 3))
        self.assertEqual(default.bias.data.shape, (3,))
        self.assertEqual(first.weight.data.shape, (2, 3))
        self.assertEqual(first.bias.data.shape, (3,))
        self.assertIsInstance(first, Module)
        self.assertEqual(first.parameters(), (first.weight, first.bias))
        np.testing.assert_array_equal(first.weight.data, second.weight.data)
        np.testing.assert_array_equal(first.bias.data, np.zeros(3))

    def test_transforms_the_last_axis_for_a_batch(self):
        layer = Linear(2, 3, np.random.default_rng(0))
        layer.weight.data[:] = [[1.0, -1.0, 0.5], [0.5, 1.0, -1.0]]
        layer.bias.data[:] = [0.0, 0.5, -0.5]
        inputs = [[2.0, 4.0], [-1.0, 3.0]]

        outputs = layer(inputs)

        np.testing.assert_array_equal(outputs.data, [[4.0, 2.5, -3.5], [0.5, 4.5, -4.0]])
        repeated = layer(np.stack([inputs, inputs]))
        self.assertEqual(repeated.data.shape, (2, 2, 3))
        np.testing.assert_array_equal(repeated.data[0], outputs.data)

    def test_gradients_accumulate_from_every_example(self):
        layer = Linear(2, 3, np.random.default_rng(0))
        layer.weight.data[:] = [[1.0, -1.0, 0.5], [0.5, 1.0, -1.0]]
        inputs = Leaf([[2.0, 4.0], [-1.0, 3.0]])

        layer(inputs).sum().wire()

        np.testing.assert_array_equal(inputs.grad, [[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_equal(layer.weight.grad, [[1.0, 1.0, 1.0], [7.0, 7.0, 7.0]])
        np.testing.assert_array_equal(layer.bias.grad, [2.0, 2.0, 2.0])

    def test_rejects_inputs_without_the_expected_feature_axis(self):
        layer = Linear(2, 3, np.random.default_rng(0))

        with self.assertRaises(ValueError):
            layer([1.0, 2.0])
        with self.assertRaises(ValueError):
            layer(np.ones((2, 4)))

    def test_requires_positive_feature_counts(self):
        with self.assertRaises(ValueError):
            Linear(0, 3, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            Linear(2, 0, np.random.default_rng(0))
