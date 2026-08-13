import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import Linear, Module, Sequential, Tanh


class TestSequential(unittest.TestCase):

    def test_applies_modules_in_order(self):
        first = Linear(2, 2, rng=np.random.default_rng(0))
        first.weight.data[:] = [[1.0, -1.0], [0.5, 0.5]]
        first.bias.data[:] = [0.5, -0.5]
        second = Linear(2, 1, rng=np.random.default_rng(0))
        second.weight.data[:] = [[2.0], [-1.0]]
        second.bias.data[:] = [0.25]
        model = Sequential(first, Tanh(), second)
        inputs = [[2.0, 4.0], [-1.0, 3.0]]

        outputs = model(inputs)

        hidden = np.tanh(np.asarray(inputs) @ first.weight.data + first.bias.data)
        expected = hidden @ second.weight.data + second.bias.data
        self.assertIsInstance(model, Module)
        np.testing.assert_allclose(outputs.data, expected)

    def test_collects_parameters_in_child_order(self):
        first = Linear(2, 3, rng=np.random.default_rng(0))
        second = Linear(3, 1, rng=np.random.default_rng(1))

        model = Sequential(Sequential(first, Tanh()), second)

        self.assertEqual(model.parameters(), (first.weight, first.bias, second.weight, second.bias))

    def test_collects_shared_parameters_once(self):
        shared = Linear(2, 2, rng=np.random.default_rng(0))

        model = Sequential(shared, Tanh(), shared)

        self.assertEqual(model.parameters(), (shared.weight, shared.bias))

    def test_empty_container_is_a_leaf_identity(self):
        outputs = Sequential()([1.0, 2.0])

        self.assertIsInstance(outputs, Leaf)
        np.testing.assert_array_equal(outputs.data, [1.0, 2.0])

    def test_backpropagates_through_every_trainable_child(self):
        first = Linear(2, 3, rng=np.random.default_rng(0))
        second = Linear(3, 1, rng=np.random.default_rng(1))
        model = Sequential(first, Tanh(), second)

        model(Leaf([[1.0, -2.0], [0.5, 3.0]])).sum().wire()

        for parameter in model.parameters():
            self.assertEqual(parameter.grad.shape, parameter.data.shape)
            self.assertTrue(np.any(parameter.grad != 0.0))

    def test_rejects_non_modules(self):
        with self.assertRaises(TypeError):
            Sequential(Linear(2, 2, rng=np.random.default_rng(0)), object())
