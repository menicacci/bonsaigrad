import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.optim import Optimizer, SGD


class TestOptimizer(unittest.TestCase):

    def test_sgd_is_an_optimizer(self):
        optimizer = SGD([Leaf([1.0])], learning_rate=0.1)

        self.assertIsInstance(optimizer, Optimizer)

    def test_rejects_no_parameters(self):
        with self.assertRaises(ValueError):
            SGD([], learning_rate=0.1)

    def test_rejects_non_leaf_parameters(self):
        with self.assertRaises(TypeError):
            SGD([1.0], learning_rate=0.1)

    def test_rejects_non_positive_learning_rates(self):
        parameter = Leaf([1.0])

        with self.assertRaises(ValueError):
            SGD([parameter], learning_rate=0.0)
        with self.assertRaises(ValueError):
            SGD([parameter], learning_rate=-0.1)
        with self.assertRaises(ValueError):
            SGD([parameter], learning_rate=float("nan"))

    def test_step_updates_each_parameter_from_its_gradient(self):
        weights = Leaf([[1.0, -2.0], [3.0, 4.0]])
        bias = Leaf([0.5, -1.0])
        weights.grad[:] = [[2.0, -4.0], [6.0, 8.0]]
        bias.grad[:] = [1.0, -2.0]

        SGD([weights, bias], learning_rate=0.25).step()

        np.testing.assert_array_equal(weights.data, [[0.5, -1.0], [1.5, 2.0]])
        np.testing.assert_array_equal(bias.data, [0.25, -0.5])

    def test_zero_grad_clears_each_parameter_gradient(self):
        weights = Leaf([[1.0, -2.0]])
        bias = Leaf([0.5, -1.0])
        weights.grad[:] = [[2.0, -4.0]]
        bias.grad[:] = [1.0, -2.0]

        SGD([weights, bias], learning_rate=0.1).zero_grad()

        np.testing.assert_array_equal(weights.grad, np.zeros_like(weights.data))
        np.testing.assert_array_equal(bias.grad, np.zeros_like(bias.data))
