import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.optim import Adam, Optimizer, SGD


class TestOptimizer(unittest.TestCase):

    def test_sgd_is_an_optimizer(self):
        optimizer = SGD([Leaf([1.0])], learning_rate=0.1)

        self.assertIsInstance(optimizer, Optimizer)

    def test_adam_is_an_optimizer(self):
        optimizer = Adam([Leaf([1.0])])

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

    def test_adam_step_uses_bias_corrected_moments(self):
        parameter = Leaf([1.0])
        optimizer = Adam([parameter], learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)

        parameter.grad[:] = [2.0]
        optimizer.step()
        parameter.grad[:] = [4.0]
        optimizer.step()

        np.testing.assert_allclose(parameter.data, [0.8034818439])

    def test_adam_rejects_invalid_hyperparameters(self):
        parameter = Leaf([1.0])

        for learning_rate in (0.0, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                Adam([parameter], learning_rate=learning_rate)
        for beta1 in (-0.1, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                Adam([parameter], beta1=beta1)
        for beta2 in (-0.1, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                Adam([parameter], beta2=beta2)
        for epsilon in (0.0, -1e-8, float("nan")):
            with self.assertRaises(ValueError):
                Adam([parameter], epsilon=epsilon)

    def test_zero_grad_clears_each_parameter_gradient(self):
        weights = Leaf([[1.0, -2.0]])
        bias = Leaf([0.5, -1.0])
        weights.grad[:] = [[2.0, -4.0]]
        bias.grad[:] = [1.0, -2.0]

        SGD([weights, bias], learning_rate=0.1).zero_grad()

        np.testing.assert_array_equal(weights.grad, np.zeros_like(weights.data))
        np.testing.assert_array_equal(bias.grad, np.zeros_like(bias.data))
