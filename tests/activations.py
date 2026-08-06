import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import Module, ReLU, Sigmoid, Tanh
from ._base import assert_grads


class TestActivations(unittest.TestCase):

    def test_relu_forward_and_backward(self):
        inputs = Leaf([-2.0, 0.0, 3.0])
        weights = Leaf([5.0, 6.0, 7.0])

        outputs = inputs.relu()
        (outputs * weights).sum().wire()

        np.testing.assert_array_equal(outputs.data, [0.0, 0.0, 3.0])
        np.testing.assert_array_equal(inputs.grad, [0.0, 0.0, 7.0])

    def test_sigmoid_forward_and_backward(self):
        inputs = Leaf([-2.0, 0.0, 2.0])
        weights = Leaf([2.0, 3.0, 5.0])

        outputs = inputs.sigmoid()
        (outputs * weights).sum().wire()

        expected = 1.0 / (1.0 + np.exp(-inputs.data))
        np.testing.assert_allclose(outputs.data, expected)
        np.testing.assert_allclose(inputs.grad, weights.data * expected * (1.0 - expected))

    def test_sigmoid_stays_finite_for_large_magnitudes(self):
        outputs = Leaf([-1_000.0, 1_000.0]).sigmoid()

        np.testing.assert_array_equal(outputs.data, [0.0, 1.0])

    def test_tanh_forward_and_backward(self):
        inputs = Leaf([-2.0, 0.0, 2.0])
        weights = Leaf([2.0, 3.0, 5.0])

        outputs = inputs.tanh()
        (outputs * weights).sum().wire()

        expected = np.tanh(inputs.data)
        np.testing.assert_allclose(outputs.data, expected)
        np.testing.assert_allclose(inputs.grad, weights.data * (1.0 - expected ** 2))

    def test_grad_checks_away_from_relu_kink(self):
        assert_grads(self, lambda values: values.relu(), [[-2.0, -0.5, 0.5, 2.0]])
        assert_grads(self, lambda values: values.sigmoid(), [[-2.0, -0.5, 0.5, 2.0]])
        assert_grads(self, lambda values: values.tanh(), [[-2.0, -0.5, 0.5, 2.0]])

    def test_activations_preserve_input_shape(self):
        values_by_shape = (0.5, [-1.0, 0.0, 1.0], np.arange(24.0).reshape(2, 3, 4) - 12.0)
        for values in values_by_shape:
            for activation in (Leaf.relu, Leaf.sigmoid, Leaf.tanh):
                with self.subTest(shape=np.shape(values), activation=activation.__name__):
                    self.assertEqual(activation(Leaf(values)).data.shape, np.shape(values))

    def test_activation_modules_wrap_leaf_operations_without_parameters(self):
        values = np.array([-2.0, 0.0, 2.0])

        for module_type, method in ((ReLU, "relu"), (Sigmoid, "sigmoid"), (Tanh, "tanh")):
            with self.subTest(module=module_type.__name__):
                module = module_type()
                self.assertIsInstance(module, Module)
                self.assertEqual(module.parameters(), ())
                np.testing.assert_allclose(module(values).data, getattr(Leaf(values), method)().data)
                inputs = Leaf(values)
                self.assertIs(module(inputs)._stems[0], inputs)
