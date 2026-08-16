import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import Module
from bonsaigrad.nn.norm import BatchNorm1d
from .._base import assert_grads


class TestBatchNorm1d(unittest.TestCase):

    def test_initializes_affine_parameters_and_running_statistics(self):
        layer = BatchNorm1d(3)

        self.assertIsInstance(layer, Module)
        self.assertTrue(layer.training)
        self.assertEqual(layer.parameters(), (layer.gamma, layer.beta))
        np.testing.assert_array_equal(layer.gamma.data, np.ones(3))
        np.testing.assert_array_equal(layer.beta.data, np.zeros(3))
        np.testing.assert_array_equal(layer.running_mean, np.zeros(3))
        np.testing.assert_array_equal(layer.running_var, np.ones(3))

    def test_normalizes_each_final_axis_feature_across_all_leading_axes(self):
        layer = BatchNorm1d(3, eps=1e-5, momentum=0.25)
        inputs = np.arange(12.0).reshape(2, 2, 3)

        outputs = layer(inputs)

        mean = inputs.mean(axis=(0, 1), keepdims=True)
        variance = ((inputs - mean) ** 2).mean(axis=(0, 1), keepdims=True)
        expected = (inputs - mean) / np.sqrt(variance + layer.eps)
        np.testing.assert_allclose(outputs.data, expected)
        np.testing.assert_allclose(layer.running_mean, 0.25 * mean.squeeze())
        np.testing.assert_allclose(layer.running_var, 0.75 + 0.25 * variance.squeeze())

    def test_applies_trainable_scale_and_shift_and_backpropagates_to_them(self):
        layer = BatchNorm1d(2)
        layer.gamma.data[:] = [2.0, -3.0]
        layer.beta.data[:] = [1.0, 4.0]
        inputs = Leaf([[1.0, 2.0], [3.0, 6.0]])
        output_weights = np.array([[2.0, -1.0], [0.5, 3.0]])

        outputs = layer(inputs)
        (outputs * output_weights).sum().wire()

        mean = inputs.data.mean(axis=0, keepdims=True)
        variance = ((inputs.data - mean) ** 2).mean(axis=0, keepdims=True)
        normalized = (inputs.data - mean) / np.sqrt(variance + layer.eps)
        np.testing.assert_allclose(outputs.data, layer.gamma.data * normalized + layer.beta.data)
        np.testing.assert_allclose(layer.gamma.grad, (output_weights * normalized).sum(axis=0))
        np.testing.assert_allclose(layer.beta.grad, output_weights.sum(axis=0))

    def test_input_gradients_match_finite_differences(self):
        layer = BatchNorm1d(2)
        inputs = np.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 2.0]])

        assert_grads(self, lambda values: layer(values), [inputs])

    def test_evaluation_uses_running_statistics_without_updating_them(self):
        layer = BatchNorm1d(2, eps=1e-5, momentum=0.5)
        layer([[2.0, 4.0], [6.0, 12.0]])
        running_mean = layer.running_mean.copy()
        running_var = layer.running_var.copy()

        self.assertIs(layer.eval(), layer)
        inputs = np.array([[10.0, 20.0], [14.0, 28.0]])
        outputs = layer(inputs)

        expected = (inputs - running_mean) / np.sqrt(running_var + layer.eps)
        np.testing.assert_allclose(outputs.data, expected)
        np.testing.assert_array_equal(layer.running_mean, running_mean)
        np.testing.assert_array_equal(layer.running_var, running_var)
        self.assertIs(layer.train(), layer)
        self.assertTrue(layer.training)

    def test_rejects_invalid_configuration_and_input_shapes(self):
        for args in ((0,), (2, 0.0), (2, 1e-5, -0.1), (2, 1e-5, 1.1)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    BatchNorm1d(*args)

        layer = BatchNorm1d(2)
        with self.assertRaises(ValueError):
            layer([1.0, 2.0])
        with self.assertRaises(ValueError):
            layer(np.ones((3, 4)))
