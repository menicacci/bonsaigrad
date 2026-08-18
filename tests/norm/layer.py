import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import Module
from bonsaigrad.nn.norm import LayerNorm
from .._base import assert_grads


class TestLayerNorm(unittest.TestCase):

    def test_normalizes_each_final_axis_independently(self):
        layer = LayerNorm(3)
        inputs = np.arange(12.0).reshape(2, 2, 3)

        outputs = layer(inputs)

        mean = inputs.mean(axis=-1, keepdims=True)
        variance = ((inputs - mean) ** 2).mean(axis=-1, keepdims=True)
        expected = (inputs - mean) / np.sqrt(variance + layer.eps)
        self.assertIsInstance(layer, Module)
        np.testing.assert_allclose(outputs.data, expected)
        np.testing.assert_allclose(outputs.data.mean(axis=-1), np.zeros((2, 2)))
        np.testing.assert_allclose(outputs.data.var(axis=-1), np.ones((2, 2)), atol=2e-5)

    def test_input_gradients_match_finite_differences(self):
        layer = LayerNorm(3)
        inputs = np.array([[1.0, -2.0, 4.0], [0.5, 3.0, -1.0]])
        output_weights = np.array([[2.0, -1.0, 0.5], [3.0, 1.0, -2.0]])

        assert_grads(self, lambda values: layer(values) * output_weights, [inputs])

    def test_applies_trainable_scale_and_shift_and_backpropagates_to_them(self):
        layer = LayerNorm(3)
        layer.gamma.data[:] = [2.0, -3.0, 0.5]
        layer.beta.data[:] = [1.0, 4.0, -2.0]
        inputs = Leaf([[1.0, 2.0, 5.0], [3.0, 6.0, 9.0]])
        output_weights = np.array([[2.0, -1.0, 0.5], [0.5, 3.0, -2.0]])

        outputs = layer(inputs)
        (outputs * output_weights).sum().wire()

        mean = inputs.data.mean(axis=-1, keepdims=True)
        variance = ((inputs.data - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (inputs.data - mean) / np.sqrt(variance + layer.eps)
        self.assertEqual(layer.parameters(), (layer.gamma, layer.beta))
        np.testing.assert_allclose(outputs.data, layer.gamma.data * normalized + layer.beta.data)
        np.testing.assert_allclose(layer.gamma.grad, (output_weights * normalized).sum(axis=0))
        np.testing.assert_allclose(layer.beta.grad, output_weights.sum(axis=0))

    def test_rejects_invalid_configuration_and_feature_shape(self):
        for args in ((0,), (3, 0.0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    LayerNorm(*args)

        with self.assertRaises(ValueError):
            LayerNorm(3)(np.ones((2, 4)))
        with self.assertRaises(ValueError):
            LayerNorm(3)(1.0)
