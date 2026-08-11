import unittest

import numpy as np

from bonsaigrad import Leaf
from bonsaigrad.nn import CrossEntropyLoss, MSELoss, Module
from ._base import assert_grads


class TestMSELoss(unittest.TestCase):
    def test_forward_is_mean_squared_difference(self):
        predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([[0.0, 2.0], [5.0, 1.0]])

        loss = MSELoss()(predictions, targets)

        np.testing.assert_allclose(loss.data, 3.5)
        self.assertEqual(loss.data.shape, ())

    def test_backward_is_twice_the_error_averaged_over_entries(self):
        predictions = Leaf([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([[0.0, 2.0], [5.0, 1.0]])

        MSELoss()(predictions, targets).wire()

        expected = 2.0 * (predictions.data - targets) / predictions.data.size
        np.testing.assert_allclose(predictions.grad, expected)

    def test_supports_broadcast_targets(self):
        predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([1.0, 3.0])

        loss = MSELoss()(predictions, targets)

        np.testing.assert_allclose(loss.data, 1.5)

    def test_is_a_parameterless_module_and_accepts_raw_predictions(self):
        loss = MSELoss()

        self.assertIsInstance(loss, Module)
        self.assertEqual(loss.parameters(), ())
        self.assertIsInstance(loss([1.0, 2.0], [0.0, 2.0]), Leaf)

    def test_grad_check(self):
        predictions = np.arange(6.0).reshape(2, 3) / 2.0
        targets = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])

        assert_grads(self, lambda x: MSELoss()(x, targets), [predictions])


class TestCrossEntropyLoss(unittest.TestCase):
    def test_forward_is_mean_negative_log_probability_of_correct_classes(self):
        logits = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
        targets = np.array([0, 1])

        loss = CrossEntropyLoss()(logits, targets)

        probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        expected = -np.log(probabilities[[0, 1], targets]).mean()
        np.testing.assert_allclose(loss.data, expected)
        self.assertEqual(loss.data.shape, ())

    def test_backward_is_softmax_minus_one_hot_averaged_over_examples(self):
        logits = Leaf([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
        targets = np.array([0, 1])

        CrossEntropyLoss()(logits, targets).wire()

        shifted = np.exp(logits.data - logits.data.max(axis=-1, keepdims=True))
        expected = shifted / shifted.sum(axis=-1, keepdims=True)
        expected[[0, 1], targets] -= 1.0
        expected /= targets.size
        np.testing.assert_allclose(logits.grad, expected)

    def test_supports_leading_axes_before_the_class_axis(self):
        logits = np.arange(24.0).reshape(2, 4, 3) / 4.0
        targets = np.array([[0, 1, 2, 0], [2, 1, 0, 2]])

        loss = CrossEntropyLoss()(logits, targets)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_normalizers = np.log(np.exp(shifted).sum(axis=-1)) + logits.max(axis=-1)
        batch, position = np.indices(targets.shape)
        expected = (log_normalizers - logits[batch, position, targets]).mean()
        np.testing.assert_allclose(loss.data, expected)

    def test_supports_one_unbatched_classification(self):
        loss = CrossEntropyLoss()([2.0, 1.0, 0.0], 0)

        expected = np.log(1.0 + np.exp(-1.0) + np.exp(-2.0))
        np.testing.assert_allclose(loss.data, expected)

    def test_stays_finite_for_large_logits(self):
        logits = [[1_000.0, 1_001.0, 1_002.0], [-1_002.0, -1_001.0, -1_000.0]]
        targets = [2, 0]

        loss = CrossEntropyLoss()(logits, targets)

        self.assertTrue(np.isfinite(loss.data))
        log_normalizer = np.log(np.exp(-2.0) + np.exp(-1.0) + 1.0)
        expected = (log_normalizer + (log_normalizer + 2.0)) / 2.0
        np.testing.assert_allclose(loss.data, expected)

    def test_is_a_parameterless_module_and_accepts_raw_logits(self):
        loss = CrossEntropyLoss()

        self.assertIsInstance(loss, Module)
        self.assertEqual(loss.parameters(), ())
        self.assertIsInstance(loss([[1.0, 2.0]], [1]), Leaf)

    def test_grad_check(self):
        logits = np.linspace(-2.0, 2.0, 24).reshape(2, 4, 3)
        targets = np.array([[0, 1, 2, 0], [2, 1, 0, 2]])

        assert_grads(self, lambda x: CrossEntropyLoss()(x, targets), [logits])

    def test_validates_target_shape_type_and_range(self):
        loss = CrossEntropyLoss()

        with self.assertRaises(ValueError):
            loss(np.ones((2, 3)), [0])
        with self.assertRaises(TypeError):
            loss(np.ones((2, 3)), [0.0, 1.0])
        with self.assertRaises(ValueError):
            loss(np.ones((2, 3)), [0, 3])
