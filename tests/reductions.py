import unittest

import numpy as np

from bonsaigrad import Leaf
from ._base import assert_grads


class TestReductions(unittest.TestCase):
    def test_sum_matches_numpy(self):
        values = np.arange(6.0).reshape(2, 3)
        np.testing.assert_array_equal(Leaf(values).sum(axis=1).data, values.sum(axis=1))

    def test_sum_all_entries_sends_one_to_each_entry(self):
        values = Leaf([[1.0, 2.0], [3.0, 4.0]])
        values.sum().wire()
        np.testing.assert_array_equal(values.grad, np.ones((2, 2)))

    def test_sum_axis_restores_removed_axis_in_backward_pass(self):
        # Each row total is used with a different scale, so that scale must reach
        # every entry in the row once the reduced axis is put back.
        values = Leaf([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weights = Leaf([10.0, 100.0])
        (values.sum(axis=1) * weights).wire()
        np.testing.assert_array_equal(values.grad, [[10.0, 10.0, 10.0], [100.0, 100.0, 100.0]])

    def test_sum_keepdims_preserves_reduced_axes(self):
        values = Leaf(np.ones((2, 3, 4)))
        out = values.sum(axis=(0, -1), keepdims=True)
        self.assertEqual(out.data.shape, (1, 3, 1))
        out.wire()
        np.testing.assert_array_equal(values.grad, np.ones((2, 3, 4)))

    def test_mean_is_sum_divided_by_entry_count(self):
        values = Leaf([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        out = values.mean(axis=1)
        np.testing.assert_array_equal(out.data, [3.0, 4.0])
        out.wire()
        np.testing.assert_array_equal(values.grad, np.full((2, 3), 1 / 3))

    def test_grad_check_sum_and_mean(self):
        assert_grads(self, lambda values: values.sum(axis=(0, 2)), [np.arange(24.0).reshape(2, 3, 4)])
        assert_grads(self, lambda values: values.mean(axis=-1), [np.arange(24.0).reshape(2, 3, 4)])

    def test_mean_composes_with_other_operations(self):
        values = Leaf([1.0, 3.0, 4.0])
        weights = Leaf([2.0, 1.0, -1.0])
        out = (values * weights).mean()
        out.wire()
        self.assertEqual(out.data.shape, ())
        np.testing.assert_array_equal(values.grad, [2 / 3, 1 / 3, -1 / 3])

    def test_logsumexp_is_stable_for_large_values(self):
        values = np.array([1000.0, 1001.0, 1002.0])
        expected = 1002.0 + np.log(np.exp(-2.0) + np.exp(-1.0) + 1.0)
        np.testing.assert_allclose(Leaf(values).logsumexp().data, expected)

    def test_logsumexp_axis_and_keepdims_match_reduction_shapes(self):
        values = Leaf(np.arange(24.0).reshape(2, 3, 4))
        out = values.logsumexp(axis=(0, -1), keepdims=True)
        self.assertEqual(out.data.shape, (1, 3, 1))

        maximum = values.data.max(axis=(0, 2), keepdims=True)
        expected = np.log(np.exp(values.data - maximum).sum(axis=(0, 2), keepdims=True)) + maximum
        np.testing.assert_allclose(out.data, expected)

    def test_logsumexp_backward_is_softmax(self):
        values = Leaf([[1.0, 2.0, 3.0], [-1.0, 1.0, 0.0]])
        weights = Leaf([2.0, -3.0])
        (values.logsumexp(axis=1) * weights).wire()

        shifted = np.exp(values.data - values.data.max(axis=1, keepdims=True))
        expected = shifted / shifted.sum(axis=1, keepdims=True) * weights.data[:, None]
        np.testing.assert_allclose(values.grad, expected)

    def test_grad_check_logsumexp(self):
        values = np.linspace(-2.0, 2.0, 24).reshape(2, 3, 4)
        assert_grads(self, lambda x: x.logsumexp(), [values])
        assert_grads(self, lambda x: x.logsumexp(axis=(0, -1)), [values])
        assert_grads(self, lambda x: x.logsumexp(axis=1, keepdims=True), [values])
