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
