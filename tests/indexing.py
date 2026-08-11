import unittest

import numpy as np

from bonsaigrad import Leaf
from ._base import assert_grads


class TestIndexing(unittest.TestCase):
    def test_basic_indexing_matches_numpy(self):
        values = np.arange(24.0).reshape(2, 3, 4)

        np.testing.assert_array_equal(Leaf(values)[1, :, ::2].data, values[1, :, ::2])

    def test_backward_returns_gradients_to_selected_entries(self):
        values = Leaf([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weights = Leaf([10.0, 100.0])

        (values[[0, 1], [2, 0]] * weights).sum().wire()

        np.testing.assert_array_equal(values.grad, [[0.0, 0.0, 10.0], [100.0, 0.0, 0.0]])

    def test_backward_accumulates_repeated_indices(self):
        values = Leaf([1.0, 2.0, 3.0])

        values[[0, 0, 2]].sum().wire()

        np.testing.assert_array_equal(values.grad, [2.0, 0.0, 1.0])

    def test_grad_check_indexing(self):
        values = np.arange(12.0).reshape(3, 4)

        assert_grads(self, lambda x: x[1:, ::2], [values])
        assert_grads(self, lambda x: x[[0, 0, 2], [1, 1, 3]], [values])
