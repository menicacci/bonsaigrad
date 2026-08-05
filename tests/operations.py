import unittest

import numpy as np

from bonsaigrad import Leaf
from ._base import assert_grads


class TestOperations(unittest.TestCase):
    def test_matmul(self):
        a, b = np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_array_equal((Leaf(a) @ Leaf(b)).data, a @ b)

    def test_matmul_gradients(self):
        # ∂L/∂a = out.grad @ bᵀ, ∂L/∂b = aᵀ @ out.grad — each stem contracts the
        # incoming gradient against the other's transpose.
        a, b = Leaf(np.ones((2, 3))), Leaf(np.arange(12.0).reshape(3, 4))
        ones = np.ones((2, 4))
        (a @ b).wire()
        np.testing.assert_array_equal(a.grad, ones @ b.data.T)
        np.testing.assert_array_equal(b.grad, a.data.T @ ones)

    def test_matmul_batched_shares_weights(self):
        # (B, n, m) @ (m, p): one weight matrix serves the whole batch, so its
        # gradient sums over the batch axis — 5 batches × 2 rows = 10 per entry.
        x, w = Leaf(np.ones((5, 2, 3))), Leaf(np.ones((3, 4)))
        (x @ w).wire()
        self.assertEqual(x.grad.shape, (5, 2, 3))
        np.testing.assert_array_equal(w.grad, np.full((3, 4), 10.0))

    def test_matmul_rejects_1d(self):
        # A vector operand needs its own backward rule; refuse rather than guess.
        with self.assertRaises(ValueError):
            Leaf([[1.0, 2.0]]) @ Leaf([1.0, 2.0])
        with self.assertRaises(ValueError):
            Leaf([1.0, 2.0]) @ Leaf([[1.0], [2.0]])

    def test_matmul_reflected(self):
        # NumPy must defer to __rmatmul__ rather than taking `@` for itself.
        out = np.ones((2, 3)) @ Leaf(np.ones((3, 4)))
        self.assertIsInstance(out, Leaf)
        self.assertEqual(out.data.shape, (2, 4))

    def test_grad_check_matmul(self):
        assert_grads(self, lambda a, b: a @ b,
                     [[[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]],
                      [[1.0, -1.0], [2.0, 0.5], [-0.5, 1.5]]])

    def test_grad_check_linear_layer(self):
        # x @ w + b: a Linear layer, contraction and bias broadcast in one graph.
        assert_grads(self, lambda x, w, b: x @ w + b,
                     [[[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]],
                      [[1.0, -1.0], [2.0, 0.5], [-0.5, 1.5]],
                      [0.25, -0.75]])

    def test_grad_check_broadcast_elementwise(self):
        assert_grads(self, lambda x, w: (x * w - 1.0) / w,
                     [[[1.0, 2.0], [3.0, 4.0]], [10.0, 100.0]])
