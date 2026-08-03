import unittest

import numpy as np

from bonsaigrad import Leaf
from ._base import assert_grads


class TestShapes(unittest.TestCase):
    def test_unbroadcast_leaves_matching_shapes_alone(self):
        grad = np.ones((2, 3))
        self.assertIs(Leaf._unbroadcast(grad, (2, 3)), grad)

    def test_unbroadcast_sums_prepended_axes(self):
        # (3,) broadcast against (2, 3) gained a leading axis; sum it away.
        out = Leaf._unbroadcast(np.ones((2, 3)), (3,))
        np.testing.assert_array_equal(out, [2.0, 2.0, 2.0])

    def test_unbroadcast_sums_stretched_axes(self):
        # (1, 3) was stretched along axis 0; the axis stays, of length 1.
        out = Leaf._unbroadcast(np.ones((2, 3)), (1, 3))
        np.testing.assert_array_equal(out, [[2.0, 2.0, 2.0]])

    def test_unbroadcast_reduces_to_scalar(self):
        out = Leaf._unbroadcast(np.ones((2, 3)), ())
        self.assertEqual(np.shape(out), ())
        self.assertEqual(float(out), 6.0)

    def test_grad_keeps_stem_shape(self):
        # The footgun: a scalar times a vector used to raise on the backward pass,
        # because a (2,) gradient cannot be written into a () grad.
        a, b = Leaf(3.0), Leaf([1.0, 2.0])
        (a * b).wire()
        self.assertEqual(a.grad.shape, ())
        self.assertEqual(b.grad.shape, (2,))

    def test_broadcast_mul_gradients(self):
        # out = a·b with a scalar: ∂L/∂a sums b over the copies it fed, ∂L/∂b is a.
        a, b = Leaf(3.0), Leaf([1.0, 2.0])
        (a * b).wire()
        self.assertEqual(float(a.grad), 3.0)
        np.testing.assert_array_equal(b.grad, [3.0, 3.0])

    def test_broadcast_add_gradients(self):
        # (3,1) + (1,4) → (3,4): each entry of a feeds a whole row, each entry of
        # b a whole column, so their gradients are the row and column counts.
        a, b = Leaf([[1.0], [2.0], [3.0]]), Leaf([[10.0, 20.0, 30.0, 40.0]])
        (a + b).wire()
        np.testing.assert_array_equal(a.grad, [[4.0], [4.0], [4.0]])
        np.testing.assert_array_equal(b.grad, [[3.0, 3.0, 3.0, 3.0]])

    def test_broadcast_row_against_matrix(self):
        # A row vector scaling every row of a matrix — the shape of a bias term.
        x = Leaf([[1.0, 2.0], [3.0, 4.0]])
        w = Leaf([10.0, 100.0])
        (x * w).wire()
        np.testing.assert_array_equal(x.grad, [[10.0, 100.0], [10.0, 100.0]])
        np.testing.assert_array_equal(w.grad, [4.0, 6.0])  # column sums of x

    def test_broadcast_node_reused(self):
        # a is both broadcast *and* used twice: 2 copies × 2 paths → 4.
        a, b = Leaf(2.0), Leaf([1.0, 1.0])
        ((a + b) + a).wire()
        self.assertEqual(float(a.grad), 4.0)

    def test_grad_check_broadcast_elementwise(self):
        assert_grads(self, lambda x, w: x * w,
                     [[[1.0, 2.0], [3.0, 4.0]], [10.0, 100.0]])
