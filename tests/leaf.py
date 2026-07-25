import unittest

import numpy as np

from bonsaigrad import Leaf
from ._base import assert_grads


class TestLeaf(unittest.TestCase):
    def test_coerces_to_float64_ndarray(self):
        leaf = Leaf(3)
        self.assertIsInstance(leaf.data, np.ndarray)
        self.assertEqual(leaf.data.dtype, np.float64)

    def test_grad_starts_at_zero(self):
        leaf = Leaf(3.0)
        self.assertEqual(float(leaf.grad), 0.0)
        self.assertEqual(leaf.grad.shape, leaf.data.shape)

    def test_repr(self):
        self.assertEqual(repr(Leaf(2.0)), "Leaf(data=2.0, grad=0.0)")

    def test_add(self):
        self.assertEqual(float((Leaf(2.0) + Leaf(3.0)).data), 5.0)

    def test_mul(self):
        self.assertEqual(float((Leaf(2.0) * Leaf(3.0)).data), 6.0)

    def test_wraps_raw_operands(self):
        self.assertEqual(float((Leaf(2.0) + 3.0).data), 5.0)
        self.assertEqual(float((Leaf(2.0) * 3.0).data), 6.0)

    def test_reflected_ops(self):
        # `2.0 + leaf` / `2.0 * leaf` route through __radd__ / __rmul__.
        self.assertEqual(float((3.0 + Leaf(2.0)).data), 5.0)
        self.assertEqual(float((3.0 * Leaf(2.0)).data), 6.0)

    def test_reflected_ops_with_numpy_operands(self):
        # NumPy must defer to __radd__/__rmul__ rather than broadcasting the
        # Leaf into an object array, which would drop it out of the graph.
        self.assertIsInstance(np.float64(3.0) + Leaf(2.0), Leaf)
        self.assertIsInstance(np.array([1.0, 2.0]) * Leaf(3.0), Leaf)

    def test_add_seeds_ones(self):
        # ∂(a+b)/∂a = ∂(a+b)/∂b = 1, and the root is seeded with grad 1.
        a, b = Leaf(2.0), Leaf(3.0)
        out = a + b
        out.bend()
        self.assertEqual(float(out.grad), 1.0)
        self.assertEqual(float(a.grad), 1.0)
        self.assertEqual(float(b.grad), 1.0)

    def test_mul_gradients(self):
        # ∂(a·b)/∂a = b, ∂(a·b)/∂b = a
        a, b = Leaf(2.0), Leaf(3.0)
        (a * b).bend()
        self.assertEqual(float(a.grad), 3.0)
        self.assertEqual(float(b.grad), 2.0)

    def test_reused_node_accumulates(self):
        # a + a: gradient must accumulate both edges, not overwrite → 2.
        a = Leaf(4.0)
        (a + a).bend()
        self.assertEqual(float(a.grad), 2.0)

    def test_squared_node_accumulates(self):
        # a * a: ∂(a²)/∂a = 2a → 8 at a=4.
        a = Leaf(4.0)
        (a * a).bend()
        self.assertEqual(float(a.grad), 8.0)

    def test_grad_check_affine(self):
        assert_grads(self, lambda a, b: a * b + a, [5.0, -2.0])

    def test_grad_check_compound(self):
        # (a + b) * (b + c): b fans out into both factors, exercising accumulation.
        assert_grads(self, lambda a, b, c: (a + b) * (b + c), [2.0, -3.0, 4.0])

    def test_grad_check_deep(self):
        assert_grads(self, lambda a, b, c, d: (a * b + c) * (a + d), [1.5, -2.0, 0.5, 3.0])
