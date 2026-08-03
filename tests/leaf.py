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

    def test_neg(self):
        self.assertEqual(float((-Leaf(2.0)).data), -2.0)

    def test_sub(self):
        self.assertEqual(float((Leaf(5.0) - Leaf(3.0)).data), 2.0)

    def test_pow(self):
        self.assertEqual(float((Leaf(2.0) ** 3).data), 8.0)

    def test_truediv(self):
        self.assertEqual(float((Leaf(6.0) / Leaf(4.0)).data), 1.5)

    def test_log(self):
        self.assertAlmostEqual(float(Leaf(np.e).log().data), 1.0)

    def test_exp(self):
        self.assertAlmostEqual(float(Leaf(1.0).exp().data), np.e)

    def test_derived_ops_desugar(self):
        # `-`/`/` add no primitives: they are built from `+`, `*` and `**`.
        self.assertEqual((Leaf(5.0) - Leaf(3.0))._op, "+")
        self.assertEqual((Leaf(6.0) / Leaf(4.0))._op, "*")
        self.assertEqual((Leaf(6.0) / Leaf(4.0))._stems[1]._op, "**-1")

    def test_pow_leaf_exponent_desugars(self):
        # aᵏ with k a Leaf is e^(k·ln a) — an `exp` apex, no backward rule of
        # its own. A constant exponent stays primitive, so it keeps its `**`.
        out = Leaf(2.0) ** Leaf(3.0)
        self.assertEqual(out._op, "exp")
        self.assertAlmostEqual(float(out.data), 8.0)
        self.assertEqual((Leaf(2.0) ** 3)._op, "**3")

    def test_wraps_raw_operands(self):
        self.assertEqual(float((Leaf(2.0) + 3.0).data), 5.0)
        self.assertEqual(float((Leaf(2.0) * 3.0).data), 6.0)
        self.assertEqual(float((Leaf(2.0) - 3.0).data), -1.0)
        self.assertEqual(float((Leaf(6.0) / 3.0).data), 2.0)

    def test_reflected_ops(self):
        # `2.0 + leaf` / `2.0 * leaf` route through __radd__ / __rmul__.
        self.assertEqual(float((3.0 + Leaf(2.0)).data), 5.0)
        self.assertEqual(float((3.0 * Leaf(2.0)).data), 6.0)
        self.assertEqual(float((3.0 - Leaf(2.0)).data), 1.0)
        self.assertEqual(float((3.0 / Leaf(2.0)).data), 1.5)
        self.assertAlmostEqual(float((3.0 ** Leaf(2.0)).data), 9.0)

    def test_reflected_ops_with_numpy_operands(self):
        # NumPy must defer to __radd__/__rmul__ rather than broadcasting the
        # Leaf into an object array, which would drop it out of the graph.
        self.assertIsInstance(np.float64(3.0) + Leaf(2.0), Leaf)
        self.assertIsInstance(np.array([1.0, 2.0]) * Leaf(3.0), Leaf)
        self.assertIsInstance(np.array([1.0, 2.0]) - Leaf(3.0), Leaf)
        self.assertIsInstance(np.array([1.0, 2.0]) / Leaf(3.0), Leaf)
        self.assertIsInstance(np.array([1.0, 2.0]) ** Leaf(3.0), Leaf)

    def test_add_seeds_ones(self):
        # ∂(a+b)/∂a = ∂(a+b)/∂b = 1, and the apex is seeded with grad 1.
        a, b = Leaf(2.0), Leaf(3.0)
        out = a + b
        out.wire()
        self.assertEqual(float(out.grad), 1.0)
        self.assertEqual(float(a.grad), 1.0)
        self.assertEqual(float(b.grad), 1.0)

    def test_mul_gradients(self):
        # ∂(a·b)/∂a = b, ∂(a·b)/∂b = a
        a, b = Leaf(2.0), Leaf(3.0)
        (a * b).wire()
        self.assertEqual(float(a.grad), 3.0)
        self.assertEqual(float(b.grad), 2.0)

    def test_reused_node_accumulates(self):
        # a + a: gradient must accumulate both edges, not overwrite → 2.
        a = Leaf(4.0)
        (a + a).wire()
        self.assertEqual(float(a.grad), 2.0)

    def test_squared_node_accumulates(self):
        # a * a: ∂(a²)/∂a = 2a → 8 at a=4.
        a = Leaf(4.0)
        (a * a).wire()
        self.assertEqual(float(a.grad), 8.0)

    def test_rest_zeroes_whole_graph(self):
        # Zeroing reaches the apex and every node below it, not just the roots.
        a, b = Leaf(2.0), Leaf(3.0)
        out = a * b
        out.wire()
        out.rest()
        self.assertEqual(float(out.grad), 0.0)
        self.assertEqual(float(a.grad), 0.0)
        self.assertEqual(float(b.grad), 0.0)

    def test_rest_makes_wire_repeatable(self):
        # Without resting, a second wire would double every stem's gradient.
        a, b = Leaf(2.0), Leaf(3.0)
        out = a * b
        out.wire()
        out.rest()
        out.wire()
        self.assertEqual(float(a.grad), 3.0)
        self.assertEqual(float(b.grad), 2.0)

    def test_grad_check_affine(self):
        assert_grads(self, lambda a, b: a * b + a, [5.0, -2.0])

    def test_grad_check_compound(self):
        # (a + b) * (b + c): b fans out into both factors, exercising accumulation.
        assert_grads(self, lambda a, b, c: (a + b) * (b + c), [2.0, -3.0, 4.0])

    def test_grad_check_deep(self):
        assert_grads(self, lambda a, b, c, d: (a * b + c) * (a + d), [1.5, -2.0, 0.5, 3.0])

    def test_grad_check_sub(self):
        # ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1 — the sign comes out of the `· -1`.
        assert_grads(self, lambda a, b: a - b, [2.0, 5.0])

    def test_grad_check_pow(self):
        # A negative base, which the `e^(k·ln a)` desugaring could not handle.
        assert_grads(self, lambda a: a ** 3, [-1.5])

    def test_grad_check_div(self):
        # ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
        assert_grads(self, lambda a, b: a / b, [3.0, 2.0])

    def test_grad_check_log(self):
        assert_grads(self, lambda a: a.log(), [2.5])

    def test_grad_check_exp(self):
        assert_grads(self, lambda a: a.exp(), [-0.5])

    def test_exp_grad_is_its_own_output(self):
        # ∂(eᵃ)/∂a = eᵃ, so the gradient reaching `a` *is* the forward value.
        a = Leaf(2.0)
        out = a.exp()
        out.wire()
        self.assertAlmostEqual(float(a.grad), float(out.data))

    def test_grad_check_log_exp_roundtrip(self):
        # e^(ln a) = a, so the gradient must come back as 1 through two ops.
        assert_grads(self, lambda a: a.log().exp(), [3.0])

    def test_grad_check_pow_leaf_exponent(self):
        # ∂(aᵏ)/∂a = k·aᵏ⁻¹ and ∂(aᵏ)/∂k = aᵏ·ln a, neither written by hand:
        # both fall out of `*`, `log` and `exp`.
        assert_grads(self, lambda a, b: a ** b, [2.0, 1.5])

    def test_grad_check_pow_reflected(self):
        assert_grads(self, lambda a: 2.0 ** a, [1.5])

    def test_grad_check_derived_compound(self):
        assert_grads(self, lambda a, b: (a - b) / (a ** 2 + 1.0), [1.5, -0.5])
