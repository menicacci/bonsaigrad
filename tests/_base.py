from collections.abc import Callable, Sequence
from typing import List
from unittest import TestCase

import numpy as np
from numpy.typing import ArrayLike

from bonsaigrad import Leaf


def numerical_grad(build: Callable[..., Leaf], xs: Sequence[ArrayLike], eps: float = 1e-6) -> List[np.ndarray]:
    """Finite-difference gradient of ``build``'s summed output w.r.t. each input.

    Perturbs one *entry* of one input at a time and runs the forward pass only,
    giving a reference the analytic ``wire`` gradients can be checked against.
    Works for any op ``Leaf`` supports, since it drives the graph through
    ``build``, and for any shape, since it walks every index.

    The output is summed because that is what ``wire`` differentiates: seeding the
    apex with ones makes every entry contribute with weight 1, exactly as ``sum``
    does. For a scalar output — the usual case — summing changes nothing.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in xs]
    grads: List[np.ndarray] = []
    for i, arr in enumerate(arrays):
        grad = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            hi, lo = [a.copy() for a in arrays], [a.copy() for a in arrays]
            hi[i][idx] += eps
            lo[i][idx] -= eps
            up = build(*[Leaf(x) for x in hi]).data.sum()
            down = build(*[Leaf(x) for x in lo]).data.sum()
            grad[idx] = (up - down) / (2 * eps)
        grads.append(grad)
    return grads


def assert_grads(test: TestCase, build: Callable[..., Leaf], xs: Sequence[ArrayLike], places: int = 5) -> None:
    """Assert ``wire``'s gradient on every root matches the numerical estimate."""
    roots = [Leaf(x) for x in xs]
    build(*roots).wire()
    for root, expected in zip(roots, numerical_grad(build, xs)):
        test.assertEqual(root.grad.shape, root.data.shape)
        np.testing.assert_allclose(root.grad, expected, atol=10 ** -places)
