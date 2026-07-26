from collections.abc import Callable
from typing import List
from unittest import TestCase

from bonsaigrad import Leaf


def numerical_grad(build: Callable[..., Leaf], xs: List[float], eps: float = 1e-6) -> List[float]:
    """Finite-difference gradient of ``build``'s scalar output w.r.t. each input.

    Perturbs one input at a time and runs the forward pass only, giving a
    reference the analytic ``wire`` gradients can be checked against. Works for
    any op ``Leaf`` supports, since it drives the graph through ``build``.
    """
    grads: List[float] = []
    for i in range(len(xs)):
        hi, lo = list(xs), list(xs)
        hi[i] += eps
        lo[i] -= eps
        up = float(build(*[Leaf(x) for x in hi]).data)
        down = float(build(*[Leaf(x) for x in lo]).data)
        grads.append((up - down) / (2 * eps))
    return grads


def assert_grads(test: TestCase, build: Callable[..., Leaf], xs: List[float], places: int = 5) -> None:
    """Assert ``wire``'s gradient on every root matches the numerical estimate."""
    roots = [Leaf(x) for x in xs]
    build(*roots).wire()
    for root, expected in zip(roots, numerical_grad(build, xs)):
        test.assertAlmostEqual(float(root.grad), expected, places=places)
