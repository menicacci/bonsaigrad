from collections.abc import Callable
from unittest import TestCase

from bonsaigrad import Leaf

# A graph builder: takes one Leaf per input and returns the output Leaf.
Build = Callable[..., Leaf]


def numerical_grad(build: Build, xs: list[float], eps: float = 1e-6) -> list[float]:
    """Finite-difference gradient of ``build``'s scalar output w.r.t. each input.

    Perturbs one input at a time and runs the forward pass only, giving a
    reference the analytic ``bend`` gradients can be checked against. Works for
    any op ``Leaf`` supports, since it drives the graph through ``build``.
    """
    grads: list[float] = []
    for i in range(len(xs)):
        hi, lo = list(xs), list(xs)
        hi[i] += eps
        lo[i] -= eps
        up = float(build(*[Leaf(x) for x in hi]).data)
        down = float(build(*[Leaf(x) for x in lo]).data)
        grads.append((up - down) / (2 * eps))
    return grads


def assert_grads(test: TestCase, build: Build, xs: list[float], places: int = 5) -> None:
    """Assert ``bend``'s gradient on every input matches the numerical estimate."""
    leaves = [Leaf(x) for x in xs]
    build(*leaves).bend()
    for leaf, expected in zip(leaves, numerical_grad(build, xs)):
        test.assertAlmostEqual(float(leaf.grad), expected, places=places)
