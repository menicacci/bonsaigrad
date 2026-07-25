from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


class Leaf:
    """A node in the computation graph.

    Parameters
    ----------
    data : array_like
        The value this node carries. Coerced to a ``float64`` ``np.ndarray``.
    _children : tuple[Leaf, ...]
        The parent nodes this one was built from. Internal; set by the ops.
    _op : str
        Label of the operation that produced this node (e.g. ``"+"``). Internal;
        used only for readable ``repr``s and, later, graph visualisation.
    """

    __array_ufunc__ = None

    def __init__(self, data: ArrayLike, _children: tuple[Leaf, ...] = (), _op: str = ""):
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple[Leaf, ...] = _children
        self._op: str = _op

    @staticmethod
    def _wrap(other: Leaf | ArrayLike) -> Leaf:
        return other if isinstance(other, Leaf) else Leaf(other)

    def __add__(self, other: Leaf | ArrayLike) -> Leaf:
        other: Leaf = self._wrap(other)
        out = Leaf(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Leaf | ArrayLike) -> Leaf:
        other: Leaf = self._wrap(other)
        out = Leaf(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def wire(self) -> None:
        topo: list[Leaf] = []
        visited: set[Leaf] = set()

        def build(node: Leaf) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)

        self.grad = np.ones_like(self.data)
        for n in reversed(topo):
            n._backward()

    def __repr__(self) -> str:
        return f"Leaf(data={self.data}, grad={self.grad})"
