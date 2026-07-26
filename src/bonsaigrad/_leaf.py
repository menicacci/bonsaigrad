from __future__ import annotations

from collections.abc import Callable
from typing import Tuple, List, Set

import numpy as np
from numpy.typing import ArrayLike


class Leaf:
    """A node in the computation graph.

    Parameters
    ----------
    data : array_like
        The value this node carries. Coerced to a ``float64`` ``np.ndarray``.
    _stems : Tuple[Leaf, ...]
        The nodes this one grew from. Internal; set by the ops.
    _op : str
        Label of the operation that produced this node (e.g. ``"+"``). Internal;
        used only for readable ``repr``s and, later, graph visualisation.
    """

    __array_ufunc__ = None

    def __init__(self, data: ArrayLike, _stems: Tuple[Leaf, ...] = (), _op: str = ""):
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._stems: Tuple[Leaf, ...] = _stems
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

    def _topo(self) -> List[Leaf]:
        """This node and every node below it, each listed after its own stems."""
        topo: List[Leaf] = []
        visited: Set[Leaf] = set()

        def build(node: Leaf) -> None:
            if node not in visited:
                visited.add(node)
                for stem in node._stems:
                    build(stem)
                topo.append(node)

        build(self)
        return topo

    def wire(self) -> None:
        topo = self._topo()

        self.grad = np.ones_like(self.data)
        for n in reversed(topo):
            n._backward()

    def rest(self) -> None:
        """Zero the gradients on this apex and every node below it."""
        for n in self._topo():
            n.grad = np.zeros_like(n.data)

    def __repr__(self) -> str:
        return f"Leaf(data={self.data}, grad={self.grad})"
