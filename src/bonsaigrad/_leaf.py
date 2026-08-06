from __future__ import annotations

from collections.abc import Callable
from typing import Tuple, List, Set, Optional, Union

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

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Sum ``grad`` back down to ``shape``, undoing a forward broadcast."""
        if grad.shape == shape:
            return grad
        grad = grad.sum(axis=tuple(range(grad.ndim - len(shape))))  # axes NumPy prepended
        stretched = tuple(i for i, n in enumerate(shape) if n == 1)  # axes stretched from 1
        return grad.sum(axis=stretched, keepdims=True) if stretched else grad

    def __add__(self, other: Leaf | ArrayLike) -> Leaf:
        other: Leaf = self._wrap(other)
        out = Leaf(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: Leaf | ArrayLike) -> Leaf:
        other: Leaf = self._wrap(other)
        out = Leaf(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, exponent: Leaf | int | float) -> Leaf:
        if isinstance(exponent, Leaf):
            return (exponent * self.log()).exp()

        out = Leaf(self.data ** exponent, (self,), f"**{exponent:g}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def log(self) -> Leaf:
        out = Leaf(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def exp(self) -> Leaf:
        out = Leaf(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Leaf:
        out = Leaf(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                axes: Tuple[int] = (axis,) if isinstance(axis, int) else axis
                for reduced_axis in sorted(a % self.data.ndim for a in axes):
                    grad = np.expand_dims(grad, reduced_axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Leaf:
        total = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            count = self.data.size
        else:
            axes: Tuple[int] = (axis,) if isinstance(axis, int) else axis
            count = np.prod([self.data.shape[a % self.data.ndim] for a in axes])
        return total / count

    def __matmul__(self, other: Leaf | ArrayLike) -> Leaf:
        other: Leaf = self._wrap(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError(f"@ needs operands of 2 axes or more, got shapes {self.data.shape} and {other.data.shape}")

        out = Leaf(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            self.grad += self._unbroadcast(out.grad @ other.data.swapaxes(-1, -2), self.data.shape)
            other.grad += self._unbroadcast(self.data.swapaxes(-1, -2) @ out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other: Leaf | ArrayLike) -> Leaf:
        return self._wrap(other) @ self

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self) -> Leaf:
        return self * -1.0

    def __sub__(self, other: Leaf | ArrayLike) -> Leaf:
        return self + (-self._wrap(other))

    def __rsub__(self, other: Leaf | ArrayLike) -> Leaf:
        return self._wrap(other) + (-self)

    def __truediv__(self, other: Leaf | ArrayLike) -> Leaf:
        return self * self._wrap(other) ** -1.0

    def __rtruediv__(self, other: Leaf | ArrayLike) -> Leaf:
        return self._wrap(other) * self ** -1.0

    def __rpow__(self, base: Leaf | ArrayLike) -> Leaf:
        return (self * self._wrap(base).log()).exp()

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
