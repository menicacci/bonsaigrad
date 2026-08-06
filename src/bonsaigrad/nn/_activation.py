from __future__ import annotations

from numpy.typing import ArrayLike

from ._module import Module
from .._leaf import Leaf


class ReLU(Module):
    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        return inputs.relu()


class Sigmoid(Module):
    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        return inputs.sigmoid()


class Tanh(Module):
    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        return inputs.tanh()
