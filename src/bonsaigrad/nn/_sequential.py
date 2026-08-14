from __future__ import annotations

from typing import Tuple

from numpy.typing import ArrayLike

from ._module import Module
from .._leaf import Leaf


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        if not all(isinstance(module, Module) for module in modules):
            raise TypeError("Sequential accepts Module instances only")

        self.modules: Tuple[Module, ...] = modules

    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        outputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        for m in self.modules:
            outputs = m(outputs)

        return outputs

    def parameters(self) -> Tuple[Leaf, ...]:
        return tuple(dict.fromkeys(p for m in self.modules for p in m.parameters()))

    def train(self) -> Sequential:
        super().train()
        for module in self.modules:
            module.train()
        return self

    def eval(self) -> Sequential:
        super().eval()
        for module in self.modules:
            module.eval()
        return self
