from __future__ import annotations

from collections.abc import Iterable

from ._optimizer import Optimizer
from .._leaf import Leaf


class SGD(Optimizer):

    def __init__(self, parameters: Iterable[Leaf], learning_rate: float):
        super().__init__(parameters)
        if not learning_rate > 0.0:
            raise ValueError("learning_rate must be positive")

        self.learning_rate: float = learning_rate

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad
