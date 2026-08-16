from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple

import numpy as np

from .._leaf import Leaf


class Optimizer(ABC):

    def __init__(self, parameters: Iterable[Leaf]):
        self.parameters: Tuple[Leaf, ...] = tuple(parameters)
        if not self.parameters:
            raise ValueError("Optimizer needs at least one parameter")
        if not all(isinstance(parameter, Leaf) for parameter in self.parameters):
            raise TypeError("Optimizer parameters must be Leaf instances")

    @abstractmethod
    def step(self) -> None:
        ...

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)
