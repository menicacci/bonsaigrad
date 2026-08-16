from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

from .._leaf import Leaf


class Module(ABC):
    """Base class for a trainable neural-network component."""

    training: bool = True

    def __init__(self) -> None:
        self.training: bool = True

    def __call__(self, *args: Any, **kwargs: Any) -> Leaf:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Leaf:
        ...

    def parameters(self) -> Tuple[Leaf, ...]:
        return ()

    def train(self) -> Module:
        self.training = True
        return self

    def eval(self) -> Module:
        self.training = False
        return self
