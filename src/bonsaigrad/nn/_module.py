from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

from .._leaf import Leaf


class Module(ABC):
    """Base class for a trainable neural-network component."""

    def __call__(self, *args: Any, **kwargs: Any) -> Leaf:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Leaf:
        ...

    def parameters(self) -> Tuple[Leaf, ...]:
        return ()
