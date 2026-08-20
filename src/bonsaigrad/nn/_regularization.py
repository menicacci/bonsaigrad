from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ._module import Module
from .._leaf import Leaf


class Dropout(Module):
    def __init__(self, p: float = 0.5, rng: Optional[np.random.Generator] = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be between 0 and 1")

        self.p: float = p
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        if not self.training:
            return inputs

        mask = self.rng.random(inputs.data.shape) >= self.p
        return inputs * mask / (1.0 - self.p)
