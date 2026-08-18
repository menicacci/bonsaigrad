from abc import ABC

import numpy as np

from .._module import Module
from ..._leaf import Leaf


class _Normalization(Module, ABC):
    def __init__(self, num_features: int, eps: float):
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.num_features: int = num_features
        self.eps: float = eps
        self.gamma: Leaf = Leaf(np.ones(num_features))
        self.beta: Leaf = Leaf(np.zeros(num_features))

    def parameters(self):
        return self.gamma, self.beta
