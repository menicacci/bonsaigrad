from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from bonsaigrad import Leaf
from bonsaigrad.nn import Module


class LayerNorm(Module):

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.num_features: int = num_features
        self.eps: float = eps
        self.gamma: Leaf = Leaf(np.ones(num_features))
        self.beta: Leaf = Leaf(np.zeros(num_features))

    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        if inputs.data.ndim < 1:
            raise ValueError(f"LayerNorm needs inputs of 1 axis or more, got shape {inputs.data.shape}")
        if inputs.data.shape[-1] != self.num_features:
            raise ValueError(f"LayerNorm with {self.num_features} features got shape {inputs.data.shape}")

        mean = inputs.mean(-1, keepdims=True)
        variance = ((inputs - mean) ** 2).mean(-1, keepdims=True)
        normalized = (inputs - mean) / (variance + self.eps) ** 0.5
        return self.gamma * normalized + self.beta

    def parameters(self) -> Tuple[Leaf, ...]:
        return self.gamma, self.beta
