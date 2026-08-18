from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from ._normalization import _Normalization
from ..._leaf import Leaf


class BatchNorm1d(_Normalization):

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(num_features=num_features, eps=eps)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between 0 and 1")

        self.momentum: float = momentum
        self.running_mean: np.ndarray = np.zeros(num_features)
        self.running_var: np.ndarray = np.ones(num_features)

    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        if inputs.data.ndim < 2:
            raise ValueError(f"BatchNorm1d needs inputs of 2 axes or more, got shape {inputs.data.shape}")
        if inputs.data.shape[-1] != self.num_features:
            raise ValueError(f"BatchNorm1d with {self.num_features} features got shape {inputs.data.shape}")

        axes: Tuple[int] = tuple(range(inputs.data.ndim - 1))
        if self.training:
            mean = inputs.mean(axis=axes, keepdims=True)
            variance = ((inputs - mean) ** 2).mean(axis=axes, keepdims=True)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean.data.squeeze()
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * variance.data.squeeze()
        else:
            mean = Leaf(self.running_mean)
            variance = Leaf(self.running_var)

        normalized = (inputs - mean) / (variance + self.eps) ** 0.5
        return self.gamma * normalized + self.beta
