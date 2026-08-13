from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._module import Module
from .initializers import WeightInitializer, fan_in_normal
from .._leaf import Leaf


class Linear(Module):
    """A trainable affine layer.

    Parameters
    ----------
    in_features : int
        The width of each input's final axis.
    out_features : int
        The number of output features the layer produces.
    rng : Optional[np.random.Generator], optional
        The generator used to initialize ``weight``. When omitted, a fresh
        generator is created. Inputs keep any leading batch axes; only their
        final axis is transformed.
    initializer : WeightInitializer, optional
        The function used to initialize ``weight``. It receives the random
        generator, input width, and output width.
    """

    def __init__(self, in_features: int, out_features: int, rng: Optional[np.random.Generator] = None, *,
                 initializer: WeightInitializer = fan_in_normal):
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")

        rng = np.random.default_rng() if rng is None else rng
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight: Leaf = Leaf(initializer(rng, in_features, out_features))
        self.bias: Leaf = Leaf(np.zeros(out_features))

    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        if inputs.data.ndim < 2:
            raise ValueError(f"Linear needs inputs of 2 axes or more, got shape {inputs.data.shape}")
        if inputs.data.shape[-1] != self.in_features:
            raise ValueError(f"Linear with {self.in_features} input features got shape {inputs.data.shape}")

        return inputs @ self.weight + self.bias

    def parameters(self) -> Tuple[Leaf, ...]:
        return self.weight, self.bias
