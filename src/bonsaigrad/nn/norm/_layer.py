from numpy.typing import ArrayLike

from ._normalization import _Normalization
from ..._leaf import Leaf


class LayerNorm(_Normalization):

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__(num_features=num_features, eps=eps)

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
