from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._module import Module
from .initializers import WeightInitializer, embedding_normal
from .._leaf import Leaf


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, rng: Optional[np.random.Generator] = None, *,
                 initializer: WeightInitializer = embedding_normal):
        super().__init__()
        if num_embeddings < 1 or embedding_dim < 1:
            raise ValueError("num_embeddings and embedding_dim must be positive")

        rng = np.random.default_rng() if rng is None else rng
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.weight: Leaf = Leaf(initializer(rng, num_embeddings, embedding_dim))

    def forward(self, indices: ArrayLike) -> Leaf:
        indices = np.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Embedding indices must be integers")
        if np.any((indices < 0) | (indices >= self.num_embeddings)):
            raise ValueError(f"Embedding indices must be between 0 and {self.num_embeddings - 1}")

        return self.weight[indices]

    def parameters(self) -> Tuple[Leaf, ...]:
        return self.weight,
