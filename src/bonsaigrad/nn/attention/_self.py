from typing import Optional, Tuple, List

import numpy as np
from numpy.typing import ArrayLike

from .._linear import Linear
from .._module import Module
from ..._leaf import Leaf


class SelfAttention(Module):

    def __init__(self, embedding_dim: int, head_dim: Optional[int] = None, causal: bool = True):
        super().__init__()
        if embedding_dim < 1 or (head_dim is not None and head_dim < 1):
            raise ValueError("embedding_dim and head_dim must be positive")

        self.embedding_dim: int = embedding_dim
        self.head_dim: int = head_dim if head_dim is not None else max(1, embedding_dim // 4)
        self.causal: bool = causal

        self.Q: Linear = Linear(embedding_dim, self.head_dim)
        self.K: Linear = Linear(embedding_dim, self.head_dim)
        self.V: Linear = Linear(embedding_dim, self.head_dim)

    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)

        scores: Leaf = self.Q(inputs) @ self.K(inputs).transpose(-2, -1) / np.sqrt(self.head_dim)
        if self.causal:
            T: int = inputs.data.shape[-2]
            allowed = np.tril(np.ones((T, T), dtype=bool))
            scores = scores + np.where(allowed, 0.0, -np.inf)

        return scores.softmax(axis=-1) @ self.V(inputs)

    def parameters(self) -> Tuple[Leaf, ...]:
        return self.Q.parameters() + self.K.parameters() + self.V.parameters()


class MultiHeadSelfAttention(Module):

    def __init__(self, num_head: int, embedding_dim: int, head_dim: Optional[int] = None, causal: bool = True):
        super().__init__()
        self.num_head: int = num_head
        self.embedding_dim: int = embedding_dim
        self.head_dim: int = head_dim if head_dim is not None else max(1, embedding_dim // 4)
        self.causal: bool = causal

        self.heads: List[SelfAttention] = [SelfAttention(embedding_dim, head_dim, causal) for _ in range(num_head)]
        self.linear = Linear(num_head * self.head_dim, embedding_dim)

    def forward(self, inputs: Leaf | ArrayLike) -> Leaf:
        inputs = inputs if isinstance(inputs, Leaf) else Leaf(inputs)
        return self.linear(Leaf.concat([h(inputs) for h in self.heads]))

    def parameters(self) -> Tuple[Leaf, ...]:
        return tuple(param for h in self.heads for param in h.parameters()) + self.linear.parameters()
