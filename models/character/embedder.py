from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from bonsaigrad import Leaf
from bonsaigrad.nn import Embedding, Linear, Module, Sequential, Tanh
from bonsaigrad.nn.norm import LayerNorm


class EmbedderCharacterModel(Module):

    def __init__(self, vocabulary_size: int, *,
                 context_length: int = 64, embedding_dim: int = 16, hidden_features: int = 128,
                 number_of_layers: int = 3, rng: Optional[np.random.Generator] = None):
        super().__init__()
        if min(vocabulary_size, context_length, embedding_dim, hidden_features, number_of_layers) < 1:
            raise ValueError("vocabulary_size, context_length, embedding_dim, hidden_features, and "
                             "number_of_layers must be positive")

        self.vocabulary_size: int = vocabulary_size
        self.context_length: int = context_length
        self.embedding_dim: int = embedding_dim
        self.hidden_features: int = hidden_features
        self.number_of_layers: int = number_of_layers

        rng: np.random.Generator = np.random.default_rng() if rng is None else rng

        def layer(input_size: int) -> Tuple[Module, ...]:
            return Linear(input_size, hidden_features, rng=rng), LayerNorm(hidden_features), Tanh()

        self.embedding: Embedding = Embedding(vocabulary_size, embedding_dim, rng=rng)
        self.hidden_layers: Sequential = Sequential(
            *layer(context_length * embedding_dim),
            *(module for _ in range(number_of_layers - 1) for module in layer(hidden_features))
        )

    def forward(self, contexts: ArrayLike) -> Leaf:
        contexts = np.asarray(contexts)
        if contexts.ndim != 2 or contexts.shape[1] != self.context_length:
            raise ValueError(f"contexts must have shape (batch, {self.context_length}), got {contexts.shape}")
        if contexts.shape[0] < 1:
            raise ValueError("contexts must contain at least one example")

        vectors: Leaf = self.embedding(contexts)
        features: Leaf = vectors.reshape(contexts.shape[0], self.context_length * self.embedding_dim)
        return self.hidden_layers(features)

    def parameters(self) -> Tuple[Leaf, ...]:
        return self.embedding.parameters() + self.hidden_layers.parameters()
