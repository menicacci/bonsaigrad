from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from bonsaigrad import Leaf
from bonsaigrad.nn import Linear, Module
from .embedder import EmbedderCharacterModel


class NextCharacterPredictor(Module):

    def __init__(self, embedder: EmbedderCharacterModel, rng: Optional[np.random.Generator] = None):
        super().__init__()
        rng = rng or np.random.default_rng()
        self.embedder: EmbedderCharacterModel = embedder
        self.vocabulary_size: int = embedder.vocabulary_size
        self.context_length: int = embedder.context_length
        self.embedding_dim: int = embedder.embedding_dim
        self.hidden_features: int = embedder.hidden_features
        self.output: Linear = Linear(embedder.hidden_features, embedder.vocabulary_size, rng)

    def forward(self, contexts: ArrayLike) -> Leaf:
        return self.output(self.embedder(contexts))

    def parameters(self) -> Tuple[Leaf, ...]:
        return self.embedder.parameters() + self.output.parameters()
