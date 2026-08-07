from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._module import Module
from .._leaf import Leaf


class MSELoss(Module):
    def forward(self, predictions: Leaf | ArrayLike, targets: ArrayLike) -> Leaf:
        predictions = predictions if isinstance(predictions, Leaf) else Leaf(predictions)
        return ((predictions - targets) ** 2).mean()


class CrossEntropyLoss(Module):
    def forward(self, logits: Leaf | ArrayLike, targets: ArrayLike) -> Leaf:
        logits = logits if isinstance(logits, Leaf) else Leaf(logits)
        targets = np.asarray(targets)

        if logits.data.ndim < 1 or logits.data.shape[-1] < 1:
            raise ValueError(f"CrossEntropyLoss needs at least one class, got shape {logits.data.shape}")
        if targets.shape != logits.data.shape[:-1]:
            raise ValueError(
                f"targets must have shape {logits.data.shape[:-1]} for logits of shape {logits.data.shape}, "
                f"got {targets.shape}"
            )
        if not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("targets must contain integer class indices")
        if np.any((targets < 0) | (targets >= logits.data.shape[-1])):
            raise ValueError(f"targets must be between 0 and {logits.data.shape[-1] - 1}")

        coordinates = tuple(np.indices(targets.shape)) + (targets,)
        correct_logits = logits[coordinates]
        return (logits.logsumexp(axis=-1) - correct_logits).mean()
