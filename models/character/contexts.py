from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

import numpy as np

from .tokenizer import CharacterTokenizer


def left_pad_context(history: np.ndarray, context_length: int, pad_id: int) -> np.ndarray:
    """Keep the newest history tokens in a left-padded fixed-width context."""
    context = np.full(context_length, pad_id, dtype=np.intp)
    history = history[-context_length:]
    if history.size:
        context[-history.size:] = history
    return context


def make_examples(phrases: Iterable[str], tokenizer: CharacterTokenizer,
                  context_length: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Build left-padded inputs and their next-token targets."""
    inputs, targets = [], []
    for phrase in phrases:
        sequence = np.asarray([*tokenizer.encode(phrase), tokenizer.eos_id], dtype=np.intp)
        for target_position in range(sequence.size):
            inputs.append(left_pad_context(sequence[:target_position], context_length, tokenizer.pad_id))
            targets.append(sequence[target_position])

    return np.stack(inputs), np.asarray(targets, dtype=np.intp)
