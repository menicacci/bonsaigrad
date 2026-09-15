"""Training and sampling helpers for character language models."""

from typing import Optional, List

import numpy as np
from numpy.typing import ArrayLike

from bonsaigrad.nn import CrossEntropyLoss
from bonsaigrad.optim import Adam
from .contexts import left_pad_context
from .model import NextCharacterPredictor
from .tokenizer import CharacterTokenizer


def train(model: NextCharacterPredictor, contexts: ArrayLike, targets: ArrayLike, *, steps: int = 2_000,
          batch_size: int = 64, learning_rate: float = 0.01, rng: Optional[np.random.Generator] = None) -> List[float]:
    contexts, targets = np.asarray(contexts), np.asarray(targets)
    if contexts.ndim != 2 or contexts.shape[1] != model.context_length:
        raise ValueError(f"contexts must have shape (examples, {model.context_length}), got {contexts.shape}")
    if contexts.shape[0] < 1:
        raise ValueError("contexts must contain at least one example")
    if targets.shape != (contexts.shape[0],):
        raise ValueError(f"targets must have shape ({contexts.shape[0]},), got {targets.shape}")
    if steps < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")

    rng = np.random.default_rng() if rng is None else rng
    loss_function = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate)
    losses = []
    for _ in range(steps):
        batch_indices = rng.integers(contexts.shape[0], size=batch_size)
        loss = loss_function(model(contexts[batch_indices]), targets[batch_indices])

        optimizer.zero_grad()
        loss.wire()
        optimizer.step()
        losses.append(float(loss.data))
    return losses


def generate(model: NextCharacterPredictor, tokenizer: CharacterTokenizer, *, prompt: str = "",
             max_new_tokens: int = 80, rng: Optional[np.random.Generator] = None) -> str:
    if model.vocabulary_size != tokenizer.size:
        raise ValueError("model vocabulary size must match tokenizer size")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    rng = np.random.default_rng() if rng is None else rng
    history = tokenizer.encode(prompt)
    completion = list(prompt)
    for _ in range(max_new_tokens):
        context = left_pad_context(np.asarray(history, dtype=np.intp), model.context_length, tokenizer.pad_id)

        logits = model(context[None, :]).data[0]
        logits[tokenizer.pad_id] = -np.inf
        shifted_logits = logits - np.max(logits)
        probabilities = np.exp(shifted_logits)
        probabilities /= probabilities.sum()
        next_token = int(rng.choice(tokenizer.size, p=probabilities))
        if next_token == tokenizer.eos_id:
            break

        history.append(next_token)
        completion.append(tokenizer.tokens[next_token])
    return "".join(completion)
