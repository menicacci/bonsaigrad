"""Character tokenization."""

from __future__ import annotations

from typing import Tuple, Dict, List, Set

import numpy as np
from numpy.typing import ArrayLike


class CharacterTokenizer:
    """Tokenize characters and map them to stable integer token IDs."""

    PAD_TOKEN = "<PAD>"
    EOS_TOKEN = "<EOS>"

    SPECIAL_TOKENS: Tuple[str, ...] = (PAD_TOKEN, EOS_TOKEN)

    def __init__(self, characters: Set[str]):
        self.tokens: Tuple[str, ...] = (*self.SPECIAL_TOKENS, *sorted(characters))
        self._ids_by_token: Dict[str, int] = {token: index for index, token in enumerate(self.tokens)}

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self._ids_by_token[self.PAD_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._ids_by_token[self.EOS_TOKEN]

    def encode(self, text: str) -> List[int]:
        """Return the token ID of each character in ``text`` as a plain list."""
        unknown = set(text).difference(self._ids_by_token)
        if unknown:
            raise ValueError(f"text contains characters outside this tokenizer: {unknown}")
        return [self._ids_by_token[character] for character in text]

    def decode(self, token_ids: ArrayLike, *, skip_special_tokens: bool = False) -> str:
        """Turn token IDs back into text, optionally omitting structural tokens."""
        token_ids = np.asarray(token_ids)
        if np.any((token_ids < 0) | (token_ids >= self.size)):
            raise ValueError(f"token IDs must be between 0 and {self.size - 1}")

        tokens = (self.tokens[int(token_id)] for token_id in token_ids.flat)
        return "".join(token for token in tokens if not skip_special_tokens or token not in self.SPECIAL_TOKENS)
