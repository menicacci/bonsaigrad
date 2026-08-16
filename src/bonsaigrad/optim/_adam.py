from __future__ import annotations

from collections.abc import Iterable
from typing import List

import numpy as np

from ._optimizer import Optimizer
from .._leaf import Leaf


class Adam(Optimizer):

    def __init__(self, parameters: Iterable[Leaf], learning_rate: float = 0.001, *,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(parameters)
        if not learning_rate > 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if not epsilon > 0.0:
            raise ValueError("epsilon must be positive")

        self.learning_rate: float = learning_rate
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.steps: int = 0
        self.first_moments: List[np.ndarray] = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.second_moments: List[np.ndarray] = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self) -> None:
        self.steps += 1
        for parameter, first_moment, second_moment in zip(
                self.parameters, self.first_moments, self.second_moments
        ):
            first_moment *= self.beta1
            first_moment += (1.0 - self.beta1) * parameter.grad
            second_moment *= self.beta2
            second_moment += (1.0 - self.beta2) * parameter.grad ** 2

            corrected_first_moment = first_moment / (1.0 - self.beta1 ** self.steps)
            corrected_second_moment = second_moment / (1.0 - self.beta2 ** self.steps)
            parameter.data -= self.learning_rate * corrected_first_moment / (
                    np.sqrt(corrected_second_moment) + self.epsilon
            )
