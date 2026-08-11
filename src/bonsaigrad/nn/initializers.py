from collections.abc import Callable

import numpy as np

WeightInitializer = Callable[[np.random.Generator, int, int], np.ndarray]


def fan_in_normal(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.normal(scale=fan_in ** -0.5, size=(fan_in, fan_out))


def xavier_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_in, fan_out))
