from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

from bonsaigrad.nn import Linear, MSELoss, Sequential, Tanh
from bonsaigrad.optim import Adam, Optimizer, SGD


SEED: int = 11

class SurfaceRegressor(Sequential):
    def __init__(self, hidden_size: int, rng: np.random.Generator):
        super().__init__(
            Linear(2, hidden_size, rng),
            Tanh(),
            Linear(hidden_size, hidden_size, rng),
            Tanh(),
            Linear(hidden_size, hidden_size, rng),
            Tanh(),
            Linear(hidden_size, 1, rng),
        )

    @classmethod
    def get(cls, hidden_size: int) -> SurfaceRegressor:
        return cls(hidden_size, np.random.default_rng(SEED))


def surface(coordinates: np.ndarray) -> np.ndarray:
    """Return a smooth two-dimensional target for coordinate pairs of shape ``(N, 2)``."""
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    o = np.sin(x) * np.cos(y) + 0.3 * np.sin(2.0 * x + y)
    return (55 * o ** 2 - 43 * o).reshape(-1, 1)


def train(model: SurfaceRegressor, optimizer: Optimizer, inputs: np.ndarray, targets: np.ndarray,
          steps: int, label: str) -> Tuple[float, float]:
    loss_function = MSELoss()
    initial_loss: float = 0.0

    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.wire()
        optimizer.step()

        if step == 0:
            initial_loss = float(loss.data)

        if (step + 1) % max(1, steps // 10) == 0 or step == 0:
            print(f"{label:>4} step {step + 1:>5}/{steps}: loss = {float(loss.data):.8f}")

    return initial_loss, float(loss.data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SGD and Adam on a two-dimensional surface.")
    parser.add_argument("--examples", type=int, default=1024, help="number of random coordinate pairs to generate")
    parser.add_argument("--holdout-examples", type=int, default=256,
                        help="number of independent coordinate pairs for evaluation")
    parser.add_argument("--steps", type=int, default=4_000, help="number of training steps per optimizer")
    parser.add_argument("--hidden-size", type=int, default=16, help="width of both hidden layers")
    parser.add_argument("--sgd-learning-rate", type=float, default=0.1, help="SGD step size")
    parser.add_argument("--adam-learning-rate", type=float, default=0.01, help="Adam step size")
    args = parser.parse_args()

    if args.examples < 1 or args.holdout_examples < 1 or args.steps < 1 or args.hidden_size < 1:
        parser.error("--examples, --holdout-examples, --steps, and --hidden-size must be positive")
    if args.sgd_learning_rate <= 0.0 or args.adam_learning_rate <= 0.0:
        parser.error("--sgd-learning-rate and --adam-learning-rate must be positive")

    data_rng = np.random.default_rng(SEED)
    inputs = data_rng.uniform(-np.pi, np.pi, size=(args.examples, 2))
    targets = surface(inputs)
    holdout_inputs = data_rng.uniform(-np.pi, np.pi, size=(args.holdout_examples, 2))
    holdout_targets = surface(holdout_inputs)

    # Recreating the generator gives both models exactly the same initial parameters.
    print("Training on o = sin(x) * cos(y) + 0.3 * sin(2x + y)")
    print("            z = 55 * o ** 2 - 43 * o\n")

    sgd_model = SurfaceRegressor.get(args.hidden_size)
    sgd_initial_loss, sgd_final_loss = train(
        sgd_model,
        SGD(sgd_model.parameters(), learning_rate=args.sgd_learning_rate),
        inputs,
        targets,
        args.steps,
        "SGD",
    )

    adam_model = SurfaceRegressor.get(args.hidden_size)
    adam_initial_loss, adam_final_loss = train(
        adam_model,
        Adam(adam_model.parameters(), learning_rate=args.adam_learning_rate),
        inputs,
        targets,
        args.steps,
        "Adam",
    )

    sgd_holdout_loss = float(MSELoss()(sgd_model(holdout_inputs), holdout_targets).data)
    adam_holdout_loss = float(MSELoss()(adam_model(holdout_inputs), holdout_targets).data)

    print("\nOptimizer comparison")
    print(f"{'optimizer':<10}{'initial loss':>16}{'final loss':>16}{'hold-out loss':>16}")
    print(f"{'SGD':<10}{sgd_initial_loss:>16.6g}{sgd_final_loss:>16.6g}{sgd_holdout_loss:>16.6g}")
    print(f"{'Adam':<10}{adam_initial_loss:>16.6g}{adam_final_loss:>16.6g}{adam_holdout_loss:>16.6g}")


if __name__ == "__main__":
    main()
