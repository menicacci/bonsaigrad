from __future__ import annotations

import argparse

import numpy as np

from bonsaigrad.nn import Linear, Sequential, Tanh


def determinants(matrices: np.ndarray) -> np.ndarray:
    """Return the determinant of each matrix with shape ``(2, 2)``."""
    return matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny network to approximate 2-by-2 determinants.")
    parser.add_argument("--examples", type=int, default=256, help="number of random matrices to generate")
    parser.add_argument("--steps", type=int, default=5_000, help="number of gradient-descent steps")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="gradient-descent step size")
    args = parser.parse_args()

    if args.examples < 1 or args.steps < 1 or args.learning_rate <= 0.0:
        parser.error("--examples, --steps, and --learning-rate must be positive")

    rng = np.random.default_rng(7)
    matrices = rng.uniform(-1.0, 1.0, size=(args.examples, 2, 2))
    inputs = matrices.reshape(args.examples, 4)
    targets = determinants(matrices).reshape(-1, 1)

    model = Sequential(
        Linear(4, 16, rng),
        Tanh(),
        Linear(16, 16, rng),
        Tanh(),
        Linear(16, 1, rng),
    )

    initial_loss = 0.0
    for step in range(args.steps):
        loss = ((model(inputs) - targets) ** 2).mean()
        loss.wire()
        print(f"step {step + 1:>5}/{args.steps}: loss = {float(loss.data):.8f}")

        if step == 0:
            initial_loss = float(loss.data)

        for parameter in model.parameters():
            parameter.data -= args.learning_rate * parameter.grad
            parameter.rest()

    print(f"loss: {initial_loss:.4f} -> {float(loss.data):.8f}")

    checks = np.array(
        [
            [[0.8, 0.0], [0.0, 0.8]],
            [[0.8, 0.0], [0.0, -0.8]],
            [[0.0, -0.8], [0.8, 0.0]],
            [[0.5, 0.75], [0.5, 0.75]],
        ]
    )
    predictions = model(checks.reshape(-1, 4)).data
    expected = determinants(checks)

    print("\nPredictions")
    for matrix, prediction, target in zip(checks, predictions, expected):
        print(f"{matrix.tolist()}: {prediction.item():>8.3f}  (expected {target:>4g})")


if __name__ == "__main__":
    main()
