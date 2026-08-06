from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

from bonsaigrad import Leaf


class MultiplicationNetwork:
    def __init__(self, hidden_size: int, rng: np.random.Generator):
        # Small random weights break symmetry and keep values entering the square moderate.
        self.input_weights = Leaf(rng.normal(scale=0.5, size=(2, hidden_size)))
        self.hidden_bias = Leaf(np.zeros(hidden_size))
        self.output_weights = Leaf(rng.normal(scale=0.5, size=(hidden_size, 1)))
        self.output_bias = Leaf(np.zeros(1))
        self.parameters = (
            self.input_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
        )

    def predict(self, inputs: np.ndarray) -> Leaf:
        hidden = (Leaf(inputs) @ self.input_weights + self.hidden_bias) ** 2
        return hidden @ self.output_weights + self.output_bias

    def train(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float, steps: int) -> Tuple[float, float]:
        initial_loss: float = 0.0

        for step in range(steps):
            loss: Leaf = ((self.predict(inputs) - targets) ** 2).mean() # Apply MSE
            loss.wire()
            print(f"step {step + 1:>5}/{steps}: loss = {float(loss.data):.8f}")

            if step == 0:
                initial_loss = float(loss.data)

            # Apply gradient descent, then clear gradients before the next graph.
            for parameter in self.parameters:
                parameter.data -= learning_rate * parameter.grad
                parameter.rest()

        return initial_loss, float(loss.data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny network to approximate multiplication.")
    parser.add_argument("--examples", type=int, default=128, help="number of random pairs to generate")
    parser.add_argument("--steps", type=int, default=5_000, help="number of gradient-descent steps")
    args = parser.parse_args()

    if args.examples < 1 or args.steps < 1:
        parser.error("--examples and --steps must be positive")

    rng = np.random.default_rng(7)
    inputs = rng.uniform(-2.0, 2.0, size=(args.examples, 2))
    targets = (inputs[:, 0] * inputs[:, 1]).reshape(-1, 1)

    model = MultiplicationNetwork(hidden_size=6, rng=rng)
    initial_loss, final_loss = model.train(inputs, targets, learning_rate=0.01, steps=args.steps)

    print(f"loss: {initial_loss:.4f} -> {final_loss:.8f}")

    checks = np.array([[-1.0, 3.0], [1.5, -0.5], [12.0, 2.0], [-13.0, -13.0]])
    predictions = model.predict(checks).data
    print("\nPredictions")
    for (left, right), prediction in zip(checks, predictions):
        print(f"{left:>4g} * {right:>4g} = {prediction.item():>8.3f}  (expected {left * right:>4g})")


if __name__ == "__main__":
    main()
