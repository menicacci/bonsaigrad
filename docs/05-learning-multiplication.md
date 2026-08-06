# 05 — Learning Multiplication

> 🧪 Runnable companion: [`examples/05_learn_multiplication.py`](../examples/05_learn_multiplication.py)

This is the first example that uses gradients to change parameter `Leaf.data`.

## From a pair to a prediction

The parameters are matrices and vectors. We will call them $W$, $\beta$, $v$,
and $\gamma$ below. Every column of $W$ belongs to one of the six hidden
values; $v$ gives one output weight to each of those values.

| symbol | parameter | structure | shape |
|:------:|:---------:|:---------:|:-----:|
| $W$ | `input_weights` | $`\begin{bmatrix} w_{11} & w_{12} & w_{13} & w_{14} & w_{15} & w_{16} \\ w_{21} & w_{22} & w_{23} & w_{24} & w_{25} & w_{26} \end{bmatrix}`$ | `(2, 6)` |
| $\beta$ | `hidden_bias` | $`\begin{bmatrix} \beta_1 & \beta_2 & \beta_3 & \beta_4 & \beta_5 & \beta_6 \end{bmatrix}`$ | `(6,)` |
| $v$ | `output_weights` | $`\begin{bmatrix} v_1 \\ v_2 \\ v_3 \\ v_4 \\ v_5 \\ v_6 \end{bmatrix}`$ | `(6, 1)` |
| $\gamma$ | `output_bias` | $`\begin{bmatrix} \gamma \end{bmatrix}`$ | `(1,)` |

For one pair, the forward pass is the same vertical sequence of matrix
operations. A square is applied to each of the six hidden values separately:

$$
\begin{gathered}
\underset{\text{input } X\; (1, 2)}{\begin{bmatrix} a & b \end{bmatrix}} \\
\overset{@\,W + \beta}{\Big\downarrow} \\
\underset{\text{linear hidden layer } Z\; (1, 6)}{\begin{bmatrix} z_1 & z_2 & z_3 & z_4 & z_5 & z_6 \end{bmatrix}} \\
\overset{\text{square, elementwise}}{\Big\downarrow} \\
\underset{\text{squared hidden layer } H\; (1, 6)}{\begin{bmatrix} h_1 & h_2 & h_3 & h_4 & h_5 & h_6 \end{bmatrix}} \\
\overset{@\,v + \gamma}{\Big\downarrow} \\
\underset{\text{prediction } \hat{Y}\; (1, 1)}{\begin{bmatrix} \hat{y} \end{bmatrix}}
\end{gathered}
$$

`ŷ` is the network's guess for `a * b`. In the code, all six hidden values are
calculated at once:

```python
hidden = (Leaf(inputs) @ self.input_weights + self.hidden_bias) ** 2
return hidden @ self.output_weights + self.output_bias
```

For a batch of `N` pairs, `inputs` has shape `(N, 2)`. The first matrix product
with `(2, 6)` weights gives `(N, 6)`: six hidden values per pair. The bias of
shape `(6,)` is broadcast across the batch. The second product with `(6, 1)`
weights gives one prediction per pair, shape `(N, 1)`.

The weights start as small random values. Randomness makes each hidden value
start differently; keeping the values small keeps the inputs to the square
moderate. The biases start at zero because the random weights already make the
hidden values different.

## Why square?

Without the square, the two layers would still only make a weighted sum of `a`
and `b`. A weighted sum can draw a line, but it cannot express their product.
The square gives the network a simple nonlinearity.

The product itself can be assembled from two squared combinations:

$$
ab = \frac{(a + b)^2 - (a - b)^2}{4}
$$

The identity only shows that its layers have enough structure to represent multiplication.

## Error, gradient, update

For every generated pair, the target is its ordinary NumPy product. To train the
network, we need one number that says how wrong all of its predictions are. We
use **mean squared error** (MSE):

$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

Here, $N$ is the number of pairs in the batch, $y_i$ is the correct product for
pair $i$, and $\hat{y}_i$ is the network's prediction. For each pair, we find
the prediction error $\hat{y}_i - y_i$, square it, and then average those
squared errors across the batch. Squaring makes every error non-negative and
makes larger mistakes matter more. The result, $\mathcal{L}_{\mathrm{MSE}}$, is
one scalar: the loss that training will try to make smaller.

The formula maps directly to the code:

```python
loss = ((predictions - targets) ** 2).mean()
```

`predictions - targets` computes every $\hat{y}_i - y_i$, `** 2` squares each
element, and `mean()` performs the division by $N$. Because `predictions` is a
`Leaf`, these operations build a computation graph ending at `loss`.

Calling `loss.wire()` then gives every parameter the amount that changing it
would change that loss. Gradient descent makes a small move in the direction
that decreases it:

```python
loss.wire()
for parameter in self.parameters:
    parameter.data -= learning_rate * parameter.grad
    parameter.rest()
```

`rest()` clears the old gradients before the next training step builds a new
computation graph. The script prints the loss before each update and, at the
end, compares a few predictions with their expected products.

## Run it

Install the project, then run the example from the repository root:

```bash
pip install -e .
python examples/05_learn_multiplication.py
```

It trains on 128 random input pairs for 5,000 steps by default. Use
`--examples` and `--steps` to make a shorter experiment while exploring:

```bash
python examples/05_learn_multiplication.py --examples 32 --steps 100
```
