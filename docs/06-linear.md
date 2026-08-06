# 06 — Linear Layers

In the multiplication example, we wrote the same useful calculation directly
with `Leaf` operations:

```python
outputs = inputs @ weight + bias
```

Neural networks use this calculation so often that it deserves a
small, named home. `Linear` owns the values that training will change and
performs the calculation that uses them.

Suppose one input has `m` features and we want `n` output features. `Linear`
stores a weight matrix $W$ and a bias vector $b$:

$$
x \in \mathbb{R}^{1 \times m},
\qquad
W \in \mathbb{R}^{m \times n},
\qquad
b \in \mathbb{R}^{n}.
$$

The forward pass is an **affine transform**:

$$
z = xW + b \in \mathbb{R}^{1 \times n}.
$$

Each column of $W$, together with the matching entry in $b$, describes one
output feature. For output $j$:

$$
z_j = \sum_{i=1}^{m} x_i W_{ij} + b_j.
$$

These outputs are often called **pre-activations**. They are the raw values
before a later activation function, such as ReLU, transforms them.

>Usually `inputs` contains several examples. With `B` examples, it has shape
>`(B, m)`. Matrix multiplication applies every column of $W$ to every row of
>$X$, then NumPy broadcasts the one bias vector across the rows:
>$$(B, m) @ (m, n) + (n,) \longrightarrow (B, n).$$


### Backward

Let $G = \partial L / \partial Z$ be the gradient arriving from the loss. For
a two-dimensional batch, the gradients of the affine transform are

$$
\frac{\partial L}{\partial X} = GW^\mathsf{T},
\qquad
\frac{\partial L}{\partial W} = X^\mathsf{T}G,
\qquad
\frac{\partial L}{\partial b} = \sum_{r=1}^{B} G_r.
$$

### Initialization

Before training starts, the scale of the random weights matters. If the
weights were all drawn with standard deviation `1`, each output would be a sum
of `m = in_features` random weighted terms. Wider layers would then start with
larger values purely because they have more inputs.

To see the scale, assume for the moment that the input features are independent,
centred, and have variance $1$, and that the weights are independent with mean
$0$ and variance $\sigma^2$. Ignoring the bias, one output is

$$
z_j = \sum_{i=1}^{m} x_i W_{ij}.
$$

The independent terms add their variances:

$$
\operatorname{Var}(z_j)
= \sum_{i=1}^{m}\operatorname{Var}(x_i W_{ij})
= m\sigma^2.
$$

We want the output variance to remain roughly $1$, like the input variance.
Setting $m\sigma^2 = 1$ gives

$$
\sigma = \frac{1}{\sqrt{m}}
= \frac{1}{\sqrt{\texttt{in_features}}}.
$$

This is the point of the formula: a wider layer gives each weight a smaller
starting scale so that all of their contributions still add up to a signal of
roughly the same size. Initialization is not about picking arbitrary small
numbers; it gives the network values that are large enough to carry information
forward, without becoming larger simply because a layer has more inputs.

## `Module` gives layers a common interface

`Linear` is a `Module`, a small convention that lets callers use every future
neural-network component in the same way:

```python
class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        return ()
```

`__call__` makes `layer(inputs)` delegate to `layer.forward(inputs)`.
`parameters()` makes the trainable `Leaf`s available for an update:

```python
for parameter in layer.parameters():
    parameter.data -= learning_rate * parameter.grad
    parameter.rest()
```

For `Linear`, `parameters()` explicitly returns `(weight, bias)`.
