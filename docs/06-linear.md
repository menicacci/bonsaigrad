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

Initialization is the rule that chooses a layer's trainable starting values
before training sees any data. Those values will be changed, but their
initial scale determines the scale of the first activations and gradients. A
poor scale can make a signal shrink or grow at every layer before learning has
had a chance to correct it.

For example, if the weights were all drawn with standard deviation `1`, each
output would be a sum of `m = in_features` random weighted terms. Wider layers
would then start with larger values purely because they have more inputs.

### Fan-in Normal Initialization

`initializers.fan_in_normal`, the default for `Linear`, draws each weight from
a normal distribution with mean $0$ and standard deviation

$$
\sigma = \frac{1}{\sqrt{\texttt{in_features}}}.
$$

Its weight variance is therefore

$$
\operatorname{Var}(W_{ij}) = \sigma^2
= \frac{1}{\texttt{in_features}}.
$$

Assume the input features are independent and centred. Ignoring the bias, one
output is

$$
z_j = \sum_{i=1}^{m} x_i W_{ij}.
$$

The independent terms add their variances:

$$
\begin{aligned}
\operatorname{Var}(z_j)
&= \texttt{in_features}\,\operatorname{Var}(x_i)\,\operatorname{Var}(W_{ij}) \\
&= \operatorname{Var}(x_i).
\end{aligned}
$$

So this initializer preserves the forward scale. The backward calculation has
the same form. If $G_j$ is an incoming gradient, then
$\partial L / \partial x_i = \sum_j G_j W_{ij}$, giving

$$
\operatorname{Var}\left(\frac{\partial L}{\partial x_i}\right)
= \frac{\texttt{out_features}}{\texttt{in_features}}
  \operatorname{Var}(G_j).
$$

When the layer is wider or narrower than its input, fan-in normal does not
preserve the backward scale. This is why Xavier initialization, below, uses
both widths of the weight matrix.


### Xavier Uniform Initialization

`initializers.xavier_uniform` uses both widths of the weight matrix. It draws
each weight from $U(-a, a)$, where

$$
a = \sqrt{\frac{6}{\texttt{in_features} + \texttt{out_features}}}.
$$

This balances the expected scale of values flowing forward with the scale of
gradients flowing backward.

For a uniform distribution $U(-a, a)$, the weight variance is

$$
\operatorname{Var}(W_{ij}) = \frac{a^2}{3}
= \frac{2}{\texttt{in_features} + \texttt{out_features}}.
$$

Using the same assumptions as before, the output variance is therefore

$$
\begin{aligned}
\operatorname{Var}(z_j)
&= \texttt{in_features}\,\operatorname{Var}(x_i)\,\operatorname{Var}(W_{ij}) \\
&= \frac{2\,\texttt{in_features}}
         {\texttt{in_features} + \texttt{out_features}}
   \operatorname{Var}(x_i).
\end{aligned}
$$

The backward calculation mirrors this one. If $G_j$ is an incoming gradient,
then $\partial L / \partial x_i = \sum_j G_j W_{ij}$, so

$$
\operatorname{Var}\left(\frac{\partial L}{\partial x_i}\right)
= \frac{2\,\texttt{out_features}}
       {\texttt{in_features} + \texttt{out_features}}
  \operatorname{Var}(G_j).
$$

When the two widths are equal, both factors are $1$: the layer preserves the
scale of both activations and gradients. When they differ, Xavier initialization
is a compromise between the forward and backward scales.

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
