# 04 — Reductions

So far, operations have either changed values entry by entry or, with `@`,
joined two tensors along a shared axis. A **reduction** instead combines entries
within one tensor. `sum` is the simplest reduction and the new primitive in this
chapter.

```python
x.sum()                         # one total for every entry in x
x.sum(axis=1)                   # one total for each row
x.sum(axis=(0, 2))              # keep only axis 1
x.sum(axis=1, keepdims=True)    # retain axis 1 with length 1
```

## Forward: remove an axis

Let `x` have shape `(2, 3)`. Reducing `axis=1` adds the three entries in each
row:

$$
y_i = \sum_j x_{ij}
$$

The axis being summed disappears, so `y` has shape `(2,)`.

| operation | input shape | output shape | output entries |
|-----------|-------------|--------------|----------------|
| `x.sum()` | `(2, 3)` | `()` | one total |
| `x.sum(axis=0)` | `(2, 3)` | `(3,)` | one per column |
| `x.sum(axis=1)` | `(2, 3)` | `(2,)` | one per row |
| `x.sum(axis=1, keepdims=True)` | `(2, 3)` | `(2, 1)` | one per row |

`keepdims=True` leaves the reduced axis in place with length `1`. The values are
the same; only the shape changes.

## Backward: restore the axis

Each `x_ij` contributes to exactly one output, `y_i`, with coefficient `1`:

$$
\frac{\partial y_i}{\partial x_{kj}} =
\begin{cases}
1 & \text{if } i = k \\
0 & \text{otherwise}
\end{cases}
$$

Give the incoming gradient on `y` the name `G`. To send it back to `x`, the
chain rule selects the one output that `x_ij` fed:

$$
\frac{\partial L}{\partial x_{ij}}
= \sum_k \frac{\partial L}{\partial y_k}
  \frac{\partial y_k}{\partial x_{ij}}
= \frac{\partial L}{\partial y_i}
= G_i
$$

For a row sum, that means copying each row-total gradient to the entries in its
row:

| forward | incoming `out.grad` | gradient on `x` |
|---------|---------------------|-----------------|
| `x.sum()` | scalar `g` | every entry is `g` |
| `x.sum(axis=0)` | `[g₀, g₁, g₂]` | each column is filled with its `gⱼ` |
| `x.sum(axis=1)` | `[g₀, g₁]` | each row is filled with its `gᵢ` |

After `x.sum(axis=1)`, `out.grad` has shape `(2,)`, while `x.grad` must have
shape `(2, 3)`. The backward pass restores the missing axis, making `(2, 1)`,
then broadcasts that one value across each row:

```python
grad = out.grad
for axis in reduced_axes:
    grad = np.expand_dims(grad, axis)
self.grad += np.broadcast_to(grad, self.data.shape)
```

The same two steps work for any reduced axes. When `keepdims=True`, those axes
are already present at length `1`, so only the broadcast is needed.

This is the mirror image of [broadcasting](02-shapes.md): broadcasting copies
values forward and sums gradients backward; `sum` combines values forward and
copies gradients backward.

## `mean`

An average is a sum divided by the number of entries reduced:

$$
\operatorname{mean}(x) = \frac{1}{n}\sum_i x_i
$$

The same backward path is therefore scaled by `1 / n`:

$$
\frac{\partial L}{\partial x_i} = \frac{1}{n}\frac{\partial L}{\partial \operatorname{mean}(x)}
$$

`Leaf.mean` does not add a second backward rule. It builds `sum` and divides by
`n`, reusing the graph operations already in place.
