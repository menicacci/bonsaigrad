# 03 — Operations

With shapes handled, we can widen the op set. But most of it needs no new
gradient rule, because a new op is only *primitive* if it cannot be written with
the ops we already have.

Alongside `+` and `*` that leaves three newcomers — `**` with a constant
exponent, `log` and `exp`. Every other op is built from those:

| op        | built from            | new backward? |
|-----------|-----------------------|---------------|
| `-a`      | `a * -1`              | no            |
| `a - b`   | `a + (-b)`            | no            |
| `a ** k`  | —                     | **yes**       |
| `a / b`   | `a * b ** -1`         | no            |
| `a.log()` | —                     | **yes**       |
| `a.exp()` | —                     | **yes**       |
| `a ** b`  | `(b * a.log()).exp()` | no            |

The three primitives come first, then the ops assembled from them.

### Negation and subtraction

`-a` is `a · -1`, so multiplication's cross-wire rule hands back `out.grad · -1`
— the sign flip we expect. Subtraction then follows for free:

```
a - b  =  a + (b · -1)

∂L/∂a = out.grad · 1  =  out.grad
∂L/∂b = out.grad · 1 · -1 = -out.grad
```

Which is the textbook rule for subtraction, assembled out of two steps we had
already proved.

### Powers

For `out = aᵏ` with `k` a constant, the power rule gives the local derivative,
and the chain rule scales the incoming gradient by it:

```
∂out/∂a = k·a^(k-1)          →      self.grad += k · a^(k-1) · out.grad
```

### Division

`a / b` is `a · b⁻¹`. The gradient w.r.t. `a` is immediate. For `b`, the signal
crosses two nodes — the multiply, then the power — and the chain rule multiplies
the local derivatives along the way:

$$
t = b^{-1}
\qquad\qquad
\texttt{out} = a \cdot t
$$

$$
\frac{\partial L}{\partial t}
= \texttt{out.grad} \cdot a
\qquad\longrightarrow\qquad
\frac{\partial L}{\partial b}
= \frac{\partial L}{\partial t} \cdot (-1)b^{-2}
= -\frac{\texttt{out.grad} \cdot a}{b^2}
$$

### Logarithm and exponential

Two more one-argument primitives, and their derivatives are the two best known
in calculus:

```
out = ln a     ∂out/∂a = 1/a       →   self.grad += (1 / a) · out.grad
out = eᵃ       ∂out/∂a = eᵃ        →   self.grad += out.data · out.grad
```

### A `Leaf` exponent

With `log` in hand, `a ** b` for a `Leaf` `b` needs no rule either:

```
aᵇ = e^(b·ln a)
```

Three nodes — a `log`, a `*`, an `exp` — and `wire` walks them without knowing
what it is differentiating. Both gradients fall out, including the awkward one:

```
∂L/∂a = out.grad · aᵇ · b/a            (through the log)
∂L/∂b = out.grad · aᵇ · ln a           (through the multiply)
```


`k ** a` — a plain number raised to a `Leaf` — is built the same way, through
`__rpow__`.

> **Aside: why keep the constant exponent primitive?**
>
> A fixed exponent and a `Leaf` exponent need different gradients. With
> `a ** k`, we only differentiate with respect to `a`, so the direct rule
> `k * a ** (k - 1)` is enough. We do not rewrite it through `log(a)`, because
> `a` can be negative while `log(a)` is defined only for `a > 0`:
>
> ```
> (-2.0) ** 3              →  -8.0
> exp(3 * log(-2.0))       →   nan
> ```
>
> For `a ** b`, where `b` is a `Leaf`, the rewrite is appropriate because its
> gradient with respect to `b` inherently contains `log(a)`.


## Matrix product

Every op so far was **elementwise**: each output entry came from entries in the
same position in its stems. `@` is different. It combines a row from the left
matrix with a column from the right one, so one input entry can help make several
output entries:

```
(n, m) @ (m, p)  →  (n, p)
```

This is the op a layer is made of. `x @ w` takes a batch of `n` examples with `m`
features and produces `n` rows of `p` outputs, mixing every input feature into
every output.

### Intuition

Take `a` of shape `(2, 3)` and `b` of shape `(3, 2)`. Their product has shape
`(2, 2)`. Write out the whole small example once, then we only need to watch one
number: `a[0,0]` (the `a₁₁` below).

$$
A =
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{bmatrix}
\qquad
B =
\begin{bmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22} \\
b_{31} & b_{32}
\end{bmatrix}
$$

$$
AB =
\begin{bmatrix}
a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31} &
a_{11}b_{12} + a_{12}b_{22} + a_{13}b_{32} \\
a_{21}b_{11} + a_{22}b_{21} + a_{23}b_{31} &
a_{21}b_{12} + a_{22}b_{22} + a_{23}b_{32}
\end{bmatrix}
$$

Only the first row contains `a₁₁`. If we nudge `a₁₁` upward by one tiny step,
the first output entry changes by `b₁₁`, the second changes by `b₁₂`, and the
second row does not change at all:

$$
\frac{\partial (AB)}{\partial a_{11}} =
\begin{bmatrix}
b_{11} & b_{12} \\
0 & 0
\end{bmatrix}
$$

The zeros are useful. They say that the bottom row of `AB` has no path back to
`a₁₁`, so no gradient from there can reach it.

In the forward pass, matrix multiplication **reuses** that number once for each
column of `b`. It helps make the first two output entries:

$$
a_{11} \xrightarrow{\times b_{11}} (AB)_{11}
\qquad\qquad
a_{11} \xrightarrow{\times b_{12}} (AB)_{12}
$$

Those are not the whole output entries: each also receives contributions from
`a[0,1]` and `a[0,2]`. But they are the only two places where `a[0,0]` appears.
It never helps make row `1`, because that row starts from `a[1,*]` instead.

Now call `wire()`. `out.grad` arrives from above with one incoming gradient per
output entry. Reverse each of the two forward paths:

$$
g_{11}
\xrightarrow{\times b_{11}}
\texttt{a.grad[0,0]}
\qquad
\qquad
g_{12}
\xrightarrow{\times b_{12}}
\texttt{a.grad[0,0]}
$$

Both arrows land at the same entry, so their values add.

This is the same rule we already know from a reused `Leaf`: one forward use means
one backward path, and several paths add together. The only new part is that the
other matrix supplies the scale on each path.

> `a[0,0]` receives the gradient from every output entry it helped make, each
> multiplied by the matching number from `b`.

For this one entry, that is:

```
a.grad[0,0] = out.grad[0,0] · b[0,0] + out.grad[0,1] · b[0,1]
```

Here is the same backward pass in the matrix picture. Give the incoming gradient
a name, `G`. Its entries line up with the entries of `AB`:

$$
G =
\begin{bmatrix}
g_{11} & g_{12} \\
g_{21} & g_{22}
\end{bmatrix}
\qquad\text{where } g_{ij} = \texttt{out.grad[i-1, j-1]}
$$

For `a₁₁`, only the entries marked by its derivative can contribute. Multiply
matching positions, then add the results:

$$
\begin{bmatrix}
g_{11} & g_{12} \\
g_{21} & g_{22}
\end{bmatrix}
\odot
\begin{bmatrix}
b_{11} & b_{12} \\
0 & 0
\end{bmatrix} =
\begin{bmatrix}
g_{11}b_{11} & g_{12}b_{12} \\
0 & 0
\end{bmatrix}
\quad\longrightarrow\quad
\texttt{a.grad[0,0]} = g_{11}b_{11} + g_{12}b_{12}
$$

Now turn around and watch `b₁₁`. It appears in the first column of `AB`, once for
each row of `A`:

$$
\frac{\partial (AB)}{\partial b_{11}} =
\begin{bmatrix}
a_{11} & 0 \\
a_{21} & 0
\end{bmatrix}
\quad\longrightarrow\quad
\texttt{b.grad[0,0]} = g_{11}a_{11} + g_{21}a_{21}
$$

So the two stems collect gradients in different directions: `a₁₁` collects across
its output **row**; `b₁₁` collects down its output **column**.

### Proof

Keep `a₁₁` in view. Its gradient needs the two numbers in the first row of `G`,
`[g₁₁, g₁₂]`, paired with the two numbers in the first row of `B`,
`[b₁₁, b₁₂]`. We need their matching products added:

$$
g_{11}b_{11} + g_{12}b_{12}
$$

But a matrix product pairs a **row** of its left input with a **column** of its
right input. `B` stores `[b₁₁, b₁₂]` as a row, so it is facing the wrong way. To
solve this, we need to apply the transpose: it does not change any values, but it
turns each row of `B` into a column:

$$
G B^\mathsf{T} =
\begin{bmatrix}
g_{11} & g_{12} \\
g_{21} & g_{22}
\end{bmatrix}
\begin{bmatrix}
b_{11} & b_{21} & b_{31} \\
b_{12} & b_{22} & b_{32}
\end{bmatrix}
$$

The top-left entry of that result is exactly the gradient we just followed:

$$
(G B^\mathsf{T})_{11} = g_{11}b_{11} + g_{12}b_{12} = \texttt{a.grad[0,0]}
$$

The same turn happens for `B`, from the other side. `b₁₁` needs the first column
of `G`, `[g₁₁, g₂₁]`, paired with the first column of `A`, `[a₁₁, a₂₁]`. A matrix
product needs the `A` values as a row on the left, so we transpose `A`:

$$
A^\mathsf{T} G =
\begin{bmatrix}
a_{11} & a_{21} \\
a_{12} & a_{22} \\
a_{13} & a_{23}
\end{bmatrix}
\begin{bmatrix}
g_{11} & g_{12} \\
g_{21} & g_{22}
\end{bmatrix}
$$

$$
(A^\mathsf{T} G)_{11} = a_{11}g_{11} + a_{21}g_{21} = \texttt{b.grad[0,0]}
$$

So the transpose is just an alignment move. It places the axis we need to add
over in the two touching positions of `@`; matrix multiplication then does the
pair-and-add work for every entry at once.

### Rule

The implementation performs all of those little path-and-add stories at once:

```python
self.grad += out.grad @ other.data.swapaxes(-1, -2)  # G @ bᵀ
other.grad += self.data.swapaxes(-1, -2) @ out.grad  # aᵀ @ G
```

The transposes simply turn the matrix that supplied each path's scale so that its
matching dimension sits next to the incoming gradient. The result must have the
same shape as the `Leaf` receiving it; that is our check that the paths were
collected in the right direction.

### Batches

`swapaxes(-1, -2)` swaps only the **last two** axes, where `.T` would reverse all
of them — wrong the moment there is a batch axis in front. NumPy treats leading
axes as a stack of matrices and broadcasts them, so `(B, n, m) @ (m, p)` works:
one weight matrix serving `B` batches of rows.

Broadcast leading axes mean copies, and copies mean sums — the same rule as
before, so `_unbroadcast` handles the batch just as it handled a bias:

```python
x = Leaf(np.ones((5, 2, 3)))  # 5 batches of 2 rows
w = Leaf(np.ones((3, 4)))  # one weight matrix, shared
(x @ w).wire()

x.grad.shape  # (5, 2, 3)
w.grad.shape  # (3, 4)  — summed over all 5 × 2 rows that used it
```

### Not 1-D

A 1-D operand is refused with a `ValueError`. NumPy would happily promote a `(m,)`
vector to a matrix, contract, then drop the axis again — but each promotion case
(`vector·vector`, `vector@matrix`, `matrix@vector`) needs a *different* backward
rule, and we have no reshape op yet to do the promotion inside the graph where the
gradient could find its way back. Refusing beats silently returning a gradient of
the wrong shape. A row of a batch is a `(1, m)` matrix, and that already works.
