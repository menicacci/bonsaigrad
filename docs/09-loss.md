# 09 — Loss

> 🔬 Runnable companion: [`notebooks/09_loss.ipynb`](../notebooks/09_loss.ipynb)

A model needs one number that says how wrong those predictions are: a **loss**.
Backpropagation starts at that number and tells every parameter how it should
change to make the next prediction a little better.

## Mean Squared Error Loss

Suppose a model makes $N$ predictions $p_1, p_2, \ldots, p_N$, with target
values $t_1, t_2, \ldots, t_N$. The **mean squared error** (MSE) is

$$
L = \frac{1}{N}\sum_{i=1}^{N}(p_i-t_i)^2.
$$

Each difference $p_i-t_i$ is the model's error on one value. Squaring has two
effects: positive and negative errors can no longer cancel each other out, and
larger errors count more heavily than smaller ones. Taking the mean makes the
scale of the loss independent of how many values are in the prediction.

For example, predictions `[1, 2, 4]` and targets `[1, 4, 3]` have errors
`[0, -2, 1]`. Their squared errors are `[0, 4, 1]`, so the loss is

$$
L = \frac{0 + 4 + 1}{3} = \frac{5}{3}.
$$

### Backward pass

Only the term $(p_i-t_i)^2$ depends on a particular prediction $p_i$. Applying
the power rule and then accounting for the final mean gives

$$
\frac{\partial L}{\partial p_i}
= \frac{2(p_i-t_i)}{N}.
$$

The sign points in the direction that would increase the error: predictions
above their targets receive a positive gradient, while predictions below their
targets receive a negative one. Gradient descent moves in the opposite
direction, so both are pulled toward their targets.

## `logsumexp`

$$
\mathrm{logsumexp}(x) = \log \sum_i e^{x_i}
$$

`Leaf.logsumexp(axis=None, keepdims=False)` can reduce all entries, one axis, or
a tuple of axes. It follows the same shape rules as the reductions from
[chapter 04](04-reductions.md): reduced axes disappear unless
`keepdims=True`.

### Forward pass

Computing `exp(x)` directly can overflow even when the final answer is a normal
floating-point value. For example, $e^{1000}$ is too large for `float64`. Let
$m = \max_i x_i$. Factoring $e^m$ out of the sum gives an equivalent expression:

$$
\log \sum_i e^{x_i}
= \log \left(e^m \sum_i e^{x_i-m}\right)
= m + \log \sum_i e^{x_i-m}
$$

Every shifted exponent is at most $e^0 = 1$, so the intermediate values do not
overflow. The maximum is retained along reduced axes while this expression is
computed, then those axes are removed unless `keepdims=True`.

### Backward pass

Start with one group of inputs and one output:

$$
y = \log \left(e^{x_1} + e^{x_2} + \cdots + e^{x_n}\right)
$$

To see how $x_i$ affects $y$, separate the operation into two steps. First form
the sum

$$
s = \sum_j e^{x_j} = e^{x_1} + e^{x_2} + \cdots + e^{x_n},
$$

then take $y = \log s$.

$$
\frac{\partial y}{\partial s}
= \frac{\partial \log s}{\partial s}
= \frac{1}{s}.
$$

Next differentiate the sum with respect to one particular input, $x_i$. Only
the term $e^{x_i}$ depends on $x_i$. Every other $x_j$ is a separate input and
is treated as a constant:

$$
\frac{\partial s}{\partial x_i}
= \frac{\partial}{\partial x_i}
  \left(e^{x_1} + \cdots + e^{x_i} + \cdots + e^{x_n}\right)
$$

$$
= 0 + \cdots + e^{x_i} + \cdots + 0
= e^{x_i}.
$$

The surviving term is $e^{x_i}$ because the exponential is its own derivative:

$$
\frac{d}{dx_i}e^{x_i} = e^{x_i}.
$$

Now the chain rule follows $x_i$ through both steps, from $x_i$ to $s$ and
then from $s$ to $y$:

$$
\frac{\partial y}{\partial x_i}
= \frac{\partial y}{\partial s}
  \frac{\partial s}{\partial x_i}
= \frac{1}{s} e^{x_i}
= \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

The result has a useful structure: the numerator is the contribution from
$x_i$, while the denominator is the contribution from every input in the
group. Each derivative is therefore one input's share of the total. The shares
are positive and add up to `1`.

We calculate these shares with the same shift used in the stable forward pass:

$$
\frac{e^{x_i}}{\sum_j e^{x_j}}
= \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
$$

The vector formed by these shares is called the **softmax** of $x$:

$$
\mathrm{softmax}(x)_i
= \frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

The gradient of `logsumexp` is therefore softmax. `logsumexp` reduces a group
of inputs to one value, while softmax contains one share for each input in that
group.

## Cross-Entropy Loss

A **classifier** chooses one answer from a fixed set. Given a photo, for example,
it might choose among `cat`, `dog`, and `tree`. Its raw output is one score for
each possible answer; the answer with the largest score is its choice.

Choosing the largest score is enough to make a decision, but it does not say how
confident that decision is. Scores `[2.0, 2.1, 1.9]` and `[2.0, 8.0, 1.0]` both
choose the second answer, but the first is uncertain while the second is very
confident. If the first answer was correct, the confident mistake should receive
a much larger correction during training.

Softmax turns the scores into probabilities so we can measure that confidence:
the probability of the correct answer tells us how good, or how confidently
wrong, the prediction was. The raw unrestricted scores are called **logits**. A
larger logit means the model considers that class more likely, but logits can be
positive or negative and do not need to add up to `1`.

For one example with $C$ classes, write the logits as

$$
x = (x_0, x_1, \ldots, x_{C-1})
$$

Softmax turns these logits into positive shares that add up to `1`. In this
classification setting, those shares are the probabilities assigned to the
classes:

$$
p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

Let $t$ be the integer index of the correct class.

Cross-entropy asks how much probability the model assigned to the correct class
and takes its negative logarithm:

$$
\ell = -\log p_t.
$$

If $p_t$ is close to `1`, then $-\log p_t$ is close to `0`. If the model gives
the correct class a tiny probability, the loss is large. The logarithm therefore
rewards confident correct predictions and strongly penalises confident wrong
ones.

### Softmax and logarithm collapse

Calculating softmax and then taking a logarithm would be both unnecessary and
numerically fragile. A very small probability can round to zero, after which
`log(0)` is negative infinity. Instead, expand the formula algebraically:

$$
\begin{aligned}
\ell
&= -\log \frac{e^{x_t}}{\sum_j e^{x_j}} \\
&= -\left(\log e^{x_t} - \log \sum_j e^{x_j}\right) \\
&= \log \sum_j e^{x_j} - x_t \\
&= \mathrm{logsumexp}(x) - x_t.
\end{aligned}
$$

### Shapes

The final axis of `logits` is always the class axis. Every leading position is
one classification problem, and `targets` contains one integer class index for
each of those positions:

| use case | `logits` shape | `targets` shape |
|---|---|---|
| one example | `(classes,)` | `()` |
| a batch | `(batch, classes)` | `(batch,)` |
| a token batch | `(batch, time, classes)` | `(batch, time)` |

For any leading shape, `np.indices(targets.shape)` creates the coordinates for
each classification position. Add `targets` as the final coordinate to gather
the corresponding correct logit:

```python
coordinates = (*np.indices(targets.shape), targets)
correct_logits = logits[coordinates]
```

This is a gather: the forward pass reads one logit per position. During the
backward pass, each incoming gradient is added back to the logit it came from;
all other logits receive no gradient from this branch.

The other term in the loss is the `logsumexp` across the class axis:

```python
normalizers = logits.logsumexp(axis=-1)
losses = normalizers - correct_logits
loss = losses.mean()
```

`normalizers` and `correct_logits` both have the same shape as `targets`, so
their difference produces one loss per position. `.mean()` turns those losses
into the single scalar from which backpropagation starts.

### Backward pass

For one example, the loss is

$$
\ell = \mathrm{logsumexp}(x) - x_t.
$$

We already found that the derivative of `logsumexp` with respect to $x_i$ is
the softmax probability $p_i$. The second term contributes `-1` only when
$i$ is the correct class:

$$
\frac{\partial \ell}{\partial x_i}
= p_i - \mathbb{1}[i=t],
$$

where $\mathbb{1}[i=t]$ is `1` for the correct class and `0` otherwise. Written
out, the gradient is

$$
\frac{\partial \ell}{\partial x_t} = p_t - 1,
\qquad
\frac{\partial \ell}{\partial x_i} = p_i \quad (i \ne t).
$$

The `logsumexp` branch sends $p_i$ to every logit. The indexing branch sends
`-1` back only to the gathered correct logit. When those two branches meet, they
produce the familiar `softmax - one_hot(target)` gradient without ever building
a one-hot array.

This gradient also explains how learning changes the prediction:

- For the correct class, $p_t-1$ is negative. Gradient descent subtracts that
  negative value, raising the correct logit.
- For every incorrect class, $p_i$ is positive. Gradient descent lowers those
  logits.

## Why Cross-Entropy Rather Than MSE?

Imagine a yes/no classifier deciding whether an image contains a cat. The image
does contain a cat, but the model assigns `1%` probability to `yes`. It is not
merely wrong: it is confidently wrong, so this is exactly the case where we
want a strong correction.

To use MSE here, we compare the predicted probability (`0.01`) with the target
probability (`1`). But that correction must then pass back through the
sigmoid—the curve that converted the model's raw logit into `0.01`. Near zero,
that curve is almost flat. Moving the raw logit a little barely changes the
probability, so MSE sends only a weak instruction back to it.

For this example, the learning signals that reach the `yes` logit are

$$
\begin{aligned}
\frac{\partial \ell_{\mathrm{CE}}}{\partial z} &= 0.01 - 1 = -0.99, \\
\frac{\partial \ell_{\mathrm{MSE}}}{\partial z} &= 2(0.01 - 1)(0.01)(1 - 0.01) \approx -0.0196.
\end{aligned}
$$

Their magnitudes are roughly fifty times apart. To make that difference feel
concrete, use a learning rate of $0.1$. The current logit is
$z \approx -4.595$, which produces a probability of $0.01$. One update gives

$$
\begin{aligned}
z_{\mathrm{CE}}' &= -4.595 - 0.1(-0.99) 
\approx -4.496, & p_{\mathrm{CE}}' &
\approx 0.0110, \\ z_{\mathrm{MSE}}' &= -4.595 - 0.1(-0.0196) 
\approx -4.593, & p_{\mathrm{MSE}}' &\approx 0.01002.
\end{aligned}
$$

One MSE update changes the predicted probability from `1.000%` to `1.002%`;
cross-entropy changes it to about `1.10%`. Neither has solved the mistake in
one step, but cross-entropy begins undoing a confident error at a useful pace.

MSE can still learn this task; it just learns needlessly slowly whenever it is
confident and wrong. Softmax has the same flat-at-the-extremes behaviour in the
multiclass case. Cross-entropy avoids that extra weakening and also has a useful
interpretation: it measures how little probability the model assigned to the
answer that actually occurred.
