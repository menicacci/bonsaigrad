# 10 — Batch Normalization

`Linear`'s initializer chooses a sensible scale for the first forward pass. But
training immediately starts changing its weights. As an earlier layer changes,
the values it hands to the next layer change their centre and spread too.

Batch Normalization, or **BatchNorm**, puts a small, trainable transformation
between those layers. During training it measures each feature in the current
mini-batch, recentres it, and scales it to a predictable range. 

This chapter follows the idea introduced in [the original Batch Normalization
paper](https://arxiv.org/abs/1502.03167), then follows the gradients through
the transformation.

---

Consider two layers with an activation between them:

$$
x \longrightarrow z = xW_1 + b_1 \longrightarrow a = f(z)
\longrightarrow aW_2 + b_2.
$$

The second layer does not receive the original data $x$. It receives $a$, the
representation made by the first layer. After every gradient update, $W_1$ and
$b_1$ change, so the distribution of values in $z$, and therefore in $a$, can
change too. 

>Here, a distribution simply means the pattern of values a feature
takes across examples: in particular, where they are centred and how spread
out they are.

The paper calls this **internal covariate shift**: the distribution of a
layer's inputs shifts because the parameters of the preceding layers are being
learned. It is "internal" because the changing inputs are inside the network,
not a change to the training dataset. In this name, *covariates* are simply
the input features a layer receives.

This is especially awkward for [sigmoid and tanh](07-activations.md). Their
slopes become small far from zero. A change that makes a pre-activation much
larger in magnitude can push the activation into that flat region, weakening
the gradient that must correct it. Careful initialization helps only at the
start; it cannot keep every later layer's inputs steady as training proceeds.

The derivatives make that weakening precise. For sigmoid,

$$
\sigma'(z) = \sigma(z)\bigl(1-\sigma(z)\bigr).
$$

At $z=0$, sigmoid is $0.5$, so its slope is $0.5(1-0.5)=0.25$. At $z=5$,
sigmoid is about $0.993$, making its slope about $0.0066$; at $z=-5$, the
slope is the same. Tanh has the analogous derivative

$$
\frac{d}{dz}\tanh(z) = 1-\tanh^2(z).
$$

Its slope is `1` at $z=0$, but only about $0.0099$ at $z=3$ or $z=-3$.

If $G = \partial L / \partial a$ is the gradient arriving at an activation
$a=f(z)$, the chain rule gives

$$
\frac{\partial L}{\partial z} = Gf'(z).
$$

So an incoming gradient of `1` becomes about `0.0066` through a sigmoid at
$z=5$, or about `0.0099` through tanh at $z=3$. The layer before the
activation receives that already-shrunken signal. Several saturated activations
in sequence multiply several small slopes together, which is why a drifting
scale can make a deep network particularly hard to correct.

BatchNorm addresses the paper's proposed cause directly: before a layer uses a
feature, express that feature relative to the other values currently in its
mini-batch.

For now, take one feature measured on $m$ examples in a mini-batch:

$$
x_1, x_2, \ldots, x_m.
$$

Its **batch mean** is its average value:

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i.
$$

Subtracting it gives each value's deviation from the centre,
$d_i = x_i - \mu_B$. The **batch variance** is the average squared deviation:

$$
v_B = \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2.
$$

Variance describes spread in squared units. Its square root, the standard
deviation, has the same units as $x$. BatchNorm divides by that scale after
recentering:

$$
\hat{x}_i = \frac{x_i-\mu_B}{\sqrt{v_B+\varepsilon}}.
$$

$\varepsilon$ is a small positive number. If all values in a batch are the
same, then $v_B=0$; adding $\varepsilon$ keeps the denominator nonzero. It
also prevents an extremely tiny variance from producing an unreasonably large
scale factor.

### Normalization is not a restriction

Always forcing every feature to mean `0` and variance `1` would be too rigid.
The next layer might benefit from a different centre or scale. BatchNorm
therefore learns one scale $\gamma$ and one shift $\beta$ per feature:

$$
y_i = \gamma\hat{x}_i + \beta.
$$

`gamma` starts at `1` and `beta` at `0`, so the first forward pass uses the
normalized values themselves. Training can then choose the scale and centre
that help the task. For a particular batch, choosing
$\gamma=\sqrt{v_B+\varepsilon}$ and $\beta=\mu_B$ would even reconstruct the
original $x_i$. The normalized representation is therefore a starting point,
not a command that every later layer must receive unit-scale values.

### Structure

`BatchNorm1d(num_features)` expects the **last** axis to be the feature axis.
It calculates a separate mean and variance for each feature, reducing every
leading axis. Nothing mixes feature `0` with feature `1`.

| input shape | reduced axes | mean and variance shape | values contributing to one feature |
|---|---|---|---|
| `(batch, features)` | `(0,)` | `(1, features)` | that feature across the batch |
| `(batch, time, features)` | `(0, 1)` | `(1, 1, features)` | that feature across every batch position and time position |

## Training and Inference

Using the current mini-batch is exactly what makes the training transformation
react to the representation the network has **now**. Computing statistics over
the whole training set at each update would be expensive, and they would be
outdated as soon as the parameters changed. A mini-batch gives an immediate,
differentiable estimate instead.

There is a consequence: the output for one example during training depends on
which other examples share its batch. That is useful for the training
transformation, but it is unsuitable at evaluation time, where an example
should receive a stable answer even when it arrives alone or in a differently
composed batch.

`BatchNorm1d` therefore keeps running estimates while it trains. After batch
$t$, with batch statistics $\mu_t$ and $v_t$, it updates them on the fly:

$$
\begin{aligned}
\text{runningMean}_{t+1} &= (1-\alpha)\,\text{runningMean}_t + \alpha\mu_t, \\
\text{runningVar}_{t+1} &= (1-\alpha)\,\text{runningVar}_t + \alpha v_t.
\end{aligned}
$$

Here $\alpha$ is `momentum`. In this implementation it is the weight of the
**new** batch: with the default `0.1`, the estimate keeps `90%` of its previous
value and takes `10%` from the batch just seen. Repeating this update creates an
exponentially weighted moving average: recent batches matter most, while older
ones fade smoothly rather than being stored and recomputed.
