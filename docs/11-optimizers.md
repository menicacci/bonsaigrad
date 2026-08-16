# 11 — Optimizers

After `loss.wire()` has propagated gradients through the graph, every trainable
`Leaf` holds a direction in which its value would increase the loss. An
**optimizer** uses those gradients to change the parameters in the opposite
direction.

`Optimizer` holds the parameters it is responsible for. `SGD` is the first
concrete optimizer: it applies the same update to each parameter. `Adam` uses
the recent history of each parameter's gradients to adapt that update.

## Stochastic Gradient Descent

For one parameter $\theta$, with gradient $\partial L / \partial \theta$, SGD
updates it as

$$
\theta \leftarrow \theta - \eta\frac{\partial L}{\partial \theta}.
$$

Here $\eta$ is the `learning_rate`. The gradient points toward increasing loss,
so subtracting it makes a small move toward lower loss. A larger learning rate
moves faster but can overshoot; a smaller one is steadier but needs more steps.

The same rule applies element by element to arrays, so one `SGD` instance can
update a layer's weight matrix and bias vector together.

## Adam

SGD has no memory: it treats every gradient the same.
That makes two things hard. A noisy gradient can pull a parameter back and
forth, and one learning rate must work for every coordinate, even when some
coordinates consistently receive much larger gradients than others.

[`Adam`](https://arxiv.org/abs/1412.6980), short for *adaptive moment
estimation*, addresses both with two running summaries of each parameter's
gradients. For a gradient $g_t$ at step $t$,

$$
\begin{aligned}
m_t &= \beta_1m_{t-1} + (1 - \beta_1)g_t, \\
v_t &= \beta_2v_{t-1} + (1 - \beta_2)g_t^2.
\end{aligned}
$$

These are exponential moving averages: recent gradients matter most, but the
past fades away rather than being discarded. $m_t$ averages signed gradients,
so it records their persistent direction. If several batches agree that a
parameter should decrease, their signal reinforces; batch-to-batch noise tends
to cancel.

$v_t$ averages *squared* gradients, so positive and negative values cannot
cancel. It estimates the usual scale of this coordinate's gradients.

Adam divides the smoothed direction by that scale:

$$
\begin{aligned}
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t}, \\
\theta &\leftarrow \theta - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
\end{aligned}
$$

The numerator keeps the update moving in the averaged direction, while the
denominator scales down coordinates whose gradients have consistently been
large. Therefore each coordinate gets its own effective step size: a steep,
high-gradient coordinate takes a more cautious step; a quiet coordinate is not
automatically left behind. In the simple case of a steady gradient $g$, the
corrected estimates approach $\hat{m}_t = g$ and
$\sqrt{\hat{v}_t} = |g|$, making the update approximately
$-\eta\,\mathrm{sign}(g)$. The gradient still chooses the direction, but its
raw scale no longer makes that coordinate's update arbitrarily large.

Both averages start at zero. At the first step, for example,
$m_1 = (1 - \beta_1)g_1$, even though $g_1$ is the only gradient observed.
Without correction, this would make the early update artificially small. The
factors $1 - \beta_1^t$ and $1 - \beta_2^t$ are exactly the missing total
weight of the moving averages, so dividing by them renormalizes the estimates.
Their effect quickly fades as more gradients are observed.

`epsilon` prevents division by zero and keeps a coordinate with an extremely
small gradient scale numerically well behaved. The operations above are
element by element for arrays, so weights in the same matrix can adapt
independently.

```python
optimizer = Adam(model.parameters())
```

Its defaults are `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`, and
`epsilon=1e-8`, the settings suggested in the paper. Larger $β$ values retain
history for longer; Adam keeps two arrays of state for every parameter: its
first and second moments. This adaptation is a practical heuristic rather than
a guarantee that every training problem will converge, but it is often a
strong default for noisy, differently scaled gradients.

## Training loop
Give the optimizer the model's parameters once, before training:

```python
optimizer = SGD(model.parameters(), learning_rate=0.03)
```

Each step then has three parts:

```python
optimizer.zero_grad()
loss = ((model(inputs) - targets) ** 2).mean()
loss.wire()
optimizer.step()
```

`zero_grad()` clears the gradients left by the previous step. `wire()` computes
the new gradients, and `step()` changes the parameter data using them. Keeping
these jobs separate makes the training loop explicit: first measure the error,
then find its gradients, then update the model.
