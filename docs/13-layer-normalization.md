# 13 — Layer Normalization

`BatchNorm1d` keeps each feature steady **across a mini-batch**. That can make
learning easier, but it also makes one example's output depend on its batch
companions and requires stored statistics at evaluation time.

[Layer Normalization](https://arxiv.org/abs/1607.06450) starts with the same problem from a different direction.
While earlier layers learn, the inputs they give a later layer can shift
together: their common centre or scale changes (**internal covariate shift**).

LayerNorm measures each example's features against one another instead of
measuring one feature across examples. For one feature vector

$$
z = (z_1, \ldots, z_H),
$$

it computes

$$
\mu = \frac{1}{H}\sum_{i=1}^{H}z_i, \qquad
v = \frac{1}{H}\sum_{i=1}^{H}(z_i-\mu)^2,
$$

then returns

$$
\hat{z}_i = \frac{z_i-\mu}{\sqrt{v+\varepsilon}}.
$$

The paper then restores flexibility with one learned gain and bias per feature:

$$
y_i = \gamma_i\hat{z}_i + \beta_i.
$$

>Suppose a preceding layer makes every feature in one example larger by the
same amount, or scales the whole feature vector. Without normalization, the
next layer must adapt to that drifting centre or magnitude. LayerNorm removes
the shared centre and divides by the shared spread, so it presents the next
layer with the features' *relative* pattern instead.

### Axis

For an input shaped `(batch, time, features)`, `LayerNorm(features)` reduces
only the final axis. Each `(batch, time)` position gets its own mean and
variance:

| normalization | values used for one set of statistics |
|---|---|
| BatchNorm | one feature across batch and time positions |
| LayerNorm | all features at one batch and time position |
