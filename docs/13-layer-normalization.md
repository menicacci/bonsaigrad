# 13 — Layer Normalization

`BatchNorm1d` keeps each feature steady **across a mini-batch**. That can make
learning easier, but it also makes one example's output depend on its batch
companions and requires stored statistics at evaluation time.

[Layer Normalization](https://arxiv.org/abs/1607.06450), introduced by Ba,
Kiros, and Hinton, starts with the same problem from a different direction.
While earlier layers learn, the inputs they give a later layer can shift
together: their common centre or scale changes. The paper calls this
**internal covariate shift**. It is a useful motivation, rather than a claim
that normalization removes every source of difficult optimization.

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

`eps` is the small positive $\varepsilon$: it keeps the denominator safe when
the features are identical.

## Why this fights shift

Suppose a preceding layer makes every feature in one example larger by the
same amount, or scales the whole feature vector. Without normalization, the
next layer must adapt to that drifting centre or magnitude. LayerNorm removes
the shared centre and divides by the shared spread, so it presents the next
layer with the features' *relative* pattern instead.

Ignoring $\varepsilon$, adding a constant to every $z_i$ changes nothing after
LayerNorm; multiplying the whole vector by a positive constant changes nothing
either. This is the paper's key reason that it can reduce the effect of
correlated activation changes. It does not make a network immune to change:
the relative differences between features can still change, and those
differences carry useful information.

## The axis makes the difference

For an input shaped `(batch, time, features)`, `LayerNorm(features)` reduces
only the final axis. Each `(batch, time)` position gets its own mean and
variance:

| normalization | values used for one set of statistics |
|---|---|
| BatchNorm | one feature across batch and time positions |
| LayerNorm | all features at one batch and time position |

Therefore LayerNorm works with a batch of one, has no running mean or variance,
and performs the same calculation during training and evaluation. Those
properties made it especially natural for the recurrent models discussed in
the paper, and later for sequence models such as GPT.

The paper's full version follows normalization with a learned per-feature gain
and bias. `BonsaiGrad`'s current `LayerNorm` stops at the normalization itself:
it has no trainable affine parameters yet. The core operation is still fully
differentiable, so its gradients flow back to the layers that produced `inputs`.
