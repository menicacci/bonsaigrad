# 14 — Dropout

A model can become very good at its training examples while making poor
predictions for new ones. This is **overfitting**: instead of learning the
patterns that produced the data, the model has learned details that happen to
be true of its particular training set.

**Regularization** is the general idea of making that memorization harder. The
goal is not to prevent a model from fitting the training data; it is to make
the learned representation useful when the data changes. Different techniques
do this in different ways. Some penalize large parameters, while others change
what the model sees during training.

[*Improving neural networks by preventing co-adaptation of feature detectors*](https://arxiv.org/abs/1207.0580),
follows the second approach; the paper proposes to randomly omit feature detectors while the
network trains.

## Co-adaptation

A hidden unit can become useful only because a few particular units are also
present. For example, one unit may detect a weak detail and another may learn
how to compensate for its mistakes. Together they can fit the training set,
but neither has learned a feature that remains useful in a different context.
The paper calls this **co-adaptation**.

Dropout breaks those dependable partnerships. On each forward pass it removes
a different random subset of activations. A unit cannot rely on its usual
companions always being there, so it is encouraged to learn a feature that
helps across many possible internal contexts.

## Training

For an activation $x_i$, let $m_i$ be a random binary mask:

$$
m_i \sim \operatorname{Bernoulli}(1-p),
$$

where $p$ is the probability of dropping the activation. The masked output is

$$
y_i = m_i x_i.
$$

When $m_i=0$, that activation contributes nothing to this pass. When $m_i=1$,
it remains. Sampling a new mask every pass therefore trains many related,
thinned versions of the same network which share their parameters.

The original paper applies the scaling at evaluation time: training uses
$m_i x_i$, then evaluation multiplies activations by their keep probability,
$1-p$. BonsaiGrad uses the equivalent **inverted dropout** form instead:

$$
y_i = \frac{m_i}{1-p}x_i.
$$

The expected value during training is unchanged:

$$
\mathbb{E}[y_i]
= \frac{\mathbb{E}[m_i]}{1-p}x_i
= \frac{1-p}{1-p}x_i
= x_i.
$$

>During training, surviving activations are larger than they would normally
>be. That is deliberate: each activation is present only with probability
>`1-p`, so scaling its occasional contribution by `1/(1-p)` keeps the average
>signal unchanged. The random zeroes are still useful because every forward
>pass asks the network to work with a different subset of its features.

So `Dropout` scales the activations that survive during training and returns
its input unchanged during evaluation. Both conventions make training and
evaluation operate at the same expected scale.

## Backward pass

The sampled mask is constant during one forward and backward pass. Therefore
the gradient is gated and scaled by exactly the same factor:

$$
\frac{\partial L}{\partial x_i}
= \frac{m_i}{1-p}\frac{\partial L}{\partial y_i}.
$$

Dropped activations receive zero gradient. Surviving activations receive the
upstream gradient scaled by $1/(1-p)$, just as their forward values were.

```python
dropout = Dropout(0.5)
outputs = dropout(inputs)  # training: each entry is dropped with probability 0.5

dropout.eval()
outputs = dropout(inputs)  # evaluation: unchanged
```
