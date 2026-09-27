# 16 — Self-Attention

**Self-Attention** gives each token a representation that includes context from
the sequence. 

It compares **queries** $Q$ with **keys** $K$ to decide how much
attention to give each token, then combines their **values** $V$ using those
weights. Here we introduce **scaled dot-product attention**, proposed by
Vaswani et al. in [*Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762),
building on earlier attention mechanisms:

$$
Y = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V.
$$

For $T$ tokens, Q, K, and V have shape $(T, d)$ in this implementation.
The attention weights have shape $(T, T)$; the output has shape $(T, d)$.

## Learning where to look

A `Linear` layer transforms each token independently. **Self-attention lets
tokens exchange information**: each position builds its output from a weighted
sum of token information:

$$
y_i = \sum_j a_{ij}v_j.
$$

Here, $v_j$ is what token $j$ contributes, and $a_{ij}$ is how much position
$i$ uses. The weights depend on the input, so a token can gather different
context in different sequences.

## Why queries, keys, and values?

Three learned linear projections produce three representations of each token:

$$
Q = XW_Q + b_Q, \qquad
K = XW_K + b_K, \qquad
V = XW_V + b_V.
$$

> It is called *self*-attention because Q, K, and V come from the same sequence.

The dot product $q_i \cdot k_j$ scores how strongly position $i$ should attend
to position $j$. Their use in the computation gives them their names:

- **Query** $q_i$: the receiving token's features for making a match.
- **Key** $k_j$: the supplying token's features for being matched.
- **Value** $v_j$: the information included in the weighted sum.

These roles come from the computation; training learns useful features for
each role.


### Why divide by $\sqrt{d}$?

A dot product sums $d$ terms, where $d$ is `head_dim`:

$$
q_i \cdot k_j = \sum_{r=1}^{d} q_{ir}k_{jr}.
$$

Suppose all query and key components are independent, with mean zero and
variance one. Each product then has mean zero and variance one. Variances of
independent terms add, giving:

$$
\operatorname{Var}(q_i \cdot k_j) = d.
$$

The mean stays zero, but the spread grows. Dividing by $\sqrt{d}$ compensates
for this because dividing by a constant divides the variance by its square:

$$
\operatorname{Var}\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)
= \frac{d}{d} = 1.
$$

Without scaling, larger score differences can make softmax concentrate almost
all its weight on one token, producing very small gradients.

### Why softmax before combining values?

[Softmax](09-loss.md) turns each row of scaled scores into nonnegative weights
that sum to one across source tokens:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}, \qquad
a_{ij} = \frac{\exp(s_{ij})}{\sum_\ell \exp(s_{i\ell})}.
$$

This gives a differentiable weighted average of values. For example, weights
$[0.1, 0.7, 0.2]$ produce $y_i = 0.1v_1 + 0.7v_2 + 0.2v_3$.

There are **two matrix multiplications**, with softmax between them:

1. $QK^\top$ computes matching scores between tokens.
2. $AV$ combines values using the weights $A = \operatorname{softmax}(S)$.

Softmax normalizes how much each source token contributes. Applying it after
$AV$ would instead normalize the output's feature coordinates.
