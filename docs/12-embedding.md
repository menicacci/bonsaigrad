# 12 — Embeddings

Models work with numbers, but many useful inputs are labels: a word, a
character, a product, or a category. 

A label has no meaningful numerical distance from another label: word `7` 
is not inherently closer to word `8`
than to word `42`. Feeding those integer labels into a `Linear` layer would
invent that relationship.

An `Embedding` learns a vector for each possible label instead. It is a
trainable lookup table with `num_embeddings` rows and `embedding_dim` columns:

$$
E \in \mathbb{R}^{V \times D}.
$$

Here $V$ is the number of labels (often called the vocabulary size), and $D$
is the number of values used to represent each label. Given an integer index
$i$, the forward pass selects its row:

$$
\operatorname{Embedding}(i) = E_i.
$$

For example, `Embedding(4, 3)` owns a table with four vectors of length
three. Looking up `[2, 0]` returns rows `2` and `0`, with shape `(2, 3)`.
More generally, an input of shape `(...,)` produces an output of shape
`(..., embedding_dim)`: the lookup adds one final vector axis.

```python
embedding = Embedding(4, 3)
vectors = embedding([2, 0])  # shape: (2, 3)
```

The input indices must be integers from `0` through
`num_embeddings - 1`. They choose rows; they are not trainable values. The
single trainable parameter is `embedding.weight`, the whole table $E$.

## Learning

Selecting a row is simple in the forward pass, but its backward behaviour is
important. If the loss sends gradient $g$ to a looked-up vector $E_i$, then
only that row receives it:

$$
\frac{\partial L}{\partial E_j} =
\begin{cases}
g & \text{if } j=i, \\
0 & \text{otherwise.}
\end{cases}
$$

## Initialization

The default `embedding_normal` initializer draws each table entry from a
mean-zero normal distribution with

$$
\sigma = \frac{1}{\sqrt{D}}.
$$

An embedding vector has $D$ independent entries, each with variance
$1 / D$. Its expected squared length is therefore about $D \cdot (1 / D)=1$.
This keeps initial vectors at a modest, comparable scale as the embedding
dimension changes; training then gives labels that appear in similar contexts
useful nearby representations.
