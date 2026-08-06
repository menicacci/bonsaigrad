# 08 — Sequential

Neural networks are often made by applying one module after another. A
`Sequential` groups those modules into one model and calls them in order:

```python
model = Sequential(
    Linear(2, 3, rng),
    Tanh(),
    Linear(3, 1, rng),
)

outputs = model(inputs)
```

Calling `model(inputs)` is the same as writing

$$
\operatorname{Linear}_2(
    \tanh(\operatorname{Linear}_1(x))
).
$$

The output of each module becomes the input to the next. `Sequential` does not
add a new operation; it simply gives this chain one name and one call.

## Learning a determinant

> 🧪 Runnable companion: [`examples/08_learn_determinant.py`](../examples/08_learn_determinant.py)

The determinant of a two-by-two matrix is a small problem with four input
features and one output:

$$
\det
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
= ad - bc.
$$

The example generates matrices whose entries are between $-1$ and $1$. Each
matrix is flattened into one input row, `[a, b, c, d]`, while ordinary NumPy
calculates the target determinant. There is no external dataset or train/test
split: the generated matrices let us focus on the learning loop itself.

The model is a chain of five modules:

```python
model = Sequential(
    Linear(4, 16, rng),
    Tanh(),
    Linear(16, 16, rng),
    Tanh(),
    Linear(16, 1, rng),
)
```

For a batch of $N$ matrices, the shapes through the model are

$$
(N, 4) \longrightarrow (N, 16) \longrightarrow (N, 16)
\longrightarrow (N, 16) \longrightarrow (N, 16) \longrightarrow (N, 1).
$$

The `Tanh` modules are essential. A determinant contains products such as
$ad$ and $bc$, so it is not a linear function of its four entries. A chain of
`Linear` layers would still be one affine transformation and could not learn
those interactions. The activations give the network nonlinear functions with
which it can approximate them.

### Run it

Install the project, then run the example from the repository root:

```bash
pip install -e .
python examples/08_learn_determinant.py
```

It trains on 256 random matrices for 5,000 steps by default. Use `--examples`
and `--steps` to make a shorter experiment while exploring:

```bash
python examples/08_learn_determinant.py --examples 32 --steps 100
```
