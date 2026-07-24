# 01 — The `Leaf`

Before everything, we
need a way to compute **gradients automatically**. The `Leaf` is that
machinery.

## Why?

Training a neural network is one idea repeated a few million times:

> Make a prediction, measure how wrong it was, then nudge every parameter a
> little in the direction that makes it *less* wrong.

"The direction that makes it less wrong" is the whole game. For a single number
that direction is obvious — if the error goes up when you increase a knob, turn
the knob down. But a network has thousands to billions of knobs, all tangled
together through layers of arithmetic. We need a way to ask, for **every knob at
once**:

> If I nudge *this* knob a tiny bit, which way and how hard does the final error
> move?

That answer, for one knob, is its **gradient**. Computing all of them, exactly
and cheaply, is what the `Leaf` exists to do.

## Gradient

Take any expression that ends in a single output — call it the *loss*, `L`. Pick
one input `x` somewhere inside. The gradient of `L` with respect to `x`, written
`∂L/∂x`, answers a simple question:

> If I increase `x` by a tiny amount, how much does `L` change, and in which
> direction?

- `∂L/∂x = 2` → nudging `x` up by `ε` pushes `L` up by about `2ε`. To *shrink*
  `L`, move `x` **down**.
- `∂L/∂x = -3` → nudging `x` up pulls `L` *down* by about `3ε`. To shrink `L`,
  move `x` **up**.
- `∂L/∂x = 0` → `x` has no first-order effect on `L` right here.

So a gradient is two things bundled together: a **direction** (its sign) and a
**strength** (its magnitude). Collect the gradient for every parameter and you
have a recipe: "to reduce the loss, move each parameter a small step *against*
its gradient."

In BonsaiGrad every `Leaf` carries its value in `data` and, after we ask for it,
its gradient in `grad`:

```python
leaf.data   # the value this node holds
leaf.grad   # ∂L/∂(this node), filled in by the backward pass
```

## Graph

We never write one giant formula and try to
differentiate it. Instead, **as an expression is built, it records itself.**

Every operation (`+`, `*`, …) produces a new `Leaf` that remembers the parents
it came from and which operation made it. So this line:

```python
a = Leaf(3.0)
b = Leaf(4.0)
f = a * b + a
```

does not just compute `15.0`. It quietly grows a little tree:

```
        f  (+)
       /  \
   (a*b)   a
    / \    ^
   a   b   |
   ^-------+
```

- The **roots** at the bottom are the inputs we created by hand (`a`, `b`).
- Each **interior node** is an intermediate result (`a*b`, then `f`).
- The single node at the **top** is the final output — the loss, once we have
  one.

This is the bonsai. The forward pass *grows* the tree from roots up to the
output. The backward pass — `bend` — sends a shaping signal from the top back
down to every root, telling each one which way it should lean. Each node stores
its parents in `_prev` and a label in `_op`:

```python
f._prev   # (Leaf(a*b), Leaf(a))
f._op     # "+"
```

## One node, four parts

Everything a node needs fits in four fields, set in `__init__`:

| field       | what it is                                                       |
|-------------|------------------------------------------------------------------|
| `data`      | the value, always a float `np.ndarray`                           |
| `grad`      | `∂L/∂(this node)`; same shape as `data`; zero until the backward pass |
| `_prev`     | the parent nodes this one was built from                         |
| `_backward` | a function that pushes this node's gradient to its parents       |

`_backward` is the interesting one. Each node knows how to hand its own gradient
back to *its* parents — and nothing more. It is a tiny, local rule. The magic is
that chaining these local rules together, in the right order, computes the
gradient of the whole tree. That chaining rule has a name.

## Chain Rule

The chain rule is the one piece of calculus we truly need. In words:

> If `L` depends on `y`, and `y` depends on `x`, then how `L` responds to `x` is
> **how `L` responds to `y`** times **how `y` responds to `x`**.

As a formula:

```
∂L/∂x = (∂L/∂y) · (∂y/∂x)
```

`∂L/∂y` is the gradient that has already arrived at `y`
from above. `∂y/∂x` is the *local* effect of `x` on `y` — something `y`'s own
operation knows. Multiply them and the gradient has travelled one more step down
the tree, from `y` to `x`. Every node does this handoff, and the signal flows
all the way to the roots.

### Addition

For `out = a + b`, nudging `a` up by `ε` moves `out` up by exactly `ε` (same for
`b`). So the local derivatives are both `1`:

```
∂out/∂a = 1        ∂out/∂b = 1
```

Plug into the chain rule — the gradient arriving at `out` is `out.grad` — and
each parent receives it **unchanged**:

```
∂L/∂a = out.grad · 1 = out.grad
∂L/∂b = out.grad · 1 = out.grad
```

Which is exactly the code:

```python
self.grad  += out.grad
other.grad += out.grad
```

Addition is a splitter: it copies the incoming gradient to every input.

### Multiplication

For `out = a * b`, nudging `a` changes `out` in proportion to `b` (and vice
versa). The local derivatives are:

```
∂out/∂a = b        ∂out/∂b = a
```

So each parent gets the incoming gradient scaled by the **other** parent's
value:

```
∂L/∂a = out.grad · b
∂L/∂b = out.grad · a
```

Again, exactly the code:

```python
self.grad  += other.data * out.grad
other.grad += self.data  * out.grad
```

Multiplication is a cross-wire: each input's gradient leans on its partner's
value.

## Why `+=`, not `=`

Notice both rules *accumulate* with `+=`. This matters whenever a node is used
more than once — like `a` in `f = a * b + a`. That `a` influences `f` down two
different paths: once through the product, once directly. The chain rule says
its total gradient is the **sum** over every path it takes to the top.

## `bend`: the backward pass

`bend` is the whole backward pass. It does three things.

**1. Put the nodes in order.** A node can only hand gradient to its parents once
it has received its own. So we must process nodes top-down: the output first,
roots last. `bend` finds this order with a depth-first walk that lists a node
only *after* all its parents, then reverses it — a standard topological sort:

```python
topo: list[Leaf] = []
visited: set[Leaf] = set()

def build(node):
    if node not in visited:
        visited.add(node)
        for parent in node._prev:
            build(parent)
        topo.append(node)

build(self)
```

**2. Seed the top.** The gradient of the output with respect to itself is,
trivially, `1` (`∂L/∂L = 1`). That is the spark that starts the relay:

```python
self.grad = np.ones_like(self.data)
```

**3. Let each node hand off its gradient.** Walk the order in reverse (top to
bottom) and fire each node's local `_backward`. By the time we reach a node, its
`grad` is already complete, so its handoff to its parents is correct:

```python
for n in reversed(topo):
    n._backward()
```

When the loop ends, **every** node — including the roots we care about — holds
its gradient. One pass, exact answers, cost proportional to the size of the
tree.

## Example

Take `f = a * b + a` with `a = 3`, `b = 4`. Let us predict the gradients before
running anything.

`f = a·b + a`, so differentiate:

```
∂f/∂a = b + 1 = 4 + 1 = 5      (a appears in the product and on its own)
∂f/∂b = a     = 3
```

Now trace what `bend` does, top to bottom. Seed `f.grad = 1`.

- `f = (a*b) + a` is an addition → copies its gradient to both parents:
  `(a*b).grad = 1`, and `a.grad += 1`.
- `(a*b)` is a multiplication → each parent gets `(a*b).grad` times the other's
  value: `a.grad += 1·b = 4`, `b.grad += 1·a = 3`.

Totals: `a.grad = 1 + 4 = 5`, `b.grad = 3`. Exactly the hand calculation — and
the `+=` on `a` is what let its two paths add up.

In code:

```python
from bonsaigrad import Leaf

a = Leaf(3.0)
b = Leaf(4.0)
f = a * b + a
f.bend()

f.data    # 15.0
a.grad    # 5.0
b.grad    # 3.0
```
