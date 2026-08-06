# BonsaiGrad

A small deep learning library built **from scratch on NumPy** — building up to a
trainable GPT. A learning project: the goal is to understand how neural networks
work under the hood, so clarity beats speed. Core abstraction is the `Leaf`: a
computation-graph node that holds a value and propagates gradients backward.

## How we work

- **As partners.** Build it together — I want to understand every step. Think out
  loud, propose and discuss before implementing, push back and flag trade-offs.
- **Step by step.** Bottom-up, one component at a time, in small increments.
  Don't build on foundations that don't exist yet.
- **Explain everything.** Each component gets a `docs/NN-slug.md` covering the
  math and the why — not just the API. Update it in the same step as the code.

## Constraints

- **Library depends on `numpy` only** — build everything else by hand, no
  autograd/NN libraries. Notebook tooling (matplotlib, jupyter) lives in the
  `notebooks` optional extra, never in core deps.
- **Not a production library.** No CI, releases, or performance work — this is a
  learning artefact. Tests run by hand with `python -m tests`.
- Keep code clean and readable — a reference implementation, not a clever one.
- Ask before large doc restructures.
