# 🌳 BonsaiGrad

A small, from-scratch deep learning library built on **NumPy** — nothing else.

The goal is understanding, not performance: every component is implemented by
hand so the mechanics stay visible.

Each step of development is documented in 📖 [`docs/`](docs/), which explains not
just *what* the code does but *why* it works.

## 🗺️ Roadmap

Build the pieces bottom-up, culminating in a trainable GPT:

- [ ] `Leaf` — the autograd node and reverse-mode backprop
- [ ] Tensor ops and broadcasting-aware gradients
- [ ] Layers (Linear, activations, LayerNorm, …) and the MLP
- [ ] Losses and optimizers (SGD, Adam)
- [ ] Attention, Transformer blocks, and a full GPT
- [ ] Training loop and a worked example
