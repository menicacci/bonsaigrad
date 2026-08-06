# 🌳 BonsaiGrad

A small, from-scratch deep learning library built on **NumPy** — nothing else.

The goal is understanding, not performance: every component is implemented by
hand so the mechanics stay visible.

Each step of development is documented in 📖 [`docs/`](docs/), which explains not
just *what* the code does but *why* it works.

## 🔬 Notebooks

The notebooks in 🔬 [`notebooks/`](notebooks/) visualise what each component does
(computation graphs, gradients, …). They need the optional `notebooks` extra —
the library itself stays NumPy-only:

```bash
pip install -e .[notebooks]
```

## 🧪 Examples

The programs in [`examples/`](examples/) are practical, runnable use cases that
put the library's pieces together.
