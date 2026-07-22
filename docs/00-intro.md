# 00 — Introduction

Welcome to BonsaiGrad. This first document sets the stage: what we are building,
why we are building it this particular way, and the mental model — the bonsai —
that ties the whole project together. Everything that follows in `docs/` assumes
the picture painted here.

## What we are building

BonsaiGrad is a small deep learning library written **from scratch on top of
NumPy**, with no other dependencies. We will implement every piece by hand.

We get there bottom-up, one layer of abstraction at a time, the goal is 
**understanding, not performance**.

## The bonsai metaphor

The name is the mental model.

A bonsai is a **full-grown tree in miniature**. It is the same tree, with the 
same trunk, branches, roots, and leaves, grown deliberately small so you can 
hold the whole thing in view and see how every part connects. The structure is 
faithful; only the scale is reduced.

BonsaiGrad is a bonsai of a real deep learning stack. We are simply growing it 
small enough that no part is hidden.

Two ideas carry through the whole project:
- **Faithful, not simplified**: We shrink the scale, never the structure.
- **Every branch visible:** If you cannot see how a gradient flows through a
  piece, that piece is not done being explained.

## How to read these docs

Each document explains both *what* the code does and *why* it works: we derive
the math, track the array shapes, and follow the gradient flow explicitly. They
are meant to be read in order — later pieces lean on earlier ones — and to be
read **alongside the code**, not instead of it.
