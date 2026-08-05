"""Plotting helpers for the scalar computation graphs in notebook 01."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from bonsaigrad import Leaf


_INK = "#2f3437"
_MUTED = "#9aa0a6"
_IDLE_EDGE = "#c9ccd1"
_ROOT_FILL = "#ffffff"
_OPERATION_FILL = "#f4f5f6"
_FORWARD = "#3f6d8e"
_FORWARD_FILL = "#ecf2f7"
_BACKWARD = "#c1543c"
_BACKWARD_FILL = "#fdece7"
_SYMBOLS = {"+": "+", "*": "×"}


def backward_levels(apex: Leaf) -> tuple[tuple[Leaf, ...], ...]:
    """Return graph nodes in backwards-valid levels, from ``apex`` to the roots.

    Every node in a level can run its local ``_backward`` function independently
    once the preceding level has finished. The helper makes the manual
    backpropagation walkthrough in the notebook explicit without duplicating
    graph traversal code there.
    """
    nodes, _ = _trace(apex)
    depths: dict[int, int] = {}
    levels: dict[int, list[Leaf]] = {}
    for node in nodes:
        levels.setdefault(_depth(node, depths), []).append(node)

    return tuple(tuple(levels[depth]) for depth in sorted(levels, reverse=True))


def draw_graph(
    apex: Leaf,
    names: Mapping[int, str] | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    edge_labels: bool = False,
    fired: Collection[int] | None = None,
    flow: Literal["up", "down"] = "down",
) -> Axes:
    """Draw the scalar computation graph that feeds ``apex``.

    ``flow="up"`` visualises values travelling from roots to the output.
    ``flow="down"`` visualises gradients travelling back to the roots. Pass the
    identities of nodes whose local backward functions have run in ``fired`` to
    show an intermediate backward-pass state.
    """
    if flow not in {"up", "down"}:
        raise ValueError(f"flow must be 'up' or 'down', got {flow!r}")

    names = names or {}
    nodes, edges = _trace(apex)
    _require_scalar_nodes(nodes)
    positions, top_depth, shoots = _layout(nodes, edges)
    is_forward = flow == "up"
    accent, accent_fill = (
        (_FORWARD, _FORWARD_FILL) if is_forward else (_BACKWARD, _BACKWARD_FILL)
    )

    fired_nodes = set(fired or ())
    live_nodes = _live_nodes(apex, nodes, shoots, fired_nodes, is_forward)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 1.5 + 1.4 * top_depth))

    for stem, shoot in edges:
        _draw_edge(
            ax,
            stem,
            shoot,
            positions,
            accent,
            is_forward,
            fired_nodes,
            edge_labels,
        )

    for node in nodes:
        _draw_node(ax, node, positions, shoots, names, live_nodes, is_forward, accent, accent_fill)

    x_positions = [position[0] for position in positions.values()]
    ax.set_xlim(min(x_positions) - 0.85, max(x_positions) + 0.85)
    ax.set_ylim(-0.5, top_depth + 0.5)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.5, color=_INK, pad=8)
    return ax


def _trace(apex: Leaf) -> tuple[list[Leaf], list[tuple[Leaf, Leaf]]]:
    nodes: list[Leaf] = []
    edges: list[tuple[Leaf, Leaf]] = []
    seen: set[int] = set()

    def walk(node: Leaf) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        nodes.append(node)
        for stem in node._stems:
            edges.append((stem, node))
            walk(stem)

    walk(apex)
    return nodes, edges


def _depth(node: Leaf, depths: dict[int, int]) -> int:
    if id(node) not in depths:
        depths[id(node)] = (
            0 if not node._stems else 1 + max(_depth(stem, depths) for stem in node._stems)
        )
    return depths[id(node)]


def _layout(
    nodes: list[Leaf], edges: list[tuple[Leaf, Leaf]]
) -> tuple[dict[int, tuple[float, int]], int, dict[int, list[Leaf]]]:
    """Place nodes by depth, then reduce edge crossings with barycentre ordering."""
    depths: dict[int, int] = {}
    levels: dict[int, list[Leaf]] = {}
    for node in nodes:
        levels.setdefault(_depth(node, depths), []).append(node)

    x_positions: dict[int, float] = {}
    shoots: dict[int, list[Leaf]] = {}
    for stem, shoot in edges:
        shoots.setdefault(id(stem), []).append(shoot)
    for level in levels.values():
        _centre_nodes(level, x_positions)

    top_depth = max(levels)
    for _ in range(4):
        for depth in sorted(levels):
            if depth == top_depth:
                continue
            levels[depth].sort(key=lambda node: _barycentre(node, shoots, x_positions))
            _centre_nodes(levels[depth], x_positions)

    positions = {
        id(node): (x_positions[id(node)], depth)
        for depth, level in levels.items()
        for node in level
    }
    return positions, top_depth, shoots


def _centre_nodes(nodes: list[Leaf], x_positions: dict[int, float]) -> None:
    for index, node in enumerate(nodes):
        x_positions[id(node)] = index - (len(nodes) - 1) / 2


def _barycentre(
    node: Leaf, shoots: Mapping[int, list[Leaf]], x_positions: Mapping[int, float]
) -> float:
    parents = shoots.get(id(node), [])
    if not parents:
        return x_positions[id(node)]
    return sum(x_positions[id(parent)] for parent in parents) / len(parents)


def _live_nodes(
    apex: Leaf,
    nodes: list[Leaf],
    shoots: Mapping[int, list[Leaf]],
    fired: set[int],
    is_forward: bool,
) -> set[int]:
    if is_forward:
        return {id(node) for node in nodes}

    live = {
        id(node)
        for node in nodes
        if any(id(shoot) in fired for shoot in shoots.get(id(node), []))
    }
    if apex.grad.item():
        live.add(id(apex))
    return live


def _draw_edge(
    ax: Axes,
    stem: Leaf,
    shoot: Leaf,
    positions: Mapping[int, tuple[float, int]],
    accent: str,
    is_forward: bool,
    fired: set[int],
    edge_labels: bool,
) -> None:
    (x0, y0), (x1, y1) = positions[id(stem)], positions[id(shoot)]
    is_live = is_forward or id(shoot) in fired
    tail, head = ((x0, y0), (x1, y1)) if is_forward else ((x1, y1), (x0, y0))
    ax.add_patch(
        FancyArrowPatch(
            tail,
            head,
            arrowstyle="-|>",
            mutation_scale=11,
            color=accent if is_live else _MUTED,
            lw=1.7 if is_live else 1.1,
            shrinkA=30,
            shrinkB=30,
            zorder=1,
        )
    )

    local_derivative = _local_derivative(shoot, stem)
    if edge_labels and not is_forward and local_derivative not in (None, 1.0):
        _draw_edge_label(ax, x0, y0, x1, y1, local_derivative, accent if is_live else _MUTED)


def _draw_edge_label(
    ax: Axes, x0: float, y0: int, x1: float, y1: int, derivative: float, color: str
) -> None:
    dx, dy = x0 - x1, y0 - y1
    norm = (dx * dx + dy * dy) ** 0.5 or 1
    ax.text(
        (x0 + x1) / 2 - dy / norm * 0.13,
        (y0 + y1) / 2 + dx / norm * 0.13,
        f"×{derivative:.4g}",
        fontsize=8.5,
        zorder=3,
        color=color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"),
    )


def _draw_node(
    ax: Axes,
    node: Leaf,
    positions: Mapping[int, tuple[float, int]],
    shoots: Mapping[int, list[Leaf]],
    names: Mapping[int, str],
    live_nodes: set[int],
    is_forward: bool,
    accent: str,
    accent_fill: str,
) -> None:
    x, y = positions[id(node)]
    is_live = id(node) in live_nodes
    lines = [_node_heading(node, names), f"data  {node.data.item():.4g}"]
    if not is_forward:
        lines.append(_gradient_line(node, shoots, is_live))

    is_operation = bool(node._stems)
    fill = (
        accent_fill if is_operation else _ROOT_FILL
    ) if is_live else (_OPERATION_FILL if is_operation else _ROOT_FILL)
    ax.text(
        x,
        y,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=8.5,
        color=_INK,
        linespacing=1.5,
        zorder=2,
        bbox=dict(
            boxstyle="round,pad=0.42",
            lw=1.7 if is_live else 1.1,
            fc=fill,
            ec=accent if is_live else _IDLE_EDGE,
        ),
    )


def _node_heading(node: Leaf, names: Mapping[int, str]) -> str:
    name = names.get(id(node))
    operation = _SYMBOLS.get(node._op, node._op)
    if node._stems:
        label = f"{name} = {operation}" if name else operation
    else:
        label = name or "root"
    return f"$\\bf{{{label}}}$"


def _gradient_line(node: Leaf, shoots: Mapping[int, list[Leaf]], is_live: bool) -> str:
    line = f"grad  {node.grad.item():.4g}"
    parents = shoots.get(id(node), [])
    if is_live and len(parents) > 1:
        terms = [_gradient_contribution(parent, node) for parent in parents]
        if abs(sum(terms) - node.grad.item()) < 1e-9:
            line += "   ({})".format(" + ".join(f"{term:.4g}" for term in terms))
    return line


def _gradient_contribution(shoot: Leaf, stem: Leaf) -> float:
    derivative = _local_derivative(shoot, stem)
    if derivative is None:
        raise ValueError(f"cannot label a gradient contribution for operation {shoot._op!r}")
    return shoot.grad.item() * derivative


def _local_derivative(node: Leaf, stem: Leaf) -> float | None:
    if node._op == "+":
        return 1.0
    if node._op == "*":
        left, right = node._stems
        return (right if left is stem else left).data.item()
    return None


def _require_scalar_nodes(nodes: Collection[Leaf]) -> None:
    if any(node.data.size != 1 for node in nodes):
        raise ValueError("draw_graph only supports scalar Leaf values")
