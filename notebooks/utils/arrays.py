"""Small plotting helpers for explaining array values and shapes."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike


def show_array(ax: Axes, values: ArrayLike, title: str, *, cmap: str = "Blues") -> None:
    """Draw ``values`` as a labelled array, keeping its original shape in the title."""
    values = np.asarray(values)
    display_values = np.atleast_2d(values)

    ax.imshow(
        display_values,
        cmap=cmap,
        vmin=display_values.min() - 1,
        vmax=display_values.max() + 1,
    )
    for (row, column), value in np.ndenumerate(display_values):
        ax.text(column, row, f"{value:g}", ha="center", va="center")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\nshape {values.shape}", fontsize=10)


def show_vector(ax: Axes, values: ArrayLike, title: str, *, color: str = "#c1543c") -> None:
    """Draw a one-dimensional array as a labelled bar chart."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"show_vector expects one-dimensional values, got shape {values.shape}")

    positions = np.arange(values.size)
    ax.bar(positions, values, color=color)
    ax.axhline(0, color="#9aa0a6", linewidth=1)
    for position, value in zip(positions, values):
        ax.text(
            position,
            value,
            f"{value:g}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    ax.set_xticks(positions, [f"[{index}]" for index in positions])
    ax.set_title(f"{title}\nshape {values.shape}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
