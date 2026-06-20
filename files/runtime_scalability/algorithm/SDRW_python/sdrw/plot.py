"""Optional density-curve plots (matplotlib)."""

from __future__ import annotations

from typing import Sequence


def plot_y_points(
    y: Sequence[float],
    line_size: int,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    xs = list(range(1, len(y) + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(xs, list(y), linewidth=line_size)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out = f"{filename}_multi_SBS_FG_SL3.png"
    plt.savefig(out)
    plt.close()
