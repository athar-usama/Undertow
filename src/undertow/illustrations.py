"""Narrative illustrations of a single real, patched example -- not data
visualizations of a metric across many examples (that's viz.py), but drawn
depictions of what the causal trace on *one* concrete case actually looks like.
Every number and position used here is read from a results/real_model/*/raw.jsonl
record, not invented for effect.
"""

from __future__ import annotations

import itertools
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D


def render_building_cutaway(
    lit_floors_by_column: dict[str, list[int]],
    n_layers: int,
    column_labels: dict[str, str],
    entry_label: str,
    exit_label: str,
    out_path: Path,
    title: str = "Where the signal actually lives, floor by floor",
) -> None:
    """A cutaway of a tower with `n_layers` floors. `lit_floors_by_column` maps a
    column key (e.g. a token position) to the list of floor indices (layers) found
    necessary there; floors not listed for any column are drawn dark. A staircase
    line is drawn through the lit floors in increasing-layer order, jumping (dashed)
    across any unlit gap between columns."""
    columns = list(lit_floors_by_column.keys())
    col_x = {c: i for i, c in enumerate(columns)}
    width_per_col = 2.2
    building_width = width_per_col * len(columns)

    fig, ax = plt.subplots(figsize=(4.2 + 1.6 * len(columns), 8.5))

    for floor in range(n_layers):
        for c in columns:
            lit = floor in lit_floors_by_column[c]
            x0 = col_x[c] * width_per_col
            color = "#e8a33d" if lit else "#232935"
            edge = "#f4c67a" if lit else "#30363f"
            rect = Rectangle((x0 + 0.12, floor), width_per_col - 0.24, 0.92,
                              facecolor=color, edgecolor=edge, linewidth=1.0, zorder=2)
            ax.add_patch(rect)

    ax.add_patch(Rectangle((0, 0), building_width, n_layers, facecolor="none",
                            edgecolor="#5b6472", linewidth=2.0, zorder=3))
    for i in range(1, len(columns)):
        ax.axvline(i * width_per_col, color="#5b6472", linewidth=1.2, zorder=3)

    stair_points = []
    for c in columns:
        for floor in sorted(lit_floors_by_column[c]):
            stair_points.append((col_x[c] * width_per_col + width_per_col / 2, floor + 0.46))
    for (x1, y1), (x2, y2) in itertools.pairwise(stair_points):
        same_column = abs(x1 - x2) < 1e-6
        ax.plot([x1, x2], [y1, y2], color="#c0392b", linewidth=2.6 if same_column else 1.8,
                 linestyle="-" if same_column else "--", zorder=5,
                 solid_capstyle="round", alpha=1.0 if same_column else 0.85)

    for c in columns:
        ax.text(col_x[c] * width_per_col + width_per_col / 2, -0.9, column_labels[c],
                ha="center", va="top", fontsize=10, color="#8a97a0")

    entry_x, entry_y = stair_points[0]
    ax.annotate(entry_label, (entry_x, entry_y), xytext=(entry_x, -2.6),
                ha="center", fontsize=10, color="#c0392b",
                arrowprops={"arrowstyle": "-|>", "color": "#c0392b", "linewidth": 1.6})
    exit_x, exit_y = stair_points[-1]
    ax.annotate(exit_label, (exit_x, exit_y), xytext=(exit_x, n_layers + 1.8),
                ha="center", fontsize=11, color="#c0392b", fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "color": "#c0392b", "linewidth": 1.8})

    ax.set_xlim(-0.6, building_width + 0.6)
    ax.set_ylim(-3.6, n_layers + 3.2)
    ax.set_yticks(range(0, n_layers, 2))
    ax.set_yticklabels([f"floor {i}" for i in range(0, n_layers, 2)], fontsize=8, color="#8a97a0")
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#0b0e13")
    ax.set_title(title, fontsize=13, color="#e8ecf1")
    fig.patch.set_facecolor("#0b0e13")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="#0b0e13")
    plt.close(fig)


def render_circuit_thumbnail(
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    chain: list[tuple[int, int]],
    n_layers: int,
    out_path: Path,
) -> None:
    """A small, tightly-cropped circuit diagram for embedding as a card image: unlike
    render_causal_graph, positions are placed at their *rank* among only the distinct
    positions that actually appear in `nodes`, not their raw token index -- a real
    model's necessary sites can sit 20+ token positions apart with nothing relevant in
    between, and plotting raw indices would render mostly empty space."""
    positions = sorted({p for _, p in nodes})
    pos_rank = {p: i for i, p in enumerate(positions)}

    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    chain_set = set(chain)
    chain_edges = set(itertools.pairwise(chain))
    for src, dst in edges:
        on_chain = (src, dst) in chain_edges
        ax.plot([pos_rank[src[1]], pos_rank[dst[1]]], [src[0], dst[0]],
                color="#c0392b" if on_chain else "#c7bfae", linewidth=2.4 if on_chain else 1.0,
                alpha=1.0 if on_chain else 0.5, zorder=2 if on_chain else 1)
    for n in nodes:
        on_chain = n in chain_set
        ax.scatter([pos_rank[n[1]]], [n[0]], s=90 if on_chain else 50,
                    color="#c0392b" if on_chain else "#8a97a0", zorder=3,
                    edgecolors="white", linewidths=1.0)
    ax.set_xlim(-0.6, len(positions) - 0.4)
    ax.set_ylim(-1, n_layers)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f4f1e8")
    fig.patch.set_facecolor("#f4f1e8")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _rotated_rect(ax, xy, width, height, angle_deg, **kwargs):
    rect = Rectangle(xy, width, height, **kwargs)
    t = Affine2D().rotate_deg_around(xy[0] + width / 2, xy[1] + height / 2, angle_deg) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    return rect


def _shadowed_rect(ax, xy, width, height, angle_deg, **kwargs):
    """A rotated rect with a soft dark offset copy behind it, for a raised-paper look."""
    shadow_xy = (xy[0] + 0.09, xy[1] - 0.09)
    _rotated_rect(ax, shadow_xy, width, height, angle_deg, facecolor="#000000",
                  edgecolor="none", alpha=0.28, zorder=kwargs.get("zorder", 2) - 1)
    return _rotated_rect(ax, xy, width, height, angle_deg, **kwargs)


def render_corkboard(
    prompt_text: str,
    highlighted_word: str,
    circuit_thumb_path: Path,
    verdict_lines: list[str],
    out_path: Path,
    title: str = "The case file",
) -> None:
    """A collaged evidence-board image: a pinned transcript card, a pinned circuit
    thumbnail, a red string between them, and a handwritten-style verdict note."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(7)
    cork = gaussian_filter(rng.uniform(0.0, 1.0, size=(160, 220)), sigma=4.0)
    cork = 0.44 + 0.14 * (cork - cork.min()) / (cork.max() - cork.min())
    ax.imshow(cork, extent=(0, 11, 0, 8), cmap="copper", vmin=0.25, vmax=0.62, zorder=0,
              aspect="auto", interpolation="bicubic")

    card1_xy = (0.6, 4.3)
    card1_w, card1_h = 4.3, 2.7
    _shadowed_rect(ax, card1_xy, card1_w, card1_h, -2.5, facecolor="#f4f1e8",
                   edgecolor="#d8d2c0", linewidth=1.0, zorder=2)
    words = prompt_text.split(" ")
    wrapped, line = [], ""
    for w in words:
        if len(line) + len(w) > 30:
            wrapped.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    wrapped.append(line)
    for i, line in enumerate(wrapped):
        segs = line.split(highlighted_word)
        color = "#c0392b" if len(segs) > 1 else "#1a1a1a"
        weight = "bold" if len(segs) > 1 else "normal"
        ax.text(card1_xy[0] + 0.3, card1_xy[1] + card1_h - 0.5 - i * 0.4, line,
                 fontsize=10.5, color=color, fontweight=weight, zorder=3, rotation=-2.5,
                 rotation_mode="anchor")
    ax.add_patch(Circle((card1_xy[0] + card1_w * 0.5, card1_xy[1] + card1_h - 0.15),
                        0.11, facecolor="#c0392b", zorder=4, edgecolor="#7a1f18", linewidth=1.0))

    thumb = plt.imread(circuit_thumb_path)
    card2_x0, card2_y0, card2_w, card2_h = 6.0, 3.7, 4.1, 3.4
    pad = 0.18
    _shadowed_rect(ax, (card2_x0, card2_y0), card2_w, card2_h, 0.0, facecolor="#f4f1e8",
                   edgecolor="#d8d2c0", linewidth=1.0, zorder=2)
    ax.imshow(
        thumb, extent=(card2_x0 + pad, card2_x0 + card2_w - pad,
                        card2_y0 + pad + 0.35, card2_y0 + card2_h - pad),
        zorder=3, aspect="auto",
    )
    ax.text(card2_x0 + card2_w / 2, card2_y0 + pad + 0.15, "the circuit that explains it",
             ha="center", va="bottom", fontsize=8.5, color="#5b6472", style="italic", zorder=3)
    pin2_xy = (card2_x0 + card2_w / 2, card2_y0 + card2_h - 0.18)
    ax.add_patch(Circle(pin2_xy, 0.11, facecolor="#c0392b", zorder=4, edgecolor="#7a1f18", linewidth=1.0))

    string_start = (card1_xy[0] + card1_w * 0.5, card1_xy[1] + card1_h - 0.15)
    ax.plot([string_start[0], pin2_xy[0]], [string_start[1], pin2_xy[1]],
            color="#a4271d", linewidth=2.0, zorder=1, alpha=0.9)

    note_w = 5.6
    verdict_wrapped: list[str] = []
    for raw_line in verdict_lines:
        verdict_wrapped.extend(textwrap.fill(raw_line, width=52).split("\n"))
    line_h = 0.42
    note_h = 0.9 + line_h * len(verdict_wrapped)
    note_xy = (2.6, 0.4)
    _shadowed_rect(ax, note_xy, note_w, note_h, 1.2, facecolor="#fff7d6",
                   edgecolor="#e0d29a", linewidth=1.0, zorder=2)
    for i, line in enumerate(verdict_wrapped):
        ax.text(note_xy[0] + 0.35, note_xy[1] + note_h - 0.5 - i * line_h, line, fontsize=10.5,
                 color="#3a2f1a", style="italic", zorder=3, rotation=1.2, rotation_mode="anchor")
    ax.add_patch(Circle((note_xy[0] + note_w * 0.5, note_xy[1] + note_h - 0.18),
                        0.1, facecolor="#c0392b", zorder=4, edgecolor="#7a1f18", linewidth=1.0))

    ax.set_title(title, fontsize=17, color="#e8ecf1", pad=14)
    fig.patch.set_facecolor("#12161d")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor="#12161d")
    plt.close(fig)
