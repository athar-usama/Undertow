"""Rendering helpers shared by scripts/make_figures.py: a causal-graph diagram and a
per-site effect-size heatmap ("trace"), both built directly from the same effect-size
and graph objects the calibration/analysis scripts already compute, no synthetic
stand-ins.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from undertow.graph import CausalGraph, Site


def token_labels(tokens: np.ndarray, answer_pos: int) -> list[str]:
    from undertow import tasks

    labels = []
    for i, t in enumerate(tokens):
        if t == tasks.BOS:
            labels.append("BOS")
        elif t == tasks.SEP:
            labels.append("SEP")
        elif t == tasks.MID:
            labels.append("MID")
        elif t == tasks.ANS:
            labels.append("ANS")
        elif t == tasks.EOS:
            labels.append("EOS")
        else:
            labels.append(str(int(t) - tasks.NUM_SPECIAL))
    return labels


def render_causal_graph(
    graph: CausalGraph,
    longest_chain: list[Site],
    seq_len: int,
    n_layers: int,
    title: str,
    out_path: Path,
    token_text: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.55), 4.2))
    chain_set = set(longest_chain)
    chain_edges = set(pairwise(longest_chain))

    for src, dst in graph.edges:
        on_chain = (src, dst) in chain_edges
        arrow = FancyArrowPatch(
            (src[1], src[0]), (dst[1], dst[0]),
            connectionstyle="arc3,rad=0.15", arrowstyle="-|>", mutation_scale=10,
            color="#c0392b" if on_chain else "#b8c4ce",
            linewidth=2.2 if on_chain else 1.0,
            zorder=3 if on_chain else 1,
            alpha=1.0 if on_chain else 0.6,
        )
        ax.add_patch(arrow)

    for node in graph.nodes:
        on_chain = node in chain_set
        ax.scatter(
            [node[1]], [node[0]],
            s=140 if on_chain else 80,
            color="#c0392b" if on_chain else "#5b7c99",
            zorder=4, edgecolors="white", linewidths=1.2,
        )

    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-0.5, n_layers - 0.5)
    ax.invert_yaxis()
    ax.set_ylabel("layer")
    ax.set_xlabel("token position")
    if token_text is not None:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(token_text, rotation=90, fontsize=7)
    ax.set_title(title, fontsize=11, wrap=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_effect_heatmap(
    effects: dict[Site, float],
    seq_len: int,
    n_layers: int,
    title: str,
    out_path: Path,
    token_text: list[str] | None = None,
) -> None:
    grid = np.zeros((n_layers, seq_len))
    for (layer, pos), e in effects.items():
        grid[layer, pos] = e

    fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.55), 3.6))
    im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_ylabel("layer")
    ax.set_xlabel("token position")
    if token_text is not None:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(token_text, rotation=90, fontsize=7)
    ax.set_title(title, fontsize=11, wrap=True)
    fig.colorbar(im, ax=ax, label="patching effect (restores correct answer)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_depth_vs_k(
    results: dict[str, dict[int, float]],
    out_path: Path,
    title: str = "MNPC-depth vs. true hop count, zero-CoT",
    xlabel: str = "true hop count k",
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    all_k = sorted({k for series in results.values() for k in series})
    ax.plot(all_k, all_k, linestyle=":", color="#9aa5ad", linewidth=1.5, label="y = k (ground truth)")
    colors = {"primary": "#c0392b", "control": "#5b7c99"}
    markers = {"primary": "o", "control": "s"}
    for i, (name, series) in enumerate(results.items()):
        ks = sorted(series)
        vals = [series[k] for k in ks]
        color = colors.get(name) or list(colors.values())[i % 2]
        marker = markers.get(name) or list(markers.values())[i % 2]
        ax.plot(ks, vals, marker=marker, color=color, linewidth=2.0, markersize=7, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean MNPC-depth")
    ax.set_title(title, fontsize=11, wrap=True)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_accuracy_collapse(
    zero_cot: dict[int, float], with_reasoning: dict[int, float], out_path: Path,
    title: str = "Zero-shot accuracy collapses once the model can't show its work",
) -> None:
    ks = sorted(zero_cot)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for k in ks:
        ax.plot([0, 1], [with_reasoning[k], zero_cot[k]], color="#9aa5ad", linewidth=1.4, zorder=1)
    ax.scatter([0] * len(ks), [with_reasoning[k] for k in ks], s=130, color="#5b7c99",
               zorder=3, edgecolors="white", linewidths=1.2, label="allowed to show work")
    ax.scatter([1] * len(ks), [zero_cot[k] for k in ks], s=130, color="#c0392b",
               zorder=3, edgecolors="white", linewidths=1.2, label="zero-CoT (answer only)")
    for k in ks:
        ax.annotate(f"k={k}", (0, with_reasoning[k]), textcoords="offset points",
                    xytext=(-12, 0), ha="right", va="center", fontsize=9, color="#8a97a0")
    ax.set_xlim(-0.6, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["with reasoning", "zero-CoT"])
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title(title, fontsize=11, wrap=True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_opacity_gap_dotplot(
    ks: list[int], gaps: list[float], out_path: Path,
    title: str = "Real-model opacity gap by chain length",
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter(ks, gaps, s=160, color="#c0392b", zorder=3, edgecolors="white", linewidths=1.2)
    ax.hlines(0, min(ks) - 0.5, max(ks) + 0.5, color="#9aa5ad", linewidth=1.0, zorder=1)
    for k, g in zip(ks, gaps):
        ax.vlines(k, 0, g, color="#c0392b", alpha=0.4, linewidth=1.5, zorder=2)
    ax.set_xticks(ks)
    ax.set_xlabel("chained addition steps k")
    ax.set_ylabel("opacity gap (MNPC-depth - 1)")
    ax.set_title(title, fontsize=11, wrap=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_scaling_comparison(
    present: list[tuple[str, str, float]],
    scaling: dict,
    out_path: Path,
    title: str = "Does capability -- and the gap -- grow with scale?",
) -> None:
    """One line per k: zero-shot accuracy (solid) and with-reasoning accuracy (dashed)
    against model parameter count, log-scaled. `present` is [(slug, display_name,
    params_billions), ...] in size order; `scaling` is the loaded scaling.json."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    params = [p for _, _, p in present]
    all_ks = sorted({int(k) for slug, _, _ in present for k in scaling[slug]["zero_cot_accuracy_by_k"]})
    palette = ["#c0392b", "#5b7c99", "#a06cd5", "#3f9c6d"]

    for i, k in enumerate(all_ks):
        color = palette[i % len(palette)]
        zero = [scaling[slug]["zero_cot_accuracy_by_k"].get(str(k)) for slug, _, _ in present]
        reason = [
            scaling[slug].get("with_reasoning_accuracy_by_k", {}).get(str(k)) for slug, _, _ in present
        ]
        zx = [p for p, v in zip(params, zero) if v is not None]
        zy = [v for v in zero if v is not None]
        rx = [p for p, v in zip(params, reason) if v is not None]
        ry = [v for v in reason if v is not None]
        if zy:
            ax.plot(zx, zy, marker="o", color=color, linewidth=2.0, markersize=7, label=f"k={k}, zero-CoT")
        if ry:
            ax.plot(rx, ry, marker="o", color=color, linewidth=1.6, markersize=6,
                     linestyle="--", alpha=0.55, label=f"k={k}, with reasoning")

    ax.set_xscale("log")
    ax.set_xticks(params)
    ax.set_xticklabels([name for _, name, _ in present])
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("model size")
    ax.set_ylabel("exact-match accuracy")
    ax.set_title(title, fontsize=11, wrap=True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
