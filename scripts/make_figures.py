"""Renders every figure used in the README/assets from already-computed results:
results/grid.jsonl (training), results/calibration.jsonl (the calibration protocol),
and results/real_model.json (the real-Qwen phase). Nothing here re-runs an experiment;
it only visualizes numbers already on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from undertow import tasks
from undertow.graph import longest_path, longest_path_length
from undertow.model import ModelConfig, TinyTransformer
from undertow.patch import build_graph, effect_sizes
from undertow.viz import (
    render_accuracy_collapse,
    render_causal_graph,
    render_depth_vs_k,
    render_effect_heatmap,
    render_opacity_gap_dotplot,
    render_scaling_comparison,
    token_labels,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

M = 32
D_MODEL, D_FF, N_HEADS, MAIN_DEPTH = 48, 192, 4, 6
BIJECTION_SEED = 123
PERMUTATION_SEED = 456


def load_model(tag: str, seq_len: int) -> TinyTransformer:
    cfg = ModelConfig(vocab_size=tasks.vocab_size(M), seq_len=seq_len, n_layers=MAIN_DEPTH,
                       d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS)
    model = TinyTransformer(cfg)
    model.load_state_dict(torch.load(CHECKPOINTS / f"{tag}.pt", map_location="cpu"))
    model.eval()
    return model


def figure_depth_vs_k() -> None:
    lines = [json.loads(l) for l in (RESULTS / "calibration.jsonl").read_text().splitlines()]
    control_series = None
    primary_series = None
    for rec in lines:
        series = {int(k): v for k, v in rec["heldout_mean_depth_by_x"].items()}
        if rec["family"] == "control_depth_vs_k":
            control_series = series
        elif rec["family"] == "primary_depth_vs_required_silent_depth":
            primary_series = series

    if control_series:
        render_depth_vs_k(
            {"control (associative, 100% accurate at every k)": control_series},
            ASSETS / "control_depth_vs_k.png",
            title="Control task: causal depth stays flat while accuracy stays perfect",
        )
    if primary_series:
        render_depth_vs_k(
            {"primary (non-affine, keyed)": primary_series},
            ASSETS / "primary_depth_vs_silent.png",
            title="Primary task: causal depth tracks silent depth, not raw hop count",
            xlabel="required silent depth (hops not shown in the CoT)",
        )


def figure_example_graphs(task_type: str, t_hops: int, cot_budget: str, tau: float, top_k: int,
                           tag_suffix: str) -> None:
    bijection = tasks.make_substitution_bijection(seed=BIJECTION_SEED, m=M)
    permutation = tasks.make_fixed_permutation(seed=PERMUTATION_SEED, m=M)
    rng = np.random.default_rng(20_000 + t_hops)
    if task_type == "primary":
        clean = tasks.sample_primary(rng, M, t_hops, cot_budget, bijection)
        hop_index = int(rng.integers(0, t_hops))
        corrupted = tasks.resample_hop_input(clean, hop_index, rng, M, "primary", bijection, None)
    else:
        clean = tasks.sample_control(rng, M, t_hops, cot_budget, permutation)
        corrupted = tasks.resample_hop_input(clean, 0, rng, M, "control", None, permutation)

    tag = f"{task_type}_k{t_hops}_{cot_budget}"
    model = load_model(tag, len(clean.tokens))
    clean_tokens = torch.from_numpy(clean.tokens[:-1]).long().unsqueeze(0)
    corrupted_tokens = torch.from_numpy(corrupted.tokens[:-1]).long().unsqueeze(0)
    clean_answer_token = int(clean.tokens[clean.answer_pos + 1])

    effects = effect_sizes(model, clean_tokens, corrupted_tokens, clean.answer_pos, clean_answer_token)
    graph = build_graph(model, clean_tokens, corrupted_tokens, effects, tau=tau, top_k=top_k)
    chain = longest_path(graph)

    labels = token_labels(clean.tokens[:-1], clean.answer_pos)
    render_causal_graph(
        graph, chain, len(clean_tokens[0]), MAIN_DEPTH,
        title=f"{task_type} task, k={t_hops}, {cot_budget}-CoT (MNPC-depth={longest_path_length(graph)})",
        out_path=ASSETS / f"circuit_{tag_suffix}.png", token_text=labels,
    )
    render_effect_heatmap(
        effects, len(clean_tokens[0]), MAIN_DEPTH,
        title=f"{task_type} task, k={t_hops}: where the answer is actually decided",
        out_path=ASSETS / f"heatmap_{tag_suffix}.png", token_text=labels,
    )


# (slug, display name, parameter count in billions) for every model this project has
# swept -- add a row here (and run the sweep) to extend the scaling comparison.
MODEL_CATALOG = [
    ("qwen2.5-0.5b-instruct", "Qwen2.5-0.5B", 0.5),
    ("qwen2.5-1.5b-instruct", "Qwen2.5-1.5B", 1.5),
    ("qwen2.5-3b-instruct", "Qwen2.5-3B", 3.0),
]


def figure_real_model() -> None:
    real_model_dir = RESULTS / "real_model"
    scaling_path = real_model_dir / "scaling.json"
    if not scaling_path.exists():
        print("skipping real-model figures: results/real_model/scaling.json not found yet")
        return
    scaling = json.loads(scaling_path.read_text())

    present = [(slug, name, params) for slug, name, params in MODEL_CATALOG if slug in scaling]
    for slug, name, _params in present:
        data = scaling[slug]
        zero_cot = {int(k): v for k, v in data["zero_cot_accuracy_by_k"].items()}
        with_reasoning = {int(k): v for k, v in data.get("with_reasoning_accuracy_by_k", {}).items()}
        if with_reasoning:
            render_accuracy_collapse(
                zero_cot, with_reasoning, ASSETS / f"real_model_accuracy_collapse_{slug}.png",
                title=f"{name}: zero-shot accuracy collapses once it can't show its work",
            )
        ks = sorted(int(k) for k in data["opacity_gap_by_k"])
        gaps = [data["opacity_gap_by_k"][str(k)] for k in ks]
        if ks:
            render_opacity_gap_dotplot(
                ks, gaps, ASSETS / f"real_model_opacity_gap_{slug}.png",
                title=f"{name}: opacity gap by chain length",
            )

    if len(present) >= 2:
        render_scaling_comparison(present, scaling, ASSETS / "real_model_scaling.png")


def main() -> None:
    figure_depth_vs_k()
    figure_example_graphs("control", 10, "zero", tau=0.4, top_k=8, tag_suffix="control_k10")
    figure_example_graphs("control", 2, "zero", tau=0.4, top_k=8, tag_suffix="control_k2")
    figure_example_graphs("primary", 1, "zero", tau=0.4, top_k=8, tag_suffix="primary_k1")
    figure_example_graphs("primary", 6, "partial", tau=0.4, top_k=8, tag_suffix="primary_k6_partial")
    figure_real_model()


if __name__ == "__main__":
    main()
