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
from undertow.graph import longest_path_length
from undertow.model import ModelConfig, TinyTransformer
from undertow.patch import build_graph, effect_sizes
from undertow.viz import (
    render_accuracy_collapse,
    render_causal_graph,
    render_depth_vs_k,
    render_effect_heatmap,
    render_opacity_gap_dotplot,
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


def longest_chain_from_graph(graph):
    preds = {n: [] for n in graph.nodes}
    for s, d in graph.edges:
        preds[d].append(s)
    dp, chain_pred = {}, {}
    for node in sorted(graph.nodes, key=lambda s: s[0]):
        best_pred, best_len = None, 0
        for p in preds[node]:
            if dp[p] > best_len:
                best_len, best_pred = dp[p], p
        dp[node] = 1 + best_len
        chain_pred[node] = best_pred
    chain = []
    if dp:
        end = max(dp, key=dp.get)
        while end is not None:
            chain.append(end)
            end = chain_pred[end]
        chain.reverse()
    return chain


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
    longest_chain = longest_chain_from_graph(graph)

    labels = token_labels(clean.tokens[:-1], clean.answer_pos)
    render_causal_graph(
        graph, longest_chain, len(clean_tokens[0]), MAIN_DEPTH,
        title=f"{task_type} task, k={t_hops}, {cot_budget}-CoT (MNPC-depth={longest_path_length(graph)})",
        out_path=ASSETS / f"circuit_{tag_suffix}.png", token_text=labels,
    )
    render_effect_heatmap(
        effects, len(clean_tokens[0]), MAIN_DEPTH,
        title=f"{task_type} task, k={t_hops}: where the answer is actually decided",
        out_path=ASSETS / f"heatmap_{tag_suffix}.png", token_text=labels,
    )


def figure_real_model() -> None:
    path = RESULTS / "real_model.json"
    if not path.exists():
        print("skipping real-model figure: results/real_model.json not found yet")
        return
    data = json.loads(path.read_text())

    zero_cot = {int(k): v for k, v in data["zero_cot_accuracy_by_k"].items()}
    with_reasoning = {int(k): v for k, v in data["with_reasoning_accuracy_by_k"].items()}
    render_accuracy_collapse(zero_cot, with_reasoning, ASSETS / "real_model_accuracy_collapse.png")

    ks = sorted(int(k) for k in data["opacity_gap_by_k"])
    gaps = [data["opacity_gap_by_k"][str(k)] for k in ks]
    if ks:
        render_opacity_gap_dotplot(ks, gaps, ASSETS / "real_model_opacity_gap.png")


def main() -> None:
    figure_depth_vs_k()
    figure_example_graphs("control", 10, "zero", tau=0.4, top_k=8, tag_suffix="control_k10")
    figure_example_graphs("control", 2, "zero", tau=0.4, top_k=8, tag_suffix="control_k2")
    figure_example_graphs("primary", 1, "zero", tau=0.4, top_k=8, tag_suffix="primary_k1")
    figure_example_graphs("primary", 6, "partial", tau=0.4, top_k=8, tag_suffix="primary_k6_partial")
    figure_real_model()


if __name__ == "__main__":
    main()
