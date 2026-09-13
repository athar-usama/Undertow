"""Runs the causal-tracing/MNPC-depth method on every synthetic checkpoint that
actually converged (see run_grid.py for why the grid is scoped the way it is), and
checks the resulting depth numbers against a calibration/held-out split.

Two separate questions, matching the two task families:

1. Control task (fixed-permutation, associative, zero-CoT, t_hops 1..10, all ~100%
   accurate): does the causal trace reveal a SHALLOW necessary chain even at t_hops=10,
   exposing the shortcut the model actually used instead of ten genuine serial steps?
   Reported as MNPC-depth vs. t_hops -- the flatter this is, the stronger the evidence.

2. Primary task (keyed, non-affine): the only cells that ever produce a correct answer
   are t_hops=1 zero-CoT and a handful of CoT-assisted cells at higher t_hops. For those,
   does MNPC-depth track "required silent depth" (t_hops minus how many intermediate
   states were handed to the model), which is the quantity the whole method is actually
   trying to measure, rather than raw t_hops?

tau/top_k are grid-searched on a calibration split and re-verified, untouched, on a
disjoint held-out split, exactly as for the control-task question -- this is the
project's one across-the-board non-negotiable, not something only applied when it's
convenient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from undertow import tasks
from undertow.graph import longest_path_length
from undertow.model import ModelConfig, TinyTransformer
from undertow.patch import build_graph, effect_sizes, subgraph_faithfulness

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"

M = 32
D_MODEL, D_FF, N_HEADS, MAIN_DEPTH = 48, 192, 4, 6
BIJECTION_SEED = 123
PERMUTATION_SEED = 456
N_EXAMPLES_PER_SPLIT = 12

TAU_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TOP_K_GRID = [4, 6, 8, 10, 12]
FAITHFULNESS_MIN = 0.5

CONTROL_T_HOPS = [1, 2, 4, 6, 8, 10]
PRIMARY_CELLS = [
    (1, "zero"), (2, "full"), (2, "partial"), (6, "partial"),
]


def load_model(tag: str, seq_len: int) -> TinyTransformer:
    cfg = ModelConfig(vocab_size=tasks.vocab_size(M), seq_len=seq_len, n_layers=MAIN_DEPTH,
                       d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS)
    model = TinyTransformer(cfg)
    model.load_state_dict(torch.load(CHECKPOINTS / f"{tag}.pt", map_location="cpu"))
    model.eval()
    return model


def make_example_pair(task_type, t_hops, cot_budget, seed, bijection, permutation):
    rng = np.random.default_rng(seed)
    if task_type == "primary":
        clean = tasks.sample_primary(rng, M, t_hops, cot_budget, bijection)
        hop_index = int(rng.integers(0, t_hops))
        corrupted = tasks.resample_hop_input(
            clean, hop_index=hop_index, rng=rng, m=M, task_type="primary",
            bijection=bijection, permutation=None,
        )
    else:
        clean = tasks.sample_control(rng, M, t_hops, cot_budget, permutation)
        corrupted = tasks.resample_hop_input(
            clean, hop_index=0, rng=rng, m=M, task_type="control",
            bijection=None, permutation=permutation,
        )
    return clean, corrupted


def mnpc_depth_for_example(model, clean, corrupted, tau, top_k):
    clean_tokens = torch.from_numpy(clean.tokens[:-1]).long().unsqueeze(0)
    corrupted_tokens = torch.from_numpy(corrupted.tokens[:-1]).long().unsqueeze(0)
    clean_answer_token = int(clean.tokens[clean.answer_pos + 1])
    effects = effect_sizes(model, clean_tokens, corrupted_tokens, clean.answer_pos, clean_answer_token)
    graph = build_graph(model, clean_tokens, corrupted_tokens, effects, tau=tau, top_k=top_k)
    faithfulness = subgraph_faithfulness(
        model, clean_tokens, corrupted_tokens, graph.nodes, clean.answer_pos, clean_answer_token
    )
    depth = longest_path_length(graph)
    return depth, faithfulness


def evaluate_cells(cells, task_type, bijection, permutation, seed_offset, tau, top_k):
    """`cells` is a list of (t_hops, cot_budget, x_value) -- x_value is what depth gets
    correlated against (t_hops for the control task, required_silent_depth for primary)."""
    depths_by_x, faithful_by_x, xs_seen = {}, {}, {}
    for t_hops, cot_budget, x_value in cells:
        tag = f"{task_type}_k{t_hops}_{cot_budget}"
        example0, _ = make_example_pair(task_type, t_hops, cot_budget, seed_offset, bijection, permutation)
        model = load_model(tag, len(example0.tokens))
        depths, faiths = [], []
        for s in range(seed_offset, seed_offset + N_EXAMPLES_PER_SPLIT):
            clean, corrupted = make_example_pair(task_type, t_hops, cot_budget, s, bijection, permutation)
            depth, faithful = mnpc_depth_for_example(model, clean, corrupted, tau, top_k)
            depths.append(depth)
            faiths.append(faithful)
        depths_by_x.setdefault(x_value, []).extend(depths)
        faithful_by_x.setdefault(x_value, []).extend(faiths)
        xs_seen[(t_hops, cot_budget)] = x_value
    return depths_by_x, faithful_by_x


def spearman_vs_x(depths_by_x):
    xs, ys = [], []
    for x, depths in depths_by_x.items():
        xs.extend([x] * len(depths))
        ys.extend(depths)
    if len(set(xs)) < 2:
        return 0.0
    rho, _ = spearmanr(xs, ys)
    return rho if rho is not None and not np.isnan(rho) else 0.0


def calibrate_family(name: str, cells, task_type: str) -> dict:
    bijection = tasks.make_substitution_bijection(seed=BIJECTION_SEED, m=M)
    permutation = tasks.make_fixed_permutation(seed=PERMUTATION_SEED, m=M)

    best = {"tau": None, "top_k": None, "rho": -2.0}
    for tau in TAU_GRID:
        for top_k in TOP_K_GRID:
            depths_by_x, faithful_by_x = evaluate_cells(
                cells, task_type, bijection, permutation, seed_offset=10_000, tau=tau, top_k=top_k
            )
            mean_faithfulness = np.mean([f for fs in faithful_by_x.values() for f in fs])
            if mean_faithfulness < FAITHFULNESS_MIN:
                continue
            rho = spearman_vs_x(depths_by_x)
            if rho > best["rho"]:
                best = {"tau": tau, "top_k": top_k, "rho": rho, "faithfulness": float(mean_faithfulness)}

    if best["tau"] is None:
        best = {"tau": TAU_GRID[len(TAU_GRID) // 2], "top_k": TOP_K_GRID[len(TOP_K_GRID) // 2], "rho": 0.0}

    calib_depths, calib_faith = evaluate_cells(
        cells, task_type, bijection, permutation, seed_offset=10_000, tau=best["tau"], top_k=best["top_k"]
    )
    heldout_depths, heldout_faith = evaluate_cells(
        cells, task_type, bijection, permutation, seed_offset=90_000, tau=best["tau"], top_k=best["top_k"]
    )

    result = {
        "family": name, "task_type": task_type,
        "chosen_tau": best["tau"], "chosen_top_k": best["top_k"],
        "calibration_rho": spearman_vs_x(calib_depths),
        "heldout_rho": spearman_vs_x(heldout_depths),
        "calibration_mean_depth_by_x": {str(k): float(np.mean(v)) for k, v in calib_depths.items()},
        "heldout_mean_depth_by_x": {str(k): float(np.mean(v)) for k, v in heldout_depths.items()},
        "calibration_faithfulness": float(np.mean([f for fs in calib_faith.values() for f in fs])),
        "heldout_faithfulness": float(np.mean([f for fs in heldout_faith.values() for f in fs])),
    }
    print(json.dumps(result, indent=2))
    with open(RESULTS / "calibration.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def main() -> None:
    control_cells = [(t, "zero", t) for t in CONTROL_T_HOPS]
    calibrate_family("control_depth_vs_k", control_cells, "control")

    primary_cells = []
    for t_hops, cot_budget in PRIMARY_CELLS:
        x = t_hops - tasks.cot_count(t_hops, cot_budget)
        primary_cells.append((t_hops, cot_budget, x))
    calibrate_family("primary_depth_vs_required_silent_depth", primary_cells, "primary")


if __name__ == "__main__":
    main()
