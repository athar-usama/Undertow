"""Trains and checkpoints exactly the synthetic runs the project's final story rests
on. Every cell here is one that a pilot sweep already confirmed converges within a
fixed 4000-step budget; see results/pilot_accuracy.json and the README for the cells
that did NOT converge in that budget (primary task, full-CoT at t_hops=6/10 and
partial-CoT at t_hops=10) -- those are reported honestly as a training-budget
limitation, not re-run here, since there is nothing to checkpoint or patch when no
example is ever answered correctly.

This replaced an earlier, much larger (task_type x t_hops x cot_budget) grid after the
data itself ruled it out: a pilot sweep of t_hops=1..3 zero-CoT (then, when that looked
like it might be a training bug, a re-check of t_hops=2 zero-CoT for 20,000 steps, at
3x the width, and at up to 16 layers instead of 6) found a hard wall, not a gradual
slope -- the primary (keyed, non-affine) task is perfectly learnable zero-CoT at
t_hops=1 and essentially unlearnable at t_hops=2 regardless of depth/width/learning
rate tried. The control (fixed-permutation) task, by contrast, solves t_hops up to 10
zero-CoT at 100% accuracy every time, exactly as expected for a task whose associative
structure admits a shortcut a fixed-dataset model can memorize once and reuse. That
contrast -- not a smooth depth-vs-k curve -- is the actual finding, so the grid below
trains only the cells that are informative and that a fixed 4000-step budget actually
resolves.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from undertow import tasks
from undertow.model import train_run

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

M = 32
BIJECTION_SEED = 123
PERMUTATION_SEED = 456
D_MODEL, D_FF, N_HEADS = 48, 192, 4
BATCH_SIZE = 128
N_STEPS = 4000
MAIN_DEPTH = 6

CONTROL_T_HOPS = [1, 2, 4, 6, 8, 10]
RUNS = [
    ("primary", 1, "zero"),
    ("primary", 2, "zero"),
    ("primary", 2, "full"),
    ("primary", 2, "partial"),
    ("primary", 6, "partial"),
]


def required_silent_depth(t_hops: int, cot_budget: str) -> int:
    return t_hops - tasks.cot_count(t_hops, cot_budget)


def run_one(task_type: str, t_hops: int, cot_budget: str, n_layers: int, tag: str,
            bijection, permutation, seed: int = 0) -> dict:
    torch.set_num_threads(8)
    t0 = time.time()
    model, accuracy = train_run(
        task_type=task_type, m=M, t_hops=t_hops, cot_budget=cot_budget, n_layers=n_layers,
        seed=seed, bijection=bijection, permutation=permutation, n_steps=N_STEPS,
        batch_size=BATCH_SIZE, d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS,
    )
    elapsed = time.time() - t0
    ckpt_path = CHECKPOINTS / f"{tag}.pt"
    torch.save(model.state_dict(), ckpt_path)
    record = {
        "tag": tag, "task_type": task_type, "t_hops": t_hops, "cot_budget": cot_budget,
        "required_silent_depth": required_silent_depth(t_hops, cot_budget),
        "n_layers": n_layers, "d_model": D_MODEL, "d_ff": D_FF, "n_heads": N_HEADS,
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE, "seed": seed,
        "accuracy": accuracy, "elapsed_sec": elapsed, "checkpoint": str(ckpt_path),
    }
    with open(RESULTS / "grid.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(record)
    return record


def main() -> None:
    bijection = tasks.make_substitution_bijection(seed=BIJECTION_SEED, m=M)
    permutation = tasks.make_fixed_permutation(seed=PERMUTATION_SEED, m=M)

    grid_file = RESULTS / "grid.jsonl"
    done_tags = set()
    if grid_file.exists():
        for line in grid_file.read_text().splitlines():
            done_tags.add(json.loads(line)["tag"])

    for t_hops in CONTROL_T_HOPS:
        tag = f"control_k{t_hops}_zero"
        if tag in done_tags:
            continue
        run_one("control", t_hops, "zero", MAIN_DEPTH, tag, None, permutation)

    for task_type, t_hops, cot_budget in RUNS:
        tag = f"{task_type}_k{t_hops}_{cot_budget}"
        if tag in done_tags:
            continue
        run_one(task_type, t_hops, cot_budget, MAIN_DEPTH, tag, bijection, None)


if __name__ == "__main__":
    main()
