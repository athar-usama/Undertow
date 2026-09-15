"""Is it specifically non-affinity, or merely having a fresh per-step key, that closes
off the associative shortcut? Trains the primary task's exact same architecture and
protocol, but with S replaced by an affine bijection (still a fresh random key every
step, still input-keyed) -- predicted, by the reasoning in tasks.make_affine_bijection,
to reopen the shortcut and become solvable zero-CoT at k=2, the same t_hops where the
non-affine primary task sits at chance. Checkpointed and reported alongside the main
grid rather than folded into it, since this is a controlled variable-isolation
experiment, not part of the core grid every figure is built from.
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
AFFINE_SEED = 123  # same seed value as the non-affine bijection, different derivation
D_MODEL, D_FF, N_HEADS, MAIN_DEPTH = 48, 192, 4, 6
BATCH_SIZE = 128
N_STEPS = 4000
T_HOPS_VALUES = [2, 4, 6, 10]


def main() -> None:
    torch.set_num_threads(8)
    affine = tasks.make_affine_bijection(seed=AFFINE_SEED, m=M)

    out_path = RESULTS / "affine_ablation.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["t_hops"] for l in out_path.read_text().splitlines() if l.strip()}

    for t_hops in T_HOPS_VALUES:
        if t_hops in done:
            continue
        t0 = time.time()
        model, accuracy = train_run(
            task_type="primary", m=M, t_hops=t_hops, cot_budget="zero", n_layers=MAIN_DEPTH,
            seed=0, bijection=affine, n_steps=N_STEPS, batch_size=BATCH_SIZE,
            d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS,
        )
        elapsed = time.time() - t0
        tag = f"primary_affine_k{t_hops}_zero"
        torch.save(model.state_dict(), CHECKPOINTS / f"{tag}.pt")
        record = {"t_hops": t_hops, "tag": tag, "accuracy": accuracy, "elapsed_sec": elapsed}
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(record)


if __name__ == "__main__":
    main()
