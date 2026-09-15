"""Collapses each model's results/real_model/<slug>/raw.jsonl and with_reasoning.json
into a per-model summary.json (zero-shot accuracy, with-reasoning accuracy, patched
MNPC-depths), then a single results/real_model/scaling.json across every model found,
for the model-size scaling comparison.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REAL_MODEL_DIR = RESULTS / "real_model"


def summarize_one(model_dir: Path) -> dict | None:
    raw_path = model_dir / "raw.jsonl"
    reasoning_path = model_dir / "with_reasoning.json"
    if not raw_path.exists():
        return None

    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    by_k = defaultdict(list)
    for rec in raw:
        by_k[rec["k"]].append(rec)

    summary = {
        "model": model_dir.name, "zero_cot_accuracy_by_k": {}, "with_reasoning_accuracy_by_k": {},
        "patched_depths_by_k": {}, "opacity_gap_by_k": {}, "n_attempts_by_k": {},
    }
    for k, recs in sorted(by_k.items()):
        n = len(recs)
        n_correct = sum(1 for r in recs if r["correct"])
        summary["n_attempts_by_k"][str(k)] = n
        summary["zero_cot_accuracy_by_k"][str(k)] = n_correct / n
        depths = [r["mnpc_depth"] for r in recs if r.get("mnpc_depth") is not None]
        summary["patched_depths_by_k"][str(k)] = depths
        if depths:
            summary["opacity_gap_by_k"][str(k)] = (sum(depths) / len(depths)) - 1

    if reasoning_path.exists():
        with_reasoning = json.loads(reasoning_path.read_text())
        for k, v in with_reasoning.items():
            summary["with_reasoning_accuracy_by_k"][k] = v["correct"] / v["total"]

    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    if not REAL_MODEL_DIR.exists():
        print("no results/real_model/ directory yet")
        return

    scaling = {}
    for model_dir in sorted(REAL_MODEL_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        summary = summarize_one(model_dir)
        if summary is None:
            continue
        scaling[model_dir.name] = summary
        print(json.dumps(summary, indent=2))

    (REAL_MODEL_DIR / "scaling.json").write_text(json.dumps(scaling, indent=2))
    print(f"scaling summary -> {REAL_MODEL_DIR / 'scaling.json'}")


if __name__ == "__main__":
    main()
