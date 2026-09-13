"""Collapses results/real_model_raw.jsonl and results/real_model_with_reasoning.json
into results/real_model.json: per-k zero-shot accuracy, per-k with-reasoning accuracy,
and per-k patched MNPC-depths (only where at least one example cleared the correctness
gate -- k=3 has none, and that absence is itself reported, not papered over).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    raw = [json.loads(l) for l in (RESULTS / "real_model_raw.jsonl").read_text().splitlines() if l.strip()]
    with_reasoning = json.loads((RESULTS / "real_model_with_reasoning.json").read_text())

    by_k = defaultdict(list)
    for rec in raw:
        by_k[rec["k"]].append(rec)

    summary = {"zero_cot_accuracy_by_k": {}, "with_reasoning_accuracy_by_k": {},
               "patched_depths_by_k": {}, "opacity_gap_by_k": {}}
    for k, recs in sorted(by_k.items()):
        n = len(recs)
        n_correct = sum(1 for r in recs if r["correct"])
        summary["zero_cot_accuracy_by_k"][str(k)] = n_correct / n
        depths = [r["mnpc_depth"] for r in recs if r.get("mnpc_depth") is not None]
        summary["patched_depths_by_k"][str(k)] = depths
        if depths:
            summary["opacity_gap_by_k"][str(k)] = (sum(depths) / len(depths)) - 1

    for k, v in with_reasoning.items():
        summary["with_reasoning_accuracy_by_k"][k] = v["correct"] / v["total"]

    (RESULTS / "real_model.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
