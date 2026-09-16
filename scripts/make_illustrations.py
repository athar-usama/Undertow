"""Renders the three narrative illustrations (building cutaway, iceberg, corkboard),
all built from the exact same real, patched example (k2_ex7, Qwen2.5-0.5B-Instruct)
so a reader can recognize the same underlying facts across three different visual
treatments. Run after real_model_sweep.py has produced that example's record.
"""

from __future__ import annotations

import json
from pathlib import Path

from undertow.illustrations import (
    render_building_cutaway,
    render_circuit_thumbnail,
    render_corkboard,
    render_iceberg,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

RAW_PATH = ROOT / "results" / "real_model" / "qwen2.5-0.5b-instruct" / "raw.jsonl"
UID = "k2_ex7"


def load_example() -> dict:
    for line in RAW_PATH.read_text().splitlines():
        rec = json.loads(line)
        if rec["uid"] == UID:
            return rec
    raise RuntimeError(f"{UID} not found in {RAW_PATH}")


def main() -> None:
    rec = load_example()
    circuit = rec["circuit"]
    n_layers = circuit["n_layers"]
    nodes = circuit["nodes"]

    positions = sorted({p for _, p in nodes})

    def describe(pos: int) -> str:
        text = rec["token_texts"][pos].strip()
        return f'token "{text}"' if text else "generation start"

    lit_floors_by_column = {
        str(pos): sorted(layer for layer, p in nodes if p == pos) for pos in positions
    }
    column_labels = {str(pos): describe(pos) for pos in positions}
    entry_pos = min(positions)
    exit_pos = max(positions)

    render_building_cutaway(
        lit_floors_by_column, n_layers, column_labels,
        entry_label=f'"{rec["token_texts"][entry_pos].strip()}" read here',
        exit_label=f'answer "{rec["generated_text"]}" produced here',
        out_path=ASSETS / "illustration_building.png",
        title="k=2 example: which of 24 floors the answer actually needed",
    )

    n_shown = len(nodes)
    render_iceberg(
        tip_label=f'"{rec["generated_text"]}"',
        tip_caption="the only token anyone outside the model ever sees",
        deep_caption=f"{n_shown} of {n_layers} layers were causally necessary; none of it was written down",
        depth_annotations=[
            (0.05, f"layers {min(l for l,p in nodes if p==exit_pos)}-{max(l for l,p in nodes if p==exit_pos)}: final commit"),
            (0.85, f"layers {min(l for l,p in nodes if p==entry_pos)}-{max(l for l,p in nodes if p==entry_pos)}: reads the 2nd addend"),
        ],
        out_path=ASSETS / "illustration_iceberg.png",
        title='Undertow: what "16" cost to produce',
    )

    thumb_path = ASSETS / "_corkboard_thumb.png"
    render_circuit_thumbnail(
        nodes=[tuple(n) for n in nodes],
        edges=[(tuple(s), tuple(d)) for s, d in circuit["edges"]],
        chain=[tuple(n) for n in circuit["chain"]],
        n_layers=n_layers,
        out_path=thumb_path,
    )

    render_corkboard(
        prompt_text=rec["prompt"],
        highlighted_word=rec["token_texts"][entry_pos].strip(),
        circuit_thumb_path=thumb_path,
        verdict_lines=[
            f"Answer given: {rec['generated_text']} (correct: {rec['answer']})",
            f"Necessary chain: {len(circuit['chain'])} layers, tokens {entry_pos} -> {exit_pos}",
            "No chain-of-thought shown. None was needed to solve it silently.",
        ],
        out_path=ASSETS / "illustration_corkboard.png",
        title="The case file: one silent answer, traced back",
    )
    thumb_path.unlink(missing_ok=True)
    print("wrote illustration_building.png, illustration_iceberg.png, illustration_corkboard.png")


if __name__ == "__main__":
    main()
