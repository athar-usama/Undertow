"""Embeds every correctly-answered, successfully-patched real-model example into
explorer/interrogation-room.html, replacing the __INTERROGATION_DATA__ placeholder
with a JSON literal. Pulls from every model under results/real_model/ that has a
raw.jsonl, so the picker spans every model this project has swept.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "real_model"
PAGE = ROOT / "explorer" / "interrogation-room.html"


def short_prompt(x0: int, coeffs: list[int]) -> str:
    return f"start at {x0}, " + ", ".join(f"+{c}" for c in coeffs)


def main() -> None:
    examples = []
    for model_dir in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        raw_path = model_dir / "raw.jsonl"
        if not raw_path.exists():
            continue
        for line in raw_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("correct") or "circuit" not in rec:
                continue
            examples.append({
                "model": model_dir.name,
                "k": rec["k"],
                "prompt_short": short_prompt(rec["x0"], rec["coeffs"]),
                "answer": rec["answer"],
                "generated_text": rec["generated_text"],
                "token_texts": rec["token_texts"],
                "token_effect": rec["token_effect"],
                "circuit": rec["circuit"],
            })

    examples.sort(key=lambda e: (e["model"], e["k"]))

    html = PAGE.read_text(encoding="utf-8")
    marker = "__INTERROGATION_DATA__"
    if marker not in html:
        raise RuntimeError("placeholder not found; interrogation-room.html may already be populated")
    html = html.replace(marker, json.dumps(examples))
    PAGE.write_text(html, encoding="utf-8")
    print(f"embedded {len(examples)} examples into {PAGE}")


if __name__ == "__main__":
    main()
