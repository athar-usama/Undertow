"""The control condition for the real-model phase: the same chained-addition family,
same model, but explicitly allowed to show its work before answering. Establishes that
a low zero-CoT accuracy reflects an opacity gap, not a plain incapacity to do the
arithmetic at all.

    python scripts/real_model_with_reasoning.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from undertow.realmodel import encode_chat, extract_last_int, load, random_coeffs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

K_VALUES = [2, 3, 4]
N_PER_K = 30
SEED = 2026  # matches real_model_sweep.py's SEED for a like-for-like comparison


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def make_prompt_with_reasoning(x0: int, coeffs: list[int]) -> str:
    steps = [f"add {c}" for c in coeffs]
    return (
        f"Start with the number {x0}. Then " + ", then ".join(steps)
        + ". Show your work step by step, then give the final resulting number."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    torch.set_num_threads(6)
    model, tokenizer = load(args.model)
    rng = random.Random(SEED)

    results: dict[int, list[bool]] = {k: [] for k in K_VALUES}
    for k in K_VALUES:
        for _ in range(N_PER_K):
            x0 = rng.randrange(1, 10)
            coeffs = random_coeffs(rng, k)
            state = x0
            for c in coeffs:
                state += c
            prompt = make_prompt_with_reasoning(x0, coeffs)
            ids = encode_chat(tokenizer, prompt)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=120, do_sample=False)
            text = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
            pred = extract_last_int(text)
            results[k].append(pred == state)
        acc = sum(results[k]) / len(results[k])
        print(f"k={k} with-reasoning accuracy={acc:.2f} ({sum(results[k])}/{len(results[k])})")

    out_dir = RESULTS / "real_model" / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {str(k): {"correct": sum(v), "total": len(v)} for k, v in results.items()}
    (out_dir / "with_reasoning.json").write_text(json.dumps(payload, indent=2))
    print(f"done: {model_slug(args.model)} -> {out_dir / 'with_reasoning.json'}")


if __name__ == "__main__":
    main()
