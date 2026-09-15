"""Real-model phase: a real open-weight model on chained-addition word problems,
zero-CoT, k in {2,3,4}. Plain addition rather than modular arithmetic keeps the task
within a small instruction-tuned model's reach while still requiring genuine
multi-step composition, and a correctness-gated method needs some correct answers to
have anything to analyze. This script uses a fixed attempt budget per k (not "keep
trying until enough succeed") and reports the yield honestly.

Runs against any model on the command line, e.g.:
    python scripts/real_model_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/real_model_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct

Results go to results/real_model/<model-slug>/raw.jsonl, checkpointed so the sweep can
be interrupted and resumed on this shared machine. For every example that is both
answered correctly and successfully patched, the record also carries the per-position
token-display text and the discovered circuit (nodes/edges/longest chain/per-token
effect), which is what scripts/make_interrogation_room.py later renders as an actual
chat transcript with the causally-necessary tokens highlighted, not a chart.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from undertow.graph import longest_path, longest_path_length
from undertow.realmodel import (
    HFPatcher,
    build_real_graph,
    coarse_to_fine_necessary_sites,
    encode_chat,
    extract_last_int,
    load,
    make_chain_prompt,
    random_coeffs,
    token_display_strings,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

K_VALUES = [2, 3, 4]
DEFAULT_ATTEMPTS_PER_K = 80  # fixed budget; correctness-gated patching only runs on the hits
TAU = 0.35
TOP_K = 10
SEED = 2026
MAX_NEW_TOKENS = 30


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def load_existing(raw_path: Path) -> list[dict]:
    if not raw_path.exists():
        return []
    return [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]


def token_effect_by_position(effects: dict, n_positions: int) -> list[float]:
    """Max effect at any layer, per token position -- what a text-overlay heatmap needs."""
    out = [0.0] * n_positions
    for (_layer, pos), e in effects.items():
        if pos < n_positions:
            out[pos] = max(out[pos], e)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--attempts-per-k", type=int, default=DEFAULT_ATTEMPTS_PER_K)
    parser.add_argument(
        "--max-patches-per-k", type=int, default=None,
        help="stop attempting once this many examples have been successfully patched for a "
             "given k, so a larger model's higher per-example patching cost stays bounded",
    )
    args = parser.parse_args()

    slug = model_slug(args.model)
    out_dir = RESULTS / "real_model" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    torch.set_num_threads(8)
    model, tokenizer = load(args.model)
    patcher = HFPatcher(model)
    rng = random.Random(SEED)
    attempts_per_k = args.attempts_per_k

    existing = load_existing(raw_path)
    done = {rec["uid"] for rec in existing}
    patches_by_k = {k: sum(1 for rec in existing if rec["k"] == k and rec.get("circuit")) for k in K_VALUES}

    for k in K_VALUES:
        for i in range(attempts_per_k):
            if args.max_patches_per_k is not None and patches_by_k[k] >= args.max_patches_per_k:
                break
            uid = f"k{k}_ex{i}"
            if uid in done:
                continue
            x0 = rng.randrange(1, 10)
            coeffs = random_coeffs(rng, k)
            example = make_chain_prompt(tokenizer, x0, coeffs)
            clean_ids = encode_chat(tokenizer, example.prompt)
            answer_pos = clean_ids.shape[1] - 1

            with torch.no_grad():
                gen = model.generate(clean_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            generated_text = tokenizer.decode(
                gen[0, clean_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            predicted = extract_last_int(generated_text)
            correct = predicted == example.answer

            record = {
                "uid": uid, "k": k, "x0": x0, "coeffs": coeffs, "answer": example.answer,
                "prompt": example.prompt, "generated_text": generated_text,
                "predicted": predicted, "correct": correct, "mnpc_depth": None,
            }

            if correct:
                corrupt_index = rng.randrange(k)
                corrupt_coeffs = list(coeffs)
                old = corrupt_coeffs[corrupt_index]
                new = rng.randrange(1, 10)
                while new == old:
                    new = rng.randrange(1, 10)
                corrupt_coeffs[corrupt_index] = new
                corrupted_example = make_chain_prompt(tokenizer, x0, corrupt_coeffs)
                corrupted_ids = encode_chat(tokenizer, corrupted_example.prompt)

                if corrupted_ids.shape[1] == clean_ids.shape[1]:
                    clean_answer_token = int(gen[0, clean_ids.shape[1]])
                    n_tokens = clean_ids.shape[1]
                    candidate_positions = list(range(max(0, n_tokens - 25), n_tokens))
                    t0 = time.time()
                    effects = coarse_to_fine_necessary_sites(
                        patcher, clean_ids, corrupted_ids, answer_pos,
                        clean_answer_token, candidate_positions, tau=TAU,
                    )
                    _, clean_cache = patcher.run_with_cache(clean_ids)
                    _, corrupted_cache = patcher.run_with_cache(corrupted_ids)
                    graph = build_real_graph(
                        patcher, clean_ids, corrupted_ids, effects, clean_cache, corrupted_cache,
                        tau=TAU, top_k=TOP_K,
                    )
                    record["mnpc_depth"] = longest_path_length(graph)
                    record["n_necessary_sites"] = len(graph.nodes)
                    record["patch_wall_sec"] = time.time() - t0
                    record["token_texts"] = token_display_strings(tokenizer, clean_ids)
                    record["token_effect"] = token_effect_by_position(effects, n_tokens)
                    record["circuit"] = {
                        "nodes": [list(n) for n in graph.nodes],
                        "edges": [[list(s), list(d)] for s, d in graph.edges],
                        "chain": [list(n) for n in longest_path(graph)],
                        "n_layers": len(clean_cache),
                        "answer_pos": answer_pos,
                    }
                    patches_by_k[k] += 1

            with open(raw_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(record["uid"], record["correct"], record.get("mnpc_depth"))

    print(f"done: {model_slug(args.model)} -> {raw_path}")


if __name__ == "__main__":
    main()
