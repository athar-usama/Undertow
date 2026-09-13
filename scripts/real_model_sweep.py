"""Real-model phase: Qwen2.5-0.5B-Instruct on chained-addition word problems, zero-CoT,
k in {2,3,4}. Plain addition rather than modular arithmetic keeps the task within a
0.5B model's reach while still requiring genuine multi-step composition, and a
correctness-gated method needs some correct answers to have anything to analyze. This
script uses a fixed attempt budget per k (not "keep trying until enough succeed") and
reports the yield honestly. Checkpointed to results/real_model_raw.jsonl so the sweep
can be interrupted and resumed on this shared machine.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from undertow.graph import longest_path_length
from undertow.realmodel import (
    HFPatcher,
    build_real_graph,
    coarse_to_fine_necessary_sites,
    encode_chat,
    extract_last_int,
    load,
    make_chain_prompt,
    random_coeffs,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
RAW_PATH = RESULTS / "real_model_raw.jsonl"

K_VALUES = [2, 3, 4]
ATTEMPTS_PER_K = 40  # fixed budget; correctness-gated patching only runs on the hits
TAU = 0.35
TOP_K = 10
SEED = 2026
MAX_NEW_TOKENS = 30


def already_done() -> set[str]:
    if not RAW_PATH.exists():
        return set()
    return {json.loads(line)["uid"] for line in RAW_PATH.read_text().splitlines() if line.strip()}


def main() -> None:
    torch.set_num_threads(8)
    model, tokenizer = load()
    patcher = HFPatcher(model)
    rng = random.Random(SEED)
    done = already_done()

    for k in K_VALUES:
        for i in range(ATTEMPTS_PER_K):
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
                "generated_text": generated_text, "predicted": predicted, "correct": correct,
                "mnpc_depth": None,
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

            with open(RAW_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(record)


if __name__ == "__main__":
    main()
