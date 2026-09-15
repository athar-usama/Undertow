"""Embeds precomputed patching results for a handful of representative examples into
explorer/index.html, replacing the __UNDERTOW_DATA__ placeholder with a JSON literal.
Run after make_figures.py / calibrate.py have produced real results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from undertow import tasks
from undertow.graph import longest_path
from undertow.model import ModelConfig, TinyTransformer
from undertow.patch import build_graph, effect_sizes, subgraph_faithfulness
from undertow.viz import token_labels

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "results" / "checkpoints"
EXPLORER_HTML = ROOT / "explorer" / "index.html"

M = 32
D_MODEL, D_FF, N_HEADS, MAIN_DEPTH = 48, 192, 4, 6
BIJECTION_SEED = 123
PERMUTATION_SEED = 456
TAU, TOP_K = 0.4, 8

CONTROL_CELLS = [(t, "zero") for t in (1, 2, 4, 6, 8, 10)]
# (2, "zero") is deliberately excluded: the primary task never learns it (chance-level
# accuracy, see README) so there is no genuine "solved computation" for patching to
# trace there -- only cells the model actually converged on belong in the explorer.
PRIMARY_CELLS = [(1, "zero"), (2, "full"), (2, "partial"), (6, "partial")]


def load_model(tag: str, seq_len: int) -> TinyTransformer:
    cfg = ModelConfig(vocab_size=tasks.vocab_size(M), seq_len=seq_len, n_layers=MAIN_DEPTH,
                       d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS)
    model = TinyTransformer(cfg)
    model.load_state_dict(torch.load(CHECKPOINTS / f"{tag}.pt", map_location="cpu"))
    model.eval()
    return model


def example_payload(task_type: str, t_hops: int, cot_budget: str) -> dict:
    bijection = tasks.make_substitution_bijection(seed=BIJECTION_SEED, m=M)
    permutation = tasks.make_fixed_permutation(seed=PERMUTATION_SEED, m=M)
    rng = np.random.default_rng(30_000 + t_hops + (7 if cot_budget != "zero" else 0))
    if task_type == "primary":
        clean = tasks.sample_primary(rng, M, t_hops, cot_budget, bijection)
        corrupted = tasks.resample_hop_input(
            clean, int(rng.integers(0, t_hops)), rng, M, "primary", bijection, None
        )
    else:
        clean = tasks.sample_control(rng, M, t_hops, cot_budget, permutation)
        corrupted = tasks.resample_hop_input(clean, 0, rng, M, "control", None, permutation)

    tag = f"{task_type}_k{t_hops}_{cot_budget}"
    model = load_model(tag, len(clean.tokens))
    clean_tokens = torch.from_numpy(clean.tokens[:-1]).long().unsqueeze(0)
    corrupted_tokens = torch.from_numpy(corrupted.tokens[:-1]).long().unsqueeze(0)
    clean_answer_token = int(clean.tokens[clean.answer_pos + 1])

    effects = effect_sizes(model, clean_tokens, corrupted_tokens, clean.answer_pos, clean_answer_token)
    graph = build_graph(model, clean_tokens, corrupted_tokens, effects, tau=TAU, top_k=TOP_K)
    faithfulness = subgraph_faithfulness(
        model, clean_tokens, corrupted_tokens, graph.nodes, clean.answer_pos, clean_answer_token
    )
    chain = longest_path(graph)
    labels = token_labels(clean.tokens[:-1], clean.answer_pos)

    return {
        "seq_len": len(clean_tokens[0]),
        "n_layers": MAIN_DEPTH,
        "nodes": [list(n) for n in graph.nodes],
        "edges": [[list(s), list(d)] for s, d in graph.edges],
        "chain": [list(n) for n in chain],
        "faithfulness": faithfulness,
        "tokens": labels,
    }


def main() -> None:
    payload = {"control": {}, "primary": {}}
    for t_hops, cot_budget in CONTROL_CELLS:
        payload["control"][str(t_hops)] = example_payload("control", t_hops, cot_budget)
    for t_hops, cot_budget in PRIMARY_CELLS:
        key = f"{t_hops}-{cot_budget}"
        payload["primary"][key] = example_payload("primary", t_hops, cot_budget)

    html = EXPLORER_HTML.read_text(encoding="utf-8")
    marker = "__UNDERTOW_DATA__"
    if marker not in html:
        raise RuntimeError("placeholder not found; explorer/index.html may already be populated")
    html = html.replace(marker, json.dumps(payload))
    EXPLORER_HTML.write_text(html, encoding="utf-8")
    print(f"embedded {sum(len(v) for v in payload.values())} examples into {EXPLORER_HTML}")


if __name__ == "__main__":
    main()
