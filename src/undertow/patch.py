"""Resample activation patching: per-site effect sizes, and a mediation test used to
turn a set of necessary sites into a directed dependency graph.

Effect size at a site (layer, position) is the standard causal-tracing quantity: how
much of the clean-minus-corrupted logit gap on the *clean* answer token is restored by
patching that one site's clean activation into an otherwise-corrupted forward pass.

The mediation test asks a *pairwise* question: having patched site A alone into the
corrupted run, how much does site B's own activation move back toward its clean value?
A large fraction means A causally feeds B, which is what turns a flat set of necessary
sites into a dependency chain rather than an unordered pile.
"""

from __future__ import annotations

import torch

from undertow.graph import CausalGraph, Site
from undertow.model import TinyTransformer


@torch.no_grad()
def run_with_cache(model: TinyTransformer, tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    cache: list[torch.Tensor] = []
    logits = model(tokens, cache=cache)
    return logits, cache


@torch.no_grad()
def effect_sizes(
    model: TinyTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    answer_pos: int,
    clean_answer_token: int,
) -> dict[Site, float]:
    """One forward pass per (layer, position) site; returns effect in [0, 1] for each."""
    clean_logits, clean_cache = run_with_cache(model, clean_tokens)
    corrupted_logits, _ = run_with_cache(model, corrupted_tokens)

    clean_logit = clean_logits[0, answer_pos, clean_answer_token].item()
    corrupted_logit = corrupted_logits[0, answer_pos, clean_answer_token].item()
    denom = clean_logit - corrupted_logit

    n_layers = len(clean_cache)
    seq_len = clean_tokens.shape[1]
    out: dict[Site, float] = {}
    for layer in range(n_layers):
        for pos in range(seq_len):
            clean_val = clean_cache[layer][:, pos, :]
            patched_logits = model(corrupted_tokens, patch={layer: {pos: clean_val}})
            patched_logit = patched_logits[0, answer_pos, clean_answer_token].item()
            if abs(denom) < 1e-6:
                effect = 0.0
            else:
                effect = (patched_logit - corrupted_logit) / denom
            out[(layer, pos)] = float(min(max(effect, 0.0), 1.0))
    return out


@torch.no_grad()
def mediation_effect(
    model: TinyTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: list[torch.Tensor],
    corrupted_cache: list[torch.Tensor],
    src: Site,
    dst: Site,
) -> float:
    """Fraction of site `dst`'s clean-vs-corrupted activation gap that closes when only
    site `src` (at an earlier layer) is patched to its clean value."""
    src_layer, src_pos = src
    dst_layer, dst_pos = dst
    clean_val = clean_cache[src_layer][:, src_pos, :]
    new_cache: list[torch.Tensor] = []
    model(corrupted_tokens, cache=new_cache, patch={src_layer: {src_pos: clean_val}})

    new_dst = new_cache[dst_layer][:, dst_pos, :]
    clean_dst = clean_cache[dst_layer][:, dst_pos, :]
    corrupted_dst = corrupted_cache[dst_layer][:, dst_pos, :]

    d0 = (corrupted_dst - clean_dst).norm().item()
    d1 = (new_dst - clean_dst).norm().item()
    if d0 < 1e-8:
        return 0.0
    return float(min(max(1.0 - d1 / d0, 0.0), 1.0))


def build_graph(
    model: TinyTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    site_effects: dict[Site, float],
    tau: float,
    top_k: int,
    mediation_threshold: float = 0.3,
) -> CausalGraph:
    """Necessary sites (effect >= tau), restricted to the top-`top_k` by effect size,
    connected by directed edges wherever the mediation test clears `mediation_threshold`."""
    necessary = [site for site, e in site_effects.items() if e >= tau]
    necessary.sort(key=lambda s: site_effects[s], reverse=True)
    necessary = necessary[:top_k]

    graph = CausalGraph()
    for site in necessary:
        graph.nodes.add(site)
    if not necessary:
        return graph

    _, clean_cache = run_with_cache(model, clean_tokens)
    _, corrupted_cache = run_with_cache(model, corrupted_tokens)

    for src in necessary:
        for dst in necessary:
            if dst[0] <= src[0]:
                continue
            m = mediation_effect(model, corrupted_tokens, clean_cache, corrupted_cache, src, dst)
            if m >= mediation_threshold:
                graph.add_edge(src, dst)
    return graph


@torch.no_grad()
def subgraph_faithfulness(
    model: TinyTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    nodes: set,
    answer_pos: int,
    clean_answer_token: int,
) -> float:
    """Patching every node in `nodes` at once, what fraction of the clean-vs-corrupted
    logit gap on the clean answer token is restored? This is the gate a reported
    MNPC-depth must clear: a graph that does not reproduce the model's behavior is not
    trustworthy evidence about how that behavior was computed."""
    clean_logits, clean_cache = run_with_cache(model, clean_tokens)
    corrupted_logits, _ = run_with_cache(model, corrupted_tokens)
    clean_logit = clean_logits[0, answer_pos, clean_answer_token].item()
    corrupted_logit = corrupted_logits[0, answer_pos, clean_answer_token].item()
    denom = clean_logit - corrupted_logit
    if abs(denom) < 1e-6:
        return 0.0

    patch: dict[int, dict[int, torch.Tensor]] = {}
    for layer, pos in nodes:
        patch.setdefault(layer, {})[pos] = clean_cache[layer][:, pos, :]

    patched_logits = model(corrupted_tokens, patch=patch)
    patched_logit = patched_logits[0, answer_pos, clean_answer_token].item()
    return float(min(max((patched_logit - corrupted_logit) / denom, 0.0), 1.0))
