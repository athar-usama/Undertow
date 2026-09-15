"""Applying the calibrated MNPC-depth method to a real, small, open-weight language
model (Qwen2.5-0.5B-Instruct). Mirrors patch.py's cache/patch API but through PyTorch
forward hooks on the model's own decoder layers, since a real HF model cannot be
rewritten as an explicit per-layer loop the way the synthetic TinyTransformer is.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from undertow.graph import CausalGraph, Site

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


def _get_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _set_hidden(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


class HFPatcher:
    """Same cache/patch contract as `undertow.model.TinyTransformer.forward`, backed by
    forward hooks on `model.model.layers` (the standard HF decoder-layer list for the
    Llama/Qwen family of architectures)."""

    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers

    @torch.no_grad()
    def run_with_cache(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        cache: list[torch.Tensor] = []

        def make_hook():
            def hook(_module, _inp, out):
                cache.append(_get_hidden(out).detach().clone())
                return out

            return hook

        handles = [layer.register_forward_hook(make_hook()) for layer in self.layers]
        try:
            out = self.model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        return out.logits, cache

    @torch.no_grad()
    def run_with_patch(
        self, input_ids: torch.Tensor, patch: dict[int, dict[int, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        cache: list[torch.Tensor] = []

        def make_hook(layer_idx: int):
            def hook(_module, _inp, out):
                hidden = _get_hidden(out)
                if layer_idx in patch:
                    hidden = hidden.clone()
                    for pos, val in patch[layer_idx].items():
                        hidden[:, pos, :] = val
                    out = _set_hidden(out, hidden)
                cache.append(hidden.detach().clone())
                return out

            return hook

        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(self.layers)]
        try:
            out = self.model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        return out.logits, cache


@dataclass
class ChainPrompt:
    prompt: str
    answer: int
    answer_prefix_len: int  # token count of the prompt encoding, before the answer digits


def make_chain_prompt(tokenizer, x0: int, coeffs: list[int]) -> ChainPrompt:
    """A chained addition word problem: start from x0, add each of `coeffs` in sequence,
    ask only for the final value (no CoT) -- the zero-CoT, real-model analogue of the
    synthetic keyed substitution chain task. Plain addition rather than modular
    arithmetic keeps the task within a small instruction-tuned model's reach while still
    requiring genuine multi-step composition, which is the quantity this method
    measures; a correctness-gated method needs some correct answers to have anything to
    analyze in the first place."""
    state = x0
    steps = []
    for c in coeffs:
        state += c
        steps.append(f"add {c}")
    instruction = (
        f"Start with the number {x0}. Then " + ", then ".join(steps)
        + ". Give only the final resulting number, with no working shown and no explanation."
    )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    prompt_ids = encoded["input_ids"]
    return ChainPrompt(prompt=instruction, answer=state, answer_prefix_len=prompt_ids.shape[1])


def encode_chat(tokenizer, content: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    return encoded["input_ids"]


def coarse_to_fine_necessary_sites(
    patcher: HFPatcher,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    answer_pos: int,
    clean_answer_token: int,
    candidate_positions: list[int],
    tau: float,
    coarse_layer_stride: int = 2,
) -> dict[Site, float]:
    """Coarse pass over every `coarse_layer_stride`-th layer across `candidate_positions`
    only (not every token), then a fine pass over every layer restricted to a small
    window around whatever positions the coarse pass flagged. This keeps a real model
    with dozens of layers and long prompts tractable (see README/PLAN for the exhaustive
    -sweep cost that makes this necessary)."""
    clean_logits, clean_cache = patcher.run_with_cache(clean_ids)
    corrupted_logits, _ = patcher.run_with_cache(corrupted_ids)
    clean_logit = clean_logits[0, answer_pos, clean_answer_token].item()
    corrupted_logit = corrupted_logits[0, answer_pos, clean_answer_token].item()
    denom = clean_logit - corrupted_logit
    n_layers = len(clean_cache)

    def effect_at(layer: int, pos: int) -> float:
        if abs(denom) < 1e-6:
            return 0.0
        patched_logits, _ = patcher.run_with_patch(
            corrupted_ids, {layer: {pos: clean_cache[layer][:, pos, :]}}
        )
        patched_logit = patched_logits[0, answer_pos, clean_answer_token].item()
        return float(min(max((patched_logit - corrupted_logit) / denom, 0.0), 1.0))

    effects: dict[Site, float] = {}
    coarse_layers = list(range(0, n_layers, coarse_layer_stride))
    for layer in coarse_layers:
        for pos in candidate_positions:
            effects[(layer, pos)] = effect_at(layer, pos)

    hot_positions = sorted({pos for (_, pos), e in effects.items() if e >= tau})
    if not hot_positions:
        return effects

    window = set()
    for pos in hot_positions:
        for p in range(max(0, pos - 1), min(clean_ids.shape[1], pos + 2)):
            window.add(p)

    for layer in range(n_layers):
        if layer in coarse_layers:
            continue
        for pos in window:
            effects[(layer, pos)] = effect_at(layer, pos)
    return effects


def build_real_graph(
    patcher: HFPatcher,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    effects: dict[Site, float],
    clean_cache: list[torch.Tensor],
    corrupted_cache: list[torch.Tensor],
    tau: float,
    top_k: int,
    mediation_threshold: float = 0.3,
) -> CausalGraph:
    necessary = sorted(
        (s for s, e in effects.items() if e >= tau), key=lambda s: effects[s], reverse=True
    )[:top_k]
    graph = CausalGraph()
    for site in necessary:
        graph.nodes.add(site)

    for src in necessary:
        for dst in necessary:
            if dst[0] <= src[0]:
                continue
            src_layer, src_pos = src
            dst_layer, dst_pos = dst
            clean_val = clean_cache[src_layer][:, src_pos, :]
            _, patched_cache = patcher.run_with_patch(corrupted_ids, {src_layer: {src_pos: clean_val}})
            new_dst = patched_cache[dst_layer][:, dst_pos, :]
            clean_dst = clean_cache[dst_layer][:, dst_pos, :]
            corrupted_dst = corrupted_cache[dst_layer][:, dst_pos, :]
            d0 = (corrupted_dst - clean_dst).norm().item()
            d1 = (new_dst - clean_dst).norm().item()
            m = 0.0 if d0 < 1e-8 else float(min(max(1.0 - d1 / d0, 0.0), 1.0))
            if m >= mediation_threshold:
                graph.add_edge(src, dst)
    return graph


def random_coeffs(rng: random.Random, k: int, high: int = 9) -> list[int]:
    return [rng.randrange(1, high + 1) for _ in range(k)]


def token_display_strings(tokenizer, ids: torch.Tensor) -> list[str]:
    """The incremental decoded text added at each token position, rather than raw BPE
    pieces -- legible for a text overlay instead of tokenizer-internal fragments."""
    id_list = ids[0].tolist()
    strings = []
    prev_text = ""
    for i in range(1, len(id_list) + 1):
        text = tokenizer.decode(id_list[:i], skip_special_tokens=False)
        strings.append(text[len(prev_text):])
        prev_text = text
    return strings


def extract_last_int(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None
