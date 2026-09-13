"""A minimal decoder-only transformer, written with an explicit per-layer loop so that
the residual stream after every layer is directly addressable for activation patching.

This is deliberately not built on `torch.nn.TransformerDecoder` or an attention library:
patching requires injecting an arbitrary clean activation at an arbitrary (layer,
position) site mid-forward-pass, which is far simpler to guarantee correct against a
loop we write ourselves than against a black-box module's internal call graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from undertow import tasks


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    n_layers: int = 6
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.triu(torch.ones(cfg.seq_len, cfg.seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, heads, t, d_head)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.causal_mask[:t, :t], float("-inf"))
        att = att.softmax(dim=-1)
        out = att @ v  # (b, heads, t, d_head)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        positions = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        return self.tok_emb(tokens) + self.pos_emb(positions)

    def forward(
        self,
        tokens: torch.Tensor,
        cache: list[torch.Tensor] | None = None,
        patch: dict[int, dict[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """`cache`, if given, is filled in-place with the residual stream after every
        layer (index = layer number). `patch`, if given, maps layer index -> {position:
        replacement activation vector} to substitute into the residual stream right
        after that layer runs, before the next layer consumes it."""
        x = self.embed(tokens)
        for layer_idx, block in enumerate(self.blocks):
            x = block(x)
            if cache is not None:
                cache.append(x.detach().clone())
            if patch is not None and layer_idx in patch:
                x = x.clone()
                for position, value in patch[layer_idx].items():
                    x[:, position, :] = value
        x = self.ln_f(x)
        return self.head(x)


def _supervise_start(tokens: np.ndarray, answer_pos: int) -> int:
    mid_positions = np.where(tokens == tasks.MID)[0]
    return int(mid_positions[0]) if len(mid_positions) else answer_pos


def make_batch(
    task_type: str, m: int, t_hops: int, cot_budget: str, batch_size: int,
    rng: np.random.Generator, bijection: np.ndarray | None, permutation: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Returns (input_tokens, target_tokens, loss_mask, seq_len, answer_pos) for one
    training batch. All examples share the same (task_type, t_hops, cot_budget), so the
    sequence length and the position of the answer symbol are identical across the
    whole batch and no padding is needed."""
    examples = []
    for _ in range(batch_size):
        if task_type == "primary":
            ex = tasks.sample_primary(rng, m, t_hops, cot_budget, bijection)
        else:
            ex = tasks.sample_control(rng, m, t_hops, cot_budget, permutation)
        examples.append(ex)

    seq_len = len(examples[0].tokens)
    answer_pos = examples[0].answer_pos
    tokens = np.stack([ex.tokens for ex in examples])
    inputs = torch.from_numpy(tokens[:, :-1]).long()
    targets = torch.from_numpy(tokens[:, 1:]).long()

    mask = torch.zeros(len(examples), seq_len - 1)
    for i, ex in enumerate(examples):
        start = _supervise_start(ex.tokens, ex.answer_pos)
        mask[i, start:] = 1.0
    return inputs, targets, mask, seq_len, answer_pos


def train_run(
    task_type: str,
    m: int,
    t_hops: int,
    cot_budget: str,
    n_layers: int,
    seed: int,
    bijection: np.ndarray | None = None,
    permutation: np.ndarray | None = None,
    n_steps: int = 4000,
    batch_size: int = 256,
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 512,
    lr: float = 3e-4,
    device: str = "cpu",
) -> tuple[TinyTransformer, float]:
    """Trains one tiny transformer on one (task_type, t_hops, cot_budget) condition.
    Returns the trained model and its exact-match accuracy on a fresh evaluation batch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    _, _, _, seq_len, _ = make_batch(task_type, m, t_hops, cot_budget, 2, rng, bijection, permutation)
    cfg = ModelConfig(
        vocab_size=tasks.vocab_size(m), seq_len=seq_len, n_layers=n_layers,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
    )
    model = TinyTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(n_steps):
        inputs, targets, mask, _, _ = make_batch(
            task_type, m, t_hops, cot_budget, batch_size, rng, bijection, permutation
        )
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        )
        loss = (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    eval_inputs, eval_targets, _, _, answer_pos = make_batch(
        task_type, m, t_hops, cot_budget, 512, rng, bijection, permutation
    )
    with torch.no_grad():
        logits = model(eval_inputs.to(device))
        preds = logits.argmax(dim=-1).cpu()
    correct = (preds[:, answer_pos] == eval_targets[:, answer_pos]).sum().item()
    accuracy = correct / preds.shape[0]
    return model, accuracy
