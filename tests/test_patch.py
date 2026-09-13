import numpy as np
import pytest
import torch

from undertow import tasks
from undertow.model import ModelConfig, TinyTransformer
from undertow.patch import build_graph, effect_sizes, run_with_cache


def _toy_model_and_example(seed=0):
    m, t_hops = 16, 4
    rng = np.random.default_rng(seed)
    bijection = tasks.make_substitution_bijection(seed=seed, m=m)
    ex = tasks.sample_primary(rng, m, t_hops, "zero", bijection)
    corrupted = tasks.resample_hop_input(
        ex, hop_index=1, rng=np.random.default_rng(seed + 1), m=m,
        task_type="primary", bijection=bijection, permutation=None,
    )
    cfg = ModelConfig(vocab_size=tasks.vocab_size(m), seq_len=len(ex.tokens), n_layers=3, d_model=32, n_heads=2)
    torch.manual_seed(seed)
    model = TinyTransformer(cfg)
    model.eval()
    clean_tokens = torch.from_numpy(ex.tokens[:-1]).long().unsqueeze(0)
    corrupted_tokens = torch.from_numpy(corrupted.tokens[:-1]).long().unsqueeze(0)
    clean_answer_token = int(ex.tokens[ex.answer_pos + 1])
    return model, ex, clean_tokens, corrupted_tokens, clean_answer_token


def test_patching_last_layer_answer_position_exactly_reproduces_clean_logit():
    model, ex, clean_tokens, corrupted_tokens, _clean_answer_token = _toy_model_and_example()
    clean_logits, clean_cache = run_with_cache(model, clean_tokens)
    clean_val = clean_cache[-1][:, ex.answer_pos, :]
    last_layer = len(clean_cache) - 1
    with torch.no_grad():
        patched_logits = model(corrupted_tokens, patch={last_layer: {ex.answer_pos: clean_val}})
    assert torch.allclose(
        patched_logits[0, ex.answer_pos], clean_logits[0, ex.answer_pos], atol=1e-5
    )


def test_effect_size_is_one_at_full_last_layer_patch_and_bounded_elsewhere():
    model, ex, clean_tokens, corrupted_tokens, clean_answer_token = _toy_model_and_example()
    effects = effect_sizes(model, clean_tokens, corrupted_tokens, ex.answer_pos, clean_answer_token)
    last_layer = model.cfg.n_layers - 1
    assert effects[(last_layer, ex.answer_pos)] == pytest.approx(1.0, abs=1e-4)
    assert all(0.0 <= v <= 1.0 for v in effects.values())


def test_no_op_when_clean_equals_corrupted():
    model, ex, clean_tokens, _, clean_answer_token = _toy_model_and_example()
    effects = effect_sizes(model, clean_tokens, clean_tokens, ex.answer_pos, clean_answer_token)
    assert all(v == 0.0 for v in effects.values())


def test_build_graph_is_a_dag_within_top_k():
    model, ex, clean_tokens, corrupted_tokens, clean_answer_token = _toy_model_and_example()
    effects = effect_sizes(model, clean_tokens, corrupted_tokens, ex.answer_pos, clean_answer_token)
    graph = build_graph(model, clean_tokens, corrupted_tokens, effects, tau=0.0, top_k=6)
    for src, dst in graph.edges:
        assert dst[0] > src[0]
    assert len(graph.nodes) <= 6
