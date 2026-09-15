"""Synthetic task generators for the primary (keyed substitution chain) and control
(fixed permutation self-iteration) tasks.

Token vocabulary (shared by both tasks):
    PAD = 0
    BOS = 1
    SEP = 2   # primary task only: marks the start of the per-hop key sequence
    MID = 3   # marks the start of verbalized chain-of-thought intermediate states
    ANS = 4   # marks the position whose next-token prediction is the final answer
    EOS = 5
    symbols   = 6 .. 6 + M - 1, representing values in Z_M

Both tasks are trained one (task_type, T, cot_budget) combination at a time, so within
a single run the sequence length is fixed and no padding is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAD, BOS, SEP, MID, ANS, EOS = 0, 1, 2, 3, 4, 5
NUM_SPECIAL = 6


def sym(value: int) -> int:
    return NUM_SPECIAL + value


def vocab_size(m: int) -> int:
    return NUM_SPECIAL + m


def make_fixed_permutation(seed: int, m: int) -> np.ndarray:
    """A single random permutation of Z_m, fixed for the whole control-task family."""
    rng = np.random.default_rng(seed)
    return rng.permutation(m).astype(np.int64)


def make_substitution_bijection(seed: int, m: int) -> np.ndarray:
    """A single random (non-affine) bijection of Z_m, fixed for the whole primary-task
    family. A generic random permutation has no closed-form affine structure, so it
    cannot be combined across steps by summing/multiplying coefficients the way a
    linear congruential map could."""
    rng = np.random.default_rng(seed + 1_000_003)
    return rng.permutation(m).astype(np.int64)


def make_affine_bijection(seed: int, m: int) -> np.ndarray:
    """An affine bijection of Z_m (x -> a*x + b mod m, a coprime to m), for the ablation
    that asks whether it's specifically non-affinity, not merely having a fresh per-step
    key, that closes off the associative shortcut. Composing this with itself stays
    affine (a*(a*x+b)+b = a^2*x + (a*b+b), i.e. the coefficients just combine), and the
    same stays true when a fresh key is added before each application: the whole T-step
    composition reduces to s_T = a^T*s_0 + sum_i a^(T-i)*(a*k_i+b) mod m, a closed-form
    linear combination over a fixed, memorizable set of powers of a -- the same kind of
    shortcut the control task's fixed permutation admits, predicted to reappear here
    even though every step still takes a fresh input key."""
    rng = np.random.default_rng(seed + 3_000_017)
    candidates = [x for x in range(1, m) if np.gcd(x, m) == 1]
    a = int(rng.choice(candidates))
    b = int(rng.integers(0, m))
    return np.array([(a * x + b) % m for x in range(m)], dtype=np.int64)


def cot_count(t_hops: int, budget: str) -> int:
    """Number of intermediate states (out of s_1..s_{T-1}) that are verbalized."""
    n_available = max(t_hops - 1, 0)
    if budget == "zero":
        return 0
    if budget == "full":
        return n_available
    if budget == "partial":
        return -(-n_available // 2)  # ceil division
    raise ValueError(f"unknown cot budget {budget!r}")


@dataclass
class Example:
    tokens: np.ndarray  # (seq_len,) int64
    answer_pos: int  # index of the ANS token; logits there predict tokens[answer_pos+1]
    trajectory: np.ndarray  # (T+1,) int64, trajectory[0]=s0 .. trajectory[T]=s_T
    hop_positions: list[int]  # token positions of the per-hop corruptible inputs
    # (for the primary task: one position per key k_i; for the control task: a single
    # entry, the position of s0, since there is no other independent per-hop input)


def _trajectory_primary(s0: int, keys: np.ndarray, bijection: np.ndarray, m: int) -> np.ndarray:
    traj = np.empty(len(keys) + 1, dtype=np.int64)
    traj[0] = s0
    state = s0
    for i, k in enumerate(keys):
        state = int(bijection[(state + int(k)) % m])
        traj[i + 1] = state
    return traj


def _trajectory_control(s0: int, t_hops: int, permutation: np.ndarray) -> np.ndarray:
    traj = np.empty(t_hops + 1, dtype=np.int64)
    traj[0] = s0
    state = s0
    for i in range(t_hops):
        state = int(permutation[state])
        traj[i + 1] = state
    return traj


def _assemble(prefix: list[int], trajectory: np.ndarray, c_shown: int) -> tuple[np.ndarray, int]:
    t_hops = len(trajectory) - 1
    tokens = list(prefix)
    if c_shown > 0:
        tokens.append(MID)
        for i in range(t_hops - c_shown, t_hops):
            tokens.append(sym(int(trajectory[i])))
    tokens.append(ANS)
    answer_pos = len(tokens) - 1
    tokens.append(sym(int(trajectory[t_hops])))
    tokens.append(EOS)
    return np.array(tokens, dtype=np.int64), answer_pos


def sample_primary(rng: np.random.Generator, m: int, t_hops: int, cot_budget: str,
                    bijection: np.ndarray) -> Example:
    s0 = int(rng.integers(0, m))
    keys = rng.integers(0, m, size=t_hops).astype(np.int64)
    trajectory = _trajectory_primary(s0, keys, bijection, m)
    prefix = [BOS, sym(s0), SEP] + [sym(int(k)) for k in keys]
    c_shown = cot_count(t_hops, cot_budget)
    tokens, answer_pos = _assemble(prefix, trajectory, c_shown)
    hop_positions = [3 + i for i in range(t_hops)]  # position of each k_i in `tokens`
    return Example(tokens=tokens, answer_pos=answer_pos, trajectory=trajectory,
                   hop_positions=hop_positions)


def sample_control(rng: np.random.Generator, m: int, t_hops: int, cot_budget: str,
                    permutation: np.ndarray) -> Example:
    s0 = int(rng.integers(0, m))
    trajectory = _trajectory_control(s0, t_hops, permutation)
    prefix = [BOS, sym(s0)]
    c_shown = cot_count(t_hops, cot_budget)
    tokens, answer_pos = _assemble(prefix, trajectory, c_shown)
    hop_positions = [1]  # only s0 itself is an independent input site
    return Example(tokens=tokens, answer_pos=answer_pos, trajectory=trajectory,
                   hop_positions=hop_positions)


def resample_hop_input(example: Example, hop_index: int, rng: np.random.Generator, m: int,
                        task_type: str, bijection: np.ndarray | None,
                        permutation: np.ndarray | None) -> Example:
    """Build a corrupted variant of `example` that differs only in the input governing
    hop `hop_index` (for the primary task: key k_{hop_index}; for the control task,
    hop_index is ignored and s0 itself is resampled, since there is no other
    independent per-hop input available)."""
    tokens = example.tokens.copy()
    if task_type == "primary":
        pos = example.hop_positions[hop_index]
        old_key = tokens[pos] - NUM_SPECIAL
        new_key = int(rng.integers(0, m - 1))
        if new_key >= old_key:
            new_key += 1
        tokens[pos] = sym(new_key)
        s0 = int(tokens[1]) - NUM_SPECIAL
        keys = np.array([tokens[3 + i] - NUM_SPECIAL for i in range(len(example.hop_positions))])
        trajectory = _trajectory_primary(s0, keys, bijection, m)
    elif task_type == "control":
        pos = example.hop_positions[0]
        old_s0 = tokens[pos] - NUM_SPECIAL
        new_s0 = int(rng.integers(0, m - 1))
        if new_s0 >= old_s0:
            new_s0 += 1
        tokens[pos] = sym(new_s0)
        t_hops = len(example.trajectory) - 1
        trajectory = _trajectory_control(new_s0, t_hops, permutation)
    else:
        raise ValueError(task_type)

    t_hops = len(trajectory) - 1
    # Recompute the CoT-region tokens (if any) and the answer token from the new
    # trajectory, keeping the sequence length identical to the clean example.
    mid_positions = np.where(example.tokens == MID)[0]
    if len(mid_positions) == 1:
        mid_pos = int(mid_positions[0])
        c_shown = example.answer_pos - mid_pos - 1
        for j in range(c_shown):
            tokens[mid_pos + 1 + j] = sym(int(trajectory[t_hops - c_shown + j]))
    tokens[example.answer_pos + 1] = sym(int(trajectory[t_hops]))
    return Example(tokens=tokens, answer_pos=example.answer_pos, trajectory=trajectory,
                   hop_positions=example.hop_positions)
