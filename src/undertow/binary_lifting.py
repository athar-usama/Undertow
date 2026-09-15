"""A classical (non-neural) reference algorithm for the control task's shortcut: given
a single fixed permutation pi, compute pi^k(s0) in O(log k) compositions instead of k,
by precomputing pi, pi^2, pi^4, pi^8, ... once and combining them according to the
binary digits of k (binary lifting / exponentiation by squaring, adapted from integer
exponentiation to permutation composition).

This exists to turn "the model is probably doing some kind of shortcut" into a
falsifiable, checkable claim: the causal depth measured on the control task (see
README) can be compared directly against the depth this actual algorithm needs, not
just against the naive y=k baseline.
"""

from __future__ import annotations

import math

import numpy as np


def compose(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """(f then g), i.e. (g o f): apply f first, then g."""
    return g[f]


def precompute_powers(pi: np.ndarray, max_k: int) -> list[np.ndarray]:
    """[pi^(2^0), pi^(2^1), pi^(2^2), ...], enough terms to cover any k <= max_k."""
    n_terms = max(1, max_k.bit_length())
    powers = [pi.copy()]
    for _ in range(n_terms - 1):
        powers.append(compose(powers[-1], powers[-1]))
    return powers


def iterate_binary_lifting(s0: int, k: int, powers: list[np.ndarray]) -> tuple[int, int]:
    """pi^k(s0), computed by applying only the power-of-two permutations whose bit is
    set in k's binary representation. Returns (result, number of compositions actually
    applied) -- the latter is this algorithm's own "causal depth", at most floor(log2(k)) + 1."""
    state = s0
    applied = 0
    bit = 0
    remaining = k
    while remaining:
        if remaining & 1:
            state = int(powers[bit][state])
            applied += 1
        remaining >>= 1
        bit += 1
    return state, applied


def theoretical_min_depth(k: int) -> int:
    """floor(log2(k)) + 1 for k >= 1, the binary-lifting depth ceiling; 0 for k=0."""
    if k <= 0:
        return 0
    return math.floor(math.log2(k)) + 1
