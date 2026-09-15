import numpy as np

from undertow.binary_lifting import iterate_binary_lifting, precompute_powers, theoretical_min_depth


def brute_force(pi: np.ndarray, s0: int, k: int) -> int:
    state = s0
    for _ in range(k):
        state = int(pi[state])
    return state


def test_binary_lifting_matches_brute_force_many_permutations_and_k():
    m = 32
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pi = rng.permutation(m).astype(np.int64)
        powers = precompute_powers(pi, max_k=64)
        for k in (0, 1, 2, 3, 4, 5, 7, 8, 10, 16, 31, 63, 64):
            for s0 in (0, 1, m - 1, rng.integers(0, m)):
                expected = brute_force(pi, int(s0), k)
                got, _applied = iterate_binary_lifting(int(s0), k, powers)
                assert got == expected, f"seed={seed} k={k} s0={s0}"


def test_applied_count_never_exceeds_theoretical_bound():
    m = 32
    rng = np.random.default_rng(0)
    pi = rng.permutation(m).astype(np.int64)
    powers = precompute_powers(pi, max_k=1000)
    for k in range(200):
        _, applied = iterate_binary_lifting(0, k, powers)
        assert applied <= theoretical_min_depth(k)


def test_theoretical_min_depth_known_values():
    assert theoretical_min_depth(0) == 0
    assert theoretical_min_depth(1) == 1
    assert theoretical_min_depth(2) == 2
    assert theoretical_min_depth(3) == 2
    assert theoretical_min_depth(4) == 3
    assert theoretical_min_depth(10) == 4
    assert theoretical_min_depth(1023) == 10
    assert theoretical_min_depth(1024) == 11


def test_precompute_powers_are_actual_powers_of_pi():
    m = 16
    rng = np.random.default_rng(1)
    pi = rng.permutation(m).astype(np.int64)
    powers = precompute_powers(pi, max_k=16)
    # powers[0] = pi^1
    assert np.array_equal(powers[0], pi)
    # powers[1] = pi^2, verified against brute force for every starting state
    for s0 in range(m):
        assert powers[1][s0] == brute_force(pi, s0, 2)
    # powers[2] = pi^4
    for s0 in range(m):
        assert powers[2][s0] == brute_force(pi, s0, 4)
