import numpy as np

from undertow import tasks


def brute_force_primary(s0, keys, bijection, m):
    state = s0
    traj = [s0]
    for k in keys:
        state = int(bijection[(state + int(k)) % m])
        traj.append(state)
    return np.array(traj, dtype=np.int64)


def brute_force_control(s0, t_hops, permutation):
    state = s0
    traj = [s0]
    for _ in range(t_hops):
        state = int(permutation[state])
        traj.append(state)
    return np.array(traj, dtype=np.int64)


def test_bijections_are_actual_bijections():
    m = 32
    pi = tasks.make_fixed_permutation(seed=0, m=m)
    s = tasks.make_substitution_bijection(seed=0, m=m)
    assert sorted(pi.tolist()) == list(range(m))
    assert sorted(s.tolist()) == list(range(m))
    # the two must not accidentally be identical (different seeds offset)
    assert not np.array_equal(pi, s)


def test_primary_trajectory_matches_brute_force():
    m, t_hops = 32, 6
    rng = np.random.default_rng(1)
    bijection = tasks.make_substitution_bijection(seed=1, m=m)
    ex = tasks.sample_primary(rng, m, t_hops, "zero", bijection)
    keys = [ex.tokens[3 + i] - tasks.NUM_SPECIAL for i in range(t_hops)]
    s0 = ex.tokens[1] - tasks.NUM_SPECIAL
    expected = brute_force_primary(s0, keys, bijection, m)
    assert np.array_equal(ex.trajectory, expected)


def test_control_trajectory_matches_brute_force():
    m, t_hops = 32, 6
    rng = np.random.default_rng(2)
    permutation = tasks.make_fixed_permutation(seed=2, m=m)
    ex = tasks.sample_control(rng, m, t_hops, "zero", permutation)
    s0 = ex.tokens[1] - tasks.NUM_SPECIAL
    expected = brute_force_control(s0, t_hops, permutation)
    assert np.array_equal(ex.trajectory, expected)


def test_answer_position_predicts_final_state():
    m, t_hops = 32, 5
    rng = np.random.default_rng(3)
    bijection = tasks.make_substitution_bijection(seed=3, m=m)
    ex = tasks.sample_primary(rng, m, t_hops, "zero", bijection)
    assert ex.tokens[ex.answer_pos] == tasks.ANS
    assert ex.tokens[ex.answer_pos + 1] == tasks.sym(int(ex.trajectory[t_hops]))
    assert ex.tokens[ex.answer_pos + 2] == tasks.EOS


def test_cot_count_zero_full_partial():
    assert tasks.cot_count(6, "zero") == 0
    assert tasks.cot_count(6, "full") == 5
    assert tasks.cot_count(6, "partial") == 3  # ceil(5/2)
    assert tasks.cot_count(1, "partial") == 0  # no intermediates available


def test_full_cot_sequence_contains_all_intermediate_states():
    m, t_hops = 32, 5
    rng = np.random.default_rng(4)
    bijection = tasks.make_substitution_bijection(seed=4, m=m)
    ex = tasks.sample_primary(rng, m, t_hops, "full", bijection)
    mid_pos = np.where(ex.tokens == tasks.MID)[0][0]
    shown = ex.tokens[mid_pos + 1: ex.answer_pos]
    expected = [tasks.sym(int(x)) for x in ex.trajectory[1:t_hops]]
    assert shown.tolist() == expected


def test_resample_hop_input_only_changes_targeted_hop_for_primary():
    m, t_hops = 32, 6
    rng = np.random.default_rng(5)
    bijection = tasks.make_substitution_bijection(seed=5, m=m)
    ex = tasks.sample_primary(rng, m, t_hops, "zero", bijection)
    corrupt_rng = np.random.default_rng(6)
    corrupted = tasks.resample_hop_input(
        ex, hop_index=2, rng=corrupt_rng, m=m, task_type="primary",
        bijection=bijection, permutation=None,
    )
    # only the targeted key token differs in the input region
    diff_positions = np.where(ex.tokens[:ex.answer_pos] != corrupted.tokens[:ex.answer_pos])[0]
    assert diff_positions.tolist() == [ex.hop_positions[2]]
    # ground truth is recomputed consistently with the new key
    s0 = corrupted.tokens[1] - tasks.NUM_SPECIAL
    keys = [corrupted.tokens[3 + i] - tasks.NUM_SPECIAL for i in range(t_hops)]
    expected = brute_force_primary(s0, keys, bijection, m)
    assert np.array_equal(corrupted.trajectory, expected)
    assert corrupted.tokens[corrupted.answer_pos + 1] == tasks.sym(int(expected[t_hops]))


def test_resample_hop_input_changes_s0_for_control():
    m, t_hops = 32, 6
    rng = np.random.default_rng(7)
    permutation = tasks.make_fixed_permutation(seed=7, m=m)
    ex = tasks.sample_control(rng, m, t_hops, "zero", permutation)
    corrupt_rng = np.random.default_rng(8)
    corrupted = tasks.resample_hop_input(
        ex, hop_index=0, rng=corrupt_rng, m=m, task_type="control",
        bijection=None, permutation=permutation,
    )
    assert corrupted.tokens[1] != ex.tokens[1]
    new_s0 = corrupted.tokens[1] - tasks.NUM_SPECIAL
    expected = brute_force_control(new_s0, t_hops, permutation)
    assert np.array_equal(corrupted.trajectory, expected)


def test_resample_hop_input_preserves_partial_cot_region():
    m, t_hops = 32, 6
    rng = np.random.default_rng(9)
    bijection = tasks.make_substitution_bijection(seed=9, m=m)
    ex = tasks.sample_primary(rng, m, t_hops, "partial", bijection)
    corrupt_rng = np.random.default_rng(10)
    corrupted = tasks.resample_hop_input(
        ex, hop_index=0, rng=corrupt_rng, m=m, task_type="primary",
        bijection=bijection, permutation=None,
    )
    assert len(corrupted.tokens) == len(ex.tokens)
    mid_pos = np.where(corrupted.tokens == tasks.MID)[0][0]
    shown = corrupted.tokens[mid_pos + 1: corrupted.answer_pos]
    c_shown = corrupted.answer_pos - mid_pos - 1
    expected = [tasks.sym(int(x)) for x in corrupted.trajectory[t_hops - c_shown:t_hops]]
    assert shown.tolist() == expected
