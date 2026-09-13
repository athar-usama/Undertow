from itertools import pairwise

import pytest

from undertow.graph import CausalGraph, longest_path_length, necessary_sites


def test_empty_graph_has_depth_zero():
    assert longest_path_length(CausalGraph()) == 0


def test_single_node_has_depth_one():
    g = CausalGraph()
    g.nodes.add((0, 1))
    assert longest_path_length(g) == 1


def test_disconnected_nodes_do_not_inflate_depth():
    g = CausalGraph()
    g.nodes.update({(0, 1), (1, 2), (2, 3)})
    assert longest_path_length(g) == 1


def test_known_chain_recovers_exact_length():
    # 0 -> 1 -> 2 -> 3 -> 4: a chain of five sites, depth should be exactly 5.
    g = CausalGraph()
    sites = [(layer, 0) for layer in range(5)]
    for a, b in pairwise(sites):
        g.add_edge(a, b)
    assert longest_path_length(g) == 5


def test_longest_path_picks_the_longer_of_two_branches():
    g = CausalGraph()
    # short branch: (0,0) -> (3,0)
    g.add_edge((0, 0), (3, 0))
    # long branch: (0,1) -> (1,1) -> (2,1) -> (3,0)
    g.add_edge((0, 1), (1, 1))
    g.add_edge((1, 1), (2, 1))
    g.add_edge((2, 1), (3, 0))
    assert longest_path_length(g) == 4


def test_edge_must_run_forward_in_layer_index():
    g = CausalGraph()
    with pytest.raises(ValueError):
        g.add_edge((2, 0), (1, 0))
    with pytest.raises(ValueError):
        g.add_edge((2, 0), (2, 1))


def test_necessary_sites_thresholding():
    effects = {(0, 0): 0.9, (1, 0): 0.4, (2, 0): 0.05}
    assert necessary_sites(effects, tau=0.5) == {(0, 0)}
    assert necessary_sites(effects, tau=0.3) == {(0, 0), (1, 0)}
