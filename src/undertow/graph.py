"""Necessity thresholding and longest-necessary-chain (MNPC-depth) extraction from a
patching effect-size table.

A "site" is a (layer, position) pair. Given per-site effect sizes in [0, 1] from
activation patching, a site is a candidate node if its effect exceeds `tau`. A directed
edge A -> B (layer(A) < layer(B)) is added when a mediation test shows that patching A
alone measurably moves B's activation toward its clean value. Edges only ever run from
a lower layer to a strictly higher layer, so the resulting graph is a DAG by
construction and layer index is already a valid topological order.

MNPC-depth (Minimum Necessary Patch-Chain depth) is the number of nodes on the longest
path through that DAG, computed by a standard topological dynamic program.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Site = tuple[int, int]  # (layer, position)


@dataclass
class CausalGraph:
    nodes: set[Site] = field(default_factory=set)
    edges: set[tuple[Site, Site]] = field(default_factory=set)

    def add_edge(self, src: Site, dst: Site) -> None:
        if dst[0] <= src[0]:
            raise ValueError(f"edge must run forward in layer index, got {src} -> {dst}")
        self.nodes.add(src)
        self.nodes.add(dst)
        self.edges.add((src, dst))


def necessary_sites(effect_sizes: dict[Site, float], tau: float) -> set[Site]:
    return {site for site, effect in effect_sizes.items() if effect >= tau}


def longest_path(graph: CausalGraph) -> list[Site]:
    """The longest directed path in `graph`, as an ordered list of sites from its
    earliest layer to its latest. Empty if the graph has no nodes."""
    if not graph.nodes:
        return []

    preds: dict[Site, list[Site]] = {n: [] for n in graph.nodes}
    for src, dst in graph.edges:
        preds[dst].append(src)

    dp: dict[Site, int] = {}
    back: dict[Site, Site | None] = {}
    for node in sorted(graph.nodes, key=lambda s: s[0]):
        best_pred, best_len = None, 0
        for p in preds[node]:
            if dp[p] > best_len:
                best_len, best_pred = dp[p], p
        dp[node] = 1 + best_len
        back[node] = best_pred

    end = max(dp, key=dp.get)
    chain: list[Site] = []
    while end is not None:
        chain.append(end)
        end = back[end]
    chain.reverse()
    return chain


def longest_path_length(graph: CausalGraph) -> int:
    """Number of nodes on the longest directed path in `graph`. A graph with no edges
    but at least one node has depth 1 (a single necessary site, no serial dependency
    between distinct sites); an empty graph has depth 0."""
    return len(longest_path(graph))
