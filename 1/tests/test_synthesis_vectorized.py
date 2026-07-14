"""Synthetic table build scales ~linearly (smoke)."""

from __future__ import annotations

import time

import networkx as nx

from dhaka_pathfind.synthesis.attributes import build_synthetic_edge_table


def _chain(n_nodes: int) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for i in range(n_nodes):
        g.add_node(i, x=90.4 + i * 1e-4, y=23.75 + i * 1e-4)
    for i in range(n_nodes - 1):
        g.add_edge(i, i + 1, length=50.0)
    return g


def test_build_time_not_quadratic_smoke():
    g1 = _chain(500)
    g2 = _chain(2000)
    t1 = time.perf_counter()
    build_synthetic_edge_table(g1, seed=1)
    e1 = time.perf_counter() - t1
    t2 = time.perf_counter()
    build_synthetic_edge_table(g2, seed=1)
    e2 = time.perf_counter() - t2
    # Allow generous margin; fail if ~16× slower for 4× edges
    assert e2 < e1 * 20
