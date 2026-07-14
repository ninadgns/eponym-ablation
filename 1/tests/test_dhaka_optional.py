"""Optional integration checks when ``data/dhaka_graph.graphml`` exists."""

from __future__ import annotations

import pytest

from dhaka_pathfind.config import graphml_path
from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.graph.load import load_graph_from_file, load_landmarks, nearest_node
from dhaka_pathfind.heuristics.ground_truth import dijkstra_dist_to_goal
from dhaka_pathfind.heuristics.registry import build_heuristic_aux, get_heuristic
from dhaka_pathfind.cost.model import make_edge_weight_fn
from dhaka_pathfind.search.algorithms import astar, ucs
from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges


@pytest.mark.slow
def test_dhaka_ucs_matches_astar_if_graph_cached():
    p = graphml_path()
    if not p.exists():
        pytest.skip("no cached Dhaka graph — run load_or_download once")

    g = load_graph_from_file(p)
    ensure_synthetic_edges(g)
    lm = load_landmarks()
    names = sorted(lm.keys())
    s = nearest_node(g, lm[names[0]]["lat"], lm[names[0]]["lon"])
    t = nearest_node(g, lm[names[-1]]["lat"], lm[names[-1]]["lon"])
    if s == t:
        pytest.skip("degenerate pair")
    ctx = TravellerContext()
    pr = CostPreset.BALANCED
    r1 = ucs(g, s, t, ctx, pr)
    r2 = astar(g, s, t, ctx, pr, heuristic_name="admissible")
    assert r1.path is not None and r2.path is not None
    assert abs(r1.path_cost - r2.path_cost) < max(1.0, 0.01 * r1.path_cost)


@pytest.mark.slow
def test_admissible_on_sample_nodes_if_graph_cached():
    p = graphml_path()
    if not p.exists():
        pytest.skip("no cached Dhaka graph")

    g = load_graph_from_file(p)
    ensure_synthetic_edges(g)
    nodes = list(g.nodes())[:200]
    goal = nodes[-1]
    ctx = TravellerContext()
    pr = CostPreset.BALANCED
    w = make_edge_weight_fn(ctx, pr)
    dist = dijkstra_dist_to_goal(g, goal, w)
    aux = build_heuristic_aux(g, ctx, pr)
    h_fn = get_heuristic("admissible")
    for n in nodes[:50]:
        if n == goal:
            continue
        h = h_fn(g, n, goal, ctx, pr, aux)
        assert h <= dist.get(n, float("inf")) + 5.0
