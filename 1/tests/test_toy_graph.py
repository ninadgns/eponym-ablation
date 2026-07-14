"""Small synthetic MultiDiGraph — no OSM download."""

from __future__ import annotations

import networkx as nx
import pytest

from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.cost.model import edge_cost, monotonicity_pair
from dhaka_pathfind.heuristics.ground_truth import dijkstra_dist_to_goal
from dhaka_pathfind.heuristics.registry import build_heuristic_aux, get_heuristic
from dhaka_pathfind.search.algorithms import astar, bidirectional_ucs, dijkstra, ucs


def _toy_graph() -> nx.MultiDiGraph:
    """4 nodes in a north–south line; ~100 m between consecutive nodes (OSMnx x=lon, y=lat)."""
    g = nx.MultiDiGraph()
    m_per_deg_lat = 111_000.0
    dlat = 100.0 / m_per_deg_lat
    base_lat, base_lon = 23.75, 90.40
    nodes = [(i + 1, {"x": base_lon, "y": base_lat + i * dlat}) for i in range(4)]
    for nid, d in nodes:
        g.add_node(nid, **d)
    synth = {
        "lanes": 3.0,
        "surface_quality": 0.7,
        "base_safety": 0.6,
        "accident_risk": 0.2,
        "lighting": 0.5,
        "crime_proxy": 0.3,
        "water_logging": 0.1,
        "incident_rate": 0.1,
        "rickshaw_allowed": 1,
        "traffic_congestion_prior": 0.4,
    }
    for u, v in [(1, 2), (2, 3), (3, 4)]:
        data = {"length": 100.0, **{f"synth_{k}": v for k, v in synth.items()}}
        g.add_edge(u, v, **data)
    return g


def test_ucs_equals_dijkstra_toy():
    g = _toy_graph()
    ctx = TravellerContext()
    pr = CostPreset.BALANCED
    r1 = ucs(g, 1, 4, ctx, pr)
    r2 = dijkstra(g, 1, 4, ctx, pr)
    assert r1.path == r2.path
    assert abs(r1.path_cost - r2.path_cost) < 1e-6


def test_ucs_equals_astar_admissible_toy():
    g = _toy_graph()
    ctx = TravellerContext()
    pr = CostPreset.BALANCED
    r1 = ucs(g, 1, 4, ctx, pr)
    r2 = astar(g, 1, 4, ctx, pr, heuristic_name="admissible")
    assert r2.path is not None
    assert abs(r1.path_cost - r2.path_cost) < 1e-3


def test_bidirectional_finds_path_toy():
    g = _toy_graph()
    ctx = TravellerContext()
    pr = CostPreset.BALANCED
    r = bidirectional_ucs(g, 1, 4, ctx, pr)
    assert r.path is not None
    assert r.path[0] == 1 and r.path[-1] == 4


def test_admissible_heuristic_vs_reverse_dijkstra_toy():
    g = _toy_graph()
    ctx = TravellerContext()
    pr = CostPreset.BALANCED
    w = lambda u, v, d: edge_cost(u, v, d, ctx, pr)
    dist = dijkstra_dist_to_goal(g, 4, w)
    aux = build_heuristic_aux(g, ctx, pr)
    h_fn = get_heuristic("admissible")
    for node in (1, 2, 3):
        h = h_fn(g, node, 4, ctx, pr, aux)
        assert h <= dist.get(node, float("inf")) + 0.02


def test_monotonicity_female_vs_male_every_edge():
    g = _toy_graph()
    pr = CostPreset.BALANCED
    female, male = monotonicity_pair()
    for u, v, k, data in g.edges(keys=True, data=True):
        cf = edge_cost(u, v, data, female, pr)
        cm = edge_cost(u, v, data, male, pr)
        assert cf > cm
