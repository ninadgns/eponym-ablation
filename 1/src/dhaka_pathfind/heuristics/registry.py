"""Named heuristics for informed search."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import networkx as nx

from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.cost.model import compute_mean_cost_per_meter, compute_min_cost_per_meter
from dhaka_pathfind.heuristics.geo import heuristic_distance_m

HeuristicAux = dict[str, Any]


class HeuristicFn(Protocol):
    def __call__(
        self,
        graph: nx.MultiDiGraph,
        node: int,
        goal: int,
        ctx: TravellerContext,
        preset: CostPreset,
        aux: HeuristicAux,
    ) -> float: ...


def _admissible(
    graph: nx.MultiDiGraph,
    node: int,
    goal: int,
    ctx: TravellerContext,
    preset: CostPreset,
    aux: HeuristicAux,
) -> float:
    """
    Admissible: straight-line distance (m) × global minimum cost/m.

    Let m* = min_e C_e / length_e. For any path P from node to goal,
    cost(P) >= m* × length(P) >= m* × d_geo(node, goal) if d_geo undercounts true
    graph shortest-path length in meters (great-circle ≤ shortest path on road net
    in same metric space only if we assume road distance >= air — generally true).

    **Proof sketch:** For each edge, C_e / len_e >= m*, so any path has total cost
    >= m* × sum(len) >= m* × d_geo when sum(len) is at least the geodesic distance
    between endpoints (road distance is ≥ great-circle in dense urban networks
    is not always true globally — for strict admissibility we rely on m* as a
    **lower bound per meter** so h = m* × d_geo <= m* × road_distance <= true cost
    **provided** road_distance >= d_geo. In practice we use this as standard
    assignment heuristic; tests compare against reverse-Dijkstra on the **same** graph.
    """
    m_star = float(aux["min_cost_per_m"])
    d = heuristic_distance_m(graph, node, goal)
    return m_star * d


def _realism(
    graph: nx.MultiDiGraph,
    node: int,
    goal: int,
    ctx: TravellerContext,
    preset: CostPreset,
    aux: HeuristicAux,
) -> float:
    """
    Non-admissible: blends geodesic with **mean** cost/m (can overestimate).
    """
    mean_cm = float(aux["mean_cost_per_m"])
    d = heuristic_distance_m(graph, node, goal)
    return mean_cm * d * 1.08


def _fast(
    graph: nx.MultiDiGraph,
    node: int,
    goal: int,
    ctx: TravellerContext,
    preset: CostPreset,
    aux: HeuristicAux,
) -> float:
    """
    Non-admissible “cheap” heuristic: slightly inflated min cost/m (may overestimate).
    """
    m_star = float(aux["min_cost_per_m"])
    d = heuristic_distance_m(graph, node, goal)
    return m_star * d * 1.25


HEURISTICS: dict[str, HeuristicFn] = {
    "admissible": _admissible,
    "realism": _realism,
    "fast": _fast,
}


def build_heuristic_aux(
    graph: nx.MultiDiGraph,
    ctx: TravellerContext,
    preset: CostPreset,
) -> HeuristicAux:
    return {
        "min_cost_per_m": compute_min_cost_per_meter(graph, ctx, preset),
        "mean_cost_per_m": compute_mean_cost_per_meter(graph, ctx, preset),
    }


def get_heuristic(name: str) -> HeuristicFn:
    if name not in HEURISTICS:
        raise KeyError(f"Unknown heuristic {name!r}; choose from {sorted(HEURISTICS)}")
    return HEURISTICS[name]
