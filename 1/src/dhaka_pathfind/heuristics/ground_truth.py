"""Reverse Dijkstra (from goal) for true remaining cost — tests and heuristic gaps."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from typing import Any

import networkx as nx


def dijkstra_dist_to_goal(
    graph: nx.MultiDiGraph,
    goal: int,
    weight: Callable[[int, int, dict[str, Any]], float],
) -> dict[int, float]:
    """
    Shortest-path distance *to* ``goal`` in the original directed graph.

    Runs Dijkstra on ``graph.reverse()`` starting at ``goal``. For each reversed
    edge ``(u, nbr)``, the corresponding original edge is ``(nbr, u)``.
    """
    rev = graph.reverse(copy=False)
    dist: dict[int, float] = {goal: 0.0}
    pq: list[tuple[float, int]] = [(0.0, goal)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist.get(u, math.inf):
            continue
        for nbr in rev.successors(u):
            for _k, data in rev[u][nbr].items():
                w = weight(nbr, u, data)
                nd = d_u + w
                if nd < dist.get(nbr, math.inf):
                    dist[nbr] = nd
                    heapq.heappush(pq, (nd, nbr))
    return dist
