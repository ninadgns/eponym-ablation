"""Six search algorithms using the same realistic edge-weight function."""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable
from itertools import count
from typing import Any

import networkx as nx

from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.cost.model import edge_cost, make_edge_weight_fn
from dhaka_pathfind.heuristics.registry import HeuristicAux, build_heuristic_aux, get_heuristic
from dhaka_pathfind.search.metrics_utils import effective_branching_factor, mean_heuristic_gap_on_path
from dhaka_pathfind.search.types import SearchMetrics, SearchResult

WeightFn = Callable[[int, int, dict[str, Any]], float]
HeuristicCallable = Callable[[nx.MultiDiGraph, int, int, TravellerContext, CostPreset, HeuristicAux], float]


def _path_cost(graph: nx.MultiDiGraph, path: list[int], w: WeightFn) -> float:
    t = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        best = math.inf
        for _k, data in graph[u][v].items():
            best = min(best, w(u, v, data))
        t += best
    return t


def _suffix_costs(graph: nx.MultiDiGraph, path: list[int], w: WeightFn) -> list[float]:
    n = len(path)
    out = [0.0] * n
    for i in range(n - 1):
        acc = 0.0
        for j in range(i, n - 1):
            u, v = path[j], path[j + 1]
            best = min(w(u, v, d) for _k, d in graph[u][v].items())
            acc += best
        out[i] = acc
    out[-1] = 0.0
    return out


def _reconstruct(came_from: dict[int, int | None], source: int, target: int) -> list[int] | None:
    if target not in came_from:
        return None
    path = [target]
    while path[-1] != source:
        p = came_from.get(path[-1])
        if p is None:
            return None
        path.append(p)
    path.reverse()
    return path


def ucs(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    ctx: TravellerContext,
    preset: CostPreset,
) -> SearchResult:
    """
    Uniform-cost search — optimal for non-negative weights.

    **Note:** For a single pair, this is algorithmically the same relaxation order as Dijkstra stopping at ``target``; see ``dijkstra`` for the named variant.
    """
    w = make_edge_weight_fn(ctx, preset)
    t0 = time.perf_counter()
    pq: list[tuple[float, int, int]] = []
    tie = count()
    heapq.heappush(pq, (0.0, next(tie), source))
    g_best: dict[int, float] = {source: 0.0}
    came: dict[int, int | None] = {source: None}
    expanded = 0
    revisits = 0

    while pq:
        g, _, u = heapq.heappop(pq)
        if g > g_best.get(u, math.inf):
            revisits += 1
            continue
        expanded += 1
        if u == target:
            path = _reconstruct(came, source, target)
            assert path is not None
            depth = len(path) - 1
            ms = SearchMetrics(
                nodes_expanded=expanded,
                revisits=revisits,
                effective_branching_factor=effective_branching_factor(expanded, depth),
                max_depth=depth,
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
            return SearchResult(path=path, path_cost=g, metrics=ms, algorithm="ucs")

        for v in graph.successors(u):
            for _k, data in graph[u][v].items():
                wgt = w(u, v, data)
                ng = g + wgt
                if ng < g_best.get(v, math.inf):
                    g_best[v] = ng
                    came[v] = u
                    heapq.heappush(pq, (ng, next(tie), v))

    ms = SearchMetrics(
        nodes_expanded=expanded,
        revisits=revisits,
        runtime_ms=(time.perf_counter() - t0) * 1000,
    )
    return SearchResult(path=None, path_cost=math.inf, metrics=ms, algorithm="ucs")


def dijkstra(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    ctx: TravellerContext,
    preset: CostPreset,
) -> SearchResult:
    """
    **Dijkstra (single-pair):** same implementation as UCS here — classic PQ relaxation
    with early exit at ``target``. Included as a separate entrypoint for coursework naming.
    """
    r = ucs(graph, source, target, ctx, preset)
    r.algorithm = "dijkstra"
    return r


def bidirectional_ucs(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    ctx: TravellerContext,
    preset: CostPreset,
) -> SearchResult:
    """Bidirectional uniform-cost search on a directed graph."""
    w = make_edge_weight_fn(ctx, preset)
    rev = graph.reverse(copy=False)
    t0 = time.perf_counter()
    tie = count()

    pq_f: list[tuple[float, int, int]] = []
    pq_b: list[tuple[float, int, int]] = []
    heapq.heappush(pq_f, (0.0, next(tie), source))
    heapq.heappush(pq_b, (0.0, next(tie), target))

    g_f: dict[int, float] = {source: 0.0}
    g_b: dict[int, float] = {target: 0.0}
    came_f: dict[int, int | None] = {source: None}
    came_b: dict[int, int | None] = {target: None}

    mu = math.inf
    meeting: int | None = None
    expanded = 0
    revisits = 0

    def pop_revisit(g: float, best: dict[int, float], u: int) -> bool:
        nonlocal revisits
        if g > best.get(u, math.inf):
            revisits += 1
            return True
        return False

    while pq_f and pq_b:
        if pq_f[0][0] + pq_b[0][0] >= mu and math.isfinite(mu):
            break

        if pq_f[0][0] <= pq_b[0][0]:
            g, _, u = heapq.heappop(pq_f)
            if pop_revisit(g, g_f, u):
                continue
            expanded += 1
            if u in g_b and g + g_b[u] < mu:
                mu = g + g_b[u]
                meeting = u
            for v in graph.successors(u):
                for _k, data in graph[u][v].items():
                    wgt = w(u, v, data)
                    ng = g + wgt
                    if ng < g_f.get(v, math.inf):
                        g_f[v] = ng
                        came_f[v] = u
                        heapq.heappush(pq_f, (ng, next(tie), v))
                        if v in g_b and ng + g_b[v] < mu:
                            mu = ng + g_b[v]
                            meeting = v
        else:
            g, _, u = heapq.heappop(pq_b)
            if pop_revisit(g, g_b, u):
                continue
            expanded += 1
            if u in g_f and g_f[u] + g < mu:
                mu = g_f[u] + g
                meeting = u
            for v in rev.successors(u):
                for _k, data in rev[u][v].items():
                    wgt = w(v, u, data)
                    ng = g + wgt
                    if ng < g_b.get(v, math.inf):
                        g_b[v] = ng
                        came_b[v] = u
                        heapq.heappush(pq_b, (ng, next(tie), v))
                        if v in g_f and g_f[v] + ng < mu:
                            mu = g_f[v] + ng
                            meeting = v

    def forward_path(came: dict[int, int | None], source: int, dest: int) -> list[int]:
        """``came[v]=u`` means ``u`` precedes ``v`` from ``source``; return ``source..dest``."""
        p = [dest]
        while p[-1] != source:
            p.append(came[p[-1]])  # type: ignore[arg-type]
        p.reverse()
        return p

    if target in g_f and (meeting is None or g_f[target] < mu):
        meeting = target
        mu = g_f[target]

    if meeting is None or not math.isfinite(mu):
        ms = SearchMetrics(
            nodes_expanded=expanded,
            revisits=revisits,
            runtime_ms=(time.perf_counter() - t0) * 1000,
        )
        return SearchResult(path=None, path_cost=math.inf, metrics=ms, algorithm="bidirectional_ucs")

    def tail_path(came: dict[int, int | None], start: int, dest: int) -> list[int]:
        """``came[v]=u`` is one step toward ``dest`` in backward tree; return ``start..dest``."""
        p = [start]
        while p[-1] != dest:
            p.append(came[p[-1]])  # type: ignore[arg-type]
        return p

    left = forward_path(came_f, source, meeting)
    right = tail_path(came_b, meeting, target)
    path = left + right[1:]
    pc = _path_cost(graph, path, w)
    depth = len(path) - 1
    ms = SearchMetrics(
        nodes_expanded=expanded,
        revisits=revisits,
        effective_branching_factor=effective_branching_factor(expanded, depth),
        max_depth=depth,
        runtime_ms=(time.perf_counter() - t0) * 1000,
    )
    return SearchResult(path=path, path_cost=pc, metrics=ms, algorithm="bidirectional_ucs")


def astar(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    ctx: TravellerContext,
    preset: CostPreset,
    heuristic_name: str = "admissible",
) -> SearchResult:
    """A* with a named heuristic from the registry."""
    w = make_edge_weight_fn(ctx, preset)
    aux = build_heuristic_aux(graph, ctx, preset)
    h_fn = get_heuristic(heuristic_name)

    def h(node: int) -> float:
        return h_fn(graph, node, target, ctx, preset, aux)

    t0 = time.perf_counter()
    tie = count()
    pq: list[tuple[float, float, int, int]] = []
    heapq.heappush(pq, (h(source), 0.0, next(tie), source))
    g_best: dict[int, float] = {source: 0.0}
    came: dict[int, int | None] = {source: None}
    expanded = 0
    revisits = 0

    while pq:
        f, g, _, u = heapq.heappop(pq)
        if g > g_best.get(u, math.inf):
            revisits += 1
            continue
        expanded += 1
        if u == target:
            path = _reconstruct(came, source, target)
            assert path is not None
            suff = _suffix_costs(graph, path, w)
            gap = mean_heuristic_gap_on_path(
                [h_fn(graph, path[i], target, ctx, preset, aux) for i in range(len(path))],
                suff,
            )
            depth = len(path) - 1
            ms = SearchMetrics(
                nodes_expanded=expanded,
                revisits=revisits,
                effective_branching_factor=effective_branching_factor(expanded, depth),
                max_depth=depth,
                runtime_ms=(time.perf_counter() - t0) * 1000,
                heuristic_mean_abs_gap=gap,
            )
            return SearchResult(path=path, path_cost=g, metrics=ms, algorithm="astar")

        for v in graph.successors(u):
            for _k, data in graph[u][v].items():
                wgt = w(u, v, data)
                ng = g + wgt
                if ng < g_best.get(v, math.inf):
                    g_best[v] = ng
                    came[v] = u
                    heapq.heappush(pq, (ng + h(v), ng, next(tie), v))

    ms = SearchMetrics(
        nodes_expanded=expanded,
        revisits=revisits,
        runtime_ms=(time.perf_counter() - t0) * 1000,
    )
    return SearchResult(path=None, path_cost=math.inf, metrics=ms, algorithm="astar")


def weighted_astar(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    ctx: TravellerContext,
    preset: CostPreset,
    heuristic_name: str = "admissible",
    weight: float = 1.35,
) -> SearchResult:
    """Weighted A*: f = g + w * h."""
    wfn = make_edge_weight_fn(ctx, preset)
    aux = build_heuristic_aux(graph, ctx, preset)
    h_fn = get_heuristic(heuristic_name)

    def h(node: int) -> float:
        return h_fn(graph, node, target, ctx, preset, aux)

    t0 = time.perf_counter()
    tie = count()
    pq: list[tuple[float, float, int, int]] = []
    heapq.heappush(pq, (weight * h(source), 0.0, next(tie), source))
    g_best: dict[int, float] = {source: 0.0}
    came: dict[int, int | None] = {source: None}
    expanded = 0
    revisits = 0

    while pq:
        f, g, _, u = heapq.heappop(pq)
        if g > g_best.get(u, math.inf):
            revisits += 1
            continue
        expanded += 1
        if u == target:
            path = _reconstruct(came, source, target)
            assert path is not None
            depth = len(path) - 1
            ms = SearchMetrics(
                nodes_expanded=expanded,
                revisits=revisits,
                effective_branching_factor=effective_branching_factor(expanded, depth),
                max_depth=depth,
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
            return SearchResult(path=path, path_cost=g, metrics=ms, algorithm="weighted_astar")

        for v in graph.successors(u):
            for _k, data in graph[u][v].items():
                wgt = wfn(u, v, data)
                ng = g + wgt
                if ng < g_best.get(v, math.inf):
                    g_best[v] = ng
                    came[v] = u
                    heapq.heappush(pq, (ng + weight * h(v), ng, next(tie), v))

    ms = SearchMetrics(
        nodes_expanded=expanded,
        revisits=revisits,
        runtime_ms=(time.perf_counter() - t0) * 1000,
    )
    return SearchResult(path=None, path_cost=math.inf, metrics=ms, algorithm="weighted_astar")


def greedy_best_first(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
    ctx: TravellerContext,
    preset: CostPreset,
    heuristic_name: str = "fast",
) -> SearchResult:
    """Greedy best-first: expand minimum h only (not optimal)."""
    wfn = make_edge_weight_fn(ctx, preset)
    aux = build_heuristic_aux(graph, ctx, preset)
    h_fn = get_heuristic(heuristic_name)

    def h(node: int) -> float:
        return h_fn(graph, node, target, ctx, preset, aux)

    t0 = time.perf_counter()
    tie = count()
    pq: list[tuple[float, int, int]] = []
    heapq.heappush(pq, (h(source), next(tie), source))
    came: dict[int, int | None] = {source: None}
    g_acc: dict[int, float] = {source: 0.0}
    expanded = 0
    closed: set[int] = set()

    while pq:
        _, _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == target:
            path = _reconstruct(came, source, target)
            assert path is not None
            pc = _path_cost(graph, path, wfn)
            depth = len(path) - 1
            ms = SearchMetrics(
                nodes_expanded=expanded,
                revisits=0,
                effective_branching_factor=effective_branching_factor(expanded, depth),
                max_depth=depth,
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
            return SearchResult(path=path, path_cost=pc, metrics=ms, algorithm="greedy_best_first")

        for v in graph.successors(u):
            if v in closed:
                continue
            for _k, data in graph[u][v].items():
                wgt = wfn(u, v, data)
                ng = g_acc[u] + wgt
                if v not in g_acc or ng < g_acc[v]:
                    g_acc[v] = ng
                    came[v] = u
                    heapq.heappush(pq, (h(v), next(tie), v))

    ms = SearchMetrics(nodes_expanded=expanded, runtime_ms=(time.perf_counter() - t0) * 1000)
    return SearchResult(path=None, path_cost=math.inf, metrics=ms, algorithm="greedy_best_first")
