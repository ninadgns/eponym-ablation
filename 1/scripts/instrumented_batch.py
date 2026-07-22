#!/usr/bin/env python
"""
Instrumented re-run of the Assignment 1 batch, separating heuristic *setup* cost
from *search* cost.

The stock batch driver (``dhaka_pathfind.analysis.batch``) records only the outer
wall clock, which folds together two very different things:

  * ``build_heuristic_aux`` -- two full O(|E|) passes over the graph computing the
    global min and mean cost-per-metre, done once per informed query;
  * the search itself.

``SearchResult.metrics.runtime_ms`` already excludes the former (the internal
``t0`` is taken after the aux build), so both halves are recoverable without
touching the library. This script records them side by side, plus a direct
timing of ``build_heuristic_aux`` in isolation, and checks whether the three
registry heuristics are distinguishable to greedy best-first search.

Writes: outputs/results/instrumented_42.csv
        outputs/results/setup_cost.csv
        outputs/results/greedy_invariance.csv
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

from dhaka_pathfind.config import OUTPUTS_DIR
from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.graph.load import load_landmarks, load_or_download, nearest_node
from dhaka_pathfind.heuristics.registry import HEURISTICS, build_heuristic_aux
from dhaka_pathfind.search.algorithms import (
    astar,
    bidirectional_ucs,
    dijkstra,
    greedy_best_first,
    ucs,
    weighted_astar,
)
from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges

SEED = 42
PAIRS = 10
PRESET = CostPreset.BALANCED
UNINFORMED = ("ucs", "dijkstra", "bidirectional_ucs")
INFORMED = ("astar", "weighted_astar", "greedy_best_first")


def run_one(graph, s, t, ctx, algo, heur):
    if algo == "ucs":
        return ucs(graph, s, t, ctx, PRESET)
    if algo == "dijkstra":
        return dijkstra(graph, s, t, ctx, PRESET)
    if algo == "bidirectional_ucs":
        return bidirectional_ucs(graph, s, t, ctx, PRESET)
    if algo == "astar":
        return astar(graph, s, t, ctx, PRESET, heuristic_name=heur)
    if algo == "weighted_astar":
        return weighted_astar(graph, s, t, ctx, PRESET, heuristic_name=heur, weight=1.35)
    if algo == "greedy_best_first":
        return greedy_best_first(graph, s, t, ctx, PRESET, heuristic_name=heur)
    raise ValueError(algo)


def main() -> None:
    graph = load_or_download()
    ensure_synthetic_edges(graph)
    lm = load_landmarks()
    names = sorted(lm)
    ctx = TravellerContext()

    n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
    print(f"graph: {n_nodes} nodes, {n_edges} edges")

    # ---- 1. heuristic setup cost in isolation -------------------------------
    setup_rows = []
    for rep in range(5):
        t0 = time.perf_counter()
        aux = build_heuristic_aux(graph, ctx, PRESET)
        dt = (time.perf_counter() - t0) * 1000
        setup_rows.append({"rep": rep, "setup_ms": dt, **{k: v for k, v in aux.items()}})
        print(f"  build_heuristic_aux rep {rep}: {dt:.1f} ms")

    # ---- 2. paired batch, setup and search timed separately -----------------
    rng = random.Random(SEED)
    pairs = []
    while len(pairs) < PAIRS:
        a, b = rng.sample(names, 2)
        sa = nearest_node(graph, lm[a]["lat"], lm[a]["lon"])
        ta = nearest_node(graph, lm[b]["lat"], lm[b]["lon"])
        if sa != ta:
            pairs.append((a, b, sa, ta))

    rows = []
    greedy_paths: dict[tuple[int, str], tuple] = {}
    for pid, (na, nb, s, t) in enumerate(pairs):
        combos = [(a, "n/a") for a in UNINFORMED]
        combos += [(a, h) for a in INFORMED for h in HEURISTICS]
        for algo, heur in combos:
            t0 = time.perf_counter()
            res = run_one(graph, s, t, ctx, algo, heur)
            total_ms = (time.perf_counter() - t0) * 1000
            search_ms = res.metrics.runtime_ms
            rows.append(
                {
                    "pair_id": pid,
                    "landmark_a": na,
                    "landmark_b": nb,
                    "algorithm": algo,
                    "heuristic_name": heur,
                    "path_cost": res.path_cost,
                    "nodes_expanded": res.metrics.nodes_expanded,
                    "revisits": res.metrics.revisits,
                    "path_len": len(res.path) if res.path else 0,
                    "total_ms": total_ms,
                    "search_ms": search_ms,
                    "setup_ms": total_ms - search_ms,
                    "found": res.path is not None,
                }
            )
            if algo == "greedy_best_first":
                greedy_paths[(pid, heur)] = tuple(res.path or ())
        print(f"  pair {pid} ({na} -> {nb}) done")

    # ---- 3. greedy invariance across the three heuristics -------------------
    inv_rows = []
    hs = list(HEURISTICS)
    for pid in range(PAIRS):
        ref = greedy_paths[(pid, hs[0])]
        inv_rows.append(
            {
                "pair_id": pid,
                "reference_heuristic": hs[0],
                **{
                    f"identical_to_{h}": (greedy_paths[(pid, h)] == ref) for h in hs[1:]
                },
                "path_len": len(ref),
            }
        )

    out = Path(OUTPUTS_DIR) / "results"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("instrumented_42.csv", rows),
        ("setup_cost.csv", setup_rows),
        ("greedy_invariance.csv", inv_rows),
    ):
        p = out / name
        with p.open("w", newline="") as fh:
            wri = csv.DictWriter(fh, fieldnames=list(data[0]))
            wri.writeheader()
            wri.writerows(data)
        print(f"wrote {p} ({len(data)} rows)")


if __name__ == "__main__":
    main()
