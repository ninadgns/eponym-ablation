#!/usr/bin/env python
"""
Instrumented re-run of the Assignment 1 batch, on a common timing basis.

Three problems with the stock batch driver (``dhaka_pathfind.analysis.batch``) that this
script fixes:

1. **Setup is folded into search.** ``build_heuristic_aux`` runs two full O(|E|) passes over
   the graph before an informed search expands anything. The driver's outer clock includes it;
   ``SearchResult.metrics.runtime_ms`` excludes it (the internal ``t0`` is taken after the aux
   build). We record both, and time the aux build separately.

2. **The timed regions are not the same region.** ``astar()`` computes ``_suffix_costs()`` --- an
   O(path_len^2) re-evaluation of the multi-factor edge cost --- plus a per-path-node heuristic
   evaluation *inside* its timed region, for the ``heuristic_mean_abs_gap`` diagnostic
   (``algorithms.py:288-303``). We time that post-processing on the identical path and subtract
   it. Only ``ucs()`` and ``weighted_astar()`` are then free of post-processing:
   ``bidirectional_ucs()`` and ``greedy_best_first()`` each call ``_path_cost()`` inside their
   timed regions too (``algorithms.py:244`` and ``:419``), which is O(path_len) and costs well
   under a millisecond -- measured by ``repro/path_cost_probe.py``, and left in place rather than
   subtracted.

3. **Single-shot timings.** Search times vary by tens of percent run to run on a laptop. We take
   ``REPS`` repetitions per (pair, configuration) and report the median, with the aux build
   memoised so repetitions are not dominated by setup.

Writes to outputs/results/: instrumented_42.csv (per-rep), setup_cost.csv,
greedy_invariance.csv, astar_postproc.csv.
"""

from __future__ import annotations

import csv
import random
import statistics as st
import time
from pathlib import Path

from dhaka_pathfind.config import OUTPUTS_DIR
from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.cost.model import make_edge_weight_fn
from dhaka_pathfind.graph.load import load_landmarks, load_or_download, nearest_node
from dhaka_pathfind.heuristics.registry import HEURISTICS, build_heuristic_aux, get_heuristic
from dhaka_pathfind.search import algorithms as algo_mod
from dhaka_pathfind.search.algorithms import (
    _suffix_costs,
    astar,
    bidirectional_ucs,
    dijkstra,
    greedy_best_first,
    ucs,
    weighted_astar,
)
from dhaka_pathfind.search.metrics_utils import mean_heuristic_gap_on_path
from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges

SEED = 42
PAIRS = 10
REPS = 5
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
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # ---- 1. heuristic setup cost, timed on its own ---------------------------
    setup_rows = []
    for rep in range(5):
        t0 = time.perf_counter()
        aux = build_heuristic_aux(graph, ctx, PRESET)
        dt = (time.perf_counter() - t0) * 1000
        setup_rows.append({"rep": rep, "setup_ms": dt, **aux})
        print(f"  build_heuristic_aux rep {rep}: {dt:.1f} ms")

    # Memoise the aux build so repetitions measure search, not setup. Patch the name the
    # algorithms module resolves, not the registry's, since it imported by value.
    _cache: dict = {}

    def _memo(g, c, p):
        key = (id(g), repr(c), p)
        if key not in _cache:
            _cache[key] = build_heuristic_aux(g, c, p)
        return _cache[key]

    algo_mod.build_heuristic_aux = _memo

    rng = random.Random(SEED)
    pairs = []
    while len(pairs) < PAIRS:
        a, b = rng.sample(names, 2)
        sa = nearest_node(graph, lm[a]["lat"], lm[a]["lon"])
        ta = nearest_node(graph, lm[b]["lat"], lm[b]["lon"])
        if sa != ta:
            pairs.append((a, b, sa, ta))

    w = make_edge_weight_fn(ctx, PRESET)
    aux = build_heuristic_aux(graph, ctx, PRESET)

    rows, pp_rows = [], []
    greedy_paths: dict[tuple[int, str], tuple] = {}

    for pid, (na, nb, s, t) in enumerate(pairs):
        combos = [(a, "n/a") for a in UNINFORMED]
        combos += [(a, h) for a in INFORMED for h in HEURISTICS]
        for algo, heur in combos:
            for rep in range(REPS):
                res = run_one(graph, s, t, ctx, algo, heur)
                search_ms = res.metrics.runtime_ms

                # Post-processing that only astar() performs inside its timed region.
                postproc_ms = 0.0
                if algo == "astar" and res.path is not None:
                    h_fn = get_heuristic(heur)
                    t0 = time.perf_counter()
                    suff = _suffix_costs(graph, res.path, w)
                    mean_heuristic_gap_on_path(
                        [h_fn(graph, res.path[i], t, ctx, PRESET, aux)
                         for i in range(len(res.path))],
                        suff,
                    )
                    postproc_ms = (time.perf_counter() - t0) * 1000
                    pp_rows.append({
                        "pair_id": pid, "heuristic": heur, "rep": rep,
                        "path_len": len(res.path), "reported_search_ms": search_ms,
                        "postproc_ms": postproc_ms,
                    })

                rows.append({
                    "pair_id": pid, "landmark_a": na, "landmark_b": nb, "rep": rep,
                    "algorithm": algo, "heuristic_name": heur,
                    "path_cost": res.path_cost,
                    "nodes_expanded": res.metrics.nodes_expanded,
                    "revisits": res.metrics.revisits,
                    "path_len": len(res.path) if res.path else 0,
                    "search_ms_reported": search_ms,
                    "postproc_ms": postproc_ms,
                    "search_ms": search_ms - postproc_ms,
                    "found": res.path is not None,
                })
                if algo == "greedy_best_first" and rep == 0:
                    greedy_paths[(pid, heur)] = tuple(res.path or ())
        print(f"  pair {pid} ({na} -> {nb}) done", flush=True)

    hs = list(HEURISTICS)
    inv_rows = [{
        "pair_id": pid, "reference_heuristic": hs[0],
        **{f"identical_to_{h}": greedy_paths[(pid, h)] == greedy_paths[(pid, hs[0])]
           for h in hs[1:]},
        "path_len": len(greedy_paths[(pid, hs[0])]),
    } for pid in range(PAIRS)]

    out = Path(OUTPUTS_DIR) / "results"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("instrumented_42.csv", rows),
        ("setup_cost.csv", setup_rows),
        ("greedy_invariance.csv", inv_rows),
        ("astar_postproc.csv", pp_rows),
    ):
        p = out / name
        with p.open("w", newline="") as fh:
            wri = csv.DictWriter(fh, fieldnames=list(data[0]))
            wri.writeheader()
            wri.writerows(data)
        print(f"wrote {p} ({len(data)} rows)")

    pp = [r["postproc_ms"] for r in pp_rows]
    print(f"\nA* post-processing subtracted: median {st.median(pp):.1f} ms "
          f"over {len(pp)} measurements")


if __name__ == "__main__":
    main()
