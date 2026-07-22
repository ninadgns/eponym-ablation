#!/usr/bin/env python
"""
Measure the `_path_cost()` call that bidirectional_ucs() and greedy_best_first() make
*inside* their timed regions -- the one asymmetry §4.2 declines to correct for.

Unlike astar()'s O(path_len^2) post-processing, which instrumented_batch.py times and
subtracts, this one is O(path_len) and small enough that subtracting it would claim a
precision we do not have. We measure it anyway, so the two figures quoted in §4.2 come
from the released code rather than from a note.

    cd 1 && uv run python ../repro/path_cost_probe.py

Same graph, context, preset and ten landmark pairs as instrumented_batch.py (SEED = 42).
"""

from __future__ import annotations

import random
import statistics as st
import time

from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.cost.model import make_edge_weight_fn
from dhaka_pathfind.graph.load import load_landmarks, load_or_download, nearest_node
from dhaka_pathfind.search.algorithms import _path_cost, bidirectional_ucs, greedy_best_first
from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges

SEED = 42
PAIRS = 10
REPS = 5
PRESET = CostPreset.BALANCED
HEUR = "admissible"


def main() -> None:
    graph = load_or_download()
    ensure_synthetic_edges(graph)
    lm = load_landmarks()
    names = sorted(lm)
    ctx = TravellerContext()
    w = make_edge_weight_fn(ctx, PRESET)

    # Identical pair selection to instrumented_batch.py.
    rng = random.Random(SEED)
    pairs = []
    while len(pairs) < PAIRS:
        a, b = rng.sample(names, 2)
        sa = nearest_node(graph, lm[a]["lat"], lm[a]["lon"])
        ta = nearest_node(graph, lm[b]["lat"], lm[b]["lon"])
        if sa != ta:
            pairs.append((a, b, sa, ta))

    for label, run in (
        ("bidirectional_ucs", lambda s, t: bidirectional_ucs(graph, s, t, ctx, PRESET)),
        ("greedy_best_first",
         lambda s, t: greedy_best_first(graph, s, t, ctx, PRESET, heuristic_name=HEUR)),
    ):
        per_pair = []
        for _na, _nb, s, t in pairs:
            res = run(s, t)
            assert res.path is not None
            reps = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                _path_cost(graph, res.path, w)
                reps.append((time.perf_counter() - t0) * 1000)
            per_pair.append(st.median(reps))
        print(f"{label:20} _path_cost: mean {st.mean(per_pair):.2f} ms over {PAIRS} pairs "
              f"(median {st.median(per_pair):.2f}, range {min(per_pair):.2f}-{max(per_pair):.2f})")


if __name__ == "__main__":
    main()
