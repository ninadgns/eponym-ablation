"""Batch comparative runs — CSV matrix (≥100 rows for default settings)."""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from pathlib import Path

import pandas as pd

from dhaka_pathfind.config import OUTPUTS_DIR, ensure_outputs_dir
from dhaka_pathfind.cost.context import CostPreset, TravellerContext
from dhaka_pathfind.graph.load import load_landmarks, load_or_download, nearest_node
from dhaka_pathfind.heuristics.registry import HEURISTICS
from dhaka_pathfind.search.algorithms import (
    astar,
    bidirectional_ucs,
    dijkstra,
    greedy_best_first,
    ucs,
    weighted_astar,
)
from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges

HEURISTIC_NAMES = list(HEURISTICS.keys())
INFORMED = {"astar", "weighted_astar", "greedy_best_first"}


def _run_one(
    graph,
    s: int,
    t: int,
    ctx: TravellerContext,
    preset: CostPreset,
    algo_name: str,
    heuristic: str | None,
):
    if algo_name == "ucs":
        return ucs(graph, s, t, ctx, preset)
    if algo_name == "dijkstra":
        return dijkstra(graph, s, t, ctx, preset)
    if algo_name == "bidirectional_ucs":
        return bidirectional_ucs(graph, s, t, ctx, preset)
    if algo_name == "astar":
        return astar(graph, s, t, ctx, preset, heuristic_name=heuristic or "admissible")
    if algo_name == "weighted_astar":
        return weighted_astar(
            graph, s, t, ctx, preset, heuristic_name=heuristic or "admissible", weight=1.35
        )
    if algo_name == "greedy_best_first":
        return greedy_best_first(graph, s, t, ctx, preset, heuristic_name=heuristic or "fast")
    raise ValueError(algo_name)


UNINFORMED_ALGOS = ("ucs", "dijkstra", "bidirectional_ucs")
INFORMED_ALGOS = ("astar", "weighted_astar", "greedy_best_first")


def run_batch(
    pairs_count: int = 10,
    seed: int = 42,
    out_csv: Path | None = None,
    preset: CostPreset = CostPreset.BALANCED,
) -> Path:
    """
    Default: 10 pairs × (3 uninformed + 3 informed × 3 heuristics) = 120 rows ≥ 100.
    """
    ensure_outputs_dir()
    graph = load_or_download()
    ensure_synthetic_edges(graph)
    lm = load_landmarks()
    names = sorted(lm.keys())
    rng = random.Random(seed)

    pair_specs: list[tuple[str, str, int, int]] = []
    while len(pair_specs) < pairs_count:
        a, b = rng.sample(names, 2)
        sa = nearest_node(graph, lm[a]["lat"], lm[a]["lon"])
        ta = nearest_node(graph, lm[b]["lat"], lm[b]["lon"])
        if sa != ta:
            pair_specs.append((a, b, sa, ta))

    ctx = TravellerContext()
    ctx_hash = hashlib.sha256(ctx.model_dump_json().encode()).hexdigest()[:16]

    rows: list[dict[str, object]] = []
    for pair_id, (na, nb, s, t) in enumerate(pair_specs):
        for algo in UNINFORMED_ALGOS:
            t0 = time.perf_counter()
            res = _run_one(graph, s, t, ctx, preset, algo, None)
            dt = (time.perf_counter() - t0) * 1000
            rows.append(
                {
                    "pair_id": pair_id,
                    "landmark_a": na,
                    "landmark_b": nb,
                    "algorithm": algo,
                    "heuristic_name": "n/a",
                    "preset": preset.value,
                    "context_hash": ctx_hash,
                    "path_cost": res.path_cost,
                    "nodes_expanded": res.metrics.nodes_expanded,
                    "revisits": res.metrics.revisits,
                    "effective_branching_factor": res.metrics.effective_branching_factor,
                    "max_depth": res.metrics.max_depth,
                    "runtime_ms": dt,
                    "heuristic_mean_abs_gap": res.metrics.heuristic_mean_abs_gap,
                    "found": res.path is not None,
                }
            )
        for algo in INFORMED_ALGOS:
            for heur in HEURISTIC_NAMES:
                t0 = time.perf_counter()
                res = _run_one(graph, s, t, ctx, preset, algo, heur)
                dt = (time.perf_counter() - t0) * 1000
                rows.append(
                    {
                        "pair_id": pair_id,
                        "landmark_a": na,
                        "landmark_b": nb,
                        "algorithm": algo,
                        "heuristic_name": heur,
                        "preset": preset.value,
                        "context_hash": ctx_hash,
                        "path_cost": res.path_cost,
                        "nodes_expanded": res.metrics.nodes_expanded,
                        "revisits": res.metrics.revisits,
                        "effective_branching_factor": res.metrics.effective_branching_factor,
                        "max_depth": res.metrics.max_depth,
                        "runtime_ms": dt,
                        "heuristic_mean_abs_gap": res.metrics.heuristic_mean_abs_gap,
                        "found": res.path is not None,
                    }
                )

    df = pd.DataFrame(rows)
    outp = out_csv or (OUTPUTS_DIR / "results" / f"batch_{seed}.csv")
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print(f"Wrote {len(df)} rows to {outp}")
    return outp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=10, help="random landmark pairs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    run_batch(pairs_count=args.pairs, seed=args.seed, out_csv=args.out)


if __name__ == "__main__":
    main()
