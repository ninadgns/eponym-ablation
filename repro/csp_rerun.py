#!/usr/bin/env python
"""
Clean re-run of the Assignment 2 CSP/COP sweep, for the paper.

Differences from ``scripts/run_experiments.py``:

  * Runs only solvers that actually exist in the codebase. ``ALL_SOLVERS`` omits
    ``AC3Solver`` even though ``fuel_csp/algorithms/ac3.py`` implements it, so we
    add it back here rather than editing the repo's registry.
  * Adds a small-n regime (5..20) where systematic search completes, so the
    scaling curve has an uncensored section; the 2 s budget censors everything
    at n >= 30 and the original sweep started at n = 10.
  * 8 seeds instead of 3, for usable error bars.
  * Records the censoring flag explicitly per run so the paper can state which
    cells are budget-limited rather than algorithm-limited.

Writes csp_rerun_raw.csv next to this script.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2"))

from fuel_csp.algorithms import ALL_SOLVERS  # noqa: E402
from fuel_csp.algorithms.min_conflicts import MinConflictsSolver  # noqa: E402
from fuel_csp.analyzer import ExperimentConfig  # noqa: E402
from fuel_csp.synthetic import GeneratorConfig, generate_problem  # noqa: E402

# NOTE: fuel_csp/algorithms/ac3.py implements an AC3Solver but is absent from
# ALL_SOLVERS and does not import (it references fuel_csp.constraints.conflict_set,
# which does not exist). No conflict-directed-backjumping solver exists anywhere in
# the tree. The five below are every solver in this codebase that runs.
SOLVERS = dict(ALL_SOLVERS)

SIZES = (5, 8, 10, 12, 15, 20, 25, 30, 40, 50)
SEEDS = (7, 13, 42, 101, 202, 303, 404, 505)
BUDGET_S = 5.0


def main() -> None:
    cfg = ExperimentConfig(time_budget_s=BUDGET_S, num_stations=6, num_slots=6)
    rows = []
    total = len(SOLVERS) * len(SIZES) * len(SEEDS)
    done = 0
    for name, cls in SOLVERS.items():
        for n in SIZES:
            for seed in SEEDS:
                problem = generate_problem(
                    GeneratorConfig(num_vehicles=n, num_stations=6, num_slots=6, seed=seed)
                )
                if cls is MinConflictsSolver:
                    solver = cls(max_steps=cfg.min_conflicts_steps, seed=seed, time_budget_s=BUDGET_S)
                else:
                    solver = cls(time_budget_s=BUDGET_S)
                res = solver.solve(problem)
                res.stats.seed = seed
                d = res.stats.as_dict()
                d["algorithm"] = name
                d["budget_s"] = BUDGET_S
                d["censored"] = res.stats.runtime_seconds >= 0.98 * BUDGET_S
                rows.append(d)
                done += 1
                if done % 40 == 0:
                    print(f"{done}/{total}", flush=True)

    out = Path(__file__).with_name("csp_rerun_raw.csv")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
