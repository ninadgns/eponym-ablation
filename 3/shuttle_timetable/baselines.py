"""Part A baselines (PROBLEM_STATEMENT.md §A.9).

`demand_proportional` is the one that matters. It is a genuinely smart heuristic — it places
departures so equal numbers of students arrive between consecutive buses — and it costs nothing
to compute. If PSO cannot beat it, that is the finding, and it goes in the report.
"""

from __future__ import annotations

import numpy as np

from shuttle_timetable.instance import DEFAULT, ShuttleConfig, arrival_rate
from shuttle_timetable.simulator import objective


def uniform_schedule(cfg: ShuttleConfig = DEFAULT) -> np.ndarray:
    """K departures on an even headway. Current practice."""
    return (np.arange(cfg.K) + 0.5) * cfg.T / cfg.K


def demand_proportional_schedule(cfg: ShuttleConfig = DEFAULT) -> np.ndarray:
    """Departure j leaves once j/K of the day's expected arrivals have shown up.

    Equal expected load per bus. Ignores capacity and the fleet constraint entirely, which is
    exactly why a search can beat it — but it is not a strawman.
    """
    grid = np.arange(0.0, cfg.T + 0.5, 0.5)
    lam = arrival_rate(grid, cfg)
    cum = np.cumsum(lam)
    cum /= cum[-1]
    quantiles = np.arange(1, cfg.K + 1) / cfg.K
    return np.interp(quantiles, cum, grid)


def random_search(
    arrival_sets: list[np.ndarray],
    cfg: ShuttleConfig,
    budget: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform sampling of the box, at the same evaluation budget as PSO/GA.

    Returns (best_x, best_so_far_curve) where the curve has one entry per evaluation.
    """
    best_x = None
    best_j = np.inf
    curve = np.empty(budget, dtype=float)
    for i in range(budget):
        x = rng.uniform(0.0, cfg.T, cfg.K)
        j = objective(x, arrival_sets, cfg)
        if j < best_j:
            best_j, best_x = j, x
        curve[i] = best_j
    return best_x, curve
