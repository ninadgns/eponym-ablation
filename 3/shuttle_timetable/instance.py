"""Part A instance: the arrival process and the round-trip time.

Every constant is defended in PROBLEM_STATEMENT.md §A.2–A.3.

The load factor N / (K*C) must stay in [0.6, 0.85]. Above 1.0 the instance is degenerate:
every schedule strands students, the objective flattens, and the search has nothing to find.
`tests/test_shuttle_instance.py` guards this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShuttleConfig:
    T: int = 780  # service window, minutes (07:00-20:00)
    K: int = 14  # trips to schedule (the budget)
    C: int = 40  # bus capacity
    B: int = 3  # fleet size

    lam0: float = 0.10  # baseline arrival rate, students/min
    sigma: float = 8.0  # class-change bump width, minutes
    # Bump centres: the 90-minute DU class cadence. Amplitudes rise through the day and
    # peak at the evening departure surge.
    bump_centres: tuple[float, ...] = (50, 140, 230, 320, 410, 500, 590, 680)
    bump_amps: tuple[float, ...] = (1.5, 2.0, 2.0, 1.5, 1.5, 2.0, 2.5, 3.0)

    R0: float = 35.0  # free-flow loop time, minutes
    R1: float = 25.0  # congestion surcharge at full rush, minutes
    congestion_centres: tuple[float, ...] = (120.0, 660.0)  # 09:00 and 18:00
    congestion_sigma: float = 45.0

    W_strand: float = 60.0  # wait charged to a student who never boards
    lambda_fleet: float = 0.5  # weight on the fleet-overload penalty


DEFAULT = ShuttleConfig()


def arrival_rate(t, cfg: ShuttleConfig = DEFAULT) -> np.ndarray:
    """Student arrival rate lambda(t) in students/min. Baseline plus Gaussian class bumps."""
    t = np.asarray(t, dtype=float)
    lam = np.full(t.shape, cfg.lam0, dtype=float)
    for c, a in zip(cfg.bump_centres, cfg.bump_amps, strict=True):
        lam += a * np.exp(-((t - c) ** 2) / (2.0 * cfg.sigma**2))
    return lam


def expected_arrivals(cfg: ShuttleConfig = DEFAULT) -> float:
    """E[N] = integral of lambda over the service window (trapezoid on a half-minute grid)."""
    grid = np.arange(0.0, cfg.T + 0.5, 0.5)
    lam = arrival_rate(grid, cfg)
    return float(np.sum(0.5 * (lam[:-1] + lam[1:]) * np.diff(grid)))


def sample_arrivals(cfg: ShuttleConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw arrival times from the non-homogeneous Poisson process by thinning.

    Returns a sorted array of arrival times in [0, T].
    """
    grid = np.arange(0.0, cfg.T + 1.0, 1.0)
    lam_max = float(arrival_rate(grid, cfg).max())  # bump centres are integers, so this is exact
    n_prop = rng.poisson(lam_max * cfg.T)
    t = rng.uniform(0.0, cfg.T, n_prop)
    accept = rng.uniform(0.0, 1.0, n_prop) < arrival_rate(t, cfg) / lam_max
    return np.sort(t[accept])


def round_trip_time(t, cfg: ShuttleConfig = DEFAULT) -> np.ndarray:
    """Loop time R(t) for a bus departing at t. Free-flow plus a congestion surcharge.

    A bus leaving at 09:00 is gone ~60 min; one leaving at noon, ~35 min.
    """
    t = np.asarray(t, dtype=float)
    g = np.zeros(t.shape, dtype=float)
    for c in cfg.congestion_centres:
        g += np.exp(-((t - c) ** 2) / (2.0 * cfg.congestion_sigma**2))
    g = np.minimum(g, 1.0)
    return cfg.R0 + cfg.R1 * g


def load_factor(cfg: ShuttleConfig = DEFAULT) -> float:
    """E[N] / (K*C). The instance is only interesting while this sits below 1."""
    return expected_arrivals(cfg) / (cfg.K * cfg.C)
