"""Particle Swarm Optimization — from scratch (PROBLEM_STATEMENT.md §A.8).

    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    x <- x + v

Kennedy & Eberhart (1995); the linearly decaying inertia weight is Shi & Eberhart (1998).

Design decisions the experiments in §A.10 rest on:

  * `topology` supports 'gbest' (fully connected) and 'ring' (k=1: neighbours i-1, i, i+1 mod n).
    The social attractor is the neighbourhood best; with 'gbest' every neighbourhood is the whole
    swarm, so the two share one code path and the comparison in A.10.3 is like-for-like.
  * `c2 = 0` severs social communication for real: the term drops out of the velocity update and
    each particle then follows only its own pbest, i.e. n independent hill-searchers. r2 is still
    drawn so the two arms consume the same RNG stream and differ only in the mechanism (A.10.2).
  * Every objective call goes through `_evaluate`, which increments the counter and writes one
    best-so-far entry into `curve`. The run stops at exactly `budget = n_particles * (1 + n_iters)`
    evaluations. The budget is the experimental control; a method that wins on more has not won.
  * Boundary handling is REPAIR: clamp x into [0, T] and zero the offending velocity component,
    so a particle pinned to a wall does not keep accumulating momentum into it. We do not reflect
    and we do not resample — one rule, stated and defended.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shuttle_timetable.instance import ShuttleConfig
from shuttle_timetable.simulator import objective


@dataclass(frozen=True)
class PSOResult:
    best_x: np.ndarray  # best schedule found
    best_j: float  # its objective value (minimised)
    curve: np.ndarray  # best-so-far objective, one entry per evaluation
    n_evals: int  # must equal the budget exactly


def pso(
    arrival_sets: list[np.ndarray],
    cfg: ShuttleConfig,
    rng: np.random.Generator,
    n_particles: int = 30,
    n_iters: int = 100,
    w_start: float = 0.9,
    w_end: float = 0.4,
    c1: float = 1.49,
    c2: float = 1.49,
    v_max_frac: float = 0.2,
    topology: str = "gbest",  # 'gbest' | 'ring'
) -> PSOResult:
    """Budget = n_particles * (1 + n_iters) = 3030 with the defaults."""
    if topology not in ("gbest", "ring"):
        raise ValueError(f"unknown topology {topology!r}; expected 'gbest' or 'ring'")

    k = cfg.K
    lo, hi = 0.0, float(cfg.T)
    v_max = v_max_frac * cfg.T
    budget = n_particles * (1 + n_iters)

    curve = np.empty(budget, dtype=float)
    n_evals = 0
    best_so_far = np.inf

    def _evaluate(x: np.ndarray) -> float:
        """The single point where the budget is spent. Nothing else may call `objective`."""
        nonlocal n_evals, best_so_far
        j = objective(x, arrival_sets, cfg)
        best_so_far = min(best_so_far, j)
        curve[n_evals] = best_so_far
        n_evals += 1
        return j

    # --- initialise ------------------------------------------------------------------
    x = rng.uniform(lo, hi, size=(n_particles, k))
    v = rng.uniform(-v_max, v_max, size=(n_particles, k))

    p_best_x = x.copy()
    p_best_j = np.array([_evaluate(xi) for xi in x])  # spends n_particles evaluations

    # --- iterate ---------------------------------------------------------------------
    for it in range(n_iters):
        w = w_start + (w_end - w_start) * (it / max(n_iters - 1, 1))  # 0.9 -> 0.4, linear

        social_x = _social_attractor(p_best_x, p_best_j, topology)

        r1 = rng.random((n_particles, k))
        r2 = rng.random((n_particles, k))  # drawn even when c2 = 0, to keep the arms comparable
        v = (
            w * v
            + c1 * r1 * (p_best_x - x)
            + c2 * r2 * (social_x - x)
        )
        np.clip(v, -v_max, v_max, out=v)
        x = x + v

        # Repair: clamp into the box and kill the velocity that drove the particle out of it.
        out = (x < lo) | (x > hi)
        np.clip(x, lo, hi, out=x)
        v[out] = 0.0

        for i in range(n_particles):
            j = _evaluate(x[i])
            if j < p_best_j[i]:
                p_best_j[i] = j
                p_best_x[i] = x[i].copy()

    best = int(np.argmin(p_best_j))
    return PSOResult(
        best_x=p_best_x[best].copy(),
        best_j=float(p_best_j[best]),
        curve=curve,
        n_evals=n_evals,
    )


def _social_attractor(
    p_best_x: np.ndarray, p_best_j: np.ndarray, topology: str
) -> np.ndarray:
    """The point each particle is pulled towards: gbest for everyone, or its ring neighbourhood's.

    Returned per-particle so both topologies drive the identical velocity update.
    """
    n = p_best_x.shape[0]
    if topology == "gbest" or n == 1:
        return np.repeat(p_best_x[np.argmin(p_best_j)][None, :], n, axis=0)

    # ring, k = 1: particle i consults {i-1, i, i+1} (mod n)
    neighbours = np.stack(
        [np.roll(p_best_j, 1), p_best_j, np.roll(p_best_j, -1)]
    )  # (3, n)
    offsets = np.array([-1, 0, 1])[neighbours.argmin(axis=0)]
    return p_best_x[(np.arange(n) + offsets) % n]
