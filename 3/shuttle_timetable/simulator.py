"""Part A simulator — this file *is* the objective definition (PROBLEM_STATEMENT.md §A.5–A.6).

Boarding is FIFO by arrival, capped at bus capacity C. A student who never boards is charged
W_strand minutes. Every student carries exactly one wait; stranding is NOT a separate penalty
term (see the note in §A.6 — double-counting, and the perverse incentive to abandon the peak
crowd, are both real traps here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shuttle_timetable.instance import DEFAULT, ShuttleConfig, round_trip_time


@dataclass(frozen=True)
class SimResult:
    waits: np.ndarray  # per-student wait, minutes (W_strand if never boarded)
    boarded: np.ndarray  # bool mask: did this student get on a bus at all?
    loads: np.ndarray  # passengers carried by each departure, in departure order
    departures: np.ndarray  # sorted departure times actually used
    concurrency: np.ndarray  # buses in service on each 1-minute tick
    n_boarded: int
    n_stranded: int  # gave up (reneged) or service ended before they boarded
    mean_wait: float
    p90_wait: float
    service_level: float  # % of students waiting <= 10 min
    fleet_penalty: float
    objective: float  # mean_wait + fleet_penalty  (MINIMISE)


def simulate(x, arrivals: np.ndarray, cfg: ShuttleConfig = DEFAULT) -> SimResult:
    """Run one timetable against one realised arrival list.

    RENEGING: a student who has waited W_strand minutes gives up and walks. This is not
    decoration — without it, a student left behind by a full bus can end up waiting *longer*
    than the W_strand we charge someone who never boards, and the objective then prefers
    stranding them to carrying them. Reneging caps every wait at W_strand and makes the
    objective monotone in service quality. See PROBLEM_STATEMENT.md §A.5.
    """
    dep = np.sort(np.clip(np.asarray(x, dtype=float), 0.0, cfg.T))
    n = arrivals.size

    waits = np.full(n, cfg.W_strand, dtype=float)
    boarded = np.zeros(n, dtype=bool)
    loads = np.zeros(dep.size, dtype=int)

    # FIFO by arrival. `head` is the earliest student not yet dealt with; `arrived` is the number
    # who have shown up by the current departure. Everyone who has arrived, not boarded and not
    # given up is eligible, so the boarding set is always a contiguous slice starting at `head`.
    head = 0
    arrived = 0
    for j, td in enumerate(dep):
        while arrived < n and arrivals[arrived] <= td:
            arrived += 1
        # Give-ups: waited the full hour before this bus showed. They keep waits == W_strand.
        while head < arrived and arrivals[head] + cfg.W_strand < td:
            head += 1
        take = min(arrived - head, cfg.C)
        if take > 0:
            waits[head : head + take] = td - arrivals[head : head + take]
            boarded[head : head + take] = True
            loads[j] = take
            head += take

    n_boarded = int(boarded.sum())
    n_stranded = int(n - n_boarded)

    # Fleet occupancy: trip j holds a bus over [t_j, t_j + R(t_j)).
    tau = np.arange(0, cfg.T + 1, dtype=float)
    ret = dep + round_trip_time(dep, cfg)
    concurrency = (
        (tau[None, :] >= dep[:, None]) & (tau[None, :] < ret[:, None])
    ).sum(axis=0)
    over = np.maximum(0, concurrency - cfg.B).astype(float)
    fleet_penalty = float(cfg.lambda_fleet * np.sum(over**2))

    mean_wait = float(waits.mean()) if n else 0.0
    p90 = float(np.percentile(waits, 90)) if n else 0.0
    service_level = 100.0 * float(np.count_nonzero(waits <= 10.0)) / n if n else 100.0

    return SimResult(
        waits=waits,
        boarded=boarded,
        loads=loads,
        departures=dep,
        concurrency=concurrency,
        n_boarded=n_boarded,
        n_stranded=n_stranded,
        mean_wait=mean_wait,
        p90_wait=p90,
        service_level=service_level,
        fleet_penalty=fleet_penalty,
        objective=mean_wait + fleet_penalty,
    )


def objective(x, arrival_sets: list[np.ndarray], cfg: ShuttleConfig = DEFAULT) -> float:
    """The optimiser's objective: mean over the M FIXED training realisations. MINIMISE.

    Averaging over a fixed set of realisations (not fresh draws) keeps this a *deterministic*
    function of x. The optimiser must not be made to fight objective noise on top of a landscape
    that is already piecewise-constant.
    """
    return float(np.mean([simulate(x, a, cfg).objective for a in arrival_sets]))


def fitness(x, arrival_sets: list[np.ndarray], cfg: ShuttleConfig = DEFAULT) -> float:
    """Population-based solvers maximise. F = -J."""
    return -objective(x, arrival_sets, cfg)
