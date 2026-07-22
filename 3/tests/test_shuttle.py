"""Part A acceptance tests (PROBLEM_STATEMENT.md, Acceptance tests).

The load-factor test is the important one: it is the guard against the instance silently
degenerating into "everything strands, the objective is a constant, the search finds nothing".
"""

import numpy as np
import pytest

from shuttle_timetable.baselines import (
    demand_proportional_schedule,
    uniform_schedule,
)
from shuttle_timetable.instance import (
    ShuttleConfig,
    arrival_rate,
    expected_arrivals,
    load_factor,
    round_trip_time,
    sample_arrivals,
)
from shuttle_timetable.simulator import objective, simulate


def test_load_factor_keeps_the_instance_non_degenerate(shuttle_cfg):
    """If N > K*C every schedule strands students and the objective goes flat."""
    lf = load_factor(shuttle_cfg)
    assert 0.6 <= lf <= 0.85, (
        f"load factor {lf:.2f} outside [0.6, 0.85]: the instance is degenerate. "
        f"E[N]={expected_arrivals(shuttle_cfg):.0f}, seats={shuttle_cfg.K * shuttle_cfg.C}"
    )


def test_sampled_arrivals_match_the_expected_count(shuttle_cfg):
    counts = [
        sample_arrivals(shuttle_cfg, np.random.default_rng(s)).size for s in range(10)
    ]
    assert abs(np.mean(counts) - expected_arrivals(shuttle_cfg)) < 20


def test_arrival_rate_peaks_at_the_bump_centres(shuttle_cfg):
    for c in shuttle_cfg.bump_centres:
        assert arrival_rate(c, shuttle_cfg) > arrival_rate(c + 3 * shuttle_cfg.sigma, shuttle_cfg)


def test_round_trip_is_longer_in_rush_hour(shuttle_cfg):
    assert round_trip_time(120.0, shuttle_cfg) == pytest.approx(60.0, abs=0.1)  # 09:00
    assert round_trip_time(660.0, shuttle_cfg) == pytest.approx(60.0, abs=0.1)  # 18:00
    assert round_trip_time(390.0, shuttle_cfg) == pytest.approx(35.0, abs=0.5)  # midday


def test_objective_is_permutation_invariant(shuttle_cfg, arrivals):
    """The K! symmetry is real — it is the source of multimodality, so assert it exists."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, shuttle_cfg.T, shuttle_cfg.K)
    a = simulate(x, arrivals, shuttle_cfg).objective
    b = simulate(rng.permutation(x), arrivals, shuttle_cfg).objective
    assert a == pytest.approx(b)


def test_capacity_one_single_trip_strands_everyone_else(shuttle_cfg, arrivals):
    cfg = ShuttleConfig(K=1, C=1, **{
        k: v for k, v in vars(shuttle_cfg).items() if k not in ("K", "C")
    })
    res = simulate(np.array([cfg.T]), arrivals, cfg)
    assert res.n_boarded == 1
    assert res.n_stranded == arrivals.size - 1


def test_no_departure_strands_everyone(shuttle_cfg, arrivals):
    res = simulate(np.zeros(shuttle_cfg.K), arrivals, shuttle_cfg)  # all buses leave at t=0
    assert res.n_boarded == 0  # nobody has arrived yet at t=0
    assert res.mean_wait == pytest.approx(shuttle_cfg.W_strand)


def test_fleet_penalty_is_zero_when_headway_exceeds_the_loop(shuttle_cfg, arrivals):
    """Departures spaced wider than the longest loop can never need a second bus."""
    max_r = float(round_trip_time(np.arange(0, shuttle_cfg.T + 1.0), shuttle_cfg).max())
    dep = np.arange(shuttle_cfg.K) * (max_r + 1.0)
    dep = dep[dep <= shuttle_cfg.T]
    res = simulate(dep, arrivals, shuttle_cfg)
    assert res.concurrency.max() <= 1
    assert res.fleet_penalty == 0.0


def test_bunched_departures_violate_the_fleet_constraint(shuttle_cfg, arrivals):
    """All 14 trips inside one loop time needs 14 buses. We have 3."""
    res = simulate(np.full(shuttle_cfg.K, 300.0), arrivals, shuttle_cfg)
    assert res.concurrency.max() == shuttle_cfg.K
    assert res.fleet_penalty > 0.0


def test_stranded_students_are_charged_exactly_once(shuttle_cfg, arrivals):
    """Objective = mean wait + fleet penalty. Stranding enters ONLY through W_strand."""
    x = uniform_schedule(shuttle_cfg)
    res = simulate(x, arrivals, shuttle_cfg)
    assert res.objective == pytest.approx(res.mean_wait + res.fleet_penalty)
    assert int((~res.boarded).sum()) == res.n_stranded
    assert np.all(res.waits[~res.boarded] == shuttle_cfg.W_strand)


def test_reneging_caps_every_wait_at_w_strand(shuttle_cfg, arrivals):
    """No boarded student may wait longer than the penalty charged to someone who walks.

    Without reneging, a student bumped by a full bus can wait ~66 min against a W_strand of 60 —
    and the objective would then prefer to strand them rather than carry them. That is a real
    perverse incentive, and this test is the guard against it coming back.
    """
    for x in (uniform_schedule(shuttle_cfg), demand_proportional_schedule(shuttle_cfg)):
        res = simulate(x, arrivals, shuttle_cfg)
        assert res.waits.max() <= shuttle_cfg.W_strand
        assert res.waits[res.boarded].max() <= shuttle_cfg.W_strand


def test_demand_proportional_beats_uniform(shuttle_cfg, arrival_sets):
    """A sanity check on the instance: if the smart heuristic is NOT better than even headway,
    the arrival process has no structure for a search to exploit and the instance is wrong."""
    j_uniform = objective(uniform_schedule(shuttle_cfg), arrival_sets, shuttle_cfg)
    j_demand = objective(demand_proportional_schedule(shuttle_cfg), arrival_sets, shuttle_cfg)
    assert j_demand < j_uniform


def test_objective_is_deterministic(shuttle_cfg, arrival_sets):
    """Fixed training realisations => the optimiser is not fighting objective noise."""
    x = uniform_schedule(shuttle_cfg)
    assert objective(x, arrival_sets, shuttle_cfg) == objective(x, arrival_sets, shuttle_cfg)
