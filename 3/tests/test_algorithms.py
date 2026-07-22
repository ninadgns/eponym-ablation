"""Contract tests for the two GRADED algorithms (PSO and GA).

These are the definition of "done". They skip while a function still raises NotImplementedError,
so the suite stays green as scaffolding and turns into a real gate the moment you start writing.

Do not weaken a test to make an implementation pass. If a test is wrong, fix the test on its
merits and say so — the reneging wart found while building this scaffold came from taking a
failing test seriously.
"""

import numpy as np
import pytest

from shuttle_timetable.baselines import demand_proportional_schedule, uniform_schedule
from shuttle_timetable.ga import genetic_algorithm
from shuttle_timetable.pso import pso
from shuttle_timetable.simulator import objective

BUDGET = 30 * 101  # 3030 — the fixed evaluation budget, the experimental control


def _skip_if_unimplemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")



def test_pso_respects_the_evaluation_budget(shuttle_cfg, arrival_sets):
    """The budget is the experimental control. A method that wins on more evaluations has not won."""
    res = _skip_if_unimplemented(
        pso, arrival_sets, shuttle_cfg, np.random.default_rng(0), n_particles=30, n_iters=100
    )
    assert res.n_evals == BUDGET
    assert res.curve.size == BUDGET


def test_pso_stays_inside_the_box(shuttle_cfg, arrival_sets):
    res = _skip_if_unimplemented(pso, arrival_sets, shuttle_cfg, np.random.default_rng(0))
    assert np.all(res.best_x >= 0.0) and np.all(res.best_x <= shuttle_cfg.T)


def test_pso_curve_is_monotone_non_increasing(shuttle_cfg, arrival_sets):
    """It is a BEST-SO-FAR curve. If it ever rises, you are plotting the wrong thing."""
    res = _skip_if_unimplemented(pso, arrival_sets, shuttle_cfg, np.random.default_rng(0))
    assert np.all(np.diff(res.curve) <= 1e-9)


def test_pso_reported_best_matches_a_fresh_evaluation(shuttle_cfg, arrival_sets):
    """Guards the classic bug: returning a gbest whose cached fitness is stale."""
    res = _skip_if_unimplemented(pso, arrival_sets, shuttle_cfg, np.random.default_rng(0))
    assert objective(res.best_x, arrival_sets, shuttle_cfg) == pytest.approx(res.best_j)


def test_pso_beats_the_demand_proportional_heuristic(shuttle_cfg, arrival_sets):
    """The heuristic is free and smart. If 3030 evaluations cannot beat it, REPORT THAT — but
    then the swarm is not earning its keep and the report must say so plainly."""
    res = _skip_if_unimplemented(pso, arrival_sets, shuttle_cfg, np.random.default_rng(0))
    j_heuristic = objective(demand_proportional_schedule(shuttle_cfg), arrival_sets, shuttle_cfg)
    assert res.best_j < j_heuristic


def test_pso_is_deterministic_given_a_seed(shuttle_cfg, arrival_sets):
    a = _skip_if_unimplemented(pso, arrival_sets, shuttle_cfg, np.random.default_rng(42))
    b = pso(arrival_sets, shuttle_cfg, np.random.default_rng(42))
    assert a.best_j == pytest.approx(b.best_j)


def test_severing_communication_is_a_real_ablation(shuttle_cfg, arrival_sets):
    """c2 = 0 must genuinely produce n independent searchers, not a cosmetically different swarm.
    Experiment A.10.2 is worthless if this is not true."""
    social = _skip_if_unimplemented(
        pso, arrival_sets, shuttle_cfg, np.random.default_rng(0), c2=1.49
    )
    solo = pso(arrival_sets, shuttle_cfg, np.random.default_rng(0), c2=0.0)
    assert solo.best_j != pytest.approx(social.best_j)


def test_ring_topology_is_supported(shuttle_cfg, arrival_sets):
    res = _skip_if_unimplemented(
        pso, arrival_sets, shuttle_cfg, np.random.default_rng(0), topology="ring"
    )
    assert res.n_evals == BUDGET


def test_ga_respects_the_same_budget(shuttle_cfg, arrival_sets):
    res = _skip_if_unimplemented(
        genetic_algorithm, arrival_sets, shuttle_cfg, np.random.default_rng(0)
    )
    assert res.n_evals == BUDGET
    assert np.all(res.best_x >= 0.0) and np.all(res.best_x <= shuttle_cfg.T)
    assert objective(res.best_x, arrival_sets, shuttle_cfg) == pytest.approx(res.best_j)


def test_ga_beats_a_uniform_timetable(shuttle_cfg, arrival_sets):
    res = _skip_if_unimplemented(
        genetic_algorithm, arrival_sets, shuttle_cfg, np.random.default_rng(0)
    )
    assert res.best_j < objective(uniform_schedule(shuttle_cfg), arrival_sets, shuttle_cfg)

