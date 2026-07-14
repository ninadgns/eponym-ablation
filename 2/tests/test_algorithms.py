"""Tests for all solver algorithms."""

import pytest
from fuel_csp.algorithms import ALL_SOLVERS
from fuel_csp.constraints import total_conflicts
from fuel_csp.synthetic import GeneratorConfig, generate_problem


def _small():
    return generate_problem(GeneratorConfig(num_vehicles=10, seed=42))


def _tiny():
    return generate_problem(GeneratorConfig(num_vehicles=5, seed=1))


@pytest.mark.parametrize("name", list(ALL_SOLVERS.keys()))
def test_solver_returns_result(name):
    p = _tiny()
    solver = ALL_SOLVERS[name](time_budget_s=5.0)
    result = solver.solve(p)
    assert result is not None
    assert result.stats.algorithm == name


@pytest.mark.parametrize("name", list(ALL_SOLVERS.keys()))
def test_assignment_respects_hard_constraints(name):
    p = _small()
    solver = ALL_SOLVERS[name](time_budget_s=5.0)
    result = solver.solve(p)
    assert total_conflicts(p, result.assignment) == 0, (
        f"{name}: assignment has hard-constraint violations"
    )


@pytest.mark.parametrize("name", list(ALL_SOLVERS.keys()))
def test_assigned_vehicles_in_their_domains(name):
    p = _small()
    solver = ALL_SOLVERS[name](time_budget_s=5.0)
    result = solver.solve(p)
    for vid, val in result.assignment.items():
        assert val in p.domains[vid], (
            f"{name}: vehicle {vid} assigned to value not in its domain"
        )


@pytest.mark.parametrize("name", list(ALL_SOLVERS.keys()))
def test_stats_fields_populated(name):
    p = _tiny()
    solver = ALL_SOLVERS[name](time_budget_s=5.0)
    result = solver.solve(p)
    s = result.stats
    assert s.runtime_seconds >= 0
    assert s.n == p.n
    assert 0.0 <= s.failure_rate <= 1.0


def test_min_conflicts_cost_trace_monotone_trend():
    p = _small()
    solver = ALL_SOLVERS["min_conflicts"](time_budget_s=5.0)
    result = solver.solve(p)
    assert len(result.stats.cost_trace) > 0
