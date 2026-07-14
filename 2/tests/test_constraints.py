"""Tests for constraint checking and objective function."""

import pytest
from fuel_csp.constraints import (
    conflicts,
    is_consistent,
    objective,
    pump_clash,
    total_conflicts,
)
from fuel_csp.problem import Assignment


def test_pump_clash_same_slot():
    a = Assignment(0, 0, 0)
    b = Assignment(0, 0, 0)
    assert pump_clash(a, b)


def test_pump_clash_different_pump():
    a = Assignment(0, 0, 0)
    b = Assignment(0, 1, 0)
    assert not pump_clash(a, b)


def test_pump_clash_different_slot():
    a = Assignment(0, 0, 0)
    b = Assignment(0, 0, 1)
    assert not pump_clash(a, b)


def test_is_consistent_no_assignment(small_problem):
    p = small_problem
    # Any value should be consistent against an empty assignment
    for i, domain in enumerate(p.domains):
        if domain:
            assert is_consistent(p, {}, i, domain[0])
            break


def test_is_consistent_pump_conflict(small_problem):
    p = small_problem
    # Find two vehicles that share a domain value
    for i in range(p.n):
        for j in range(i + 1, p.n):
            vals_i = {(a.station_id, a.pump_id, a.slot_id): a for a in p.domains[i]}
            for a in p.domains[j]:
                key = (a.station_id, a.pump_id, a.slot_id)
                if key in vals_i:
                    clashing_i = vals_i[key]
                    assignment = {i: clashing_i}
                    assert not is_consistent(p, assignment, j, a)
                    return
    pytest.skip("No overlapping domain values found in this fixture")


def test_objective_increases_with_unassigned(small_problem):
    p = small_problem
    full = {i: p.domains[i][0] for i in range(p.n) if p.domains[i]}
    partial = {i: v for i, v in list(full.items())[: len(full) // 2]}
    assert objective(p, partial) > objective(p, full) - 1e-9


def test_total_conflicts_empty():
    from fuel_csp.synthetic import GeneratorConfig, generate_problem
    p = generate_problem(GeneratorConfig(num_vehicles=5, seed=1))
    assert total_conflicts(p, {}) == 0


