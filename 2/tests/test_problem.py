"""Tests for problem formulation and domain construction."""

import pytest
from fuel_csp.problem import Assignment, Problem, Station, Vehicle
from fuel_csp.synthetic import GeneratorConfig, generate_problem


def test_domains_built(small_problem):
    p = small_problem
    assert len(p.domains) == p.n
    for d in p.domains:
        assert isinstance(d, list)


def test_domain_values_respect_fuel_type(small_problem):
    p = small_problem
    for i, v in enumerate(p.vehicles):
        for a in p.domains[i]:
            s = p.stations[a.station_id]
            assert s.stocks(v.fuel_type) >= v.demand_liters


def test_domain_values_respect_reachability(small_problem):
    p = small_problem
    for i, v in enumerate(p.vehicles):
        for a in p.domains[i]:
            assert p.distance_km(i, a.station_id) <= v.range_km + 1e-9


def test_domain_values_respect_time_windows(small_problem):
    p = small_problem
    for i, v in enumerate(p.vehicles):
        for a in p.domains[i]:
            s = p.stations[a.station_id]
            assert s.open_slot <= a.slot_id < s.close_slot
            assert v.earliest_slot <= a.slot_id <= v.latest_slot


def test_constraint_graph_symmetric(small_problem):
    p = small_problem
    for i, nbrs in enumerate(p.neighbours):
        for j in nbrs:
            assert i in p.neighbours[j], f"Constraint graph not symmetric: {i}-{j}"


def test_distance_km_nonneg(small_problem):
    p = small_problem
    for i in range(p.n):
        for s in p.stations:
            assert p.distance_km(i, s.sid) >= 0
