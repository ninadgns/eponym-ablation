"""Hard-constraint checking and soft-objective evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from fuel_csp.problem import Assignment, Problem


@dataclass
class ConsistencyCounter:
    checks: int = 0

    def tick(self) -> None:
        self.checks += 1


def pump_clash(a: Assignment, b: Assignment) -> bool:
    return (
        a.station_id == b.station_id
        and a.pump_id == b.pump_id
        and a.slot_id == b.slot_id
    )


def is_consistent(
    problem: Problem,
    assignment: dict[int, Assignment],
    i: int,
    val: Assignment,
    counter: ConsistencyCounter | None = None,
) -> bool:
    """Check val for variable i against the partial assignment.

    Checks pump-exclusivity and incremental supply capacity.
    """
    if counter is not None:
        counter.tick()

    sid = val.station_id
    ft = problem.vehicles[i].fuel_type
    demand = problem.vehicles[i].demand_liters
    used = demand

    for j, other in assignment.items():
        if j == i:
            continue
        # pump clash
        if pump_clash(val, other):
            return False
        # accumulate fuel drawn at same station / same fuel type
        if other.station_id == sid and problem.vehicles[j].fuel_type == ft:
            used += problem.vehicles[j].demand_liters

    if used > problem.stations[sid].stocks(ft) + 1e-6:
        return False
    return True


def conflicts(
    problem: Problem,
    assignment: dict[int, Assignment],
    i: int,
    val: Assignment,
) -> int:
    """Count hard-constraint violations val would introduce. Used by Min-Conflicts."""
    n = 0
    sid = val.station_id
    ft = problem.vehicles[i].fuel_type
    used = problem.vehicles[i].demand_liters

    for j, other in assignment.items():
        if j == i:
            continue
        if pump_clash(val, other):
            n += 1
        if other.station_id == sid and problem.vehicles[j].fuel_type == ft:
            used += problem.vehicles[j].demand_liters

    if used > problem.stations[sid].stocks(ft) + 1e-6:
        n += 1
    return n


def total_conflicts(problem: Problem, assignment: dict[int, Assignment]) -> int:
    """Count all hard-constraint violations in a complete assignment."""
    n = 0
    seen: dict[tuple[int, int, int], int] = {}
    for i, a in assignment.items():
        key = (a.station_id, a.pump_id, a.slot_id)
        if key in seen:
            n += 1
        else:
            seen[key] = i

    drawn: dict[tuple[int, str], float] = defaultdict(float)
    for i, a in assignment.items():
        v = problem.vehicles[i]
        drawn[(a.station_id, v.fuel_type)] += v.demand_liters
    for (sid, ft), used in drawn.items():
        if used > problem.stations[sid].stocks(ft) + 1e-6:
            n += 1
    return n


def objective(problem: Problem, assignment: dict[int, Assignment]) -> float:
    """COP soft objective J(S). Lower is better."""
    w = problem.weights
    total_dist = 0.0
    total_wait = 0.0
    prio_penalty = 0.0
    n_unassigned = 0

    assigned = set(assignment)
    for i, v in enumerate(problem.vehicles):
        if i not in assigned:
            n_unassigned += 1
            prio_penalty += v.priority * 5.0
            continue
        a = assignment[i]
        total_dist += problem.distance_km(i, a.station_id)
        total_wait += float(a.slot_id)
        if v.kind == "ambulance":
            prio_penalty += float(a.slot_id) ** 2 * v.priority
        else:
            prio_penalty += float(a.slot_id) * max(0, v.priority - 1)

    return (
        w["distance"] * total_dist
        + w["wait"] * total_wait
        + w["priority"] * prio_penalty
        + w["unassigned"] * n_unassigned
    )


