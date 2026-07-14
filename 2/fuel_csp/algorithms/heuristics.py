"""Variable-ordering and value-ordering heuristics."""

from __future__ import annotations

from fuel_csp.problem import Assignment, Problem


# ---------------------------------------------------------------------------
# Variable ordering
# ---------------------------------------------------------------------------

def mrv(
    problem: Problem,
    unassigned: list[int],
    live_domains: list[list[Assignment]],
) -> int:
    """Minimum Remaining Values — pick the variable with the smallest domain.

    Tie-break: degree heuristic (most constrained neighbours among unassigned
    variables wins). This uses the precomputed constraint graph rather than
    approximating degree with priority.
    """
    unassigned_set = set(unassigned)
    best = unassigned[0]
    best_remaining = len(live_domains[best])
    best_degree = sum(1 for nb in problem.neighbours[best] if nb in unassigned_set)

    for vid in unassigned[1:]:
        remaining = len(live_domains[vid])
        degree = sum(1 for nb in problem.neighbours[vid] if nb in unassigned_set)
        # MRV primary, degree secondary (more constraints = pick first)
        if (remaining, -degree) < (best_remaining, -best_degree):
            best = vid
            best_remaining = remaining
            best_degree = degree
    return best


def priority_first(problem: Problem, unassigned: list[int]) -> int:
    """Highest-priority vehicle first (ambulance before car)."""
    return max(unassigned, key=lambda i: (problem.vehicles[i].priority, -i))


def degree_heuristic(
    problem: Problem,
    unassigned: list[int],
) -> int:
    """Pick the variable involved in the most constraints with unassigned peers."""
    unassigned_set = set(unassigned)
    return max(
        unassigned,
        key=lambda i: sum(1 for nb in problem.neighbours[i] if nb in unassigned_set),
    )


# ---------------------------------------------------------------------------
# Value ordering
# ---------------------------------------------------------------------------

def lcv_sort(
    problem: Problem,
    i: int,
    assignment: dict[int, Assignment],
    candidate_values: list[Assignment],
    live_domains: list[list[Assignment]],
) -> list[Assignment]:
    """Least Constraining Value — prefer values that eliminate fewest choices
    from the remaining variables' live domains."""
    scored: list[tuple[int, float, Assignment]] = []
    for v in candidate_values:
        clash = 0
        for j in range(problem.n):
            if j == i or j in assignment:
                continue
            for other in live_domains[j]:
                if (
                    other.station_id == v.station_id
                    and other.pump_id == v.pump_id
                    and other.slot_id == v.slot_id
                ):
                    clash += 1
        # Tie-break: closer station + earlier slot is cheaper
        tie = problem.distance_km(i, v.station_id) + 0.1 * v.slot_id
        scored.append((clash, tie, v))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [v for _, _, v in scored]


def cost_sort(
    problem: Problem,
    i: int,
    candidate_values: list[Assignment],
) -> list[Assignment]:
    """Sort by single-variable cost (distance + slot-wait). Fast tie-breaker."""
    return sorted(
        candidate_values,
        key=lambda a: (problem.distance_km(i, a.station_id) + 0.3 * a.slot_id, a.slot_id),
    )
