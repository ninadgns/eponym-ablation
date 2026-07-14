"""
Backtracking solver family.

Four configurations share the _BTBase engine, differing only in variable
ordering, value ordering, and whether forward checking is on.

  1. BasicBacktracking   — pure recursion, no heuristics
  2. BacktrackingMRV     — MRV variable ordering (degree tie-break)
  3. BacktrackingLCV     — LCV value ordering + priority variable order
  4. BacktrackingFC      — Forward Checking + MRV + LCV (composite)

The degree tie-break in MRV uses the precomputed constraint graph (real
edges) rather than priority as a proxy.

Forward checking prunes supply headroom in O(1) using a running dict
rather than re-summing on every value trial.

All solvers implement COP graceful failure: when a variable's domain
empties, it is skipped (not hard-failed) and the best partial snapshot
is returned at the end.
"""

from __future__ import annotations

from time import perf_counter

from fuel_csp.algorithms.base import Solver, SolverResult, SolverStats, Timer
from fuel_csp.algorithms.heuristics import cost_sort, lcv_sort, mrv, priority_first
from fuel_csp.constraints import ConsistencyCounter, is_consistent, objective
from fuel_csp.problem import Assignment, Problem


def _record_best(
    problem: Problem,
    assignment: dict[int, Assignment],
    best: dict,
) -> None:
    if len(assignment) < best["count"]:
        return
    cost = objective(problem, assignment)
    if len(assignment) > best["count"] or cost < best["objective"]:
        best["count"] = len(assignment)
        best["objective"] = cost
        best["assignment"] = dict(assignment)


class _BTBase(Solver):
    """Shared recursive engine for all backtracking configurations."""

    use_forward_checking: bool = False

    def solve(self, problem: Problem) -> SolverResult:
        stats = self._new_stats(problem)
        counter = ConsistencyCounter()
        best: dict = {"count": -1, "objective": float("inf"), "assignment": {}}
        live_domains = [list(d) for d in problem.domains]

        t0 = perf_counter()
        with Timer(stats):
            self._recurse(problem, {}, set(), live_domains, stats, counter, best, t0)

        assignment = dict(best["assignment"])
        stats.constraint_checks = counter.checks
        stats.num_assigned = len(assignment)
        stats.num_unassigned = problem.n - stats.num_assigned
        stats.failure_rate = stats.num_unassigned / max(1, problem.n)
        stats.objective = objective(problem, assignment)
        stats.success = stats.num_unassigned == 0
        return SolverResult(stats=stats, assignment=assignment)

    # -- hooks subclasses override ------------------------------------------

    def _select_var(
        self,
        problem: Problem,
        unassigned: list[int],
        live_domains: list[list[Assignment]],
    ) -> int:
        return unassigned[0]

    def _order_values(
        self,
        problem: Problem,
        i: int,
        assignment: dict[int, Assignment],
        live_domains: list[list[Assignment]],
    ) -> list[Assignment]:
        return live_domains[i]

    # -- recursion ----------------------------------------------------------

    def _recurse(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        skipped: set[int],
        live_domains: list[list[Assignment]],
        stats: SolverStats,
        counter: ConsistencyCounter,
        best: dict,
        t0: float,
    ):
        _record_best(problem, assignment, best)

        if len(assignment) == problem.n:
            return True

        if self._budget_exceeded(t0):
            return True

        unassigned = [
            i for i in range(problem.n)
            if i not in assignment and i not in skipped
        ]
        if not unassigned:
            return False

        var = self._select_var(problem, unassigned, live_domains)
        values = self._order_values(problem, var, assignment, live_domains)

        if not values:
            # COP graceful skip
            skipped.add(var)
            ret = self._recurse(problem, assignment, skipped, live_domains, stats, counter, best, t0)
            skipped.discard(var)
            return ret

        for val in values:
            stats.nodes_expanded += 1

            if not is_consistent(problem, assignment, var, val, counter):
                stats.backtracks += 1
                continue

            assignment[var] = val

            if self.use_forward_checking:
                snapshot = [list(d) for d in live_domains]
                fc_ok = self._forward_check(problem, assignment, live_domains, var, val)
                if not fc_ok:
                    live_domains[:] = snapshot
                    del assignment[var]
                    stats.backtracks += 1
                    continue

                result = self._recurse(problem, assignment, skipped, live_domains, stats, counter, best, t0)
                live_domains[:] = snapshot
            else:
                result = self._recurse(problem, assignment, skipped, live_domains, stats, counter, best, t0)

            if result is True:
                return True

            del assignment[var]
            stats.backtracks += 1

            if self._budget_exceeded(t0):
                return True

        # COP skip: allow skipping this var (forward-checking solvers only)
        if self.use_forward_checking:
            skipped.add(var)
            ret = self._recurse(problem, assignment, skipped, live_domains, stats, counter, best, t0)
            skipped.discard(var)
            return ret
        return False

    # -- forward checking ---------------------------------------------------

    def _forward_check(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        live_domains: list[list[Assignment]],
        var: int,
        val: Assignment,
    ) -> bool:
        """Prune live domains of unassigned variables after assigning var=val.

        Computes committed supply draw at (station, fuel_type) once, then
        uses it for every candidate value — O(1) per value instead of
        O(|assignment|).

        Returns False if a high-priority variable's domain is emptied.
        """
        var_v = problem.vehicles[var]
        supply_drawn: dict[tuple[int, str], float] = {}
        for k, ka in assignment.items():
            key = (ka.station_id, problem.vehicles[k].fuel_type)
            supply_drawn[key] = supply_drawn.get(key, 0.0) + problem.vehicles[k].demand_liters

        for j in range(problem.n):
            if j == var or j in assignment:
                continue
            pruned: list[Assignment] = []
            for other in live_domains[j]:
                # Pump clash with the newly assigned value
                if (
                    other.station_id == val.station_id
                    and other.pump_id == val.pump_id
                    and other.slot_id == val.slot_id
                ):
                    continue
                # Supply check using precomputed draws
                if other.station_id == val.station_id:
                    o_v = problem.vehicles[j]
                    if o_v.fuel_type == var_v.fuel_type:
                        key = (val.station_id, var_v.fuel_type)
                        drawn = supply_drawn.get(key, 0.0) + o_v.demand_liters
                        if drawn > problem.stations[val.station_id].stocks(var_v.fuel_type) + 1e-6:
                            continue
                pruned.append(other)

            was_nonempty = len(live_domains[j]) > 0
            live_domains[j] = pruned

            if not pruned and was_nonempty and problem.vehicles[j].priority >= var_v.priority:
                return False

        return True


# ---------------------------------------------------------------------------
# The four concrete algorithm classes
# ---------------------------------------------------------------------------


class BasicBacktracking(_BTBase):
    """Pure backtracking — input variable order, input value order, no heuristics."""
    name = "basic_backtracking"


class BacktrackingMRV(_BTBase):
    """BT + MRV variable ordering with true degree tie-break."""
    name = "bt_mrv"

    def _select_var(self, problem, unassigned, live_domains):
        return mrv(problem, unassigned, live_domains)


class BacktrackingLCV(_BTBase):
    """BT + LCV value ordering; highest-priority vehicle picked first."""
    name = "bt_lcv"

    def _select_var(self, problem, unassigned, live_domains):
        return priority_first(problem, unassigned)

    def _order_values(self, problem, i, assignment, live_domains):
        return lcv_sort(problem, i, assignment, live_domains[i], live_domains)


class BacktrackingFC(_BTBase):
    """Forward Checking + MRV (degree tie-break) + LCV + cost tie-break."""
    name = "bt_fc_mrv_deg"
    use_forward_checking = True

    def _select_var(self, problem, unassigned, live_domains):
        return mrv(problem, unassigned, live_domains)

    def _order_values(self, problem, i, assignment, live_domains):
        lcv = lcv_sort(problem, i, assignment, live_domains[i], live_domains)
        return cost_sort(problem, i, lcv)
