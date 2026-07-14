"""
Min-Conflicts local search with tabu list.

Improvement over a vanilla Min-Conflicts:
  * Short-term tabu list keeps a sliding window of (variable, value) pairs
    that were recently assigned. This prevents the solver from cycling back
    to the same conflicted state within the tabu tenure, which is the main
    failure mode of plain Min-Conflicts on dense instances.
  * Random restarts every `random_restart_every` steps if conflicts remain,
    so the solver escapes deep local minima.
  * Returns the best feasible (conflict-free) partial assignment seen, not
    just the final one, so the COP graceful-failure guarantee holds even
    when the step budget expires.
"""

from __future__ import annotations

import random
from collections import deque
from time import perf_counter

from fuel_csp.algorithms.base import Solver, SolverResult, Timer
from fuel_csp.constraints import conflicts, objective, total_conflicts
from fuel_csp.problem import Assignment, Problem


class MinConflictsSolver(Solver):
    name = "min_conflicts"

    def __init__(
        self,
        max_steps: int = 4000,
        random_restart_every: int = 800,
        tabu_tenure: int = 20,
        seed: int = 42,
        time_budget_s: float = 10.0,
    ) -> None:
        super().__init__(time_budget_s=time_budget_s)
        self.max_steps = max_steps
        self.random_restart_every = random_restart_every
        self.tabu_tenure = tabu_tenure
        self.seed = seed

    def solve(self, problem: Problem) -> SolverResult:
        stats = self._new_stats(problem, seed=self.seed)
        rng = random.Random(self.seed)

        # Variables with non-empty domains are the only ones we can assign.
        assignable = [i for i in range(problem.n) if problem.domains[i]]

        # Tabu list: deque of (variable_id, Assignment) with fixed tenure.
        tabu: deque[tuple[int, Assignment]] = deque(maxlen=self.tabu_tenure)

        t0 = perf_counter()
        with Timer(stats):
            assignment = self._random_assignment(problem, assignable, rng)
            best = dict(assignment)
            best_conf = total_conflicts(problem, best)
            best_cost = objective(problem, best)
            stats.cost_trace.append(best_cost)

            for step in range(self.max_steps):
                if self._budget_exceeded(t0):
                    break
                stats.repair_steps = step + 1

                conflicted = self._conflicted_vars(problem, assignment)
                if not conflicted:
                    break  # solution is conflict-free

                vid = rng.choice(conflicted)
                stats.nodes_expanded += 1

                tabu_set = set(tabu)
                new_val = self._least_conflicting_value(
                    problem, assignment, vid, tabu_set,
                )
                if new_val is None:
                    if vid in assignment:
                        del assignment[vid]
                else:
                    tabu.append((vid, assignment.get(vid)))  # remember old value
                    assignment[vid] = new_val

                cur_conf = total_conflicts(problem, assignment)
                cur_cost = objective(problem, assignment)
                stats.cost_trace.append(cur_cost)

                if (cur_conf, cur_cost) < (best_conf, best_cost):
                    best = dict(assignment)
                    best_conf = cur_conf
                    best_cost = cur_cost

                if step > 0 and step % self.random_restart_every == 0 and best_conf > 0:
                    assignment = self._random_assignment(problem, assignable, rng)
                    tabu.clear()

        final = self._extract_feasible(problem, best)
        stats.num_assigned = len(final)
        stats.num_unassigned = problem.n - stats.num_assigned
        stats.failure_rate = stats.num_unassigned / max(1, problem.n)
        stats.objective = objective(problem, final)
        stats.constraint_checks = stats.repair_steps
        stats.success = (total_conflicts(problem, final) == 0 and stats.num_unassigned == 0)
        stats.hit_step_budget = stats.repair_steps >= self.max_steps and not stats.success
        return SolverResult(stats=stats, assignment=final)

    # -- helpers ------------------------------------------------------------

    def _random_assignment(
        self,
        problem: Problem,
        assignable: list[int],
        rng: random.Random,
    ) -> dict[int, Assignment]:
        return {i: rng.choice(problem.domains[i]) for i in assignable}

    def _conflicted_vars(
        self, problem: Problem, assignment: dict[int, Assignment]
    ) -> list[int]:
        return [
            i for i, val in assignment.items()
            if conflicts(problem, assignment, i, val) > 0
        ]

    def _least_conflicting_value(
        self,
        problem: Problem,
        assignment: dict[int, Assignment],
        vid: int,
        tabu_set: set[tuple[int, Assignment]],
    ) -> Assignment | None:
        dom = problem.domains[vid]
        if not dom:
            return None

        best_val: Assignment | None = None
        best_key: tuple[int, float] = (10**9, 10**9)

        for val in dom:
            # Skip tabu assignments unless they would be aspiration-eligible
            # (better than the best known conflict count overall).
            if (vid, val) in tabu_set:
                c = conflicts(problem, assignment, vid, val)
                if c > 0:
                    continue  # tabu and not improving — skip

            c = conflicts(problem, assignment, vid, val)
            # Tie-break: closer station + earlier slot gives lower COP cost
            d = problem.distance_km(vid, val.station_id) + 0.3 * val.slot_id
            key = (c, d)
            if key < best_key:
                best_key = key
                best_val = val

        return best_val

    def _extract_feasible(
        self, problem: Problem, assignment: dict[int, Assignment]
    ) -> dict[int, Assignment]:
        """Drop the minimum set of vehicles to make the assignment conflict-free.

        Priority order: ambulance > bus > truck > car ≈ motorbike.
        Higher-priority vehicles are kept when pump slots or supply collide.
        """
        # 1. Resolve pump clashes — keep highest-priority vehicle per slot.
        slot_owner: dict[tuple[int, int, int], int] = {}
        order = sorted(
            assignment.keys(),
            key=lambda i: (-problem.vehicles[i].priority, i),
        )
        out: dict[int, Assignment] = {}
        for i in order:
            a = assignment[i]
            key = (a.station_id, a.pump_id, a.slot_id)
            if key not in slot_owner:
                slot_owner[key] = i
                out[i] = a

        # 2. Resolve supply violations — drop low-priority vehicles first.
        from collections import defaultdict
        drawn: dict[tuple[int, str], float] = defaultdict(float)
        keep: dict[int, Assignment] = {}
        for i in sorted(out, key=lambda i: (-problem.vehicles[i].priority, i)):
            a = out[i]
            v = problem.vehicles[i]
            new_draw = drawn[(a.station_id, v.fuel_type)] + v.demand_liters
            cap = problem.stations[a.station_id].stocks(v.fuel_type)
            if new_draw <= cap + 1e-6:
                drawn[(a.station_id, v.fuel_type)] = new_draw
                keep[i] = a
        return keep
