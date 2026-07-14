"""Shared solver infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from fuel_csp.problem import Assignment, Problem


@dataclass
class SolverStats:
    algorithm: str = ""
    n: int = 0
    seed: int = 0
    nodes_expanded: int = 0
    backtracks: int = 0
    constraint_checks: int = 0
    repair_steps: int = 0
    runtime_seconds: float = 0.0
    objective: float = 0.0
    num_assigned: int = 0
    num_unassigned: int = 0
    failure_rate: float = 0.0
    success: bool = False
    hit_step_budget: bool = False
    cost_trace: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "cost_trace"}


@dataclass
class SolverResult:
    stats: SolverStats
    assignment: dict[int, Assignment] = field(default_factory=dict)

    def pretty(self) -> str:
        s = self.stats
        return (
            f"{s.algorithm:<26s} | N={s.n:>3d} "
            f"| J={s.objective:>9.2f} "
            f"| assigned={s.num_assigned:>3d}/{s.n:<3d} "
            f"| bt={s.backtracks:>6d} "
            f"| bj={s.backjumps:>5d} "
            f"| nodes={s.nodes_expanded:>6d} "
            f"| t={s.runtime_seconds*1000:>7.1f}ms"
        )


class Timer:
    def __init__(self, stats: SolverStats) -> None:
        self._stats = stats
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stats.runtime_seconds = perf_counter() - self._start


class Solver:
    name: str = "abstract"

    def __init__(self, time_budget_s: float = 10.0) -> None:
        self.time_budget_s = time_budget_s

    def solve(self, problem: Problem) -> SolverResult:
        raise NotImplementedError

    def _new_stats(self, problem: Problem, seed: int = 0) -> SolverStats:
        return SolverStats(algorithm=self.name, n=problem.n, seed=seed)

    def _budget_exceeded(self, t0: float) -> bool:
        return (perf_counter() - t0) >= self.time_budget_s
