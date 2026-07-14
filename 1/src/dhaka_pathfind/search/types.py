"""Search result and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchMetrics:
    nodes_expanded: int = 0
    revisits: int = 0
    effective_branching_factor: float = 0.0
    max_depth: int = 0
    runtime_ms: float = 0.0
    heuristic_mean_abs_gap: float | None = None


@dataclass
class SearchResult:
    path: list[int] | None
    path_cost: float
    metrics: SearchMetrics = field(default_factory=SearchMetrics)
    algorithm: str = ""
