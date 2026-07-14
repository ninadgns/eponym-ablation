"""Helpers for effective branching factor and heuristic gap."""

from __future__ import annotations

import math


def effective_branching_factor(nodes_expanded: int, depth: int) -> float:
    """Rough Michie-style estimate when exact inversion is expensive."""
    if depth <= 0 or nodes_expanded <= 0:
        return 0.0
    return float(nodes_expanded ** (1.0 / depth))


def mean_heuristic_gap_on_path(
    h_values: list[float],
    actual_suffix_costs: list[float],
) -> float:
    """Mean |h(s) - actual_remaining| along the final path (same length lists)."""
    if not h_values or len(h_values) != len(actual_suffix_costs):
        return 0.0
    return float(sum(abs(a - b) for a, b in zip(h_values, actual_suffix_costs)) / len(h_values))
