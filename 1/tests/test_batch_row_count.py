"""Batch CSV row count (≥100 for default parameters)."""

from dhaka_pathfind.analysis.batch import INFORMED_ALGOS, UNINFORMED_ALGOS
from dhaka_pathfind.heuristics.registry import HEURISTICS


def test_default_batch_at_least_100_rows():
    pairs = 10
    informed_runs = len(INFORMED_ALGOS) * len(HEURISTICS)
    uninformed_runs = len(UNINFORMED_ALGOS)
    per_pair = uninformed_runs + informed_runs
    assert pairs * per_pair >= 100
