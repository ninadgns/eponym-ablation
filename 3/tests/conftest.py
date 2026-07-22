"""Shared fixtures."""

import numpy as np
import pytest

from shuttle_timetable.instance import ShuttleConfig, sample_arrivals


@pytest.fixture(scope="session")
def shuttle_cfg() -> ShuttleConfig:
    return ShuttleConfig()


@pytest.fixture(scope="session")
def arrivals(shuttle_cfg) -> np.ndarray:
    return sample_arrivals(shuttle_cfg, np.random.default_rng(0))


@pytest.fixture(scope="session")
def arrival_sets(shuttle_cfg) -> list[np.ndarray]:
    """The M = 3 FIXED training realisations (seeds 0, 1, 2)."""
    return [sample_arrivals(shuttle_cfg, np.random.default_rng(s)) for s in (0, 1, 2)]
