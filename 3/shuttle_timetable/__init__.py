"""Part A — shuttle departure timetable under a finite fleet."""

from shuttle_timetable.instance import (
    ShuttleConfig,
    arrival_rate,
    expected_arrivals,
    round_trip_time,
    sample_arrivals,
)
from shuttle_timetable.simulator import SimResult, objective, simulate

__all__ = [
    "ShuttleConfig",
    "arrival_rate",
    "expected_arrivals",
    "round_trip_time",
    "sample_arrivals",
    "SimResult",
    "simulate",
    "objective",
]
