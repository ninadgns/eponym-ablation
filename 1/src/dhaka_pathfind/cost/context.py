"""Traveller context and cost-weight presets."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"
    NONBINARY = "nonbinary"


class Social(StrEnum):
    ALONE = "alone"
    ACCOMPANIED = "accompanied"


class AgeGroup(StrEnum):
    CHILD = "child"
    ADULT = "adult"
    ELDERLY = "elderly"


class Vehicle(StrEnum):
    WALK = "walk"
    RICKSHAW = "rickshaw"
    CNG = "cng"
    MOTORBIKE = "motorbike"
    CAR = "car"
    BUS = "bus"


class TimeOfDay(StrEnum):
    EARLY = "early"  # ~5–8
    MIDDAY = "midday"
    RUSH_MORNING = "rush_morning"
    RUSH_EVENING = "rush_evening"
    LATE_NIGHT = "late_night"


class Weather(StrEnum):
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    STORM = "storm"
    HEAT = "heat"


class CostPreset(StrEnum):
    BALANCED = "balanced"
    SPEED = "speed"
    SAFETY = "safety"
    COMFORT = "comfort"


_PRESET_WEIGHTS: dict[CostPreset, dict[str, float]] = {
    CostPreset.BALANCED: {
        "intrinsic": 1.0,
        "dynamic": 1.0,
        "traveller": 1.0,
        "risk": 1.0,
    },
    CostPreset.SPEED: {
        "intrinsic": 0.85,
        "dynamic": 1.35,
        "traveller": 0.75,
        "risk": 0.7,
    },
    CostPreset.SAFETY: {
        "intrinsic": 1.05,
        "dynamic": 1.0,
        "traveller": 1.4,
        "risk": 1.65,
    },
    CostPreset.COMFORT: {
        "intrinsic": 1.1,
        "dynamic": 1.15,
        "traveller": 1.15,
        "risk": 1.2,
    },
}


class TravellerContext(BaseModel):
    """Who / when / weather — used with edge attributes to compute ``C_edge``."""

    gender: Gender = Gender.MALE
    social: Social = Social.ALONE
    age: AgeGroup = AgeGroup.ADULT
    vehicle: Vehicle = Vehicle.CAR
    time_of_day: TimeOfDay = TimeOfDay.MIDDAY
    weather: Weather = Weather.CLEAR

    model_config = {"frozen": True}

    def stable_hash(self) -> str:
        """Short deterministic id for batch CSV rows."""
        s = self.model_dump_json(sort_keys=True)
        return hex(abs(hash(s)) % (16**12))[2:]


def preset_weights(preset: CostPreset) -> dict[str, float]:
    return dict(_PRESET_WEIGHTS[preset])


def context_key(ctx: TravellerContext) -> tuple[Any, ...]:
    return (
        ctx.gender,
        ctx.social,
        ctx.age,
        ctx.vehicle,
        ctx.time_of_day,
        ctx.weather,
    )
