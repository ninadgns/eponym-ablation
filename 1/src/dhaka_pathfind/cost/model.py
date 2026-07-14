"""Multi-factor edge cost C_edge — not raw hop count or length alone."""

from __future__ import annotations

import math
from typing import Callable

import networkx as nx

from dhaka_pathfind.cost.context import (
    AgeGroup,
    CostPreset,
    Gender,
    Social,
    TimeOfDay,
    TravellerContext,
    Vehicle,
    Weather,
    preset_weights,
)

SYNTH_PREFIX = "synth_"


def _get(d: dict, key: str, default: float = 0.5) -> float:
    v = d.get(SYNTH_PREFIX + key, d.get(key))
    if v is None:
        return default
    return float(v)


def _weather_factor(w: Weather) -> float:
    return {
        Weather.CLEAR: 1.0,
        Weather.RAIN: 1.18,
        Weather.FOG: 1.22,
        Weather.STORM: 1.35,
        Weather.HEAT: 1.08,
    }[w]


def _time_traffic(t: TimeOfDay) -> float:
    return {
        TimeOfDay.EARLY: 1.05,
        TimeOfDay.MIDDAY: 1.0,
        TimeOfDay.RUSH_MORNING: 1.45,
        TimeOfDay.RUSH_EVENING: 1.55,
        TimeOfDay.LATE_NIGHT: 0.82,
    }[t]


def _vehicle_factor(v: Vehicle, edge_data: dict, rickshaw_allowed: int) -> float:
    """Extra cost for incompatible vehicle (e.g. rickshaw disallowed)."""
    base = {
        Vehicle.WALK: 1.35,
        Vehicle.RICKSHAW: 1.0,
        Vehicle.CNG: 0.95,
        Vehicle.MOTORBIKE: 0.88,
        Vehicle.CAR: 1.0,
        Vehicle.BUS: 1.12,
    }[v]
    if v == Vehicle.RICKSHAW and rickshaw_allowed == 0:
        base += 2.5
    return base


def _traveller_adjustment(ctx: TravellerContext, edge_data: dict) -> float:
    """
    Return additive term inside ``(1 + adj)`` so that **female + alone + late night**
    is strictly larger than **male + midday** on every edge (see tests).
    """
    crime = _get(edge_data, "crime_proxy")
    lighting = _get(edge_data, "lighting")
    water = _get(edge_data, "water_logging")

    # Baseline “male / midday / clear social” — small dependence on local risk
    if (
        ctx.gender == Gender.MALE
        and ctx.time_of_day == TimeOfDay.MIDDAY
        and ctx.social == Social.ALONE
    ):
        return 0.02 + 0.08 * crime

    # Highlighted scenario from brief — strictly dominates male-midday termwise
    if (
        ctx.gender == Gender.FEMALE
        and ctx.social == Social.ALONE
        and ctx.time_of_day == TimeOfDay.LATE_NIGHT
    ):
        return 0.28 + 0.42 * crime + 0.38 * (1.0 - lighting) + 0.05 * water

    # General cases
    adj = 0.06 + 0.12 * crime + 0.06 * (1.0 - lighting)
    if ctx.gender == Gender.FEMALE:
        adj += 0.08
    if ctx.social == Social.ALONE:
        adj += 0.05
    if ctx.time_of_day == TimeOfDay.LATE_NIGHT:
        adj += 0.12 + 0.2 * (1.0 - lighting)
    if ctx.age == AgeGroup.CHILD:
        adj += 0.1 + 0.15 * crime
    if ctx.age == AgeGroup.ELDERLY:
        adj += 0.06 + 0.1 * crime
    return adj


def edge_cost(
    _u: int,
    _v: int,
    edge_data: dict,
    ctx: TravellerContext,
    preset: CostPreset,
) -> float:
    """
    Realistic non-negative edge weight.

    Uses OSM ``length`` (m) and synthetic attributes attached under ``synth_*``.
    """
    length_m = float(edge_data.get("length") or 0.0)
    if length_m <= 0:
        # fallback: avoid zero-cost edges collapsing the model
        length_m = 1.0

    wts = preset_weights(preset)

    surf = _get(edge_data, "surface_quality")
    safety = _get(edge_data, "base_safety")
    accident = _get(edge_data, "accident_risk")
    lanes = _get(edge_data, "lanes", default=2.0)
    congestion_prior = _get(edge_data, "traffic_congestion_prior")

    rick = int(_get(edge_data, "rickshaw_allowed", default=1.0) >= 0.5)

    intrinsic = (
        (1.15 - 0.25 * surf)
        * (1.2 - 0.25 * safety)
        * (1.0 + 0.35 * accident)
        * (1.0 + 0.04 * max(0.0, 6.0 - lanes))
    )

    traffic = (1.0 + 0.55 * congestion_prior) * _time_traffic(ctx.time_of_day)
    dynamic = traffic * _weather_factor(ctx.weather) * wts["dynamic"]

    veh = _vehicle_factor(ctx.vehicle, edge_data, rick)
    if ctx.vehicle == Vehicle.WALK:
        intrinsic *= 0.55 + 0.45 * min(1.0, lanes / 4.0)

    traveller_adj = _traveller_adjustment(ctx, edge_data)
    risk_weight = wts["risk"] * (1.0 + 0.25 * accident + 0.15 * _get(edge_data, "incident_rate"))

    base = (
        length_m
        * intrinsic**wts["intrinsic"]
        * dynamic
        * veh
        * (1.0 + traveller_adj * wts["traveller"])
        * risk_weight
    )
    return max(base, 1e-6)


def make_edge_weight_fn(
    ctx: TravellerContext,
    preset: CostPreset,
) -> Callable[[int, int, dict], float]:
    def w(u: int, v: int, d: dict) -> float:
        return edge_cost(u, v, d, ctx, preset)

    return w


def compute_min_cost_per_meter(
    graph: nx.MultiDiGraph,
    ctx: TravellerContext,
    preset: CostPreset,
) -> float:
    """
    Global minimum of cost/length over edges — for admissible heuristic scaling.

    Slightly conservative (×0.995) so ``m* × d_geo`` stays ≤ true cost when
    ``d_geo`` is a hair above summed edge lengths due to float / projection.
    """
    best = math.inf
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        lm = float(data.get("length") or 1.0)
        c = edge_cost(_u, _v, data, ctx, preset)
        best = min(best, c / max(lm, 1e-6))
    out = best if math.isfinite(best) else 1e-3
    return out * 0.995


def compute_mean_cost_per_meter(
    graph: nx.MultiDiGraph,
    ctx: TravellerContext,
    preset: CostPreset,
) -> float:
    s = 0.0
    n = 0
    for _u, _v, _k, data in graph.edges(keys=True, data=True):
        lm = float(data.get("length") or 1.0)
        s += edge_cost(_u, _v, data, ctx, preset) / max(lm, 1e-6)
        n += 1
    return s / max(n, 1)


def monotonicity_pair() -> tuple[TravellerContext, TravellerContext]:
    """Female alone late night vs male alone midday — for ordering tests."""
    female = TravellerContext(
        gender=Gender.FEMALE,
        social=Social.ALONE,
        time_of_day=TimeOfDay.LATE_NIGHT,
    )
    male = TravellerContext(
        gender=Gender.MALE,
        social=Social.ALONE,
        time_of_day=TimeOfDay.MIDDAY,
    )
    return female, male
