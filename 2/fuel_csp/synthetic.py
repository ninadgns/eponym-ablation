"""Deterministic synthetic problem generator.

Stations are deliberately scarce so larger instances force real backtracking
and the COP graceful-failure path. Each instance is reproducible given
(num_vehicles, seed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from fuel_csp.problem import FUEL_TYPES, Problem, Station, Vehicle


@dataclass
class GeneratorConfig:
    num_vehicles: int
    num_stations: int = 6
    num_slots: int = 6
    grid_size_km: float = 20.0
    pumps_min: int = 1
    pumps_max: int = 3
    reserve_min: float = 80.0
    reserve_max: float = 220.0
    demand_min: float = 15.0
    demand_max: float = 45.0
    range_min: float = 8.0
    range_max: float = 22.0
    ambulance_rate: float = 0.10
    bus_rate: float = 0.15
    truck_rate: float = 0.15
    seed: int = 42


def _pick_kind(rng: random.Random, cfg: GeneratorConfig) -> str:
    r = rng.random()
    if r < cfg.ambulance_rate:
        return "ambulance"
    if r < cfg.ambulance_rate + cfg.bus_rate:
        return "bus"
    if r < cfg.ambulance_rate + cfg.bus_rate + cfg.truck_rate:
        return "truck"
    return "car" if rng.random() < 0.5 else "motorbike"


def _fuel_for(kind: str, rng: random.Random) -> str:
    if kind in ("ambulance", "bus"):
        return "diesel"
    if kind == "truck":
        return "diesel" if rng.random() < 0.7 else "octane"
    if kind == "car":
        return rng.choice(("petrol", "octane"))
    return "petrol"  # motorbike


def _demand_for(kind: str, rng: random.Random, cfg: GeneratorConfig) -> float:
    base = rng.uniform(cfg.demand_min, cfg.demand_max)
    scale = {"ambulance": 0.6, "bus": 1.5, "truck": 1.8, "car": 0.7, "motorbike": 0.25}[kind]
    return round(base * scale, 1)


def _range_for(kind: str, rng: random.Random, cfg: GeneratorConfig) -> float:
    base = rng.uniform(cfg.range_min, cfg.range_max)
    if kind == "ambulance":
        base *= 0.7
    if kind == "motorbike":
        base *= 0.9
    return round(base, 1)


def _slot_window(kind: str, rng: random.Random, num_slots: int) -> tuple[int, int]:
    if kind == "ambulance":
        return 0, max(1, num_slots // 3)
    lo = rng.randrange(0, max(1, num_slots // 2))
    hi = rng.randrange(lo, num_slots)
    return lo, hi


def generate_problem(cfg: GeneratorConfig) -> Problem:
    rng = random.Random(cfg.seed)

    stations: list[Station] = []
    for sid in range(cfg.num_stations):
        x = rng.uniform(0, cfg.grid_size_km)
        y = rng.uniform(0, cfg.grid_size_km)
        pumps = rng.randint(cfg.pumps_min, cfg.pumps_max)
        reserves = {
            ft: round(rng.uniform(cfg.reserve_min, cfg.reserve_max), 1)
            for ft in FUEL_TYPES
        }
        if rng.random() < 0.25:
            reserves[rng.choice(FUEL_TYPES)] = 0.0
        stations.append(
            Station(
                sid=sid, x=x, y=y, pumps=pumps,
                open_slot=0, close_slot=cfg.num_slots,
                reserves=reserves,
            )
        )

    vehicles: list[Vehicle] = []
    for vid in range(cfg.num_vehicles):
        kind = _pick_kind(rng, cfg)
        vehicles.append(
            Vehicle(
                vid=vid, kind=kind,
                fuel_type=_fuel_for(kind, rng),
                x=rng.uniform(0, cfg.grid_size_km),
                y=rng.uniform(0, cfg.grid_size_km),
                range_km=_range_for(kind, rng, cfg),
                demand_liters=_demand_for(kind, rng, cfg),
                earliest_slot=_slot_window(kind, rng, cfg.num_slots)[0],
                latest_slot=_slot_window(kind, rng, cfg.num_slots)[1],
            )
        )

    problem = Problem(vehicles=vehicles, stations=stations, num_slots=cfg.num_slots)
    problem.build_domains()
    problem.build_constraint_graph()
    return problem
