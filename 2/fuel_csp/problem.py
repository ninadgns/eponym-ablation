"""
CSP/COP problem definition for the urban fuel-crisis allocator.

Variables  X  : one per vehicle (x_0 ... x_{N-1})
Domain    D_i : feasible (station_id, pump_id, slot_id) triples after
                per-variable hard-constraint pre-filtering.
Constraints C : pump-exclusivity and supply-capacity checked during search.

Soft objective J(S) (the COP layer):
  J = w_dist * total_distance
    + w_wait * total_wait_time
    + w_prio * priority_penalty   (ambulances penalised heavily for late slots)
    + w_unassigned * #unassigned  (partial solutions are valid COP outputs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

FUEL_TYPES: tuple[str, ...] = ("petrol", "diesel", "octane")

PRIORITY: dict[str, int] = {
    "ambulance": 5,
    "bus": 3,
    "truck": 2,
    "car": 1,
    "motorbike": 1,
}


@dataclass(frozen=True)
class Vehicle:
    vid: int
    kind: str
    fuel_type: str
    x: float
    y: float
    range_km: float
    demand_liters: float
    earliest_slot: int
    latest_slot: int

    @property
    def priority(self) -> int:
        return PRIORITY[self.kind]


@dataclass(frozen=True)
class Station:
    sid: int
    x: float
    y: float
    pumps: int
    open_slot: int
    close_slot: int
    reserves: dict[str, float]
    name: str = ""

    def stocks(self, fuel: str) -> float:
        return self.reserves.get(fuel, 0.0)


@dataclass(frozen=True)
class Assignment:
    station_id: int
    pump_id: int
    slot_id: int

    def __str__(self) -> str:
        return f"S{self.station_id}/P{self.pump_id}@T{self.slot_id}"


@dataclass
class Problem:
    """Container for one CSP/COP instance."""

    vehicles: list[Vehicle]
    stations: list[Station]
    num_slots: int
    domains: list[list[Assignment]] = field(default_factory=list)
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 1.0,
            "wait": 0.5,
            "priority": 10.0,
            "unassigned": 100.0,
        }
    )
    # Precomputed constraint graph: neighbours[i] = set of j that share at
    # least one feasible station with i. Built lazily by build_constraint_graph().
    neighbours: list[set[int]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.vehicles)

    def distance_km(self, vehicle_id: int, station_id: int) -> float:
        v = self.vehicles[vehicle_id]
        s = self.stations[station_id]
        return math.hypot(v.x - s.x, v.y - s.y)

    def build_domains(self) -> None:
        self.domains = [self._domain_for(i) for i in range(self.n)]

    def _domain_for(self, i: int) -> list[Assignment]:
        v = self.vehicles[i]
        out: list[Assignment] = []
        for s in self.stations:
            if s.stocks(v.fuel_type) < v.demand_liters:
                continue
            if self.distance_km(i, s.sid) > v.range_km:
                continue
            slot_lo = max(v.earliest_slot, s.open_slot)
            slot_hi = min(v.latest_slot, s.close_slot - 1)
            if slot_lo > slot_hi:
                continue
            for slot in range(slot_lo, slot_hi + 1):
                for pump in range(s.pumps):
                    out.append(Assignment(s.sid, pump, slot))
        return out

    def build_constraint_graph(self) -> None:
        """Compute the neighbour sets — used by the degree heuristic.

        i and j are neighbours if they could ever clash: either they share a
        feasible station (supply clash or pump clash) or their domains overlap
        on any (station, pump, slot) triple.
        """
        n = self.n
        self.neighbours = [set() for _ in range(n)]
        # Index domain values for fast lookup
        val_sets: list[set[tuple[int, int, int]]] = [
            {(a.station_id, a.pump_id, a.slot_id) for a in self.domains[i]}
            for i in range(n)
        ]
        station_sets: list[set[int]] = [
            {a.station_id for a in self.domains[i]} for i in range(n)
        ]
        for i in range(n):
            for j in range(i + 1, n):
                # pump-exclusivity constraint: any shared (s,p,t) triple
                if val_sets[i] & val_sets[j]:
                    self.neighbours[i].add(j)
                    self.neighbours[j].add(i)
                    continue
                # supply constraint: same station, same fuel type
                if (
                    station_sets[i] & station_sets[j]
                    and self.vehicles[i].fuel_type == self.vehicles[j].fuel_type
                ):
                    self.neighbours[i].add(j)
                    self.neighbours[j].add(i)


def iter_pairs(seq: Iterable[int]) -> Iterable[tuple[int, int]]:
    items = list(seq)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]
