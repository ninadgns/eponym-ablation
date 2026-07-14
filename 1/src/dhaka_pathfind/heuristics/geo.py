"""Great-circle distance between graph nodes (WGS84: x=lon, y=lat)."""

from __future__ import annotations

import math

import networkx as nx

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def node_lat_lon(graph: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    d = graph.nodes[node]
    return float(d["y"]), float(d["x"])  # lat, lon


def heuristic_distance_m(graph: nx.MultiDiGraph, a: int, b: int) -> float:
    la, loa = node_lat_lon(graph, a)
    lb, lob = node_lat_lon(graph, b)
    return haversine_m(la, loa, lb, lob)
