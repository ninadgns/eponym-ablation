"""Download or load cached OSM road graph for Dhaka."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
import requests
import yaml

from dhaka_pathfind.config import (
    DATA_DIR,
    DEFAULT_NETWORK_TYPE,
    DHAKA_BBOX,
    GRAPHML_FILENAME,
    ensure_data_dir,
    landmarks_path,
)

_METADATA_SUFFIX = ".meta.json"


def _metadata_path(graphml: Path) -> Path:
    return graphml.with_name(graphml.name + _METADATA_SUFFIX)


def load_landmarks() -> dict[str, dict[str, float]]:
    """Load name -> {lat, lon} from YAML."""
    p = landmarks_path()
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {str(k): {"lat": float(v["lat"]), "lon": float(v["lon"])} for k, v in raw.items()}


def nearest_node(
    graph: nx.MultiDiGraph,
    lat: float,
    lon: float,
) -> int:
    """Snap (lat, lon) to nearest graph node. OSMnx uses x=lon, y=lat."""
    return ox.distance.nearest_nodes(graph, X=lon, Y=lat)


def download_graph(
    bbox: tuple[float, float, float, float] | None = None,
    network_type: str = DEFAULT_NETWORK_TYPE,
) -> nx.MultiDiGraph:
    """Fetch a fresh MultiDiGraph from OSM for the bbox."""
    south, west, north, east = bbox or DHAKA_BBOX
    ox.settings.use_cache = True
    # OSMnx 2.x: ``bbox`` is ``(left, bottom, right, top)`` = (west, south, east, north).
    bbox_osm = (west, south, east, north)
    last: Exception | None = None
    for attempt in range(4):
        try:
            g = ox.graph_from_bbox(bbox_osm, network_type=network_type)
            return g  # type: ignore[no-any-return]
        except (requests.RequestException, OSError, TimeoutError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    assert last is not None
    raise last


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    ensure_data_dir()
    ox.save_graphml(graph, path)
    meta = {
        "bbox": DHAKA_BBOX,
        "network_type": DEFAULT_NETWORK_TYPE,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
    }
    with open(_metadata_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_graph_from_file(path: Path) -> nx.MultiDiGraph:
    return ox.load_graphml(path)  # type: ignore[no-any-return]


def load_or_download(
    graphml: Path | None = None,
    force_download: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    network_type: str = DEFAULT_NETWORK_TYPE,
) -> nx.MultiDiGraph:
    """
    Return Dhaka road graph: from cache if present unless ``force_download``.

    Persists to ``data/dhaka_graph.graphml`` on first download.
    """
    path = graphml or (DATA_DIR / GRAPHML_FILENAME)
    if not force_download and path.exists():
        return load_graph_from_file(path)

    g = download_graph(bbox=bbox, network_type=network_type)
    save_graph(g, path)
    return g


def resolve_named_or_coords(
    graph: nx.MultiDiGraph,
    name_or_lat: str,
    lon_or_none: float | None,
) -> int:
    """
    If lon is None, treat first arg as landmark name; else lat/lon pair.
    """
    if lon_or_none is not None:
        return nearest_node(graph, float(name_or_lat), float(lon_or_none))
    landmarks = load_landmarks()
    if name_or_lat not in landmarks:
        raise KeyError(f"Unknown landmark {name_or_lat!r}; known: {sorted(landmarks)}")
    p = landmarks[name_or_lat]
    return nearest_node(graph, p["lat"], p["lon"])
