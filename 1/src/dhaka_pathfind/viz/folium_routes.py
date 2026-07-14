"""Folium maps for route polylines (WGS84)."""

from __future__ import annotations

from pathlib import Path

import folium
import networkx as nx

from dhaka_pathfind.config import OUTPUTS_DIR, ensure_outputs_dir
from dhaka_pathfind.graph.load import nearest_node


def _coords_along_path(graph: nx.MultiDiGraph, path: list[int]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = min(
            graph[u][v].values(),
            key=lambda d: float(d.get("length") or 0.0),
        )
        geom = data.get("geometry")
        if geom is not None:
            try:
                xs, ys = geom.xy
                for lon, lat in zip(xs, ys):
                    out.append((lat, lon))
                continue
            except Exception:
                pass
        la1, lo1 = float(graph.nodes[u]["y"]), float(graph.nodes[u]["x"])
        la2, lo2 = float(graph.nodes[v]["y"]), float(graph.nodes[v]["x"])
        out.extend([(la1, lo1), (la2, lo2)])
    return out


def make_route_map(
    graph: nx.MultiDiGraph,
    path: list[int] | None,
    title: str = "Route",
) -> folium.Map:
    ensure_outputs_dir()
    if path and len(path) >= 2:
        mid = path[len(path) // 2]
        lat0 = float(graph.nodes[mid]["y"])
        lon0 = float(graph.nodes[mid]["x"])
    else:
        lat0, lon0 = 23.81, 90.41
    m = folium.Map(location=(lat0, lon0), zoom_start=12, tiles="cartodbpositron")
    folium.LayerControl().add_to(m)
    if path and len(path) >= 2:
        coords = _coords_along_path(graph, path)
        folium.PolyLine(coords, color="blue", weight=4, opacity=0.85, popup=title).add_to(m)
        folium.Marker((coords[0][0], coords[0][1]), popup="start").add_to(m)
        folium.Marker((coords[-1][0], coords[-1][1]), popup="end").add_to(m)
    return m


# Distinct colors for the six algorithms (Streamlit “all six” mode).
ROUTE_COLORS: dict[str, str] = {
    "ucs": "#2563eb",
    "dijkstra": "#16a34a",
    "bidirectional_ucs": "#9333ea",
    "astar": "#dc2626",
    "weighted_astar": "#ea580c",
    "greedy_best_first": "#0891b2",
}

# Extra colors for compound layer names (e.g. ``astar:admissible``).
_ROUTE_COLOR_FALLBACK = [
    "#c026d3",
    "#0d9488",
    "#ca8a04",
    "#4f46e5",
    "#be123c",
    "#15803d",
    "#0369a1",
    "#b45309",
]


def route_color(name: str) -> str:
    if name in ROUTE_COLORS:
        return ROUTE_COLORS[name]
    base = name.split(":", 1)[0] if ":" in name else name
    if base in ROUTE_COLORS:
        i = abs(hash(name)) % len(_ROUTE_COLOR_FALLBACK)
        return _ROUTE_COLOR_FALLBACK[i]
    i = abs(hash(name)) % len(_ROUTE_COLOR_FALLBACK)
    return _ROUTE_COLOR_FALLBACK[i]


def make_endpoint_picker_map(
    origin: tuple[float, float],
    dest: tuple[float, float],
    landmarks: dict[str, dict[str, float]] | None = None,
) -> folium.Map:
    """
    Map for choosing origin / destination by click (used with ``st_folium``).

    Shows green (origin) and red (destination) markers, optional landmark
    reference circles, and ``LatLngPopup`` for click coordinates.
    """
    ensure_outputs_dir()
    lat_o, lon_o = origin
    lat_d, lon_d = dest
    lat_c = (lat_o + lat_d) / 2.0
    lon_c = (lon_o + lon_d) / 2.0
    m = folium.Map(location=(lat_c, lon_c), zoom_start=12, tiles="cartodbpositron")
    folium.LatLngPopup().add_to(m)
    folium.Marker(
        (lat_o, lon_o),
        popup="Origin",
        tooltip="Origin",
        icon=folium.Icon(color="green"),
    ).add_to(m)
    folium.Marker(
        (lat_d, lon_d),
        popup="Destination",
        tooltip="Destination",
        icon=folium.Icon(color="red"),
    ).add_to(m)
    if landmarks:
        ref = folium.FeatureGroup(name="Landmarks (reference)", show=True)
        for name, d in landmarks.items():
            folium.CircleMarker(
                location=(d["lat"], d["lon"]),
                radius=5,
                color="#64748b",
                weight=1,
                fill=True,
                fill_opacity=0.55,
                tooltip=name,
            ).add_to(ref)
        ref.add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
    return m


def make_multi_route_map(
    graph: nx.MultiDiGraph,
    named_paths: list[tuple[str, list[int] | None]],
) -> folium.Map:
    """
    One Folium map with one colored polyline per (name, path), plus layer control.

    ``named_paths`` order is preserved; paths that are ``None`` or too short are skipped.
    """
    ensure_outputs_dir()
    valid = [(n, p) for n, p in named_paths if p is not None and len(p) >= 2]
    if not valid:
        m = folium.Map(location=(23.81, 90.41), zoom_start=12, tiles="cartodbpositron")
        folium.LayerControl().add_to(m)
        return m

    mid_node = valid[0][1][len(valid[0][1]) // 2]
    lat0 = float(graph.nodes[mid_node]["y"])
    lon0 = float(graph.nodes[mid_node]["x"])
    m = folium.Map(location=(lat0, lon0), zoom_start=12, tiles="cartodbpositron")

    for name, path in valid:
        color = route_color(name)
        coords = _coords_along_path(graph, path)
        fg = folium.FeatureGroup(name=name, show=True)
        folium.PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.88,
            popup=f"{name}",
            tooltip=name,
        ).add_to(fg)
        fg.add_to(m)

    path0 = valid[0][1]
    c0 = _coords_along_path(graph, path0)
    if c0:
        folium.Marker((c0[0][0], c0[0][1]), popup="start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker((c0[-1][0], c0[-1][1]), popup="end", icon=folium.Icon(color="red")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def save_route_map(
    graph: nx.MultiDiGraph,
    path: list[int] | None,
    out_path: str | Path,
    title: str = "Route",
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    m = make_route_map(graph, path, title=title)
    m.save(str(p))
    return p


def quick_map_between_landmarks(
    graph: nx.MultiDiGraph,
    start_name: str,
    end_name: str,
    out_html: str | None = None,
) -> Path:
    """Convenience: resolve two YAML landmark names and save a map (no routing)."""
    from dhaka_pathfind.graph.load import load_landmarks

    lm = load_landmarks()
    s = nearest_node(graph, lm[start_name]["lat"], lm[start_name]["lon"])
    t = nearest_node(graph, lm[end_name]["lat"], lm[end_name]["lon"])
    path = [s, t]
    outp = Path(out_html or (OUTPUTS_DIR / "maps" / "landmarks_line.html"))
    return save_route_map(graph, path, outp, title=f"{start_name}–{end_name}")
