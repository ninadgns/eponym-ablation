"""Streamlit UI for Dhaka pathfinding (run: streamlit run streamlit_app.py)."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, TypedDict

import streamlit as st
from streamlit_folium import st_folium

from dhaka_pathfind.cost.context import (
    AgeGroup,
    CostPreset,
    Gender,
    Social,
    TimeOfDay,
    TravellerContext,
    Vehicle,
    Weather,
)
from dhaka_pathfind.graph.load import load_landmarks, load_or_download, nearest_node
from dhaka_pathfind.search.algorithms import (
    astar,
    bidirectional_ucs,
    dijkstra,
    greedy_best_first,
    ucs,
    weighted_astar,
)
from dhaka_pathfind.heuristics.registry import HEURISTICS
from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges
from dhaka_pathfind.viz.folium_routes import (
    make_endpoint_picker_map,
    make_multi_route_map,
    make_route_map,
)

# Allow running without install: repo root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


ALGOS = {
    "ucs": ucs,
    "dijkstra": dijkstra,
    "bidirectional_ucs": bidirectional_ucs,
    "astar": astar,
    "weighted_astar": weighted_astar,
    "greedy_best_first": greedy_best_first,
}

MODE_ONE = "One algorithm"
MODE_SIX = "All six algorithms"

INFORMED_ALGOS = frozenset({"astar", "weighted_astar", "greedy_best_first"})
HEURISTIC_NAMES: list[str] = list(HEURISTICS.keys())

HEURISTIC_SINGLE = "Single heuristic"
HEURISTIC_ALL_PARALLEL = "All heuristics (parallel compare)"

ENDPOINT_LANDMARKS = "Landmarks"
ENDPOINT_MAP = "Map (click to set)"
_FOLIUM_PICKER_PREV = "_folium_picker_prev_click_norm"


class MapSpecSingle(TypedDict):
    kind: Literal["single"]
    path: list[int] | None
    label: str


class MapSpecMulti(TypedDict):
    kind: Literal["multi"]
    named_paths: list[tuple[str, list[int] | None]]


MapSpec = MapSpecSingle | MapSpecMulti


@st.cache_resource
def _graph():
    g = load_or_download()
    ensure_synthetic_edges(g)
    return g


def _normalized_map_click(lc: object) -> tuple[float, float] | None:
    """Stable (lat, lon) from ``st_folium`` ``last_clicked``, or ``None``."""
    if not isinstance(lc, dict):
        return None
    lat, lng = lc.get("lat"), lc.get("lng")
    if lng is None:
        lng = lc.get("lon")
    if lat is None or lng is None:
        return None
    return (round(float(lat), 6), round(float(lng), 6))


def _run_heuristic_variants_parallel(
    fn,
    graph,
    s: int,
    t: int,
    ctx: TravellerContext,
    preset: CostPreset,
    heuristic_names: list[str],
):
    """Run one informed search function once per heuristic (thread pool, shared graph)."""

    def one(h: str):
        return h, fn(graph, s, t, ctx, preset, heuristic_name=h)

    n = max(1, len(heuristic_names))
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(one, heuristic_names))


def main() -> None:
    st.set_page_config(page_title="Dhaka Pathfind", layout="wide")
    st.title("Dhaka realistic pathfinding")
    graph = _graph()
    lm = load_landmarks()
    names = sorted(lm.keys())
    if not names:
        st.error("No landmarks in data/landmarks.yaml — cannot initialise endpoints.")
        st.stop()

    st.subheader("1. Origin and destination")
    endpoint_source = st.radio(
        "Set endpoints",
        [ENDPOINT_LANDMARKS, ENDPOINT_MAP],
        horizontal=True,
        key="ep_source",
        help="Map mode: pick **Origin** or **Destination**, click the map; routing still snaps to the nearest graph node.",
    )
    if st.session_state.get("_ep_source_track") != endpoint_source:
        st.session_state[_FOLIUM_PICKER_PREV] = None
    st.session_state["_ep_source_track"] = endpoint_source

    if endpoint_source == ENDPOINT_LANDMARKS:
        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("From", names, index=0, key="ep_from_lm")
        with c2:
            b = st.selectbox(
                "To",
                names,
                index=min(1, len(names) - 1),
                key="ep_to_lm",
            )
        st.session_state["map_o_lat"] = lm[a]["lat"]
        st.session_state["map_o_lon"] = lm[a]["lon"]
        st.session_state["map_d_lat"] = lm[b]["lat"]
        st.session_state["map_d_lon"] = lm[b]["lon"]
        s = nearest_node(graph, lm[a]["lat"], lm[a]["lon"])
        t = nearest_node(graph, lm[b]["lat"], lm[b]["lon"])
    else:
        init_o = lm[names[0]]
        init_d = lm[names[min(1, len(names) - 1)]]
        if "map_o_lat" not in st.session_state:
            st.session_state["map_o_lat"] = init_o["lat"]
            st.session_state["map_o_lon"] = init_o["lon"]
            st.session_state["map_d_lat"] = init_d["lat"]
            st.session_state["map_d_lon"] = init_d["lon"]

        st.radio(
            "Click sets",
            ["Origin", "Destination"],
            horizontal=True,
            key="map_pick_role",
            help="Choose which endpoint to move, then click empty map (or use the lat/lng popup).",
        )
        o_lat = float(st.session_state["map_o_lat"])
        o_lon = float(st.session_state["map_o_lon"])
        d_lat = float(st.session_state["map_d_lat"])
        d_lon = float(st.session_state["map_d_lon"])
        picker_map = make_endpoint_picker_map(
            (o_lat, o_lon),
            (d_lat, d_lon),
            landmarks=lm,
        )
        picker_data = st_folium(
            picker_map,
            height=420,
            use_container_width=True,
            key="endpoint_picker_map",
            returned_objects=["last_clicked"],
        )
        lc = picker_data.get("last_clicked") if isinstance(picker_data, dict) else None
        curr = _normalized_map_click(lc)
        prev = st.session_state.get(_FOLIUM_PICKER_PREV)
        if curr != prev:
            st.session_state[_FOLIUM_PICKER_PREV] = curr
            if curr is not None:
                lat, lon = curr
                role = st.session_state.get("map_pick_role", "Origin")
                if role == "Origin":
                    st.session_state["map_o_lat"] = lat
                    st.session_state["map_o_lon"] = lon
                else:
                    st.session_state["map_d_lat"] = lat
                    st.session_state["map_d_lon"] = lon
                st.rerun()

        o_lat = float(st.session_state["map_o_lat"])
        o_lon = float(st.session_state["map_o_lon"])
        d_lat = float(st.session_state["map_d_lat"])
        d_lon = float(st.session_state["map_d_lon"])
        st.caption(
            f"**Origin** `{o_lat:.5f}, {o_lon:.5f}` · **Destination** `{d_lat:.5f}, {d_lon:.5f}`"
        )
        s = nearest_node(graph, o_lat, o_lon)
        t = nearest_node(graph, d_lat, d_lon)

    st.divider()
    st.subheader("2. Cost and traveller parameters")
    row_ctx1 = st.columns(3)
    with row_ctx1[0]:
        _g = st.selectbox("Gender", list(Gender))
    with row_ctx1[1]:
        _soc = st.selectbox("Social", list(Social))
    with row_ctx1[2]:
        _age = st.selectbox("Age", list(AgeGroup))
    row_ctx2 = st.columns(3)
    with row_ctx2[0]:
        _veh = st.selectbox("Vehicle", list(Vehicle))
    with row_ctx2[1]:
        _tod = st.selectbox("Time of day", list(TimeOfDay))
    with row_ctx2[2]:
        _wx = st.selectbox("Weather", list(Weather))
    ctx = TravellerContext(
        gender=_g,
        social=_soc,
        age=_age,
        vehicle=_veh,
        time_of_day=_tod,
        weather=_wx,
    )

    preset_col, _preset_spacer = st.columns([1, 2])
    with preset_col:
        preset = st.selectbox("Preset", list(CostPreset))

    st.divider()
    st.subheader("3. Search mode and heuristics")
    mode = st.radio(
        "Search mode",
        [MODE_ONE, MODE_SIX],
        horizontal=True,
        help="Single: one algorithm, one path on the map. All six: table + six colored routes (layer toggles).",
    )

    algo = None
    if mode == MODE_ONE:
        algo_col, _algo_spacer = st.columns([1, 2])
        with algo_col:
            algo = st.selectbox("Algorithm", list(ALGOS.keys()))

    use_heuristic_ui = mode == MODE_SIX or (
        mode == MODE_ONE and algo in INFORMED_ALGOS
    )
    heuristic_mode = HEURISTIC_SINGLE
    heuristic_choice = HEURISTIC_NAMES[0]
    if use_heuristic_ui:
        heuristic_mode = st.radio(
            "Heuristic mode",
            [HEURISTIC_SINGLE, HEURISTIC_ALL_PARALLEL],
            horizontal=True,
            help="All heuristics: runs admissible, realism, and fast together via a thread pool (same graph in memory).",
        )
        if heuristic_mode == HEURISTIC_SINGLE:
            hcol, _h_spacer = st.columns([1, 2])
            with hcol:
                heuristic_choice = st.selectbox(
                    "Heuristic",
                    HEURISTIC_NAMES,
                    help="Used for A*, weighted A*, and greedy best-first.",
                )

    # Store only path data (serializable). Rebuild Folium on each run — storing
    # ``folium.Map`` in session_state often yields a blank iframe after reruns.
    if st.button("Run", type="primary"):
        st.session_state["run_id"] = int(st.session_state.get("run_id", 0)) + 1
        rows: list[dict] = []
        if mode == MODE_ONE:
            assert algo is not None
            fn = ALGOS[algo]
            if algo in INFORMED_ALGOS and heuristic_mode == HEURISTIC_ALL_PARALLEL:
                pairs = _run_heuristic_variants_parallel(
                    fn, graph, s, t, ctx, preset, HEURISTIC_NAMES
                )
                named_paths: list[tuple[str, list[int] | None]] = []
                for h_name, res in pairs:
                    label = f"{algo}:{h_name}"
                    rows.append(
                        {
                            "algorithm": label,
                            "path_cost": res.path_cost,
                            "expanded": res.metrics.nodes_expanded,
                            "ms": res.metrics.runtime_ms,
                        }
                    )
                    named_paths.append(
                        (label, list(res.path) if res.path else None)
                    )
                st.session_state["route_rows"] = rows
                st.session_state["map_mode"] = "multi"
                st.session_state["map_spec"] = {
                    "kind": "multi",
                    "named_paths": named_paths,
                }
            else:
                if algo in INFORMED_ALGOS:
                    res = fn(
                        graph,
                        s,
                        t,
                        ctx,
                        preset,
                        heuristic_name=heuristic_choice,
                    )
                else:
                    res = fn(graph, s, t, ctx, preset)
                rows.append(
                    {
                        "algorithm": algo,
                        "path_cost": res.path_cost,
                        "expanded": res.metrics.nodes_expanded,
                        "ms": res.metrics.runtime_ms,
                    }
                )
                st.session_state["route_rows"] = rows
                st.session_state["map_mode"] = "single"
                spec: MapSpec = {
                    "kind": "single",
                    "path": list(res.path) if res.path else None,
                    "label": str(algo),
                }
                st.session_state["map_spec"] = spec
        else:

            def run_labeled(
                item: tuple[str, object, str | None],
            ):
                label, search_fn, h = item
                if h is None:
                    return label, search_fn(graph, s, t, ctx, preset)
                return label, search_fn(
                    graph, s, t, ctx, preset, heuristic_name=h
                )

            tasks: list[tuple[str, object, str | None]] = []
            for name, search_fn in ALGOS.items():
                if name in INFORMED_ALGOS:
                    if heuristic_mode == HEURISTIC_ALL_PARALLEL:
                        for h in HEURISTIC_NAMES:
                            tasks.append((f"{name}:{h}", search_fn, h))
                    else:
                        tasks.append((name, search_fn, heuristic_choice))
                else:
                    tasks.append((name, search_fn, None))

            n_workers = max(1, len(tasks))
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(run_labeled, tasks))

            named_paths_multi: list[tuple[str, list[int] | None]] = []
            for label, res in results:
                rows.append(
                    {
                        "algorithm": label,
                        "path_cost": res.path_cost,
                        "expanded": res.metrics.nodes_expanded,
                        "ms": res.metrics.runtime_ms,
                    }
                )
                named_paths_multi.append(
                    (label, list(res.path) if res.path else None)
                )
            st.session_state["route_rows"] = rows
            st.session_state["map_mode"] = "multi"
            multi_spec: MapSpec = {
                "kind": "multi",
                "named_paths": named_paths_multi,
            }
            st.session_state["map_spec"] = multi_spec

    if "route_rows" in st.session_state:
        st.subheader("Results")
        st.caption(
            "Showing the last successful Run (updates when you click Run again)."
        )
        st.dataframe(st.session_state["route_rows"], width="stretch")

    if "map_spec" in st.session_state:
        mm = st.session_state.get("map_mode", "single")
        is_multi = mm in ("multi", "six")
        st.subheader("Route map" if not is_multi else "Route maps (toggle layers)")
        spec = st.session_state["map_spec"]
        if spec["kind"] == "single":
            folium_map = make_route_map(graph, spec["path"], title=spec["label"])
        else:
            folium_map = make_multi_route_map(graph, spec["named_paths"])

        rid = st.session_state.get("run_id", 0)
        # Interactive map (preferred)
        st_folium(
            folium_map,
            height=560,
            use_container_width=True,
            key=f"route_map_{rid}",
        )
        # Static HTML fallback if the bidirectional iframe fails (tiles still load here).
        with st.expander("Map not showing? Open static preview", expanded=False):
            st.iframe(folium_map._repr_html_(), height=580)


if __name__ == "__main__":
    main()
