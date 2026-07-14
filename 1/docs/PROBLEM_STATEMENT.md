# Formal problem statement (implementation mapping)

## Goal

Build a **realistic pathfinding simulation** on Dhaka’s road network: OSM graph via **OSMnx**, **vectorized synthetic** edge attributes, a **multi-factor cost** \(C_{\text{edge}}\), **six** search algorithms using that cost (not hop count or raw length alone), **multiple heuristics** (one admissible with conservative scaling, two non-admissible), **comparative analysis** (≥100 runs in the default batch), **visualization** (Folium + matplotlib/seaborn), **CLI** and **Streamlit** UI, and documentation.

## Non-goals

- Live traffic APIs (synthetic dynamics only).
- Production-scale global routing.
- Cities other than Dhaka (only `config.DHAKA_BBOX` + landmarks are city-specific).

## Success criteria ↔ repository

| Criterion | Location |
|-----------|----------|
| OSM via Python package | `dhaka_pathfind/graph/load.py` (OSMnx) |
| Cached graph | `data/dhaka_graph.graphml` (generated) |
| Synthetic attributes | `dhaka_pathfind/synthesis/attributes.py` |
| Cost + presets + context | `dhaka_pathfind/cost/` |
| 6 algorithms | `dhaka_pathfind/search/algorithms.py` |
| Heuristics | `dhaka_pathfind/heuristics/registry.py` |
| Reverse-Dijkstra ground truth | `dhaka_pathfind/heuristics/ground_truth.py` |
| Batch ≥100 rows (default 10 pairs × 12 algorithm/heuristic rows) | `dhaka_pathfind/analysis/batch.py` |
| Plots | `dhaka_pathfind/analysis/plots.py` |
| Folium | `dhaka_pathfind/viz/folium_routes.py` |
| CLI / UI | `dhaka_pathfind/cli.py`, `streamlit_app.py` |
| End-to-end | `./run.sh all` |
| Tests (toy + monotonicity + admissibility slack) | `tests/` |

## Female-alone-late-night vs male-midday

Enforced in `cost/model.py` and tested in `tests/test_toy_graph.py`.
