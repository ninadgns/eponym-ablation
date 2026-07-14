# Dhaka realistic pathfinding

Multi-factor routing over OpenStreetMap (Dhaka): synthetic edge attributes, traveller context, six search algorithms (UCS, Dijkstra, bidirectional UCS, A\*, weighted A\*, greedy best-first), batch CSV analysis, Folium maps, Streamlit UI.

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Commands

| Command | Purpose |
|--------|---------|
| `./run.sh all` | Sync, cache OSM graph + synthetic edges, pytest, batch (≥100 CSV rows), plots, sample map |
| `./run.sh test` | `pytest` |
| `./run.sh analyze` | Batch CSV (`python -m dhaka_pathfind.analysis.batch --help`) |
| `./run.sh ui` | `streamlit run streamlit_app.py` |
| `./run.sh cli` | `dhaka-route` Typer CLI |

First run downloads the Dhaka bbox graph via OSMnx and writes `data/dhaka_graph.graphml` (gitignored).

## Examples

```bash
uv run dhaka-route query --from Shahbag --to Motijheel --algorithm astar --map outputs/maps/q.html
uv run python -m dhaka_pathfind.analysis.batch --pairs 10 --seed 42
```

See [docs/TEACHING_GUIDE.md](docs/TEACHING_GUIDE.md) for metrics and troubleshooting.
