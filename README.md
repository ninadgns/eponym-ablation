# AI Lab

Course assignments for the AI Lab, both set in Dhaka.

| Folder | Assignment | Method |
|--------|------------|--------|
| [`1/`](1/) | Dhaka realistic pathfinding | Uninformed and informed search (UCS, Dijkstra, bidirectional UCS, A\*, weighted A\*, greedy best-first) over the OpenStreetMap road graph, under a multi-factor cost model |
| [`2/`](2/) | Urban fuel-crisis allocator | Constraint satisfaction / optimisation (backtracking, MRV, LCV, forward checking, AC-3, min-conflicts) |

## Setup

Each assignment is a self-contained [uv](https://github.com/astral-sh/uv) project with its own
`pyproject.toml` and lockfile. Set one up by syncing inside its folder:

```bash
cd 1 && uv sync    # or: cd 2 && uv sync
```

Each folder has a `run.sh` entry point and its own `README.md` with the details.

## A note on large files

Assignment 1 depends on a ~58M OpenStreetMap graph (`1/data/dhaka_graph.graphml`), a cached OSM
extract, and a derived Parquet edge table. These are generated artefacts and are **not** tracked in
git — see `1/.gitignore`. Regenerate them from the scripts in `1/` rather than expecting a clone to
carry them.
