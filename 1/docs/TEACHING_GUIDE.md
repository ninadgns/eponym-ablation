# Teaching guide

## Install

```bash
uv sync
```

Use `uv.lock` for reproducible dependency resolution.

## One-shot pipeline

```bash
chmod +x run.sh   # once
./run.sh all
```

This will:

1. Download/cache the Dhaka driving graph (first run only; needs network).
2. Build `data/dhaka_edges.parquet` with synthetic columns (seeded).
3. Run `pytest`.
4. Write `outputs/results/batch_*.csv` and figures under `outputs/figures/`.
5. Write `outputs/maps/example_ucs.html`.

## Metrics (batch CSV)

- **path_cost** — sum of `edge_cost` along the returned path.
- **nodes_expanded** — priority-queue pops that are not stale (see `search/algorithms.py`).
- **revisits** — pops where `g` is worse than the settled best for that node.
- **effective_branching_factor** — coarse \(N^{1/\text{depth}}\) style estimate.
- **heuristic_mean_abs_gap** — for A\*, mean \(\|h(s)-\text{suffix cost}\|\) on the final path.

## Slow tests (full Dhaka graph)

Default `pytest` excludes `@pytest.mark.slow` tests (see `pyproject.toml`). To run them:

```bash
uv run pytest -m slow
```

## Common pitfalls

- **Missing graph**: run any command that calls `load_or_download()` once online.
- **Parquet mismatch**: delete `data/dhaka_edges.parquet` if the OSM graph changes.
- **Admissible heuristic**: uses a **conservative** minimum cost/m (see `cost/model.py`) so \(h\) stays ≤ reverse-Dijkstra distances in tests modulo floating error.

## CLI / UI

- `dhaka-route query --from Shahbag --to Motijheel --algorithm ucs`
- `streamlit run streamlit_app.py`
