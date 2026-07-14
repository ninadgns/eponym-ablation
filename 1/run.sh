#!/usr/bin/env bash
# Dhaka pathfinding — entry script (see docs/phases/phase-12-run-sh-and-end-to-end-integration.md)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

UV="${UV:-uv}"

cmd="${1:-help}"

case "$cmd" in
  all)
    echo "==> uv sync"
    "$UV" sync
    echo "==> Ensure graph + synthetic edges"
    "$UV" run python -c "from dhaka_pathfind.graph.load import load_or_download; from dhaka_pathfind.synthesis.attributes import ensure_synthetic_edges; g=load_or_download(); ensure_synthetic_edges(g)"
    echo "==> pytest"
    "$UV" run pytest -q
    echo "==> batch analysis (>=100 rows)"
    "$UV" run python -m dhaka_pathfind.analysis.batch --pairs 10 --seed 42
    echo "==> plots from latest CSV"
    "$UV" run python -m dhaka_pathfind.analysis.plots --latest
    echo "==> example folium map"
    "$UV" run python -m dhaka_pathfind.sample_map
    echo "==> done."
    ;;
  test)
    "$UV" sync
    "$UV" run pytest -q "${@:2}"
    ;;
  analyze)
    "$UV" sync
    "$UV" run python -m dhaka_pathfind.analysis.batch "${@:2}"
    ;;
  ui)
    "$UV" sync
    "$UV" run streamlit run streamlit_app.py "${@:2}"
    ;;
  cli)
    "$UV" sync
    "$UV" run dhaka-route "${@:2}"
    ;;
  help|*)
    echo "Usage: ./run.sh {all|test|analyze|ui|cli|help}"
    echo "  all      - sync, cache graph+edges, pytest, batch, plots, sample map"
    echo "  test     - pytest"
    echo "  analyze  - batch CSV (pass args to module)"
    echo "  ui       - Streamlit app"
    echo "  cli      - dhaka-route Typer CLI"
    ;;
esac
