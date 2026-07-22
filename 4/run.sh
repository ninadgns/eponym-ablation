#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

unset VIRTUAL_ENV
uv sync --all-extras --quiet

case "${1:-help}" in
    test)   exec uv run pytest tests/ "${@:2}" ;;
    run)    exec uv run python scripts/run.py "${@:2}" ;;
    all)
        uv run pytest tests/ -q
        uv run python scripts/run.py
        echo "Done. Results in results/"
        ;;
    *)
        cat <<'HELP'
signal-control — AI Lab Assignment 4

./run.sh test     Run the test suite
./run.sh run      Signal control: VI vs Q-learning vs baselines       -> results/
./run.sh all      test + run

Read PROBLEM_STATEMENT.md first — it is where the parameters are defended.
Both graded algorithms (value_iteration, q_learning) are implemented; README.md has the results.
HELP
        ;;
esac
