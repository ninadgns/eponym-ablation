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
    test)        exec uv run pytest tests/ "${@:2}" ;;
    experiments) exec uv run python scripts/run_experiments.py "${@:2}" ;;
    report)      exec uv run python scripts/generate_report.py "${@:2}" ;;
    all)
        uv run pytest tests/ -q
        uv run python scripts/run_experiments.py
        uv run python scripts/generate_report.py
        echo "Done. Results in results/"
        ;;
    *)
        cat <<'HELP'
fuel-crisis-csp — usage

./run.sh test              Run unit tests
./run.sh experiments       Scalability sweep (CSVs + plots)
./run.sh report            Regenerate REPORT.md
./run.sh all               test + experiments + report
./run.sh experiments --sizes 10 20 30 --budget 3.0
HELP
        ;;
esac
