#!/usr/bin/env python
"""Build results/REPORT.md from the latest CSVs."""

from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def main() -> None:
    import pandas as pd

    raw_path = RESULTS / "experiments_raw.csv"
    summary_path = RESULTS / "experiments_summary.csv"
    if not raw_path.exists():
        print("No CSV found — run experiments first: ./run.sh experiments")
        return

    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)

    lines: list[str] = [
        "# Fuel Crisis CSP/COP — Results Report",
        "",
        "## Problem definition",
        "",
        "**Variables** — one per vehicle. **Domain** — feasible (station, pump, slot) triples.",
        "",
        "**Hard constraints**",
        "1. Fuel-type compatibility",
        "2. Reachability (vehicle range ≥ road distance to station)",
        "3. Pump exclusivity — no two vehicles share the same (station, pump, slot)",
        "4. Supply capacity — cumulative demand ≤ station reserve per fuel type",
        "5. Operating hours / vehicle time windows",
        "",
        "**Soft objective J(S)** (COP)",
        "```",
        "J = w_dist · total_distance",
        "  + w_wait · total_wait_time",
        "  + w_prio · priority_penalty  (ambulances penalised quadratically for late slots)",
        "  + w_unassigned · #unassigned_vehicles",
        "```",
        "",
        "## Algorithms",
        "",
        "| ID | Algorithm | Key improvement |",
        "|---|---|---|",
        "| 1 | Basic Backtracking | Baseline — input order, no heuristics |",
        "| 2 | BT + MRV | Degree tie-break uses precomputed constraint graph |",
        "| 3 | BT + LCV | Least constraining value — max downstream options |",
        "| 4 | BT + FC + MRV | Forward checking with O(1) supply check per value |",
        "| 5 | Min-Conflicts | Local search with tabu list to avoid cycling |",
        "",
        "## Scalability plots",
        "",
        "![Runtime](plots/runtime_vs_n.png)",
        "![Nodes](plots/nodes_vs_n.png)",
        "![Backtracks](plots/backtracks_vs_n.png)",
        "![Objective](plots/objective_vs_n.png)",
        "![Failure rate](plots/failure_rate_vs_n.png)",
        "![Heuristic bars](plots/heuristic_bars.png)",
        "",
        "## Summary table (mean over seeds)",
        "",
        summary.to_markdown(index=False),
        "",
        "## Key observations",
        "",
    ]

    # Auto-generate observations from data
    by_algo = raw.groupby("algorithm").agg(
        nodes=("nodes_expanded", "mean"),
        bt=("backtracks", "mean"),
        bj=("backjumps", "mean"),
        rt=("runtime_seconds", "mean"),
        obj=("objective", "mean"),
        fr=("failure_rate", "mean"),
    )

    best_nodes = by_algo["nodes"].idxmin()
    best_bt = by_algo["bt"].idxmin()
    best_obj = by_algo["obj"].idxmin()
    worst_nodes = by_algo["nodes"].idxmax()

    lines += [
        f"- **Fewest nodes expanded**: `{best_nodes}` "
        f"(avg {by_algo.loc[best_nodes,'nodes']:.0f} nodes)",
        f"- **Fewest backtracks**: `{best_bt}` "
        f"(avg {by_algo.loc[best_bt,'bt']:.0f} backtracks)",
        f"- **Best solution quality**: `{best_obj}` "
        f"(avg J={by_algo.loc[best_obj,'obj']:.1f})",
        f"- **Most nodes (baseline)**: `{worst_nodes}` "
        f"(avg {by_algo.loc[worst_nodes,'nodes']:.0f} nodes) — "
        "illustrates exponential blow-up without heuristics.",
        "",
    ]

    report = "\n".join(lines)
    out = RESULTS / "REPORT.md"
    out.write_text(report)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
