#!/usr/bin/env python
"""Run the full scalability sweep and write CSVs + plots."""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 42])
    parser.add_argument("--budget", type=float, default=2.0, help="Per-run time budget (s)")
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--slots", type=int, default=6)
    args = parser.parse_args()

    from fuel_csp.analyzer import ExperimentConfig, run_matrix, save_csvs, summarise
    from fuel_csp.visualizer import generate_all_plots

    cfg = ExperimentConfig(
        sizes=tuple(args.sizes),
        seeds=tuple(args.seeds),
        time_budget_s=args.budget,
        num_stations=args.stations,
        num_slots=args.slots,
    )

    logging.info("Running %d configurations …", len(cfg.sizes) * len(cfg.seeds) * 5)
    df = run_matrix(cfg)
    paths = save_csvs(df, RESULTS)
    logging.info("CSVs written: %s", paths)

    summary = summarise(df)
    generate_all_plots(summary, df, RESULTS / "plots")
    logging.info("Plots written to %s/plots/", RESULTS)

    # Print a human-readable summary table
    print("\n=== Summary (mean over seeds) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
