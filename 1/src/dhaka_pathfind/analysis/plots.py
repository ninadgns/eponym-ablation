"""Plots from batch CSV (regenerate figures without re-running search)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dhaka_pathfind.config import OUTPUTS_DIR, ensure_outputs_dir


def latest_csv(results_dir: Path | None = None) -> Path:
    d = results_dir or (OUTPUTS_DIR / "results")
    files = sorted(d.glob("batch_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("no batch_*.csv in outputs/results")
    return files[0]


def plot_algorithm_bars(df: pd.DataFrame, out: Path) -> None:
    ensure_outputs_dir()
    g = (
        df.groupby("algorithm", as_index=False)["nodes_expanded"]
        .mean()
        .sort_values("nodes_expanded")
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=g, x="algorithm", y="nodes_expanded")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_heuristic_heatmap(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["heuristic_name"] != "n/a"].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(
        index="algorithm",
        columns="heuristic_name",
        values="nodes_expanded",
        aggfunc="mean",
    )
    plt.figure(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_pred_vs_runtime(df: pd.DataFrame, out: Path) -> None:
    sub = df.dropna(subset=["heuristic_mean_abs_gap"])
    if sub.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=sub, x="runtime_ms", y="heuristic_mean_abs_gap", hue="algorithm")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def generate_all(csv_path: Path | None = None) -> None:
    ensure_outputs_dir()
    p = csv_path or latest_csv()
    df = pd.read_csv(p)
    fig_dir = OUTPUTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_algorithm_bars(df, fig_dir / "algorithm_nodes_expanded.png")
    plot_heuristic_heatmap(df, fig_dir / "heuristic_heatmap.png")
    plot_pred_vs_runtime(df, fig_dir / "gap_vs_runtime.png")
    print(f"Figures written to {fig_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--latest", action="store_true", help="use newest batch_*.csv")
    args = ap.parse_args()
    csv_path = args.csv
    if args.latest or csv_path is None:
        csv_path = latest_csv()
    generate_all(csv_path)


if __name__ == "__main__":
    main()
