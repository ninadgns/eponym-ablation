"""Plot generation for the experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALGO_ORDER = [
    "basic_backtracking",
    "bt_mrv",
    "bt_lcv",
    "bt_fc_mrv_deg",
    "min_conflicts",
]

ALGO_LABELS = {
    "basic_backtracking": "Basic BT",
    "bt_mrv": "BT+MRV",
    "bt_lcv": "BT+LCV",
    "bt_fc_mrv_deg": "BT+FC+MRV",
    "min_conflicts": "Min-Conflicts",
}

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a",
    "#984ea3", "#ff7f00", "#a65628", "#f781bf",
]


def _color_map() -> dict[str, str]:
    return {
        algo: COLORS[i % len(COLORS)]
        for i, algo in enumerate(ALGO_ORDER)
    }


def _fig(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_metric_vs_n(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    log_scale: bool = False,
) -> None:
    fig, ax = _fig(title, "N (vehicles)", ylabel)
    cmap = _color_map()
    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo].sort_values("n")
        if sub.empty:
            continue
        col = f"{metric}_mean"
        if col not in sub.columns:
            col = metric
        if col not in sub.columns:
            continue
        ax.plot(
            sub["n"], sub[col],
            marker="o", label=ALGO_LABELS.get(algo, algo),
            color=cmap.get(algo, "gray"),
        )
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heuristic_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Aggregate per-algorithm mean across all N values."""
    agg = (
        df.groupby("algorithm")
        .agg(
            nodes=("nodes_expanded", "mean"),
            backtracks=("backtracks", "mean"),
            runtime=("runtime_seconds", "mean"),
            objective=("objective", "mean"),
        )
        .reindex([a for a in ALGO_ORDER if a in df["algorithm"].unique()])
    )

    metrics = ["nodes", "backtracks", "runtime", "objective"]
    labels = ["Nodes expanded", "Backtracks", "Runtime (s)", "Objective J(S)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    cmap = _color_map()

    for ax, metric, label in zip(axes, metrics, labels):
        vals = agg[metric]
        algos = vals.index.tolist()
        colors = [cmap.get(a, "gray") for a in algos]
        bars = ax.bar(
            range(len(algos)), vals.values,
            color=colors, edgecolor="black", linewidth=0.5,
        )
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(
            [ALGO_LABELS.get(a, a) for a in algos],
            rotation=35, ha="right", fontsize=8,
        )
        # Annotate bars
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.01,
                f"{h:.2g}",
                ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle("Per-algorithm aggregate (mean over all N and seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(cost_traces: dict[str, list[float]], out_path: Path) -> None:
    """Min-Conflicts convergence curve."""
    fig, ax = _fig(
        "Min-Conflicts convergence — cost vs. repair step",
        "Repair step", "Objective J(S)",
    )
    for label, trace in cost_traces.items():
        ax.plot(trace, label=label, alpha=0.85)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_plots(summary: pd.DataFrame, raw: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_vs_n(
        summary, "runtime_s_mean", "Wall time (s)", "Execution time vs. N",
        out_dir / "runtime_vs_n.png",
    )
    plot_metric_vs_n(
        summary, "nodes_mean", "Nodes expanded", "Nodes expanded vs. N",
        out_dir / "nodes_vs_n.png", log_scale=True,
    )
    plot_metric_vs_n(
        summary, "backtracks_mean", "Backtracks", "Backtracks vs. N",
        out_dir / "backtracks_vs_n.png", log_scale=True,
    )
    plot_metric_vs_n(
        summary, "objective_mean", "Objective J(S)", "Solution quality vs. N",
        out_dir / "objective_vs_n.png",
    )
    plot_metric_vs_n(
        summary, "failure_rate_mean", "Failure rate", "Failure rate vs. N",
        out_dir / "failure_rate_vs_n.png",
    )
    plot_heuristic_bars(raw, out_dir / "heuristic_bars.png")
