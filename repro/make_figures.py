#!/usr/bin/env python
"""
Emit the paper's two figures from released CSVs. No optimiser or search runs here --- both
figures read artifacts the experiment drivers already wrote, so a figure can never drift from
the table beside it.

    python repro/make_figures.py            # writes paper/figures/*.pdf

Inputs:
  fig:weff    1/outputs/results/instrumented_42.csv, setup_cost.csv   (instrumented_batch.py)
  fig:pso     3/results/tables/A2_communication_curves.csv            (3/scripts/run.py)

Vector PDF rather than PNG: arXiv builds the paper from source, and a 150-dpi raster of a line
plot looks like a 150-dpi raster of a line plot.
"""

from __future__ import annotations

import collections
import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# --------------------------------------------------------------------- fig:weff
def figure_weff() -> None:
    """§4: eleven configurations collapse onto one scalar axis, and setup dwarfs all of them."""
    res = ROOT / "1" / "outputs" / "results"
    rows = list(csv.DictReader((res / "instrumented_42.csv").open()))
    setup = list(csv.DictReader((res / "setup_cost.csv").open()))

    m_star = float(setup[0]["min_cost_per_m"])
    mean_cm = float(setup[0]["mean_cost_per_m"])
    setup_ms = st.median(float(r["setup_ms"]) for r in setup)

    # Same effective-weight algebra as search_analyze.py (Prop. 2).
    K = {"admissible": 1.0, "fast": 1.25, "realism": 1.08 * mean_cm / m_star}
    ALGO_W = {"ucs": 0.0, "dijkstra": 0.0, "bidirectional_ucs": 0.0,
              "astar": 1.0, "weighted_astar": 1.35, "greedy_best_first": float("inf")}

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[(r["algorithm"], r["heuristic_name"])][r["pair_id"]].append(r)
    opt = {pid: float(v[0]["path_cost"]) for pid, v in by[("ucs", "n/a")].items()}

    cfgs = {}
    for (algo, heur), pairs in by.items():
        if algo == "dijkstra":  # `return ucs(...)` with the label overwritten -- one row, not two
            continue
        gaps = [100 * (float(v[0]["path_cost"]) - opt[pid]) / opt[pid] for pid, v in pairs.items()]
        cfgs[(algo, heur)] = {
            "w": ALGO_W[algo] if heur == "n/a" else ALGO_W[algo] * K[heur],
            "nodes": st.mean(float(v[0]["nodes_expanded"]) for v in pairs.values()),
            "ms": st.mean(st.median(float(x["search_ms"]) for x in v) for v in pairs.values()),
            "exact": all(g < 1e-9 for g in gaps),
        }

    # The sweep is the w_eff axis of Prop. 2: uniform-cost at 0, the informed arms between,
    # greedy at infinity. Bidirectional UCS is *not* on it -- it is a different algorithm that
    # happens to also sit at w_eff = 0 -- so it gets its own marker rather than a line segment
    # implying a sweep it is not part of.
    bi = cfgs.pop(("bidirectional_ucs", "n/a"))
    sweep = sorted(cfgs.values(), key=lambda c: c["w"])
    finite = [c for c in sweep if c["w"] < float("inf")]
    inf_pts = [c for c in sweep if c["w"] == float("inf")]
    x_inf = max(c["w"] for c in finite) * 1.45          # a place to draw "infinity"
    x_break = (max(c["w"] for c in finite) + x_inf) / 2  # where the axis is cut

    fig, (ax_n, ax_t) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)

    for ax, key, ylab in ((ax_n, "nodes", "nodes expanded"), (ax_t, "ms", "search time (ms)")):
        xs = [c["w"] for c in finite] + [x_inf]
        ys = [c[key] for c in finite] + [inf_pts[0][key]]
        ax.plot(xs[:-1], ys[:-1], "-", color="C0", lw=1.1, zorder=2)
        ax.plot(xs[-2:], ys[-2:], ":", color="C0", lw=1.1, zorder=2)  # the jump to infinity
        for c, x in zip(finite + inf_pts[:1], xs, strict=True):
            ax.plot(x, c[key], "o", ms=4.5, zorder=4, color="C0",
                    mfc="C0" if c["exact"] else "white", mew=1.1)
        ax.plot(0, bi[key], "D", ms=5, color="C2", zorder=5)
        ax.axvline(x_break, color="0.75", ls=":", lw=0.8, zorder=1)
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.set_xticks([0, 1, 2, 3, x_inf])
        ax.set_xticklabels(["0", "1", "2", "3", r"$\infty$"])
        ax.set_xlabel(r"effective weight $w_{\mathrm{eff}}$")

    # The whole point of the right panel: every search time sits far below the one-off setup.
    ax_t.axhline(setup_ms, color="C1", lw=1.2, ls="--", zorder=3)
    ax_t.annotate(f"heuristic setup: {setup_ms:.0f} ms, paid once per context,\n"
                  f"charged to no arm by the node metric",
                  xy=(0.52, setup_ms), xycoords=("axes fraction", "data"),
                  xytext=(0, -5), textcoords="offset points",
                  ha="center", va="top", color="C1", fontsize=7.5)

    ax_n.annotate("uniform-cost", xy=(0, cfgs[("ucs", "n/a")]["nodes"]),
                  xytext=(7, 1), textcoords="offset points", fontsize=7.5, color="0.3")
    ax_n.annotate("bidirectional", xy=(0, bi["nodes"]), xytext=(7, -9),
                  textcoords="offset points", fontsize=7.5, color="C2")
    ax_n.annotate("greedy", xy=(x_inf, inf_pts[0]["nodes"]), xytext=(-3, 7),
                  textcoords="offset points", fontsize=7.5, color="0.3", ha="right")

    handles = [
        plt.Line2D([], [], color="C0", marker="o", ms=4.5, lw=1.1, label="on the sweep, exact"),
        plt.Line2D([], [], color="C0", marker="o", ms=4.5, lw=1.1, mfc="white",
                   label="on the sweep, suboptimal"),
        plt.Line2D([], [], color="C2", marker="D", ms=5, lw=0, label="bidirectional UCS (exact)"),
    ]
    ax_n.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.2,
                borderpad=0.1, handletextpad=0.5)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "weff.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(OUT / 'weff.pdf').relative_to(ROOT)}  "
          f"({len(finite)} finite-weight points, setup {setup_ms:.0f} ms)")


# ---------------------------------------------------------------------- fig:pso
def figure_pso() -> None:
    """§6: the communication ablation, on the axis the budget is denominated in."""
    src = ROOT / "3" / "results" / "tables" / "A2_communication_curves.csv"
    if not src.exists():
        sys.exit(f"missing {src.relative_to(ROOT)} — run `cd 3 && ./run.sh all` first")
    rows = list(csv.DictReader(src.open()))
    evals = [float(r["eval"]) for r in rows]

    ARMS = [
        ("a", "C3", "(a) 1 particle, whole budget alone"),
        ("b", "C1", r"(b) 30 particles, $c_2 = 0$ (no sharing)"),
        ("d", "C2", "(d) random search"),
        ("c", "C0", r"(c) 30 particles sharing one $\mathtt{gbest}$"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for key, colour, label in ARMS:
        med = [float(r[f"{key}_median"]) for r in rows]
        lo = [float(r[f"{key}_q25"]) for r in rows]
        hi = [float(r[f"{key}_q75"]) for r in rows]
        ax.plot(evals, med, color=colour, lw=1.4, label=label,
                zorder=4 if key == "c" else 3)
        ax.fill_between(evals, lo, hi, color=colour, alpha=0.15, linewidth=0)

    ax.set_xlabel("objective evaluations (identical 3,030 budget for every arm)")
    ax.set_ylabel(r"best-so-far $J$ (min of mean student wait)")
    ax.set_xlim(0, max(evals))
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "pso_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(OUT / 'pso_ablation.pdf').relative_to(ROOT)}  ({len(rows)} evaluations)")


if __name__ == "__main__":
    # Same convention as check_labels.py: the paper lives on the `paper` branch, and off it there
    # is nowhere for a figure to go. Skip rather than scatter a stray paper/ directory across main.
    if not OUT.parent.exists():
        print("SKIP — no paper/ on this branch; the paper source lives on `paper`.")
        print("       To draw the figures:  git switch paper && python repro/make_figures.py")
        sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    figure_weff()
    figure_pso()
