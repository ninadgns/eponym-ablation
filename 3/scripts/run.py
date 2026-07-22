"""Part A experiment driver (PROBLEM_STATEMENT.md §A.10).

Produces results/tables/*.csv and results/plots/*.png. Every run takes an explicit seed; nothing
here touches an unseeded RNG, so the whole file reproduces bit-for-bit on re-run.

  1. Convergence      PSO vs GA vs random search vs the two heuristics, 30 seeds, median + IQR.
  2. Communication    1 particle | 30 particles with c2=0 | 30 particles sharing gbest.
                      All at the SAME 3030-evaluation budget. This is the headline ablation.
  3. Topology         gbest vs ring(k=1), 30 paired seeds. Report mean AND variance, and the
                      iteration at which each stagnates.
  4. Generalisation   Best schedule scored on the 30 HELD-OUT arrival realisations (seeds 100-129).
  5. Sensitivity      K in 10..18, B in 2..4. Where does the fleet constraint start to bind?

Statistics: Wilcoxon signed-rank for location, Levene for variance, on 30 paired seeds. Report
p-values. No claim of "better" without a test.

Runs are dispatched across processes (`--jobs`); results are keyed by seed, so the answer does not
depend on the order they finish in.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from shuttle_timetable.baselines import (
    demand_proportional_schedule,
    random_search,
    uniform_schedule,
)
from shuttle_timetable.ga import genetic_algorithm
from shuttle_timetable.instance import (
    ShuttleConfig,
    arrival_rate,
    round_trip_time,
    sample_arrivals,
)
from shuttle_timetable.pso import pso
from shuttle_timetable.simulator import objective, simulate

TRAIN_SEEDS = (0, 1, 2)  # the M=3 fixed training realisations
TEST_SEEDS = tuple(range(100, 130))  # held-out
RUN_SEEDS = tuple(range(30))  # 30 paired algorithm seeds

BUDGET = 3030  # the experimental control: every method gets exactly this many evaluations
SENS_SEEDS = tuple(range(5))  # sensitivity sweep is 27 cells; 5 seeds each keeps it honest but cheap

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"


def training_sets(cfg: ShuttleConfig) -> list[np.ndarray]:
    return [sample_arrivals(cfg, np.random.default_rng(s)) for s in TRAIN_SEEDS]


def test_sets(cfg: ShuttleConfig) -> list[np.ndarray]:
    return [sample_arrivals(cfg, np.random.default_rng(s)) for s in TEST_SEEDS]


# --------------------------------------------------------------------------- job dispatch


def _job(spec: tuple) -> tuple[np.ndarray, float, np.ndarray]:
    """One optimiser run. Module-level and picklable so it can cross a process boundary."""
    kind, seed, cfg, sets, kw = spec
    rng = np.random.default_rng(seed)
    if kind == "pso":
        r = pso(sets, cfg, rng, **kw)
        return r.best_x, r.best_j, r.curve
    if kind == "ga":
        r = genetic_algorithm(sets, cfg, rng, **kw)
        return r.best_x, r.best_j, r.curve
    if kind == "random":
        best_x, curve = random_search(sets, cfg, BUDGET, rng)
        return best_x, float(curve[-1]), curve
    raise ValueError(f"unknown job kind {kind!r}")


def _map(specs: list[tuple], jobs: int) -> list[tuple[np.ndarray, float, np.ndarray]]:
    if jobs == 1:
        return [_job(s) for s in specs]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_job, specs))


def _runs(kind: str, cfg, sets, seeds, jobs: int, **kw):
    return _map([(kind, s, cfg, sets, kw) for s in seeds], jobs)


# --------------------------------------------------------------------------- helpers


def _curve_band(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and inter-quartile band across seeds, evaluation by evaluation."""
    c = np.vstack(curves)
    return np.median(c, axis=0), np.percentile(c, 25, axis=0), np.percentile(c, 75, axis=0)


def _stagnation_iter(curve: np.ndarray, pop: int = 30) -> int:
    """The iteration of the LAST strict improvement in the best-so-far curve.

    A run that stops improving at iteration 40 has spent 60 iterations' worth of budget learning
    nothing; that is what we mean by stagnation, and it is what separates a ring from a gbest.
    """
    improved = np.flatnonzero(np.diff(curve) < -1e-9)
    return int(improved[-1] // pop) if improved.size else 0


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired signed-rank test. Returns (statistic, p). Identical vectors => p = 1 by convention."""
    if np.allclose(a, b):
        return float("nan"), 1.0
    w = stats.wilcoxon(a, b)
    return float(w.statistic), float(w.pvalue)


def _metrics(x: np.ndarray, sets: list[np.ndarray], cfg: ShuttleConfig) -> dict[str, float]:
    """Deployment metrics, averaged over a set of arrival realisations."""
    res = [simulate(x, a, cfg) for a in sets]
    return {
        "objective": float(np.mean([r.objective for r in res])),
        "mean_wait_min": float(np.mean([r.mean_wait for r in res])),
        "p90_wait_min": float(np.mean([r.p90_wait for r in res])),
        "service_level_pct": float(np.mean([r.service_level for r in res])),
        "stranded_pct": float(np.mean([100.0 * r.n_stranded / r.waits.size for r in res])),
        "fleet_penalty": float(np.mean([r.fleet_penalty for r in res])),
        "max_concurrency": float(np.max([r.concurrency.max() for r in res])),
    }


def _save(df: pd.DataFrame, name: str) -> None:
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"\n--- {name} ---")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[saved] {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- experiments


def exp1_convergence(cfg, train, jobs) -> dict[str, list]:
    print("\n[1/5] convergence: PSO vs GA vs random search, 30 seeds")
    out = {
        "PSO (gbest)": _runs("pso", cfg, train, RUN_SEEDS, jobs),
        "GA (real-coded)": _runs("ga", cfg, train, RUN_SEEDS, jobs),
        "Random search": _runs("random", cfg, train, RUN_SEEDS, jobs),
    }

    j_uni = objective(uniform_schedule(cfg), train, cfg)
    j_dem = objective(demand_proportional_schedule(cfg), train, cfg)

    rows = [
        {
            "method": name,
            "median_J": float(np.median([r[1] for r in runs])),
            "mean_J": float(np.mean([r[1] for r in runs])),
            "sd_J": float(np.std([r[1] for r in runs], ddof=1)),
            "best_J": float(np.min([r[1] for r in runs])),
            "worst_J": float(np.max([r[1] for r in runs])),
            "evals": BUDGET,
        }
        for name, runs in out.items()
    ]
    for name, j in (("Uniform headway", j_uni), ("Demand-proportional", j_dem)):
        rows.append(
            {
                "method": name,
                "median_J": j,
                "mean_J": j,
                "sd_J": 0.0,
                "best_J": j,
                "worst_J": j,
                "evals": 0,
            }
        )
    _save(pd.DataFrame(rows), "A1_convergence_summary")

    pso_j = np.array([r[1] for r in out["PSO (gbest)"]])
    ga_j = np.array([r[1] for r in out["GA (real-coded)"]])
    rnd_j = np.array([r[1] for r in out["Random search"]])
    tests = [
        ("PSO vs GA", *_wilcoxon(pso_j, ga_j)),
        ("PSO vs random search", *_wilcoxon(pso_j, rnd_j)),
        ("GA vs random search", *_wilcoxon(ga_j, rnd_j)),
        ("PSO vs demand-proportional", *_wilcoxon(pso_j, np.full(len(RUN_SEEDS), j_dem))),
    ]
    _save(
        pd.DataFrame(
            [
                {"comparison": n, "wilcoxon_W": w, "p_value": p, "median_diff": d}
                for (n, w, p), d in zip(
                    tests,
                    [
                        float(np.median(pso_j - ga_j)),
                        float(np.median(pso_j - rnd_j)),
                        float(np.median(ga_j - rnd_j)),
                        float(np.median(pso_j - j_dem)),
                    ],
                    strict=True,
                )
            ]
        ),
        "A1_convergence_tests",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    evals = np.arange(1, BUDGET + 1)
    for (name, runs), colour in zip(out.items(), ["C0", "C1", "C2"], strict=True):
        med, lo, hi = _curve_band([r[2] for r in runs])
        ax.plot(evals, med, color=colour, label=f"{name} (median)")
        ax.fill_between(evals, lo, hi, color=colour, alpha=0.18, linewidth=0)
    ax.axhline(j_dem, color="k", ls="--", lw=1, label=f"Demand-proportional ({j_dem:.1f})")
    ax.axhline(j_uni, color="grey", ls=":", lw=1, label=f"Uniform headway ({j_uni:.1f})")
    ax.set_xlabel("objective evaluations (budget = 3030)")
    ax.set_ylabel("best-so-far $J$  (mean wait + fleet penalty, min)")
    ax.set_title("A.10.1 Convergence — median and IQR over 30 seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "A1_convergence.png", dpi=150)
    plt.close(fig)
    return out


def exp2_communication(cfg, train, gbest_runs, jobs) -> None:
    print("\n[2/5] ablation: does the swarm's communication do any work?")
    solo = _runs("pso", cfg, train, RUN_SEEDS, jobs, n_particles=1, n_iters=BUDGET - 1)
    no_social = _runs("pso", cfg, train, RUN_SEEDS, jobs, c2=0.0)
    rnd = _runs("random", cfg, train, RUN_SEEDS, jobs)

    arms = {
        "(a) 1 particle, whole budget alone": solo,
        "(b) 30 particles, c2=0 (no sharing)": no_social,
        "(c) 30 particles sharing one gbest": gbest_runs,
        "(d) random search (reference)": rnd,
    }
    _save(
        pd.DataFrame(
            [
                {
                    "arm": name,
                    "median_J": float(np.median([r[1] for r in runs])),
                    "mean_J": float(np.mean([r[1] for r in runs])),
                    "sd_J": float(np.std([r[1] for r in runs], ddof=1)),
                    "evals": BUDGET,
                }
                for name, runs in arms.items()
            ]
        ),
        "A2_communication_ablation",
    )

    j = {k: np.array([r[1] for r in v]) for k, v in arms.items()}
    keys = list(arms)
    _save(
        pd.DataFrame(
            [
                {
                    "comparison": f"{keys[i]}  vs  {keys[k]}",
                    "wilcoxon_W": w,
                    "p_value": p,
                    "median_diff": float(np.median(j[keys[i]] - j[keys[k]])),
                }
                for i, k in [(1, 3), (2, 1), (2, 0), (0, 3)]
                for w, p in [_wilcoxon(j[keys[i]], j[keys[k]])]
            ]
        ),
        "A2_communication_tests",
    )

    evals = np.arange(1, BUDGET + 1)
    bands = {name: _curve_band([r[2] for r in runs]) for name, runs in arms.items()}

    # Persist the median/IQR bands so the paper's figure can be redrawn without re-running the
    # optimiser. The arm labels are the CSV's column prefixes: a, b, c, d.
    curves = {"eval": evals}
    for name, (med, lo, hi) in bands.items():
        arm = name[1]
        curves[f"{arm}_median"], curves[f"{arm}_q25"], curves[f"{arm}_q75"] = med, lo, hi
    path = TABLES / "A2_communication_curves.csv"
    pd.DataFrame(curves).to_csv(path, index=False)
    print(f"[saved] {path.relative_to(ROOT)} ({len(evals)} rows, median + IQR per arm)")

    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, colour) in zip(arms, ["C3", "C1", "C0", "C2"], strict=True):
        med, lo, hi = bands[name]
        ax.plot(evals, med, color=colour, label=name)
        ax.fill_between(evals, lo, hi, color=colour, alpha=0.15, linewidth=0)
    ax.set_xlabel("objective evaluations (budget = 3030)")
    ax.set_ylabel("best-so-far $J$ (min)")
    ax.set_title("A.10.2 Is it the population, or the communication?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "A2_communication.png", dpi=150)
    plt.close(fig)


def exp3_topology(cfg, train, gbest_runs, jobs) -> None:
    print("\n[3/5] topology: fully connected vs ring (k=1), 30 paired seeds")
    ring_runs = _runs("pso", cfg, train, RUN_SEEDS, jobs, topology="ring")

    g_j = np.array([r[1] for r in gbest_runs])
    r_j = np.array([r[1] for r in ring_runs])
    g_stag = np.array([_stagnation_iter(r[2]) for r in gbest_runs])
    r_stag = np.array([_stagnation_iter(r[2]) for r in ring_runs])

    _save(
        pd.DataFrame(
            [
                {
                    "topology": name,
                    "mean_J": float(j.mean()),
                    "median_J": float(np.median(j)),
                    "sd_J": float(j.std(ddof=1)),
                    "var_J": float(j.var(ddof=1)),
                    "mean_stagnation_iter": float(st.mean()),
                    "median_stagnation_iter": float(np.median(st)),
                }
                for name, j, st in (
                    ("gbest (fully connected)", g_j, g_stag),
                    ("ring (k=1)", r_j, r_stag),
                )
            ]
        ),
        "A3_topology",
    )

    w, p = _wilcoxon(g_j, r_j)
    lev_stat, lev_p = stats.levene(g_j, r_j)
    _save(
        pd.DataFrame(
            [
                {
                    "test": "Wilcoxon signed-rank (location, paired)",
                    "statistic": w,
                    "p_value": p,
                    "note": "H0: gbest and ring have the same median final J",
                },
                {
                    "test": "Levene (variance)",
                    "statistic": float(lev_stat),
                    "p_value": float(lev_p),
                    "note": "H0: equal variance across seeds",
                },
            ]
        ),
        "A3_topology_tests",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    evals = np.arange(1, BUDGET + 1)
    for (name, runs), colour in zip(
        (("gbest", gbest_runs), ("ring (k=1)", ring_runs)), ["C0", "C4"], strict=True
    ):
        med, lo, hi = _curve_band([r[2] for r in runs])
        axes[0].plot(evals, med, color=colour, label=name)
        axes[0].fill_between(evals, lo, hi, color=colour, alpha=0.18, linewidth=0)
    axes[0].set_xlabel("evaluations")
    axes[0].set_ylabel("best-so-far $J$ (min)")
    axes[0].set_title("A.10.3 gbest vs ring — median and IQR")
    axes[0].legend()

    axes[1].boxplot([g_j, r_j], tick_labels=["gbest", "ring (k=1)"])
    axes[1].set_ylabel("final $J$ (min)")
    axes[1].set_title(f"Spread over 30 seeds (Levene p = {lev_p:.3f})")
    fig.tight_layout()
    fig.savefig(PLOTS / "A3_topology.png", dpi=150)
    plt.close(fig)


def exp4_generalisation(cfg, train, test, gbest_runs, ga_runs) -> np.ndarray:
    print("\n[4/5] generalisation: 30 held-out arrival realisations (seeds 100-129)")
    pso_best = gbest_runs[int(np.argmin([r[1] for r in gbest_runs]))][0]
    ga_best = ga_runs[int(np.argmin([r[1] for r in ga_runs]))][0]

    schedules = {
        "PSO (best of 30 seeds)": pso_best,
        "GA (best of 30 seeds)": ga_best,
        "Demand-proportional": demand_proportional_schedule(cfg),
        "Uniform headway": uniform_schedule(cfg),
    }
    rows = []
    for name, x in schedules.items():
        tr = _metrics(x, train, cfg)
        te = _metrics(x, test, cfg)
        rows.append(
            {
                "schedule": name,
                "train_J": tr["objective"],
                "heldout_J": te["objective"],
                "optimism_gap": te["objective"] - tr["objective"],
                "heldout_mean_wait_min": te["mean_wait_min"],
                "heldout_p90_wait_min": te["p90_wait_min"],
                "heldout_service_level_pct": te["service_level_pct"],
                "heldout_stranded_pct": te["stranded_pct"],
                "heldout_fleet_penalty": te["fleet_penalty"],
            }
        )
    _save(pd.DataFrame(rows), "A4_generalisation")

    # Paired over the 30 held-out realisations: is the advantage real out of sample?
    per_real = {
        name: np.array([simulate(x, a, cfg).objective for a in test])
        for name, x in schedules.items()
    }
    w, p = _wilcoxon(per_real["PSO (best of 30 seeds)"], per_real["Demand-proportional"])
    _save(
        pd.DataFrame(
            [
                {
                    "comparison": "PSO vs demand-proportional, paired over 30 held-out realisations",
                    "wilcoxon_W": w,
                    "p_value": p,
                    "median_diff_J": float(
                        np.median(
                            per_real["PSO (best of 30 seeds)"] - per_real["Demand-proportional"]
                        )
                    ),
                }
            ]
        ),
        "A4_generalisation_tests",
    )
    _plot_timetable(cfg, pso_best, demand_proportional_schedule(cfg), test[0])
    return pso_best


def exp5_sensitivity(cfg, train, jobs) -> None:
    print("\n[5/5] sensitivity: K in 10..18, B in 2..4")
    k_values = range(10, 19)
    b_values = (2, 3, 4)

    specs, index = [], []
    for k in k_values:
        for b in b_values:
            c = ShuttleConfig(**{**vars(cfg), "K": k, "B": b})
            for seed in SENS_SEEDS:
                specs.append(("pso", seed, c, train, {}))
                index.append((k, b, c))
    results = _map(specs, jobs)

    rows = []
    for (k, b, c), (x, j, _curve) in zip(index, results, strict=True):
        m = _metrics(x, train, c)
        rows.append({"K": k, "B": b, "seed_J": j, **m, "seats": k * c.C})
    raw = pd.DataFrame(rows)

    agg = (
        raw.groupby(["K", "B"], as_index=False)
        .agg(
            J=("seed_J", "mean"),
            J_sd=("seed_J", "std"),
            mean_wait_min=("mean_wait_min", "mean"),
            service_level_pct=("service_level_pct", "mean"),
            stranded_pct=("stranded_pct", "mean"),
            fleet_penalty=("fleet_penalty", "mean"),
            max_concurrency=("max_concurrency", "mean"),
        )
        .round(4)
    )
    _save(agg, "A5_sensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for b, colour in zip(b_values, ["C3", "C0", "C2"], strict=True):
        sub = agg[agg.B == b]
        axes[0].plot(sub.K, sub.J, "o-", color=colour, label=f"B = {b}")
        axes[1].plot(sub.K, sub.service_level_pct, "o-", color=colour, label=f"B = {b}")
        axes[2].plot(sub.K, sub.fleet_penalty, "o-", color=colour, label=f"B = {b}")
    axes[0].set_ylabel("$J$ (min)")
    axes[0].set_title("Objective")
    axes[1].set_ylabel("% of students waiting $\\leq$ 10 min")
    axes[1].set_title("Service level")
    axes[2].set_ylabel("fleet penalty at the optimum")
    axes[2].set_title("Where the fleet constraint binds")
    for ax in axes:
        ax.set_xlabel("K (trips)")
        ax.legend(fontsize=8)
    fig.suptitle("A.10.5 Sensitivity to the trip budget K and the fleet size B (5 seeds/cell)")
    fig.tight_layout()
    fig.savefig(PLOTS / "A5_sensitivity.png", dpi=150)
    plt.close(fig)


def _plot_timetable(cfg, x_pso, x_dem, arrivals) -> None:
    """The picture that makes the solution legible: arrivals, departures, loads, fleet occupancy."""
    res = simulate(x_pso, arrivals, cfg)
    grid = np.arange(0.0, cfg.T + 1.0)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(grid, np.searchsorted(arrivals, grid), color="C7", label="cumulative arrivals")
    axes[0].plot(
        grid, arrival_rate(grid, cfg) * 50, color="C8", lw=0.9, alpha=0.7,
        label=r"$\lambda(t)$ (scaled $\times$50)",
    )
    for i, t in enumerate(np.sort(x_pso)):
        axes[0].axvline(t, color="C0", lw=1.2, alpha=0.9, label="PSO departure" if i == 0 else None)
    for i, t in enumerate(np.sort(x_dem)):
        axes[0].axvline(
            t, color="k", lw=1.0, ls="--", alpha=0.5,
            label="demand-proportional" if i == 0 else None,
        )
    axes[0].set_ylabel("students")
    axes[0].set_title(
        f"PSO timetable on a held-out day — mean wait {res.mean_wait:.1f} min, "
        f"service level {res.service_level:.0f}%, {res.n_stranded} stranded"
    )
    axes[0].legend(fontsize=8, loc="upper left")

    left_behind = _left_behind_counts(res, arrivals, cfg)
    axes[1].bar(res.departures, res.loads, width=9, color="C0", label="boarded")
    axes[1].bar(
        res.departures, left_behind, width=9, bottom=res.loads, color="C3",
        label="left behind at that departure",
    )
    axes[1].axhline(cfg.C, color="k", ls=":", lw=1, label=f"capacity C = {cfg.C}")
    axes[1].set_ylabel("passengers")
    axes[1].legend(fontsize=8)

    axes[2].step(np.arange(res.concurrency.size), res.concurrency, where="post", color="C0")
    axes[2].axhline(cfg.B, color="C3", ls="--", lw=1.2, label=f"fleet size B = {cfg.B}")
    axes[2].fill_between(
        np.arange(res.concurrency.size), cfg.B, res.concurrency,
        where=res.concurrency > cfg.B, step="post", color="C3", alpha=0.3,
    )
    axes[2].plot(grid, round_trip_time(grid, cfg) / 20.0, color="C8", lw=0.9, alpha=0.6,
                 label="$R(t)$ (scaled)")
    axes[2].set_ylabel("buses in service $n(\\tau)$")
    axes[2].set_xlabel("minutes after 07:00")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS / "A4_timetable.png", dpi=150)
    plt.close(fig)


def _left_behind_counts(res, arrivals: np.ndarray, cfg: ShuttleConfig) -> np.ndarray:
    """Students still queueing after each departure: they had arrived, had not yet given up, and
    did not fit on this bus — the capacity cliff, made visible."""
    out = np.zeros(res.departures.size, dtype=int)
    board_time = arrivals + res.waits  # only meaningful where res.boarded
    for j, td in enumerate(res.departures):
        left = (
            (arrivals <= td)
            & (arrivals + cfg.W_strand >= td)  # had not yet reneged when this bus left
            & (~res.boarded | (board_time > td))  # never boarded, or boarded a LATER bus
        )
        out[j] = int(left.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Part A experiments (PROBLEM_STATEMENT.md §A.10)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    cfg = ShuttleConfig()
    train, test = training_sets(cfg), test_sets(cfg)
    print(
        f"Part A — K={cfg.K} trips, C={cfg.C} seats, B={cfg.B} buses, T={cfg.T} min; "
        f"N per training day = {[a.size for a in train]}; budget = {BUDGET} evaluations; "
        f"jobs = {args.jobs}"
    )

    conv = exp1_convergence(cfg, train, args.jobs)
    exp2_communication(cfg, train, conv["PSO (gbest)"], args.jobs)
    exp3_topology(cfg, train, conv["PSO (gbest)"], args.jobs)
    best_x = exp4_generalisation(cfg, train, test, conv["PSO (gbest)"], conv["GA (real-coded)"])
    exp5_sensitivity(cfg, train, args.jobs)

    np.savetxt(TABLES / "A_best_schedule.csv", np.sort(best_x), delimiter=",", header="departure_min_after_0700", comments="")
    print(f"\nDone. Tables -> {TABLES.relative_to(ROOT)}, plots -> {PLOTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
