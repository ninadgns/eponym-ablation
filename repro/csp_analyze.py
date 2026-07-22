#!/usr/bin/env python
"""Summarise the CSP re-run for Section 5 of the paper."""

import csv
import collections
import statistics as st
from pathlib import Path

from scipy.stats import wilcoxon

RAW = Path(__file__).with_name("csp_rerun_raw.csv")
rows = list(csv.DictReader(RAW.open()))
for r in rows:
    r["n"] = int(r["n"])
    r["seed"] = int(r["seed"])
    for k in ("nodes_expanded", "backtracks", "constraint_checks", "repair_steps"):
        r[k] = int(r[k])
    for k in ("runtime_seconds", "objective", "failure_rate"):
        r[k] = float(r[k])
    r["success"] = r["success"] == "True"
    r["censored"] = r["censored"] == "True"

ALGS = ["basic_backtracking", "bt_mrv", "bt_lcv", "bt_fc_mrv_deg", "min_conflicts"]
SIZES = sorted({r["n"] for r in rows})
SEEDS = sorted({r["seed"] for r in rows})
print(f"{len(rows)} runs | sizes {SIZES} | seeds {SEEDS}\n")

g = {(r["algorithm"], r["n"]): [] for r in rows}
for r in rows:
    g[(r["algorithm"], r["n"])].append(r)

print("=" * 100)
print(f"{'algorithm':20}{'n':>4}{'succ':>7}{'cens':>7}{'time_s':>9}{'nodes':>10}{'backtr':>10}{'checks':>11}{'obj':>10}")
print("=" * 100)
for a in ALGS:
    for n in SIZES:
        v = g[(a, n)]
        print(
            f"{a:20}{n:4d}"
            f"{sum(x['success'] for x in v)/len(v):7.2f}"
            f"{sum(x['censored'] for x in v)/len(v):7.2f}"
            f"{st.mean(x['runtime_seconds'] for x in v):9.3f}"
            f"{st.mean(x['nodes_expanded'] for x in v):10.0f}"
            f"{st.mean(x['backtracks'] for x in v):10.0f}"
            f"{st.mean(x['constraint_checks'] for x in v):11.0f}"
            f"{st.mean(x['objective'] for x in v):10.1f}"
        )
    print("-" * 100)

# --- the uncensored regime -------------------------------------------------
unc = sorted({n for n in SIZES if all(not r["censored"] for r in rows if r["n"] == n)})
print(f"\nFully uncensored sizes (no run hit the budget): {unc}")

print("\n--- success rate by algorithm x n (does ordering change WHAT is solved?) ---")
print(f"{'n':>4}" + "".join(f"{a[:14]:>16}" for a in ALGS))
for n in SIZES:
    print(f"{n:4d}" + "".join(f"{sum(x['success'] for x in g[(a,n)])/len(g[(a,n)]):16.2f}" for a in ALGS))

print("\n--- identical success SETS? (per (n,seed), which algs solved it) ---")
mismatch = 0
total = 0
for n in SIZES:
    for s in SEEDS:
        got = {a: next(r["success"] for r in g[(a, n)] if r["seed"] == s) for a in ALGS}
        sysalgs = [a for a in ALGS if a != "min_conflicts"]
        total += 1
        if len({got[a] for a in sysalgs}) > 1:
            mismatch += 1
            print(f"   n={n} seed={s}: " + " ".join(f"{a}={got[a]}" for a in sysalgs))
print(f"systematic solvers disagree on {mismatch}/{total} instances")

# --- paired tests on the uncensored regime ---------------------------------
print("\n--- paired Wilcoxon, nodes expanded, uncensored sizes only, vs basic_backtracking ---")
for a in ALGS[1:]:
    xs, ys = [], []
    for n in unc:
        for s in SEEDS:
            b = next(r for r in g[("basic_backtracking", n)] if r["seed"] == s)
            o = next(r for r in g[(a, n)] if r["seed"] == s)
            xs.append(b["nodes_expanded"])
            ys.append(o["nodes_expanded"])
    if len(set(x - y for x, y in zip(xs, ys))) <= 1:
        print(f"  {a:20} all-equal differences, test skipped")
        continue
    w, p = wilcoxon(xs, ys)
    med = st.median([y - x for x, y in zip(xs, ys)])
    direction = "FEWER" if med < 0 else "MORE"
    print(f"  {a:20} median delta = {med:+10.1f} nodes ({direction}), W={w:.0f}, p={p:.4f}  n={len(xs)}")

print("\n--- cost of an attempt: mean over ALL runs ---")
for a in ALGS:
    v = [r for r in rows if r["algorithm"] == a]
    print(
        f"  {a:20} nodes={st.mean(r['nodes_expanded'] for r in v):10.0f}"
        f"  checks={st.mean(r['constraint_checks'] for r in v):11.0f}"
        f"  time={st.mean(r['runtime_seconds'] for r in v):7.3f}s"
        f"  solved={sum(r['success'] for r in v)}/{len(v)}"
    )

# --- solved-only comparison and the censoring inversion ---------------------
print("\n--- nodes on instances EVERY systematic solver solved (no censoring) ---")
SYS = [a for a in ALGS if a != "min_conflicts"]
common = [
    (n, s) for n in SIZES for s in SEEDS
    if all(next(r["success"] for r in g[(a, n)] if r["seed"] == s) for a in ALGS)
]
print(f"{len(common)} instances solved by all five solvers")
for a in ALGS:
    v = [next(r for r in g[(a, n)] if r["seed"] == s) for n, s in common]
    print(f"  {a:20} median nodes={st.median(r['nodes_expanded'] for r in v):9.1f}"
          f"  mean={st.mean(r['nodes_expanded'] for r in v):10.1f}"
          f"  median time={1000*st.median(r['runtime_seconds'] for r in v):8.2f} ms")

base = [next(r for r in g[("basic_backtracking", n)] if r["seed"] == s) for n, s in common]
for a in ALGS[1:]:
    v = [next(r for r in g[(a, n)] if r["seed"] == s) for n, s in common]
    d = [x["nodes_expanded"] - y["nodes_expanded"] for y, x in zip(base, v)]
    if all(x == 0 for x in d):
        print(f"  {a:20} identical node counts on all {len(d)} instances")
        continue
    w, p = wilcoxon([y["nodes_expanded"] for y in base], [x["nodes_expanded"] for x in v])
    print(f"  {a:20} median delta vs basic = {st.median(d):+9.1f} nodes, W={w:.0f}, p={p:.4f}")

print("\n--- censoring inverts the node metric: throughput on fully censored cells ---")
print(f"{'algorithm':20}{'n=30':>14}{'n=40':>14}{'n=50':>14}   (nodes/second, all runs censored)")
for a in SYS:
    row = ""
    for n in (30, 40, 50):
        v = g[(a, n)]
        row += f"{st.mean(r['nodes_expanded']/r['runtime_seconds'] for r in v):14,.0f}"
    print(f"{a:20}{row}")
print("\nA censored run's node count measures how fast the solver grinds, not how well it prunes.")
