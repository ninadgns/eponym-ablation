#!/usr/bin/env python
"""
Produce Table 2 and the §4 numbers from the instrumented Assignment 1 batch.

Run `1/scripts/instrumented_batch.py` first. From the repo root:

    cd 1 && uv run python scripts/instrumented_batch.py
           && uv run python ../repro/search_analyze.py

Timing basis (all three matter; see the batch script's docstring):
  * setup   -- `build_heuristic_aux`, timed on its own, from setup_cost.csv
  * search  -- reported search time MINUS the post-processing that only astar() performs
               inside its timed region, so every arm reports the same region
  * per (pair, config) we take the median over 5 repetitions, then the mean over pairs
"""

from __future__ import annotations

import collections
import csv
import statistics as st
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "1" / "outputs" / "results"

rows = list(csv.DictReader((RESULTS / "instrumented_42.csv").open()))
setup = list(csv.DictReader((RESULTS / "setup_cost.csv").open()))
inv = list(csv.DictReader((RESULTS / "greedy_invariance.csv").open()))
pp = list(csv.DictReader((RESULTS / "astar_postproc.csv").open()))

m_star = float(setup[0]["min_cost_per_m"])
mean_cm = float(setup[0]["mean_cost_per_m"])
setup_ms = [float(r["setup_ms"]) for r in setup]
SETUP = st.median(setup_ms)

print(f"m*   = {m_star:.6f} cost/m")
print(f"mean = {mean_cm:.6f} cost/m   (ratio {mean_cm/m_star:.3f})")
print(f"setup: {min(setup_ms):.0f}-{max(setup_ms):.0f} ms over {len(setup_ms)} reps, "
      f"median {SETUP:.0f} ms")
ppv = [float(r["postproc_ms"]) for r in pp]
print(f"astar post-processing subtracted: median {st.median(ppv):.1f} ms "
      f"(range {min(ppv):.1f}-{max(ppv):.1f}) over {len(ppv)} measurements\n")

K = {"admissible": 1.0, "fast": 1.25, "realism": 1.08 * mean_cm / m_star}
ALGO_W = {"ucs": 0.0, "dijkstra": 0.0, "bidirectional_ucs": 0.0,
          "astar": 1.0, "weighted_astar": 1.35, "greedy_best_first": float("inf")}
INFORMED = {"astar", "weighted_astar", "greedy_best_first"}

by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by[(r["algorithm"], r["heuristic_name"])][r["pair_id"]].append(r)

opt = {pid: float(v[0]["path_cost"]) for pid, v in by[("ucs", "n/a")].items()}

summary = {}
for (algo, heur), pairs in by.items():
    w = ALGO_W[algo] if heur == "n/a" else ALGO_W[algo] * K[heur]
    per_pair_ms, per_pair_nodes, gaps, spread = [], [], [], []
    for pid, reps in pairs.items():
        ms = [float(x["search_ms"]) for x in reps]
        per_pair_ms.append(st.median(ms))
        spread.append((max(ms) - min(ms)) / st.median(ms))
        per_pair_nodes.append(float(reps[0]["nodes_expanded"]))
        gaps.append(100 * (float(reps[0]["path_cost"]) - opt[pid]) / opt[pid])
    search = st.mean(per_pair_ms)
    setup_for = SETUP if algo in INFORMED else 0.0
    summary[(algo, heur)] = {
        "w": w, "nodes": st.mean(per_pair_nodes), "search": search,
        "setup": setup_for, "total": search + setup_for, "gap": st.mean(gaps),
        "n_opt": sum(1 for x in gaps if x < 1e-9), "n": len(gaps),
        "spread": st.median(spread),
    }

print(f"{'configuration':34}{'w_eff':>8}{'nodes':>9}{'search':>9}{'setup':>9}{'total':>9}"
      f"{'gap%':>9}{'opt':>8}{'spread':>9}")
for (algo, heur), s in sorted(summary.items(), key=lambda kv: (kv[1]["w"], -kv[1]["nodes"])):
    label = algo if heur == "n/a" else f"{algo} + {heur}"
    w = "inf" if s["w"] == float("inf") else f"{s['w']:.3f}"
    print(f"{label:34}{w:>8}{s['nodes']:9.0f}{s['search']:9.1f}{s['setup']:9.1f}"
          f"{s['total']:9.1f}{s['gap']:9.3f}{s['n_opt']:5d}/{s['n']:<2d}{100*s['spread']:8.0f}%")

cols = [c for c in inv[0] if c.startswith("identical_to_")]
n_ident = sum(all(r[c] == "True" for c in cols) for r in inv)
print(f"\nProp 1 -- greedy best-first: identical paths on {n_ident}/{len(inv)} queries "
      f"across {len(cols)+1} heuristics")

finite = sorted((s for s in summary.values() if 0 < s["w"] < float("inf")),
                key=lambda s: s["w"])
ws = [s["w"] for s in finite]
for name in ("nodes", "search", "gap"):
    rho, p = spearmanr(ws, [s[name] for s in finite])
    print(f"Prop 2 -- Spearman(w_eff, {name:6}) = {rho:+.4f}  (p = {p:.2e}, n = {len(finite)})")

ucs = summary[("ucs", "n/a")]
dij = summary[("dijkstra", "n/a")]
bi = summary[("bidirectional_ucs", "n/a")]
astar = summary[("astar", "admissible")]
wa_adm = summary[("weighted_astar", "admissible")]
wa_fast = summary[("weighted_astar", "fast")]

print(f"\nnodes  UCS/A*  = {ucs['nodes']/astar['nodes']:.2f}x   (A* wins)")
print(f"total  A*/UCS  = {astar['total']/ucs['total']:.2f}x   (UCS wins)")
print(f"search UCS/A*  = {ucs['search']/astar['search']:.2f}x")
print(f"  -> node count overstates the time benefit by "
      f"{(ucs['nodes']/astar['nodes'])/(ucs['search']/astar['search']):.2f}x, excluding setup")
print(f"per-node: UCS {1000*ucs['search']/ucs['nodes']:.2f} us | "
      f"A*+adm {1000*astar['search']/astar['nodes']:.2f} us | "
      f"wA*+adm {1000*wa_adm['search']/wa_adm['nodes']:.2f} us")
print(f"setup is {100*astar['setup']/astar['total']:.1f}% of A*'s total runtime")

for name, base in (("UCS", ucs), ("bidirectional UCS", bi)):
    for lbl, arm in (("A*+admissible", astar), ("wA*+fast", wa_fast)):
        save = base["search"] - arm["search"]
        verdict = (f"{arm['setup']/save:.1f} queries" if save > 0
                   else "never (slower even excluding setup)")
        print(f"break-even {lbl:14} vs {name:18} = {verdict}")

exact = {k: s for k, s in summary.items() if s["n_opt"] == s["n"]}
best = min(exact.items(), key=lambda kv: kv[1]["total"])
best_inf = min((kv for kv in exact.items() if kv[1]["w"] > 0), key=lambda kv: kv[1]["total"])
print(f"\nfastest exact overall:  {best[0][0]} at {best[1]['total']:.1f} ms total")
print(f"fastest exact informed: {best_inf[0][0]}+{best_inf[0][1]} at {best_inf[1]['total']:.1f} ms "
      f"({best_inf[1]['total']/best[1]['total']:.1f}x slower)")
print(f"\nucs vs dijkstra -- dijkstra() is `return ucs(...)` with the label overwritten: "
      f"nodes {ucs['nodes']:.0f} vs {dij['nodes']:.0f}, "
      f"search {ucs['search']:.1f} vs {dij['search']:.1f} ms (timing noise on identical code)")
