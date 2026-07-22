#!/usr/bin/env python
"""
Produce Table 2 and the §4 numbers from the instrumented Assignment 1 batch.

Run `1/scripts/instrumented_batch.py` first. From the repo root:

    cd 1 && uv run python scripts/instrumented_batch.py
           && uv run python ../repro/search_analyze.py
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

m_star = float(setup[0]["min_cost_per_m"])
mean_cm = float(setup[0]["mean_cost_per_m"])
setup_ms = [float(r["setup_ms"]) for r in setup]

print(f"m*        = {m_star:.6f} cost/m")
print(f"mean      = {mean_cm:.6f} cost/m   (ratio {mean_cm/m_star:.3f})")
print(f"setup, isolated over {len(setup_ms)} reps: "
      f"{min(setup_ms):.0f}-{max(setup_ms):.0f} ms, median {st.median(setup_ms):.0f} ms")

# Effective weight of each heuristic relative to the admissible one, h_adm = m* * d_geo.
# admissible = m*d ; fast = 1.25 m*d ; realism = 1.08 * mean * d.
K = {"admissible": 1.0, "fast": 1.25, "realism": 1.08 * mean_cm / m_star}
ALGO_W = {
    "ucs": 0.0, "dijkstra": 0.0, "bidirectional_ucs": 0.0,
    "astar": 1.0, "weighted_astar": 1.35, "greedy_best_first": float("inf"),
}

opt = {r["pair_id"]: float(r["path_cost"]) for r in rows if r["algorithm"] == "ucs"}
g = collections.defaultdict(list)
for r in rows:
    g[(r["algorithm"], r["heuristic_name"])].append(r)


def w_eff(algo: str, heur: str) -> float:
    base = ALGO_W[algo]
    return base if heur == "n/a" else base * K[heur]


summary = {}
for (algo, heur), v in g.items():
    gaps = [100 * (float(r["path_cost"]) - opt[r["pair_id"]]) / opt[r["pair_id"]] for r in v]
    summary[(algo, heur)] = {
        "w": w_eff(algo, heur),
        "nodes": st.mean(float(r["nodes_expanded"]) for r in v),
        "search": st.mean(float(r["search_ms"]) for r in v),
        "setup": st.mean(float(r["setup_ms"]) for r in v),
        "total": st.mean(float(r["total_ms"]) for r in v),
        "gap": st.mean(gaps),
        "n_opt": sum(1 for x in gaps if x < 1e-9),
        "n": len(v),
    }

print(f"\n{'configuration':34}{'w_eff':>8}{'nodes':>9}{'search':>9}{'setup':>9}{'total':>9}{'gap%':>9}{'opt':>7}")
for (algo, heur), s in sorted(summary.items(), key=lambda kv: (kv[1]["w"], -kv[1]["nodes"])):
    label = algo if heur == "n/a" else f"{algo} + {heur}"
    w = "inf" if s["w"] == float("inf") else f"{s['w']:.3f}"
    print(f"{label:34}{w:>8}{s['nodes']:9.0f}{s['search']:9.1f}{s['setup']:9.1f}"
          f"{s['total']:9.1f}{s['gap']:9.3f}{s['n_opt']:4d}/{s['n']}")

# --- Proposition 1: greedy best-first is scale-invariant --------------------
cols = [c for c in inv[0] if c.startswith("identical_to_")]
n_ident = sum(all(r[c] == "True" for c in cols) for r in inv)
print(f"\nProp 1 -- greedy best-first returns identical paths on {n_ident}/{len(inv)} queries "
      f"across {len(cols) + 1} heuristics")

# --- Prop 2: everything is a monotone function of one scalar ----------------
finite = [s for s in summary.values() if 0 < s["w"] < float("inf")]
finite.sort(key=lambda s: s["w"])
ws = [s["w"] for s in finite]
for name in ("nodes", "search", "gap"):
    rho, p = spearmanr(ws, [s[name] for s in finite])
    print(f"Prop 2 -- Spearman(w_eff, {name:6}) = {rho:+.4f}  (p = {p:.2e}, n = {len(finite)})")

# --- the metric reversal ----------------------------------------------------
ucs = summary[("ucs", "n/a")]
bi = summary[("bidirectional_ucs", "n/a")]
astar = summary[("astar", "admissible")]
wa_fast = summary[("weighted_astar", "fast")]

print(f"\nnodes ratio   UCS / A*        = {ucs['nodes']/astar['nodes']:.2f}x  (A* wins)")
print(f"total ratio   A* / UCS        = {astar['total']/ucs['total']:.2f}x  (UCS wins)")
print(f"search ratio  UCS / A*        = {ucs['search']/astar['search']:.2f}x")
print(f"  -> node count overstates the benefit by "
      f"{(ucs['nodes']/astar['nodes'])/(ucs['search']/astar['search']):.2f}x even excluding setup")
print(f"per-node cost UCS = {1000*ucs['search']/ucs['nodes']:.2f} us, "
      f"A* = {1000*astar['search']/astar['nodes']:.2f} us")
print(f"setup is {100*astar['setup']/astar['total']:.1f}% of A*'s total runtime")

save = ucs["search"] - astar["search"]
print(f"\nbreak-even vs UCS:               {astar['setup']/save:.1f} queries "
      f"(saves {save:.1f} ms/query, setup {astar['setup']:.0f} ms)")
save_bi = bi["search"] - astar["search"]
print(f"break-even vs bidirectional UCS: "
      + ("never — A* is slower even excluding setup" if save_bi <= 0
         else f"{astar['setup']/save_bi:.1f} queries"))
save_wf = bi["search"] - wa_fast["search"]
print(f"  first informed config that ever beats bidirectional UCS: weighted A* + fast, "
      f"break-even {wa_fast['setup']/save_wf:.1f} queries")

exact = {k: s for k, s in summary.items() if s["n_opt"] == s["n"]}
best = min(exact.items(), key=lambda kv: kv[1]["total"])
best_inf = min((kv for kv in exact.items() if kv[1]["w"] > 0), key=lambda kv: kv[1]["total"])
print(f"\nfastest exact method (total):    {best[0][0]} at {best[1]['total']:.1f} ms")
print(f"fastest exact informed method:   {best_inf[0][0]} + {best_inf[0][1]} "
      f"at {best_inf[1]['total']:.1f} ms  ({best_inf[1]['total']/best[1]['total']:.1f}x slower)")
