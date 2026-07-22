# Fuel Crisis CSP/COP — Results Report

> **Regenerated 2026-07-22 from `repro/csp_rerun.py`.** A previous version of this file tabulated
> two additional solvers, `bt_cbj_fc_mrv` (conflict-directed backjumping) and `bt_ac3_mrv` (AC-3
> preprocessing). Neither is runnable in this codebase: there is no CBJ implementation anywhere in
> the tree, and `fuel_csp/algorithms/ac3.py` is absent from `ALL_SOLVERS` and fails to import
> (it references `fuel_csp.constraints.conflict_set`, which does not exist). Those rows have been
> removed. Everything below comes from solvers that actually execute.

## Problem definition

**Variables** — one per vehicle. **Domain** — feasible (station, pump, slot) triples.

**Hard constraints**
1. Fuel-type compatibility
2. Reachability (vehicle range >= road distance to station)
3. Pump exclusivity — no two vehicles share the same (station, pump, slot)
4. Supply capacity — cumulative demand <= station reserve per fuel type
5. Operating hours / vehicle time windows

**Soft objective J(S)** (COP)
```
J = w_dist · total_distance
  + w_wait · total_wait_time
  + w_prio · priority_penalty  (ambulances penalised quadratically for late slots)
  + w_unassigned · #unassigned_vehicles
```

## Algorithms

| ID | Algorithm | Key idea |
|---|---|---|
| 1 | Basic Backtracking | Baseline — input order, no heuristics |
| 2 | BT + MRV | Minimum remaining values, degree tie-break from a precomputed constraint graph |
| 3 | BT + LCV | Least constraining value — maximise downstream options |
| 4 | BT + FC + MRV | Forward checking with an O(1) supply check per value |
| 5 | Min-Conflicts | Local repair with a tabu list to avoid cycling |

## Experimental setup

10 instance sizes x 8 seeds x 5 solvers = 400 runs, 5 s budget per run. Every run records whether
it exhausted the budget (`censored`), because that determines whether its node count means
anything — see "Reading this table" below.

## Results (mean over 8 seeds)

| algorithm | n | success | censored | runtime_s | nodes | backtracks | checks | objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `basic_backtracking` | 5 | 6/8 | 0/8 | 0.234 | 31,136 | 31,132 | 31,136 | 123.5 |
| `basic_backtracking` | 8 | 6/8 | 1/8 | 0.672 | 134,489 | 134,482 | 134,489 | 216.3 |
| `basic_backtracking` | 10 | 4/8 | 4/8 | 2.500 | 272,493 | 272,484 | 272,493 | 288.2 |
| `basic_backtracking` | 12 | 4/8 | 4/8 | 2.500 | 320,161 | 320,150 | 320,161 | 326.7 |
| `basic_backtracking` | 15 | 4/8 | 4/8 | 2.500 | 375,556 | 375,542 | 375,556 | 367.4 |
| `basic_backtracking` | 20 | 2/8 | 6/8 | 3.750 | 219,636 | 219,618 | 219,636 | 578.7 |
| `basic_backtracking` | 25 | 1/8 | 7/8 | 4.375 | 530,210 | 530,187 | 530,210 | 776.9 |
| `basic_backtracking` | 30 | 0/8 | 8/8 | 5.000 | 815,326 | 815,300 | 815,326 | 1,100.6 |
| `basic_backtracking` | 40 | 0/8 | 8/8 | 5.000 | 899,747 | 899,717 | 899,747 | 2,488.3 |
| `basic_backtracking` | 50 | 0/8 | 8/8 | 5.000 | 842,358 | 842,327 | 842,358 | 4,216.2 |
| `bt_mrv` | 5 | 6/8 | 0/8 | 0.161 | 29,710 | 29,706 | 29,710 | 123.5 |
| `bt_mrv` | 8 | 6/8 | 1/8 | 0.672 | 97,541 | 97,534 | 97,541 | 219.7 |
| `bt_mrv` | 10 | 4/8 | 4/8 | 2.500 | 341,065 | 341,056 | 341,065 | 288.3 |
| `bt_mrv` | 12 | 4/8 | 4/8 | 2.500 | 278,077 | 278,066 | 278,077 | 319.2 |
| `bt_mrv` | 15 | 4/8 | 4/8 | 2.500 | 251,881 | 251,867 | 251,881 | 371.6 |
| `bt_mrv` | 20 | 2/8 | 6/8 | 3.750 | 352,502 | 352,483 | 352,502 | 586.7 |
| `bt_mrv` | 25 | 1/8 | 7/8 | 4.375 | 330,201 | 330,178 | 330,201 | 761.2 |
| `bt_mrv` | 30 | 0/8 | 8/8 | 5.000 | 414,617 | 414,589 | 414,617 | 1,010.7 |
| `bt_mrv` | 40 | 0/8 | 7/8 | 4.375 | 555,053 | 555,022 | 555,053 | 2,177.3 |
| `bt_mrv` | 50 | 0/8 | 7/8 | 4.375 | 579,327 | 579,290 | 579,327 | 3,151.9 |
| `bt_lcv` | 5 | 6/8 | 0/8 | 0.303 | 31,528 | 31,525 | 31,528 | 118.9 |
| `bt_lcv` | 8 | 6/8 | 1/8 | 0.688 | 96,910 | 96,903 | 96,910 | 249.1 |
| `bt_lcv` | 10 | 4/8 | 4/8 | 2.501 | 231,060 | 231,050 | 231,060 | 325.4 |
| `bt_lcv` | 12 | 4/8 | 4/8 | 2.501 | 275,355 | 275,344 | 275,355 | 358.1 |
| `bt_lcv` | 15 | 4/8 | 4/8 | 2.501 | 219,928 | 219,914 | 219,928 | 403.3 |
| `bt_lcv` | 20 | 2/8 | 6/8 | 3.766 | 206,071 | 206,053 | 206,071 | 663.7 |
| `bt_lcv` | 25 | 0/8 | 8/8 | 5.000 | 370,490 | 370,468 | 370,490 | 1,120.1 |
| `bt_lcv` | 30 | 0/8 | 8/8 | 5.000 | 323,806 | 323,780 | 323,806 | 1,310.2 |
| `bt_lcv` | 40 | 0/8 | 8/8 | 5.000 | 300,139 | 300,108 | 300,139 | 2,236.4 |
| `bt_lcv` | 50 | 0/8 | 8/8 | 5.000 | 175,517 | 175,488 | 175,517 | 4,222.0 |
| `bt_fc_mrv_deg` | 5 | 6/8 | 0/8 | 0.298 | 34,434 | 34,430 | 34,434 | 109.3 |
| `bt_fc_mrv_deg` | 8 | 6/8 | 1/8 | 0.756 | 67,496 | 67,489 | 67,496 | 193.8 |
| `bt_fc_mrv_deg` | 10 | 4/8 | 4/8 | 2.501 | 183,365 | 183,356 | 183,365 | 261.5 |
| `bt_fc_mrv_deg` | 12 | 4/8 | 4/8 | 2.501 | 167,366 | 167,354 | 167,366 | 289.3 |
| `bt_fc_mrv_deg` | 15 | 4/8 | 4/8 | 2.501 | 139,702 | 139,688 | 139,702 | 324.5 |
| `bt_fc_mrv_deg` | 20 | 2/8 | 6/8 | 3.751 | 158,330 | 158,312 | 158,330 | 552.1 |
| `bt_fc_mrv_deg` | 25 | 1/8 | 7/8 | 4.376 | 166,082 | 166,059 | 166,082 | 763.3 |
| `bt_fc_mrv_deg` | 30 | 0/8 | 8/8 | 5.000 | 150,331 | 150,304 | 150,331 | 958.9 |
| `bt_fc_mrv_deg` | 40 | 0/8 | 8/8 | 5.000 | 114,759 | 114,722 | 114,759 | 1,333.4 |
| `bt_fc_mrv_deg` | 50 | 0/8 | 8/8 | 5.000 | 82,944 | 82,901 | 82,944 | 2,155.0 |
| `min_conflicts` | 5 | 6/8 | 0/8 | 0.000 | 0 | 0 | 1 | 204.7 |
| `min_conflicts` | 8 | 6/8 | 0/8 | 0.000 | 0 | 0 | 1 | 344.3 |
| `min_conflicts` | 10 | 3/8 | 0/8 | 0.000 | 1 | 0 | 2 | 426.6 |
| `min_conflicts` | 12 | 4/8 | 0/8 | 0.000 | 1 | 0 | 2 | 495.0 |
| `min_conflicts` | 15 | 4/8 | 0/8 | 0.000 | 1 | 0 | 2 | 544.1 |
| `min_conflicts` | 20 | 1/8 | 0/8 | 0.001 | 4 | 0 | 5 | 773.9 |
| `min_conflicts` | 25 | 1/8 | 0/8 | 0.001 | 7 | 0 | 8 | 1,037.7 |
| `min_conflicts` | 30 | 0/8 | 0/8 | 0.002 | 11 | 0 | 12 | 1,162.9 |
| `min_conflicts` | 40 | 0/8 | 0/8 | 0.006 | 21 | 0 | 22 | 1,614.4 |
| `min_conflicts` | 50 | 0/8 | 0/8 | 0.425 | 1,141 | 0 | 1,142 | 2,141.5 |

## Reading this table

**Censoring.** At n >= 30 every systematic run exhausts the budget and nothing is solved. A censored
run's node count measures *how fast the solver grinds*, not how well it prunes — on fully censored
cells `basic_backtracking` sustains ~168k nodes/s against `bt_fc_mrv_deg`'s ~17k. Forward
checking's apparent node advantage at large n is 10x lower throughput against a shared stopwatch,
not better pruning. Do not read the large-n rows as a scaling law.

**The uncensored comparison.** On the 24 instances every solver completed:

| solver | median nodes | median time | vs. basic (paired Wilcoxon) |
|---|---:|---:|---|
| `basic_backtracking` | 13.5 | 0.09 ms | — |
| `bt_mrv` | 13.5 | 0.14 ms | +0.0 nodes, p = 0.27 |
| `bt_lcv` | 10.5 | 1.09 ms | −3.0 nodes, p = 1e−4 |
| `bt_fc_mrv_deg` | 9.0 | 1.27 ms | −5.0 nodes, p < 1e−4 |
| `min_conflicts` | 1.0 | 0.08 ms | −13.5 nodes, p < 1e−4 |

## Key observations

- **The orderings prune as advertised, and cost more than they save.** LCV removes a median of 3
  nodes and forward checking 5, both significant — while running 12x and 14x slower in wall clock.
  MRV alone saves no nodes at all (p = 0.27) and still costs 1.6x. At these instance sizes the
  per-node price of the ordering exceeds the value of the nodes it saves.
- **The orderings do not change what gets solved.** 27/80, 27/80, 26/80, 27/80 instances solved.
  The four systematic solvers disagree on the outcome of 1 of 80 instances (n=25, seed 303), and
  that one goes *against* the heuristic — `bt_lcv` fails where the other three succeed.
- **Min-conflicts dominates on cost.** 119 nodes and 0.044 s per attempt against basic
  backtracking's 444,111 nodes and 3.15 s — 3,700x fewer nodes, 72x less time — for 25 solved
  against 27. A 93% success rate at 0.03% of the search cost, by repairing one complete assignment
  rather than rebuilding a partial prefix on every backtrack.

## Plots

![Runtime](plots/runtime_vs_n.png)
![Nodes](plots/nodes_vs_n.png)
![Backtracks](plots/backtracks_vs_n.png)
![Objective](plots/objective_vs_n.png)
![Failure rate](plots/failure_rate_vs_n.png)
![Heuristic bars](plots/heuristic_bars.png)

> Note: the plots above were generated by `scripts/run_experiments.py` from an earlier sweep
> (5 sizes x 3 seeds, 2 s budget) and are not regenerated from the table above. Regenerate with
> `uv run python scripts/run_experiments.py` if you need them consistent.
