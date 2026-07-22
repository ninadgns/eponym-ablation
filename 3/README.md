# Assignment 3 — Population-Based Search

**University of Dhaka campus mobility: when should the 14 shuttle trips depart, given a 3-bus
fleet?** Particle Swarm Optimization and a real-coded GA, both from scratch.

| | |
|---|---|
| **Unit** | Population-based / swarm |
| **Problem** | Offline design of a continuous departure-time vector under a fleet constraint |
| **Method** | PSO + real-coded GA, from scratch |
| **Code** | [`shuttle_timetable/`](shuttle_timetable/) |

The companion assignment, [`../4/`](../4/), is the decision-making-under-uncertainty half of the
same setting (adaptive signal control). The two are separate problems sharing a setting.

The formal statement is in **[`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md)**. Read it before
touching the code — it is where the parameters are defended, and the viva is on the defence, not
the code.

## Usage

```bash
./run.sh test        # pytest — 23 tests, no skips
./run.sh run         # shuttle timetable experiments -> results/   (~5 min)
./run.sh all         # test + run
```

The driver fans its seeds out across processes (`--jobs`, default 8). Results are keyed by seed,
so the numbers do not depend on the order the workers finish in.

Every experiment takes an explicit seed and reproduces bit-for-bit on re-run.

---

## The headline results

*Written from the numbers, after the fact. Several of them disconfirm the conclusion the assignment
led us to expect — the GA outruns the swarm, the fleet constraint never binds, and the ring's
celebrated steadiness is not statistically there. Those are the interesting ones, and they are
stated first rather than buried.*

### The communication is the mechanism, not the population

The ablation demanded by §A.10.2 gives an unusually clean answer. At an identical 3,030-evaluation
budget, over 30 paired seeds:

| Arm | median $J$ (min) | verdict |
|---|---|---|
| 1 particle spending the whole budget alone | 26.10 | **worse than random search** (p < 0.001) |
| 30 particles, $c_2 = 0$ — no sharing | 23.79 | **indistinguishable from random search** (p = 0.22) |
| 30 particles sharing one `gbest` | **15.91** | beats both (p < 0.001) |
| Random search (reference) | 24.56 | — |

Thirty independent searchers are worth nothing over one random sampler. Thirty searchers that
*talk* are worth 8.6 minutes of every student's waiting time. The population is not doing the work;
the social term is. A single particle is actively **worse** than random search, because without a
`gbest` to pull it, inertia just walks it into a corner of the box.

**The GA beats PSO, and the swarm loses its own assignment.** Median $J$ = 15.15 (GA) vs 15.91
(PSO), Wilcoxon signed-rank p = 0.008 over 30 paired seeds. The expected ordering was the other
way. The likely reason is in the encoding and is worth the viva: the objective is invariant under
permutation of the departure-time vector (§A.4), so gene *position* carries no meaning, and PSO's
velocity update — which moves particle $i$'s coordinate $k$ toward `gbest`'s coordinate $k$ — is
built on a correspondence that does not exist. BLX-$\alpha$ suffers the same exposure but its
per-gene interval sampling is less committed to the alignment, and Gaussian mutation gives it a
local-search channel PSO lacks once inertia has decayed.

Both, however, comfortably beat the heuristics, and the advantage **survives out of sample** on the
30 held-out arrival days (seeds 100–129):

| Schedule | train $J$ | held-out $J$ | mean wait | service level | stranded |
|---|---|---|---|---|---|
| PSO (best of 30 seeds) | 14.23 | **16.26** | 16.3 min | 47.1 % | 8.8 % |
| GA (best of 30 seeds) | 14.42 | 16.40 | 16.4 min | 44.1 % | 7.9 % |
| Demand-proportional | 23.47 | 22.92 | 22.9 min | 46.3 % | 23.0 % |
| Uniform headway | 32.14 | 32.40 | 32.4 min | 13.5 % | 9.6 % |

PSO beats the demand-proportional heuristic on every one of the 30 held-out days (p < 0.001), with
an optimism gap of only 2.0 min — it is not merely overfitting its three training realisations.
Note the honest wrinkle: PSO's *service level* is barely better than the heuristic's (47.1 % vs
46.3 %); what it actually buys is a collapse in **stranding**, from 23.0 % to 8.8 %. It carries the
evening surge that the heuristic abandons.

**`gbest` beats the ring, and the ring's famed steadiness does not show up.** Mean $J$ 16.23 vs
17.63 (Wilcoxon p < 0.001). The ring's variance is lower (1.10 vs 1.77) in the direction the
literature predicts, but **Levene's test does not reject equal variance (p = 0.49)**, so on 30 seeds
that difference is not established. Both topologies are still improving when the budget runs out
(last improvement at iteration ≈ 99 of 100), so neither has stagnated: the binding constraint here
is the evaluation budget, not convergence.

**The fleet constraint never binds at an optimum.** Across all 27 cells of the $K \in \{10..18\}
\times B \in \{2,3,4\}$ sweep, the fleet penalty at the found optimum is **exactly zero** and peak
concurrency never exceeds $B$ — the optimiser simply routes around the constraint rather than
paying the quadratic penalty. The constraint is real but shows up as a *shadow cost*: at $K \le 12$,
cutting the fleet to $B = 2$ costs ≈ 1.5 min of mean wait against $B = 3$, while $B = 4$ buys
nothing (differences fall inside the seed noise, sd ≈ 1 min). A fourth bus is not worth buying; a
third is.

---

## What is where

| Module | State |
|---|---|
| `shuttle_timetable/instance.py`, `simulator.py`, `baselines.py` | done, tested |
| `shuttle_timetable/pso.py` | **done** — gbest + ring, $c_2 = 0$ ablation, budget-exact |
| `shuttle_timetable/ga.py` | **done** — tournament, BLX-$\alpha$, Gaussian mutation, elitism |

`tests/test_algorithms.py` is the contract for the two graded algorithms. It is not weakened
anywhere.

Results land in `results/tables/*.csv` and `results/plots/*.png`.
