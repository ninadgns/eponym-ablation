# Part A — Algorithms Explained

Part A optimizes a single black-box objective $J(\mathbf{x})$ (mean student wait + fleet-overload
penalty, see `PROBLEM_STATEMENT.md` §A.6). Two population-based optimizers do the real work — **PSO**
and a **real-coded GA** — plus a set of **baselines** to measure them against. All of them share one
hard rule: **exactly 3,030 objective evaluations**, spent in one auditable place per file. Every
comparison is decided by *mechanism*, never by one method secretly getting more evaluations.

Files: `shuttle_timetable/pso.py`, `ga.py`, `baselines.py`.

---

## 1. Particle Swarm Optimization — `pso.py`

A swarm of 30 candidate timetables (each a point $\mathbf{x} \in \mathbb{R}^{14}$) flies through the
search box, each remembering its own best and being pulled toward a socially-shared best.

**The core update** (`pso.py:92–98`), applied every iteration to every particle:

$$
v \leftarrow \underbrace{w\,v}_{\text{momentum}}
  + \underbrace{c_1 r_1(\text{pbest}-x)}_{\text{cognitive: my own best}}
  + \underbrace{c_2 r_2(\text{social}-x)}_{\text{social: neighbourhood best}},
\qquad x \leftarrow x+v
$$

- **Inertia $w$ decays linearly $0.9 \to 0.4$** (`pso.py:86`) — early iterations explore widely,
  later ones settle. This is the Shi & Eberhart (1998) refinement of the original Kennedy &
  Eberhart (1995) rule.
- $c_1 = c_2 = 1.49$; $r_1, r_2$ are fresh uniform randoms per component, so each dimension is
  pulled independently.
- **Velocity clamped** to $\pm 0.2T$ (`pso.py:97`) so particles can't jump across the whole day in
  one step.
- **Boundary handling is repair** (`pso.py:100–103`): if a component leaves $[0,T]$, clamp it back
  *and zero that velocity component* — so a particle pinned to a wall doesn't keep accumulating
  momentum into it. Not reflection, not resampling: one rule, stated and defended.

**Two design hooks the experiments depend on:**

- **`topology`** (`_social_attractor`, `pso.py:120`): the "social" attractor is either the
  whole-swarm best (`gbest`, fully connected) or, for the ring topology, the best among a particle's
  3 neighbours $\{i-1, i, i+1\}$. Both go through one code path, so the A.10.3 comparison is
  like-for-like. Returned per-particle so the velocity update is identical between the two.
- **`c2 = 0`** severs social communication entirely (A.10.2 ablation) — 30 independent
  hill-searchers with no sharing. Crucially, `r2` is *still drawn* (`pso.py:91`) so both arms consume
  the same RNG stream and differ only in the mechanism, not the randomness.

**Budget** = `n_particles * (1 + n_iters)` = 30 + 30×100 = **3,030**, spent only inside `_evaluate`
(`pso.py:68`), which is also the single place the best-so-far convergence curve is written.

---

## 2. Real-coded Genetic Algorithm — `ga.py`

A population of 30 timetables evolves by selection → crossover → mutation, at the **identical
3,030-eval budget** so the cross-family comparison against PSO is honest.

- **Selection — tournament, size 3** (`_tournament`, `ga.py:110`): pick 3 individuals at random,
  the one with lowest $J$ wins. Cheap selection pressure, no fitness scaling needed.
- **Crossover — BLX-α, α=0.5** (`_blx_alpha`, `ga.py:116`): for each gene, sample uniformly from
  the parents' interval *widened by* $\alpha \cdot d$ on each side. The widening is what lets
  offspring explore *outside* the parents' bounding box — without it the population contracts
  monotonically and the search collapses. (Eshelman & Schaffer 1993.)
- **Mutation — Gaussian**, $\sigma = 0.05T$, rate $1/K$ per gene (`ga.py:90–91`): on average one
  gene per child gets a small nudge.
- **Elitism — keep top 2** (`ga.py:96–98`): the 2 best of the *current* population survive
  unchanged, guaranteeing the best-so-far never regresses.

**The subtle budget accounting** (`ga.py:11–17`): each generation breeds a *full* pool of 30
offspring and evaluates all 30, so a generation costs exactly 30 evals → 30×(1+100) = 3,030,
matching PSO. The next population is then the 2 (already-evaluated, free-to-carry) elites plus the
best 28 offspring. Breeding only 28 offspring instead would have quietly spent 2,830 evals and
handed the GA a *different* budget — the one thing A.8 forbids.

**An honest weakness the code documents** (`ga.py:19–26`): the chromosome *is* the departure-time
vector, but $J$ is permutation-invariant (A.4), so gene position is meaningless. BLX-α is a
positional operator on a non-positional encoding — it can blend two good schedules into one that's
good nowhere. PSO's velocity update has the identical exposure. Sorting inside the simulator hides
this, it doesn't fix it. Worth stating in the report rather than papering over.

---

## 3. Baselines — `baselines.py`

Not "algorithms" in the metaheuristic sense, but the yardsticks A.9 requires:

- **Uniform headway** (`uniform_schedule`): 14 departures evenly spaced — current practice, the
  strawman-that-isn't.
- **Demand-proportional** (`demand_proportional_schedule`): departures placed at the $K$ quantiles
  of the cumulative arrival curve, so equal expected load per bus. This is the *smart* heuristic and
  the one that matters — it ignores capacity and the fleet constraint (which is why a search *can*
  beat it), but if PSO can't, that's the finding.
- **Random search** (`random_search`): 3,030 uniform samples from the box — the "is the swarm doing
  anything at all?" control.

(A.9 also lists a best-of-K local-restart hill-climber; that one lives in the experiments script
rather than `baselines.py`.)

---

## The through-line

PSO and GA are both from-scratch population methods chosen because $J$ is non-differentiable,
piecewise-constant, multimodal, and black-box (A.7) — nothing for a gradient method to descend. The
whole experimental design hangs on the fixed 3,030-eval budget being spent in exactly one auditable
place in each file, so every comparison is decided by *mechanism*, never by one method secretly
getting more evaluations.
