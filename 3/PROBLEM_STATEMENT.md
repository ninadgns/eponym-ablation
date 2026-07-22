# Assignment 3 — Formal Problem Statement

**Unit:** Population-based search.
**Setting:** University of Dhaka campus mobility — designing the shuttle timetable.

This is an offline design problem over a continuous decision vector. Its companion assignment
(`../4/`, adaptive signal control) is an online sequential control problem under uncertainty;
the two are **separate problems sharing a setting**, and that distinction is the point.

---

## Notes for the implementing agent

Read these before writing code.

1. **Both algorithms are implemented from scratch.** No `pyswarms`, no `deap`. NumPy, SciPy (for
   statistics and sparse linear algebra only), pandas, and matplotlib. Implementing PSO and GA
   *is* the assignment.
2. **`../../dhaka-pathfinder/` is a classmate's independent submission.** Do not read from, copy,
   import, or paraphrase its code, documents, or report. It is named here only so it can be
   avoided. This is an individual submission.
3. **Instances are synthetic but motivated.** No field data is required. Every parameter below has
   a stated real-world justification; keep those justifications in the code as constants with
   comments, because they are defended at the viva.
4. **Determinism.** Every experiment takes an explicit seed. No wall-clock or unseeded RNG anywhere.
   Results must reproduce exactly on re-run.
5. **Report honest results.** If an ablation kills the expected conclusion, the disconfirmation is
   the finding. Do not tune the instance until the preferred algorithm wins.

---

# Part A — Shuttle Departure Timetable under a Finite Fleet

## A.1 Informal description

The DU shuttle serves one boarding stop on a loop route. Students arrive at the stop continuously
through the day, in bursts driven by the class timetable. A fixed budget of $K$ trips must be
scheduled across the service day. Buses hold $C$ passengers; when more students are waiting than
will fit, the surplus is **left behind** and waits for the next departure. A bus that departs is
unavailable until it completes its loop, and the loop takes longer in traffic — so with a fleet of
only $B$ buses, departures cannot be bunched arbitrarily.

Choose the **continuous departure times** to minimise student waiting time.

This is a **continuous, constrained, single-objective scheduling problem with capacity cliffs**.

## A.2 Given data (problem instance)

| Symbol | Meaning | Reference value |
|---|---|---|
| $T$ | service window length (minutes), 07:00–20:00 | $780$ |
| $K$ | number of trips to schedule (budget) | $14$ |
| $C$ | bus capacity (seated + standing) | $40$ |
| $B$ | fleet size (buses available) | $3$ |
| $\lambda(t)$ | student arrival rate at the stop (students/min) | see A.3 |
| $R(t)$ | round-trip (loop) time for a bus departing at $t$ (min) | see A.3 |
| $W_{\text{strand}}$ | wait charged to a student who never boards (min) | $60$ |
| $\lambda_{\text{fleet}}$ | weight on the fleet-overload penalty | $0.5$ |

> **Instance sanity check — do this before anything else.** Total seat supply is
> $K \cdot C = 14 \times 40 = 560$. The expected number of students must sit **comfortably below**
> that, or every schedule strands passengers, the objective is dominated by a constant, and the
> problem degenerates into noise. The generator in A.3 yields $\mathbb{E}[N] \approx 400$, a load
> factor of $71\%$ — enough slack that the fleet is adequate *in aggregate*, while the peaks still
> overwhelm individual buses. **That gap is where the entire problem lives.** Assert
> $0.6 \le N / (KC) \le 0.85$ in a test.

## A.3 Instance generators

**Arrival rate $\lambda(t)$.** A baseline rate plus Gaussian bumps at class-change times. DU class
slots run on a 90-minute cadence; model bumps at $t \in \{50, 140, 230, 320, 410, 500, 590, 680\}$
minutes after 07:00, with a heavier evening bump (departure surge) at $t = 680$:

$$
\lambda(t) = \lambda_0 + \sum_{m=1}^{8} \alpha_m \exp\!\left(-\frac{(t - c_m)^2}{2\sigma^2}\right),
\qquad \lambda_0 = 0.10,\;\; \sigma = 8 .
$$

with bump centres $c_m = (50, 140, 230, 320, 410, 500, 590, 680)$ minutes after 07:00 and
amplitudes $\alpha_m = (1.5,\, 2.0,\, 2.0,\, 1.5,\, 1.5,\, 2.0,\, 2.5,\, 3.0)$ — rising through the
day and heaviest at the evening departure surge.

A bump of amplitude $\alpha$ contributes $\alpha\,\sigma\sqrt{2\pi} \approx 20\alpha$ students, so
the evening bump alone delivers ≈ 60 students inside a ±16-minute window — against a 40-seat bus.
**That is the capacity cliff, and it is the point of the instance.** Baseline plus bumps gives
$\mathbb{E}[N] \approx 400$.

**Realised arrivals.** Draw $N$ student arrival times as a non-homogeneous Poisson process with
rate $\lambda(t)$ (thinning). This yields a concrete arrival list $a_1 \le \dots \le a_N$,
$N \approx 400$.

**Round-trip time $R(t)$.** Base loop plus congestion:
$$
R(t) = R_0 + R_1 \cdot g(t), \qquad R_0 = 35,\;\; R_1 = 25,
$$
where $g(t) \in [0,1]$ is a congestion profile peaking in the morning (08:00–10:00) and evening
(17:00–19:00) rushes. So a bus leaving at 08:30 is gone ~60 min; one leaving at noon, ~40 min.

## A.4 Decision variables

$$
\mathbf{x} = (t_1, \dots, t_K) \in [0, T]^K \subset \mathbb{R}^K,
$$
the departure times. The search space is the box $\mathcal{S} = [0,T]^K$, $K = 14$.

The vector is **unordered**: the simulator sorts internally. This means the objective is invariant
under permutation of the components, so there are $K!$ symmetric copies of every optimum. This is
deliberate — it is a genuine and defensible source of multimodality, and it is the kind of
structure a population handles and a local method does not.

## A.5 Simulation semantics (defines the objective)

Given $\mathbf{x}$, sort departures $t_{(1)} \le \dots \le t_{(K)}$. Maintain a FIFO queue of
waiting students. Process events in time order:

1. **Student arrival** at $a_i$: join the back of the queue.
2. **Reneging.** Any queued student who has now waited $W_{\text{strand}}$ minutes **gives up and
   walks**. They leave the queue and are charged $w_i = W_{\text{strand}}$.
3. **Departure** at $t_{(j)}$: board the first $\min(|\text{queue}|, C)$ students in FIFO order.
   Each boarding student $i$ records wait $w_i = t_{(j)} - a_i$. The remainder stay queued.
4. **End of service** at $T$: every student still queued is **stranded**, and is charged
   $w_i = W_{\text{strand}}$.

A student who arrives after the last departure is stranded by definition.

> **Why reneging is not decoration.** Drop step 2 and a student bumped by a full bus can end up
> waiting ~66 minutes — *longer* than the $W_{\text{strand}} = 60$ charged to someone who never
> boards at all. The objective would then prefer to **strand** that student rather than carry
> them, which is a perverse incentive sitting right in the middle of the search space. Reneging
> caps every wait at $W_{\text{strand}}$, makes the objective monotone in service quality, and is
> also simply what a student who has waited an hour actually does. This was found by a test, not
> by inspection; `test_reneging_caps_every_wait_at_w_strand` is the guard.

**Fleet occupancy.** Trip $j$ occupies a bus over $[\,t_{(j)},\; t_{(j)} + R(t_{(j)})\,)$. Define
concurrency on a 1-minute grid:
$$
n(\tau) = \bigl|\{\, j : t_{(j)} \le \tau < t_{(j)} + R(t_{(j)}) \,\}\bigr|, \qquad \tau = 0, 1, \dots, T.
$$

## A.6 Objective

Every student carries exactly one wait: $w_i = t_{(j)} - a_i$ if they board trip $j$, and
$w_i = W_{\text{strand}}$ if they never board. **Minimise**

$$
\boxed{\;
J(\mathbf{x}) \;=\; \underbrace{\frac{1}{N}\sum_{i=1}^{N} w_i}_{\text{mean wait over \emph{all} students}}
\;+\; \lambda_{\text{fleet}} \sum_{\tau=0}^{T} \bigl[\max\bigl(0,\; n(\tau) - B\bigr)\bigr]^{2}
\;}
$$
$$
\text{subject to}\qquad 0 \le t_j \le T, \qquad j = 1, \dots, K.
$$


> **Why stranding is not a separate penalty term.** It is tempting to add
> $\mu \cdot (\text{stranded fraction})$ on top. Do not: a stranded student would then be charged
> twice, once through $W_{\text{strand}}$ in the mean and once through the extra term. Worse, if you
> instead take the mean over *boarded* students only, a schedule that strands the peak crowd scores
> a **better** mean wait than one that carries them — the objective would reward abandoning
> passengers. Charging every student one wait, with $W_{\text{strand}} = 60$ min stand-in for "gave
> up and walked", is monotone in stranding and has neither pathology. Report the stranded
> percentage as a *metric*, not as an objective term.

- The **box constraint** is enforced by **repair** (clamp to $[0,T]$, and zero the corresponding
  velocity component in PSO).
- The **fleet constraint** ($n(\tau) \le B$) is a soft **quadratic penalty** — zero when feasible,
  growing with the breach.
- The **trip-count constraint** (exactly $K$ trips) is structural: baked into $\dim \mathbf{x} = K$,
  so it cannot be violated.

Population-based solvers maximise, so use fitness $F(\mathbf{x}) = -J(\mathbf{x})$.

**Noise control.** Fix $M = 3$ arrival realisations (seeds 0, 1, 2) at the start of a run and define
$J$ as the mean over those three. $J$ is then a **deterministic** function of $\mathbf{x}$ — the
optimiser is not fighting objective noise. Report final schedules on **30 held-out realisations**
(seeds 100–129) to show the schedule generalises rather than overfitting three draws.

**Deployment metric.** Alongside the soft objective, report **service level** — the percentage of
students who wait $\le 10$ minutes — as the figure a transport office would actually recognise.

## A.7 Why the problem is hard (landscape characterisation)

$J$ is:

- **Non-differentiable and piecewise-constant.** Which departure a given student catches is a step
  function of $\mathbf{x}$; the left-behind count is $\max(0, |\text{queue}| - C)$ over integers; the
  stranded count is an integer. Between the boundaries where a student's boarding bus flips, $J$ is
  constant in $\mathbf{x}$ — so $\nabla J = 0$ almost everywhere and is undefined on the boundaries.
  There is nothing for a gradient method to descend.
- **Multimodal.** $K! = 14! \approx 8.7 \times 10^{10}$ symmetric relabellings of any optimum, plus
  the capacity cliffs (a departure moved one minute earlier can dump 40 students' worth of wait
  onto the next bus), plus the fleet-feasibility boundary carving the box into basins.
- **Black-box.** Evaluation requires the discrete-event simulation above. No closed form, no
  usable inverse; only sampling.

Discretising time to 1-minute slots and selecting $K$ of them is a covering/scheduling problem with
$\binom{780}{14} \approx 3 \times 10^{29}$ candidates, is NP-hard, and throws away the continuous
optimum anyway. Hence a **population-based metaheuristic** is the appropriate solver: each individual is a
complete candidate timetable in $\mathbb{R}^{14}$.

## A.8 Algorithms to implement (from scratch)

**Particle Swarm Optimization.** Standard velocity update
$$
v \leftarrow w\,v + c_1 r_1 (\text{pbest} - x) + c_2 r_2 (\text{gbest} - x), \qquad x \leftarrow x + v,
$$
with linearly decaying inertia $w: 0.9 \to 0.4$, $c_1 = c_2 = 1.49$, velocity clamped to
$\pm 0.2\,T$, boundary repair by clamping. Swarm size 30, 100 iterations
→ **3,030 fitness evaluations** (30 initial + 30 × 100).

**Genetic Algorithm**, real-coded, at the *identical* 3,030-evaluation budget: tournament selection
(size 3), BLX-$\alpha$ crossover ($\alpha = 0.5$), Gaussian mutation ($\sigma = 0.05\,T$, rate
$1/K$), elitism (keep top 2). This covers the GA lecture content and gives a genuine
cross-family comparison rather than a strawman.

**Fixed evaluation budget is the load-bearing experimental control.** Every method and every
ablation gets exactly 3,030 evaluations. A method that wins on more evaluations has not won.




## A.9 Baselines (same budget where applicable)

| Baseline | What it is |
|---|---|
| Uniform headway | $K$ departures evenly spaced across $[0,T]$ — current practice |
| Demand-proportional | departures at the $K$ quantiles of the cumulative arrival curve, so equal numbers of students arrive between consecutive buses — a genuinely smart heuristic, not a strawman |
| Random search | 3,030 uniform samples from the box |
| Best-of-$K$ restarts, local | hill-climb from random starts, same budget |

The demand-proportional baseline matters: if PSO cannot beat it, say so.

## A.10 Experiments

1. **Convergence.** Best-so-far fitness vs. evaluations, 30 seeds, median with IQR band. PSO vs GA
   vs random search vs the two heuristics.
2. **Does the swarm's communication do any work?** At the same 3,030-evaluation budget, compare:
   (a) 1 particle spending the whole budget alone; (b) 30 particles with $c_2 = 0$, i.e. thirty
   searchers that never share anything; (c) 30 particles sharing one `gbest`. If (b) is no better
   than random search, the population is not the mechanism — the *communication* is. State whichever
   way it falls.
3. **Topology.** Fully connected (`gbest`) vs ring ($k=1$) at the same budget, 30 paired seeds.
   Report mean *and* variance, and the iteration at which each stagnates. Expect the ring to be
   slower but steadier; verify rather than assert.
4. **Generalisation.** Optimised schedule scored on the 30 held-out arrival realisations, against
   the two heuristics. Does the advantage survive out of sample?
5. **Sensitivity.** Sweep $K \in \{10,\dots,18\}$ and $B \in \{2,3,4\}$; show where the fleet
   constraint starts binding and the timetable's shape changes qualitatively.

**Statistics.** 30 paired seeds throughout. Wilcoxon signed-rank for location, Levene or F-test for
variance. Report $p$-values. No claim of "better" without a test.

## A.11 Figures and tables to produce

- Timetable visualisation: cumulative arrival curve with departure times overlaid, bus-load bars,
  and left-behind counts — the picture that makes the solution legible.
- Convergence curves (median + IQR).
- Ablation table (A.10.2) and topology table (A.10.3).
- Fleet-occupancy plot $n(\tau)$ against $B$, showing feasibility.

---
---


---

# Deliverables and repo layout

```
3/
├── README.md                  # results-first summary, written last
├── PROBLEM_STATEMENT.md       # this file
├── pyproject.toml             # uv project, mirroring 1/ and 2/
├── run.sh                     # ./run.sh test | run | all
├── shuttle_timetable/
│   ├── instance.py            # λ(t), R(t), arrival sampling
│   ├── simulator.py           # simulation semantics and the objective
│   ├── pso.py                 # PSO from scratch
│   ├── ga.py                  # real-coded GA from scratch
│   └── baselines.py           # uniform, demand-proportional, random search
├── scripts/
│   └── run.py                 # A.10 (1)–(5)
├── docs/
│   └── algorithms.md
├── tests/
├── results/
│   ├── plots/
│   └── tables/
└── report/                    # LaTeX, from the instructor's template
```

## Acceptance tests (write these first)

- **Load factor $N / (K \cdot C)$ lies in $[0.6, 0.85]$.** If it exceeds 1 the instance is
  degenerate — every schedule strands students and the objective goes flat. Guard this first.
- A schedule of $K$ departures with all students boarding immediately has $\bar w \to 0$.
- With $C = 1$ and $K = 1$, exactly one student boards and $N - 1$ are stranded.
- The objective is invariant under permutation of $\mathbf{x}$ (the $K!$ symmetry is real).
- Moving one departure later never decreases the wait of a student who boarded it.
- Fleet penalty is exactly zero for a schedule with headway $> \max_t R(t)$.

## Report mapping (instructor's template)

`report/main.tex` §"Assignment 3" needs *Problem Formulation* / *Methodology* / *Results and
Discussion*. Sections A.1–A.7 above are the Problem Formulation; A.8–A.9 are the Methodology; the
experiments are the Results. The Literature Review section covering all assignments is separate —
cite the seminal PSO and GA papers plus the swarm-topology literature. Every update equation cites
its original paper.
