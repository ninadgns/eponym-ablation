# Assignment 4 — Formal Problem Statement

**Unit:** Decision making under uncertainty.
**Setting:** University of Dhaka campus mobility — controlling the signal at the campus-gate
intersection.

This is an online sequential control problem under uncertainty. Its companion assignment
(`../3/`, shuttle timetabling) is an offline design problem over a continuous decision vector;
the two are **separate problems sharing a setting**, and that distinction is the point.

---

## Notes for the implementing agent

Read these before writing code.

1. **Both algorithms are implemented from scratch.** No `stable-baselines`, no `mdptoolbox`.
   NumPy, SciPy (for statistics and sparse linear algebra only), pandas, and matplotlib.
   Implementing value iteration and Q-learning *is* the assignment.
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

# Part B — Adaptive Signal Control at the Campus-Gate Intersection

## B.1 Informal description

The intersection at a DU campus gate has two competing approaches: **A**, the main road, and **B**,
the campus/side road. One of them has green at any moment. Every tick, the controller either
**holds** the current green or **switches** it.

Three things make this a decision problem rather than a timer:

1. **Switching is not free.** The changeover burns a clearance interval in which *neither* approach
   discharges. Thrash the signal and the intersection loses capacity to lost time.
2. **Arrivals are random and unequal.** The main road is heavier, but the side road starves if it is
   never served — so a fixed rule that always favours A is not optimal either.
3. **Queues spill back.** A queue that exceeds the link's storage blocks upstream traffic, and that
   is far more costly than ordinary delay.

Holding green on a long queue that is already discharging can be better than switching to a shorter
one, *because* of the clearance you would pay and the arrivals about to land. That is lookahead,
and it is what a greedy rule cannot do.

This is a **finite, discounted, continuing Markov decision process**.

## B.2 Given data (problem instance)

| Symbol | Meaning | Value |
|---|---|---|
| $\Delta$ | tick length (seconds) | $10$ |
| $Q_{\max}$ | queue truncation per approach (vehicles) | $12$ |
| $\mu$ | saturation discharge per tick on the green approach (veh) | $5$ (≈1800 veh/h) |
| $e_{\min}$ | minimum green (ticks) — safety | $2$ (20 s) |
| $E_{\max}$ | maximum green (ticks) — anti-starvation | $6$ (60 s) |
| $A_{\max}$ | arrival truncation per approach per tick | $8$ |
| $\lambda_A(p)$ | mean arrivals/tick, main road | $2.0$ peak / $1.0$ off-peak / $0.3$ night |
| $\lambda_B(p)$ | mean arrivals/tick, side road | $1.0$ peak / $0.6$ off-peak / $0.15$ night |
| $c_{\text{switch}}$ | cost of a changeover (lost time), in queue-vehicle units | $3.0$ |
| $c_{\text{spill}}$ | cost per vehicle of spillback beyond $Q_{\max}$ | $8.0$ |
| $\gamma$ | discount factor | $0.99$ |

At peak the combined demand is $3.0$ veh/tick. A phase that holds ~4 ticks and then pays 1 clearance
tick discharges on 4 ticks in 5, giving an effective capacity of $\tfrac{4}{5}\mu = 4.0$ veh/tick —
a utilisation of $\rho \approx 0.75$. The intersection is **congested but stable**, which is exactly
the regime where control quality shows up. Verify this when you build
the instance: if it saturates, every policy spills and the MDP is degenerate; if it is empty, every
policy is optimal and the MDP is vacuous. Neither is a usable assignment.

## B.3 State and actions

$$
s = (q_A,\; q_B,\; \phi,\; e,\; p), \qquad
q_A, q_B \in \{0,\dots,Q_{\max}\},\;\;
\phi \in \{A, B\},\;\;
e \in \{0,\dots,E_{\max}\},\;\;
p \in \{\text{peak},\text{off},\text{night}\}
$$

- $q_A, q_B$ — queue lengths, truncated;
- $\phi$ — which approach currently holds green;
- $e$ — ticks elapsed in the current phase (capped);
- $p$ — traffic regime, a slowly-mixing exogenous Markov chain (mean dwell ≈ 1 h, i.e. transition
  probability ≈ $1/360$ per tick, cycling peak → off → night → peak).

$$
|\mathcal{S}| = 13 \times 13 \times 2 \times 7 \times 3 = 7{,}098 .
$$

Small enough to solve exactly; large enough that Q-learning must actually generalise its experience.

$$
\mathcal{A} = \{\textsf{hold},\; \textsf{switch}\}
$$

**Action availability is state-dependent:**

$$
\mathcal{A}(s) =
\begin{cases}
\{\textsf{hold}\} & e < e_{\min} \quad \text{(minimum green not yet served)}\\
\{\textsf{switch}\} & e = E_{\max} \quad \text{(maximum green reached)}\\
\{\textsf{hold},\;\textsf{switch}\} & \text{otherwise.}
\end{cases}
$$

This yields $11{,}154$ legal $(s,a)$ pairs out of $14{,}196$.

> **A point to get right, because it is easy to get wrong.** These masks are **hard engineering
> constraints** — minimum green is a pedestrian-safety rule, maximum green prevents side-road
> starvation. They **do** change $V^*$: they remove actions that are genuinely superior in the
> unconstrained problem but illegal in the real one. Do **not** argue that masking "cannot change
> the optimum because a duplicate action cannot change a max" — that argument applies to masking
> *no-op duplicates*, which is not what is happening here. The honest claim is: the masks define the
> feasible policy class, and we solve the constrained problem because the unconstrained solution is
> not deployable.

## B.4 Dynamics

Within a tick, in this order:

1. **Controller acts.** $a \in \mathcal{A}(s)$.
2. **Discharge.**
   - $a = \textsf{hold}$: the green approach $\phi$ discharges $d_\phi = \min(q_\phi, \mu)$ vehicles;
     the red approach discharges nothing. Phase timer $e' = \min(e+1, E_{\max})$, $\phi' = \phi$.
   - $a = \textsf{switch}$: **neither approach discharges this tick** (the clearance interval).
     $\phi' = \lnot\phi$, $e' = 0$.
3. **Arrivals.** $\text{arr}_A \sim \text{Poisson}(\lambda_A(p))$ and
   $\text{arr}_B \sim \text{Poisson}(\lambda_B(p))$, independent, each truncated to
   $\{0,\dots,A_{\max}\}$ and **renormalised** (not clipped — clipping puts spurious mass on the
   endpoint and biases the model).
4. **Queue update, with spillback.** For each approach $z \in \{A, B\}$:
   $$
   \tilde q_z = q_z - d_z + \text{arr}_z, \qquad
   q'_z = \min(\tilde q_z,\; Q_{\max}), \qquad
   \text{spill}_z = \max(0,\; \tilde q_z - Q_{\max}).
   $$
5. **Regime.** $p' \sim P(\cdot \mid p)$, independent of everything else.

The hour does not appear in the state and the task never terminates: this is a **continuing** MDP.

## B.5 Reward, and a subtlety that matters

$$
r(s, a, w) = -\Bigl(\; \underbrace{(q_A + q_B)}_{\text{delay, by Little's law}}
\;+\; c_{\text{switch}}\,\mathbb{1}[a = \textsf{switch}]
\;+\; c_{\text{spill}}\,(\text{spill}_A + \text{spill}_B) \;\Bigr),
\qquad w = (\text{arr}_A, \text{arr}_B, p').
$$

Delay is charged on the queue **entering** the tick: summing $q$ over ticks and multiplying by
$\Delta$ is total vehicle-delay, which is the quantity a traffic engineer reports.

**The subtlety.** The realised spillback depends on the realised arrivals, and **the arrivals are
not recoverable from $s'$** — once a queue is truncated at $Q_{\max}$, $q'_z = Q_{\max}$ tells you
nothing about how many vehicles were turned away. So $r$ is **not** a function of $(s, a, s')$.

This is the standard *disturbance* form of an MDP, not a defect. Value Iteration only ever needs the
**expected** reward,
$$
R(s,a) = \mathbb{E}_w\bigl[r(s,a,w)\bigr]
= -\Bigl( (q_A + q_B) + c_{\text{switch}}\mathbb{1}[a=\textsf{switch}]
+ c_{\text{spill}} \textstyle\sum_{z}\sum_{k} \Pr[\text{arr}_z = k]\,\max(0,\, q_z - d_z + k - Q_{\max}) \Bigr),
$$
which you tabulate once; the Bellman operator remains a $\gamma$-contraction. **Q-learning must
consume the realised $r$**, including the realised spill. Feeding it $\mathbb{E}[\text{spill}]$ would
leak the arrival distribution into a supposedly model-free agent and destroy the entire comparison
the assignment exists to make.

## B.6 The optimization problem

Find a stationary policy $\pi : \mathcal{S} \to \mathcal{A}$ with $\pi(s) \in \mathcal{A}(s)$
maximising
$$
\boxed{\;
V^\pi(s) = \mathbb{E}\Bigl[\; \textstyle\sum_{k=0}^{\infty} \gamma^k\, r_{t+k} \;\Big|\; s_t = s,\; \pi \Bigr]
\;}
$$
satisfying the Bellman optimality equation
$$
V^*(s) = \max_{a \in \mathcal{A}(s)} \Bigl[\, R(s,a) + \gamma \textstyle\sum_{s'} P(s' \mid s, a)\, V^*(s') \,\Bigr].
$$

$\gamma = 0.99$ at $\Delta = 10$ s gives an effective horizon $1/(1-\gamma) = 100$ ticks $\approx$
17 minutes — long enough to span several signal cycles and a queue build-and-clear, which is the
timescale the problem's structure lives on. State this justification; do not leave $\gamma$
unmotivated. Note honestly that the regime $p$ mixes on the scale of an hour, far beyond this
horizon, so $p$ acts as a quasi-static context rather than something the agent anticipates — the
lookahead value in this problem comes from the clearance cost and queue growth *within* a regime.

## B.7 Algorithms to implement (from scratch)

**Value Iteration** (model-based, exact). Tabulate $R(s,a)$ and sparse $P(s'|s,a)$. Iterate to
$\lVert V_{k+1} - V_k \rVert_\infty < \epsilon(1-\gamma)/(2\gamma)$. Then:
- report the **empirical contraction rate** and check it against the theoretical $\gamma$;
- **cross-check** the greedy policy by solving $(I - \gamma P_\pi)V^\pi = R_\pi$ exactly (a sparse
  linear solve) and confirming $\lVert V^* - V^{\pi^*}\rVert_\infty \approx 0$.

**Q-learning** (model-free). $\epsilon$-greedy over the *legal* action set, $\epsilon$ decaying to a
floor; polynomial learning rate $\alpha_n = (1+n)^{-0.7}$ per state-action visit count; exploring
starts. Episodes of fixed length (e.g. 24 h of ticks). Track state-action coverage — a claim about
Q-learning's performance is worthless if it never visited half the legal pairs.

**Certainty-equivalence VI** (the arm that can falsify the obvious thesis). Estimate $\hat{P}$ and
$\hat{R}$ from **Q-learning's own samples**, then plan on the estimated model. If a learned model
beats the learner on identical data, then "model-free wins when the model is wrong" is the wrong
lesson — and you must report that.

## B.8 Baselines

| Policy | What it knows |
|---|---|
| Value Iteration (optimal) | the exact $P$ |
| Fixed-time (Webster-style) | offline cycle and split from mean arrival rates; ignores the queue entirely — this is what is actually deployed |
| Longest-queue-first greedy | switches whenever the red queue exceeds the green queue, subject to min-green |
| Myopic greedy ($\gamma = 0$) | the formally-correct one-step-optimal policy |
| Random (legal actions) | nothing |
| Always hold | nothing |

The fixed-time and longest-queue baselines are the ones that matter. If neither is beaten by a
meaningful margin, the MDP is vacuous and the instance needs rethinking — report that rather than
hiding it.

> **A result you get for free, and should not waste.** In this MDP,
> $R(s, \textsf{hold}) \ge R(s, \textsf{switch})$ in **every** state. The delay term $(q_A + q_B)$
> is charged on the queue *entering* the tick, so it is identical under both actions; holding
> discharges the green approach, which weakly reduces expected spillback; and switching pays
> $c_{\text{switch}} \ge 0$ on top. Holding therefore dominates on immediate reward — **and this
> holds even at $c_{\text{switch}} = 0$.**
>
> Two consequences. First, the myopic ($\gamma = 0$) policy **provably never switches** unless
> maximum green forces it, so it collapses onto *always-hold*; keep both rows in the table but say
> that they coincide, and say why. Second — and this is the one to lead with — **every voluntary
> switch in the optimal policy is purely anticipatory**, paid for entirely out of discounted future
> value, because the immediate reward never once argues for it. That is a far stronger claim than
> "lookahead helps by $x\%$", it is provable rather than merely measured, and it is exactly what
> experiment B.10.6 is trying to demonstrate empirically. It was found by a test
> (`test_myopic_greedy_never_switches_voluntarily`), not by inspection.

## B.9 Evaluation protocol

**Score every policy the same way: exactly.** For each $\pi$, solve
$(I - \gamma P_\pi) V^\pi = R_\pi$ under the **true** model. This removes Monte-Carlo noise from the
comparison completely — policies are ranked by exact expected return, not by which one drew a
luckier rollout.

Report **regret** $= 100 \cdot (V^* - V^\pi) / |V^*|$, plus rollout statistics that a traffic
engineer would recognise: mean delay per vehicle (s), throughput (veh/h), switches per hour,
spillback events per hour, and 95th-percentile queue.

## B.10 Experiments

1. **VI correctness.** Convergence, empirical contraction rate vs $\gamma$, exact-linear-solve
   cross-check.
2. **Does this problem actually need lookahead?** Full baseline table by exact regret. The myopic
   greedy policy is the key comparison: it is *formally correct* for $\gamma = 0$, so if it loses
   badly, the loss is attributable to lookahead and nothing else.
3. **Learning from experience.** Q-learning vs certainty-equivalence VI on the *same* sample budget,
   5 seeds. Report coverage, regret with error bars, and per-state action-optimality of the learned
   policy.
4. **How wrong can the model be?** Sweep the assumed arrival rates by a factor of $0.25 \dots 2.0$
   against the truth, and plan with the wrong model. The question — *how wrong must a model be before
   learning from scratch beats planning with it?* — is the scientific core of Part B. Dhaka does not
   publish intersection arrival rates, so this is not hypothetical.
5. **Hyperparameter sweep.** Learning-rate schedule (polynomial vs constant), $\epsilon$ floor,
   optimistic vs pessimistic $Q$ initialisation, $\gamma$. Report means and standard deviations.
6. **Evidence of anticipation.** Pick states where the optimal action is counter-intuitive — e.g.
   holding green on the main road while the side queue is longer — and decompose
   $Q(s,\textsf{hold}) - Q(s,\textsf{switch})$ into its immediate-reward and discounted-future
   components. Show that the future term is what flips the decision.

## B.11 Figures and tables to produce

- Baseline regret table (B.8).
- Policy slice heatmaps: optimal action over $(q_A, q_B)$ for fixed $(\phi, e, p)$ — the picture that
  shows the switching curve, and how it moves between peak and night.
- Q-learning learning curves (regret vs samples), with certainty-equivalence overlaid.
- Model-misspecification curve (regret vs assumed-rate multiplier), with the Q-learning regret drawn
  as a horizontal line — the crossing point *is* the answer to experiment 4.

---
---


---

# Deliverables and repo layout

```
4/
├── README.md                  # results-first summary, written last
├── PROBLEM_STATEMENT.md       # this file
├── pyproject.toml             # uv project, mirroring 1/, 2/ and 3/
├── run.sh                     # ./run.sh test | run | all
├── signal_control/
│   ├── mdp.py                 # state space, P, R, simulator
│   ├── value_iteration.py     # VI + certainty equivalence
│   ├── q_learning.py          # Q-learning from scratch
│   ├── baselines.py           # fixed-time, longest-queue-first, myopic greedy
│   └── evaluation.py          # exact policy evaluation, regret, rollout stats
├── scripts/
│   └── run.py                 # B.10 (1)–(6)
├── docs/
│   └── algorithms.md
├── tests/
├── results/
│   ├── plots/
│   └── tables/
└── report/                    # LaTeX, from the instructor's template
```

## Acceptance tests (write these first)

- Rows of $P(\cdot \mid s,a)$ sum to 1 for every legal $(s,a)$.
- Truncated Poisson pmf sums to 1 (renormalised, not clipped).
- `switch` is absent from $\mathcal{A}(s)$ iff $e < e_{\min}$; `hold` absent iff $e = E_{\max}$.
- VI's greedy policy value equals the exact linear-solve value to $< 10^{-6}$.
- With $c_{\text{switch}} = 0$ and $e_{\min} = 0$, the optimal policy reduces to serving the longer
  queue — a known-answer sanity check on the whole pipeline.
- With $\gamma = 0$, VI's policy equals the myopic greedy baseline exactly.

## Report mapping (instructor's template)

`report/main.tex` §"Assignment 4" needs *Problem Formulation* / *Methodology* / *Results and
Discussion*. Sections B.1–B.6 above are the Problem Formulation; B.7–B.9 are the Methodology; the
experiments are the Results. The Literature Review section covering all assignments is separate —
cite the standard MDP/Q-learning sources and the adaptive-signal-control literature. Every update
equation cites its original paper.
