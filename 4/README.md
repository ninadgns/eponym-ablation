# Assignment 4 — Decision Making Under Uncertainty

**University of Dhaka campus mobility: when should the campus-gate signal switch, given random
arrivals?** Value Iteration and Q-learning, both from scratch.

| | |
|---|---|
| **Unit** | Decision making under uncertainty |
| **Problem** | Online sequential control of a signalised intersection modelled as an MDP |
| **Method** | Value Iteration + Q-learning, from scratch |
| **Code** | [`signal_control/`](signal_control/) |

The companion assignment, [`../3/`](../3/), is the population-based-search half of the same setting
(shuttle timetabling). The two are separate problems sharing a setting.

The formal statement is in **[`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md)**. Read it before
touching the code — it is where the parameters are defended, and the viva is on the defence, not
the code.

## Usage

```bash
./run.sh test        # pytest — 28 tests, no skips
./run.sh run         # signal control experiments -> results/   (~10 min)
./run.sh all         # test + run
```

The driver fans its seeds out across processes (`--jobs`, default 8). Results are keyed by seed,
so the numbers do not depend on the order the workers finish in.

Every experiment takes an explicit seed and reproduces bit-for-bit on re-run.

---

## The headline results

*Written from the numbers, after the fact. The central one disconfirms the conclusion the assignment
led us to expect — a model-based planner beats the model-free learner on the learner's own data.
That is the interesting one, and it is stated plainly rather than buried.*

### The model matters more than the learning, and that is the disconfirmation

**Value Iteration is correct.** 2,528 sweeps to $\epsilon = 10^{-8}$; empirical contraction rate
**0.9898** against the theoretical $\gamma = 0.99$; and $V^*$ agrees with the exact linear solve
$(I - \gamma P_\pi)V^\pi = R_\pi$ of its own greedy policy to **5 × 10⁻⁹**.

**Lookahead is worth a lot, and every bit of it is anticipation.** Exact regret under the true model:

| Policy | regret | switches/h | mean delay |
|---|---|---|---|
| Value Iteration (optimal) | 0 % | 84.2 | 26.9 s/veh |
| Longest-queue-first | 9.9 % | 97.0 | 29.0 s/veh |
| Fixed-time (Webster-style) | 18.1 % | 90.0 | 31.7 s/veh |
| Myopic greedy ($\gamma = 0$) | 45.9 % | 51.4 | 40.8 s/veh |
| Always hold | 45.9 % | 51.4 | 40.8 s/veh |
| Random (legal) | 43.8 % | 97.6 | 38.1 s/veh |

Myopic greedy and always-hold are **the same policy**, exactly as proved in
`signal_control/baselines.py`: $R(s,\textsf{hold}) \ge R(s,\textsf{switch})$ in every one of the
4,056 states that have a real choice — the minimum immediate advantage of holding is $+3.000$,
precisely $c_{\text{switch}}$. The immediate reward **never once** argues for switching. Yet the
optimal policy switches voluntarily in **650 states**, and in every one of them the discounted-future
term is what flips the sign. Every voluntary switch in this MDP is bought entirely out of future
value. That is not measured, it is proved, and B.10.6 shows the decomposition.

The mirror image is where longest-queue-first loses its 9.9 %: in **1,230 states the optimal policy
holds the green while the *red* queue is longer** — the reactive rule switches in every one of them
and is wrong every time (`results/tables/B6_counterintuitive_holds.csv`). At $(q_A, q_B) = (11, 12)$
with green on A in peak, LQF sees the side road one vehicle worse off and switches; $V^*$ holds,
because the main road is mid-discharge and the clearance tick would throw away five vehicles of
capacity to save one vehicle of queue. The switching curve is emphatically **not** the diagonal.

**The falsification.** §B.7 warned that certainty-equivalence is "the arm that can falsify the
obvious thesis", and it did. On **identical data** — same seeds, same 400,000 transitions, same
order — planning on the learned model beats learning the values directly:

| Method (400k samples, 5 seeds) | regret | action-optimality |
|---|---|---|
| Q-learning (default hyperparameters) | 12.49 % ± 0.75 | 65.7 % |
| Q-learning (tuned by the B.10.5 sweep) | 2.72 % ± 0.23 | — |
| Certainty-equivalence VI (same data) | **1.74 % ± 0.41** | 89.9 % |

The tuned row is load-bearing and is the reason to distrust the naive version of this result. The
B.10.5 sweep shows the *default* configuration is far from the learner's best — initialising
$Q_0 \approx \bar{V}^*$ instead of the optimistic $Q_0 = 0$, together with a step exponent of 1.0
instead of 0.7, takes regret from 12.5 % to 2.7 % at the same budget (of which the initialisation
is the larger share: 12.6 % → 5.5 % on its own, in the sweep) — so every comparison below is run
against the **tuned** learner. Beating a handicapped Q-learner would have proved nothing about
models, and it would have inflated CE's margin from "modest but real" (1.74 vs 2.72) to a spurious
sevenfold. The honest claim is the smaller one:
**on identical data, planning on the estimated model still beats learning the values directly** —
CE converts the same 400k transitions into 90 % action-optimality against the learner's 66 %. The
reason is where the sample budget goes. Q-learning spends each transition on **one** Bellman backup
and can only propagate value backwards along trajectories it actually walked. CE spends each
transition on $\hat{P}$ and $\hat{R}$, and then plans to convergence on the estimate for free —
thousands of sweeps, every state, no further samples. Once the model is roughly right, extra
planning is cheap and extra experience is not. Coverage was 68.1 % of the 11,154 legal pairs, so
this is a claim about a learner that saw two thirds of its problem, not one reporting its own
initialisation.

And so the answer to §B.10.4 — *how wrong must a model be before learning from scratch beats
planning with it?* — is sharp, and it has a genuine crossing point:

> **The model has to be wrong about whether traffic exists at all.**

Planning with arrival rates wrong by a factor of **10 too small or 2 too large** still beats a tuned
Q-learner trained on 400,000 real transitions (regret ≤ 1.8 % against its 2.7 %). The *only* model
in the sweep that loses to learning from scratch is $\lambda \times 0$ — a planner that believes no
vehicle ever arrives (3.58 %). The crossing point therefore sits between multipliers 0.0 and 0.1,
which is to say: nowhere a traffic engineer could plausibly land.

The reason is structural. The optimal switching curve is driven by the queue dynamics, the discharge
rate $\mu$, and the clearance cost $c_{\text{switch}}$ — all of which the misspecified planner still
has exactly right. The arrival rates only modulate *how far ahead* it is worth anticipating. Getting
the structure right is what matters; getting the rates right barely does. That is a useful thing to
be able to tell a transport office that has no arrival data and never will.

The sweep also confirms what the theory demands: the polynomial step size $\alpha_n = (1+n)^{-p}$
converges and a **constant $\alpha$ does not** (24.5 % and 21.8 % regret at $\alpha$ = 0.1 and 0.5,
against 12.6 % for the baseline schedule), and removing exploring starts collapses coverage from
52 % to 34 % and degrades regret with it. The single largest lever, though, was not the step size
but the **initialisation**: $Q_0 = 0$ is *optimistic* here (every reward is negative, so $V^* < 0$
everywhere), and the optimism costs far more than it buys.

---

## What is where

| Module | State |
|---|---|
| `signal_control/mdp.py`, `evaluation.py`, `baselines.py` | done, tested |
| `signal_control/value_iteration.py` | **done** — VI + certainty-equivalence planning |
| `signal_control/q_learning.py` | **done** — model-free, never touches $P$ or $R$ |

`tests/test_algorithms.py` is the contract for the two graded algorithms. It is not weakened
anywhere: `test_q_learning_never_touches_the_model` greps the learner's own source for `.P` and
`.R`, and it passes because the learner really does see nothing but `mdp.step()`.

Results land in `results/tables/*.csv` and `results/plots/*.png`.
