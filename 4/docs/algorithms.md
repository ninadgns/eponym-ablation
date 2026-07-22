# Part B — Algorithms Explained (plain-English version)

Part B is about controlling **one traffic light** at a campus gate. Two roads meet there: road **A**
(the busy main road) and road **B** (the quieter side road). Only one of them can have a green light
at a time. Every 10 seconds the controller makes one decision: **hold** the current green, or
**switch** it to the other road.

The goal is to keep total waiting cars as low as possible over the long run. That's harder than it
sounds, because:

- **Switching wastes time.** When the light changes there's a moment where *both* roads are red (cars
  clearing the intersection). Flip the light too often and you throw away capacity.
- **Traffic is random and lopsided.** The main road is busier, but if you *always* favor it, the side
  road never clears.
- **Queues can overflow.** If a queue gets too long it blocks traffic further back, which is very
  expensive.

So sometimes the smart move is to **keep** a road green even when the other road's queue looks longer
— because you'd waste time switching, and more cars are about to arrive anyway. Planning around the
future like that is the whole point. A dumb "just serve the longest queue right now" rule can't do it.

---

## First, the vocabulary (read this once)

Everything below is built on a handful of terms. Here they are in plain language, using the traffic
light as the example.

**MDP (Markov Decision Process).** A formal way to describe *any* "make a decision, the world
reacts, repeat" problem. It has four pieces:

- **State** — a snapshot of everything you need to know to decide right now. Here the state is
  `(qA, qB, phi, e, p)`:
  - `qA`, `qB` — how many cars are waiting on each road (capped at 12),
  - `phi` — which road currently has the green light (A or B),
  - `e` — how many ticks the current green has been on (so we can enforce min/max green times),
  - `p` — the traffic regime: `peak`, `off-peak`, or `night` (rush hour vs. 3am).

  There are 7,098 possible states in total — small enough to handle exactly, big enough to be
  interesting.

- **Action** — what you can do: `hold` or `switch`. Sometimes your hand is forced (see "masks"
  below).

- **Transition (the "dynamics")** — given a state and an action, what state comes next. It's *random*:
  new cars arrive according to chance, so the same action from the same state can lead to different
  next states. The rules for this live in the simulator.

- **Reward** — a score you get each tick telling you how good/bad that tick was. Here it's *negative*
  (a penalty): you lose points for every waiting car, extra points every time you switch, and a big
  penalty if a queue overflows. Since it's always negative, "best" means "least bad."

**"Markov"** just means: the next state depends only on the *current* state and action, not on the
whole history of how you got here. The current snapshot is enough. That's what makes the state
definition above complete.

**Policy** (written $\pi$) — a strategy: a rule that says, for *every* possible state, which action to
take. "If road A has 8 cars, road B has 3, and it's peak hour, then hold." A policy is the *answer*
we're solving for — the finished traffic-control brain.

**Discount factor** ($\gamma$, gamma = 0.99). The problem never ends (traffic runs forever), so if we
just added up all future rewards we'd get infinity and couldn't compare policies. Instead we make
future rewards count slightly less than immediate ones: a reward $k$ ticks from now is multiplied by
$0.99^k$. With $\gamma = 0.99$ this means the controller effectively "looks ahead" about 100 ticks
(~17 minutes) — long enough to plan across a few light cycles, which is the timescale that matters.

**Value** (written $V$). The value of a state under a policy is the *total discounted reward you
expect to collect* if you start in that state and follow the policy forever. High value = good state
to be in. `V*` (V-star) is the value under the *best possible* policy. Finding `V*` basically means
solving the problem.

**Q-value** (written $Q(s,a)$). Almost the same as value, but it also fixes the *first* action: "if
I'm in state `s`, take action `a` right now, and then behave optimally after that — how good is that?"
Once you know the Q-values, the best policy is just "in each state, pick the action with the highest
Q." This is the star of Q-learning below.

**The Bellman equation.** The single idea the whole field rests on. It says the value of a state
equals *the reward you get now* plus *the discounted value of wherever you land next*, assuming you
act optimally:

> value of a state = best possible (immediate reward + 0.99 × value of the next state)

It's recursive — value defined in terms of value — but that's exactly what lets a computer solve it by
repeating the update until the numbers stop changing.

**Model-based vs. model-free.** This distinction is the *entire scientific point* of Part B:

- **Model-based** = you *know the rules of the world* — the exact probabilities of what happens next
  and the exact expected rewards. With that knowledge you can compute the perfect policy by pure math,
  without ever "trying" anything. Value Iteration is model-based.
- **Model-free** = you *don't* know the rules. You just interact with the world, observe what actually
  happens, and learn from experience — like learning to drive by driving, not by reading the physics
  of the engine. Q-learning is model-free.

The big question Part B asks: **when is it better to learn from scratch (model-free) than to plan
using a model that might be wrong?**

**Regret.** How much worse a policy is than the best possible one, as a percentage. Regret = 0 means
you found the optimal policy. Higher = worse. It's how we grade every policy.

**Action masks (min-green / max-green).** Real safety rules, not math tricks:
- **Minimum green** — once a light turns green it must stay green a little while (pedestrians are
  crossing). So `switch` is *illegal* for the first couple of ticks.
- **Maximum green** — a light can't stay green forever or the other road starves. So past a limit,
  `hold` becomes *illegal* and you're *forced* to switch.

These masks shrink the set of legal actions in certain states. The code respects them everywhere.

---

Now the three algorithms. They're three different ways to arrive at a good policy: one that **knows
the rules**, one that **knows nothing and learns**, and one that **learns an estimate of the rules and
then plans with it**.

Files: `signal_control/value_iteration.py`, `q_learning.py`, `baselines.py`, `evaluation.py`.

---

## 1. Value Iteration — the "I know the rules, let me calculate" method

File: `value_iteration.py`. This is **model-based and exact**: we hand it the true dynamics and
rewards, and it computes the perfect policy by repeatedly applying the Bellman equation.

**How it works, step by step:**

1. Start with a guess for the value of every state (all zeros).
2. Do one "sweep": for every state, update its value using the Bellman rule — reward now, plus
   0.99 × the (probability-weighted average) value of where you might land next, picking whichever
   action gives the best result (`value_iteration.py:69`).
3. Repeat. Each sweep the values get closer to the true `V*`.
4. Stop when the values barely change between sweeps (`value_iteration.py:79`). There's a precise
   math rule for "barely" (`value_iteration.py:56`) that guarantees you're within a tiny tolerance of
   the exact answer.
5. Read off the policy: in each state, the action the Bellman step preferred (`value_iteration.py:83`).

**Why it needs the model.** Look at step 2 — "the average value of where you might land next" requires
knowing the *probabilities* of each next state. That's the model. No model, no averaging, no Value
Iteration. This dependence is the whole reason Part B contrasts it with Q-learning.

**Two details worth knowing:**

- **Illegal actions are blocked with negative infinity** (`value_iteration.py:61`). An illegal action
  looks, to the raw math, like a "do nothing, score 0" option — and since every real reward here is
  negative, the algorithm would *love* that fake zero and pick it. Setting illegal actions to $-\infty$
  makes sure they're never chosen.
- **A free correctness check** (`value_iteration.py:72`). Theory predicts the values should shrink
  toward the answer at a rate of exactly 0.99 (the discount factor) per sweep. The code measures the
  actual shrink rate; if it lands on 0.99, that's strong evidence the whole thing is implemented
  correctly. (This is experiment B.10.1.)

---

## 2. Q-learning — the "I know nothing, let me learn by trying" method

File: `q_learning.py`. This is **model-free**: the agent is *not* told the probabilities or rewards.
It just acts, sees what happens, and slowly builds up its Q-values from experience (Watkins & Dayan,
1992).

**How it works:** the agent keeps a big table of Q-values, one per (state, action) pair, all starting
at zero. Then it lives through millions of simulated ticks. On each tick:

1. From its current state, pick an action (mostly the best-known one, sometimes a random one — see
   "explore vs. exploit" below).
2. Take the action; the simulator returns the *actual* next state and the *actual* reward
   (`q_learning.py:95`).
3. Nudge the Q-value for what it just did toward "reward I just got + 0.99 × best Q-value available in
   the new state" (`q_learning.py:102-103`). Over millions of nudges, the table converges to the
   truth.

**The one rule that must not be broken:** the agent may only use the *realized* reward it actually
observed — including the actual overflow that happened (`q_learning.py:102`). If we secretly fed it the
*expected* (average) overflow, we'd be leaking knowledge of the traffic model into a method that's
supposed to know nothing — and the whole "model-free vs. model-based" experiment would be a lie.
There's literally a test that scans this function's code to make sure it never peeks at the model.

**A few knobs, in plain terms:**

- **Explore vs. exploit** (`q_learning.py:85-93`). If the agent always did what it currently thinks is
  best, it'd never discover better options. So with probability $\epsilon$ ("epsilon") it picks a
  random legal action instead. $\epsilon$ starts high (explore a lot early) and shrinks to a small
  floor (mostly exploit later). This is called $\epsilon$-greedy.
- **Learning rate** ($\alpha$, "alpha") (`q_learning.py:99`). How big each nudge is. It shrinks the
  more often a (state, action) pair has been visited — big adjustments when you're new to a situation,
  fine-tuning once you've seen it many times. There's a mathematical reason it must shrink this
  particular way to be guaranteed to converge; a *constant* nudge size never settles down (the code
  can demonstrate this failure on purpose in the B.10.5 experiment).
- **Exploring starts** (`q_learning.py:82`). Each training episode begins in a *random* state, so the
  agent is forced to experience the whole map — not just the comfortable states its current strategy
  likes. Otherwise huge parts of the state space go unvisited and its policy there is garbage.
- **Coverage** (`q_learning.py:112`). The code reports what fraction of legal (state, action) pairs
  the agent actually tried. If it never visited half of them, any claim about how good it is would be
  meaningless — so we measure and report it honestly.

The agent also **records every experience** it had — the full stream of (state, action, next state,
reward) — because the third algorithm needs that data.

---

## 3. Certainty-equivalence VI — the "let me learn the rules from experience, then calculate" method

File: `value_iteration.py`, function `certainty_equivalence` (line 93). This is the clever middle
ground, and it exists to **challenge the naive assumption** that model-free is automatically the way
to go when you don't start with a model.

**The idea:** take Q-learning's recorded experiences and use them to *estimate* the model —
count how often each transition happened to guess the probabilities, average the observed rewards to
guess the rewards (`value_iteration.py:138-147`). Then run ordinary Value Iteration on that
*estimated* model.

**Why it's a fair fight:** it's fed *exactly* the same experiences Q-learning had — same data, same
order, same random seed (`value_iteration.py:100-102`). So if it does better than Q-learning, the
difference is purely about *what each method does with the same data*, not about one getting more
practice. If planning-with-an-estimated-model beats learning-from-scratch on identical data, then
"just learn model-free" was the wrong lesson — and Part B reports that honestly (experiment B.10.3).

**One honest judgment call** (`value_iteration.py:149`): what should the estimated model assume about
(state, action) pairs the agent *never tried*? The code assumes the worst — that an untried action is
probably bad — so the planner won't blindly reach for options there's no evidence about. This is a
deliberate, stated choice ("pessimism"); other choices would quietly turn it into a different
experiment.

---

## 4. The baselines — simple rules to beat

File: `baselines.py`. These are the "is our fancy method actually worth it?" comparison points from
§B.8. If the smart methods can't beat these, the problem wasn't worth solving and we say so.

- **Fixed-time (Webster-style)** (`baselines.py:77`) — a dumb timer. Green for a fixed number of
  ticks based on average demand, *completely ignoring how many cars are actually waiting*. This is
  what's really installed at most intersections today, which is exactly why it's the one to beat.
- **Longest-queue-first** (`baselines.py:47`) — the obvious "smart" reflex: switch to whichever road
  has more cars right now. Reactive, no lookahead.
- **Myopic greedy** (`baselines.py:61`) — picks whatever action looks best *for the next tick only*
  (i.e. pretends $\gamma = 0$, no future). This is the key comparison, explained next.
- **Random** and **always-hold** — the "does anything at all beat noise?" controls.

**The neat result to lead with** (`baselines.py:61-72`): in *this* problem, `hold` always scores at
least as well as `switch` for the *immediate* tick — switching costs time and never helps right now.
So the myopic (next-tick-only) policy **never chooses to switch** unless forced by the max-green rule;
it collapses into "always hold." The powerful conclusion: **every deliberate switch the optimal policy
makes is purely about the future** — it's paid for entirely by better outcomes later, never by any
immediate gain. That's not just measured, it's provable, and it's the cleanest evidence that this
problem genuinely needs lookahead (experiment B.10.6).

---

## 5. The grader — `evaluation.py`

Every policy — from Value Iteration, Q-learning, the baselines, all of them — is scored **the exact
same way** so comparisons are fair (§B.9).

Instead of running each policy in the simulator and seeing who got luckier, the code solves a single
equation that gives each policy's *exact* expected value under the true model (`evaluation.py:25`).
No luck, no noise — a policy's score is its true long-run quality, period. From that we compute
**regret** (how far from optimal, as a %) (`evaluation.py:34`).

Alongside regret, it also reports the numbers a traffic engineer actually cares about
(`evaluation.py:40`): average delay per car, cars-per-hour throughput, how often the light switches,
how often queues overflow, and the 95th-percentile queue length. Those are the human-readable
consequences; regret is the rigorous score.

---

## The big picture

Three ways to get a traffic-control strategy: **know the rules and calculate** (Value Iteration),
**know nothing and learn by doing** (Q-learning), and **learn the rules from experience, then
calculate** (certainty-equivalence). Part B's real question is the crossover between them: *how wrong
does a model have to be before learning from scratch beats planning with it?* — a real question,
because nobody publishes actual Dhaka intersection traffic rates.

Two disciplines make the comparison trustworthy and the code enforces both mechanically: Q-learning
must **never** secretly peek at the true rules (a test checks this), and **every** policy is graded by
the same exact calculation — so any difference we measure is a difference of *method*, never of luck
or of who got more data.
