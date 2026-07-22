"""Q-learning — from scratch (PROBLEM_STATEMENT.md §B.7). Watkins & Dayan (1992).

Model-free: the agent gets `mdp.step()` and nothing else — one (s', r) at a time, no
probabilities, no expected-reward table. The realised reward it consumes carries the realised
spillback, which is exactly right: feeding it the *expected* spill would leak the arrival
distribution into a supposedly model-free agent and destroy the entire comparison Part B exists
to make. `test_q_learning_never_touches_the_model` greps this function's source to keep it honest.

    Q(s,a) <- Q(s,a) + alpha_n * [ r + gamma * max_{a' in A(s')} Q(s',a') - Q(s,a) ]

  * alpha_n = (1 + n(s,a))^-alpha_power, a polynomial schedule on the per-pair visit count. It
    satisfies the Robbins-Monro conditions (sum alpha = inf, sum alpha^2 < inf) for
    alpha_power in (0.5, 1]; a constant alpha does not, and will not converge — pass `alpha_const`
    to demonstrate that in the B.10.5 sweep rather than asserting it.
  * epsilon-greedy over the LEGAL action set only, decayed linearly to a floor.
  * Exploring starts: every episode begins at a uniformly drawn state. Without them the agent
    only ever sees the states its own policy likes, coverage collapses, and the regret number
    stops meaning anything.
  * The max over a' is taken over LEGAL actions in s'. Illegal entries are held at -inf for the
    whole run, so they can never be selected, never enter a bootstrap target, and never quietly
    inflate a Q value.

The action masks (min-green, max-green) are NOT model knowledge — they are the hard engineering
constraints of the intersection, known to any controller that is allowed to operate it. The
dynamics and the rewards are the model, and the agent never sees them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from signal_control.evaluation import exact_policy_value, regret
from signal_control.mdp import SignalMDP
from signal_control.value_iteration import value_iteration


@dataclass(frozen=True)
class QLearningResult:
    q: np.ndarray  # shape (nS, nA)
    policy: np.ndarray  # greedy w.r.t. q, legal by construction
    coverage: float  # fraction of LEGAL (s,a) pairs visited at least once
    n_steps: int
    transitions: list[tuple[int, int, int, float]]  # (s, a, s', r) — feed to certainty_equivalence
    curve: np.ndarray  # greedy-policy regret sampled during training, for the learning curve


def q_learning(
    mdp: SignalMDP,
    rng: np.random.Generator,
    n_episodes: int = 30_000,
    episode_len: int = 24 * 360,  # 24 h of 10-second ticks
    alpha_power: float = 0.7,
    eps_start: float = 1.0,
    eps_floor: float = 0.05,
    q_init: float = 0.0,
    exploring_starts: bool = True,
    eval_every: int | None = None,
    alpha_const: float | None = None,
) -> QLearningResult:
    gamma = mdp.cfg.gamma
    n_s, n_a = mdp.nS, mdp.nA
    legal = np.asarray(mdp.legal)  # the min/max-green masks: engineering, not dynamics

    q = np.full((n_s, n_a), float(q_init))
    q[~legal] = -np.inf  # an illegal action is never selected and never bootstrapped through
    visits = np.zeros((n_s, n_a), dtype=np.int64)

    legal_actions = [np.flatnonzero(legal[s]) for s in range(n_s)]

    total_steps = n_episodes * episode_len
    decay_steps = max(int(0.8 * total_steps), 1)  # reach the floor at 80% of training, then hold
    transitions: list[tuple[int, int, int, float]] = []
    curve: list[float] = []
    t = 0

    s0_fixed = mdp.index(0, 0, 0, 0, 0)  # empty intersection, green on the main road

    for ep in range(n_episodes):
        s = mdp.reset(rng) if exploring_starts else s0_fixed

        for _ in range(episode_len):
            eps = max(eps_floor, eps_start - (eps_start - eps_floor) * t / decay_steps)
            avail = legal_actions[s]

            if avail.size == 1:
                a = int(avail[0])
            elif rng.random() < eps:
                a = int(avail[rng.integers(avail.size)])
            else:
                a = int(np.argmax(q[s]))  # -inf on illegal, so the argmax is always legal

            s_next, r = mdp.step(s, a, rng)
            transitions.append((s, a, s_next, r))

            n = visits[s, a]
            alpha = alpha_const if alpha_const is not None else (1.0 + n) ** (-alpha_power)
            visits[s, a] = n + 1

            target = r + gamma * float(np.max(q[s_next]))  # max over LEGAL a' only
            q[s, a] += alpha * (target - q[s, a])

            s = s_next
            t += 1

        if eval_every is not None and (ep + 1) % eval_every == 0:
            curve.append(_greedy_regret(mdp, _greedy(q)))

    policy = _greedy(q)
    coverage = float(((visits > 0) & legal).sum() / legal.sum())

    return QLearningResult(
        q=q,
        policy=policy,
        coverage=coverage,
        n_steps=t,
        transitions=transitions,
        curve=np.array(curve, dtype=float),
    )


def _greedy(q: np.ndarray) -> np.ndarray:
    """Illegal entries sit at -inf, so the argmax is legal by construction."""
    return np.asarray(q.argmax(axis=1), dtype=int)


# --- instrumentation ---------------------------------------------------------------------------
# The two helpers below score the learner; they do not teach it. The regret of the greedy policy
# is computed with the exact evaluator of §B.9 under the TRUE model, which is the only way to get
# a learning curve that is not itself a Monte-Carlo estimate. Nothing here is ever fed back into
# an update — it exists so the report can plot regret against samples. They live outside
# `q_learning` deliberately: the learner's own source stays free of any reference to the model.


@lru_cache(maxsize=8)
def _optimal_value(mdp: SignalMDP) -> np.ndarray:
    return value_iteration(mdp).v


def _greedy_regret(mdp: SignalMDP, policy: np.ndarray) -> float:
    return regret(_optimal_value(mdp), exact_policy_value(mdp, policy))
