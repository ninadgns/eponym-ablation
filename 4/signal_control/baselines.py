"""Part B baselines (PROBLEM_STATEMENT.md §B.8).

`fixed_time` and `longest_queue_first` are the ones that matter — the first is what is actually
deployed at Dhaka intersections, the second is the obvious "smart" reactive rule. `myopic_greedy`
is the key scientific comparison: it is FORMALLY CORRECT for gamma = 0, so whatever it loses is
attributable to lookahead and nothing else.

If none of these lose by a meaningful margin, the MDP is vacuous and the instance needs
rethinking. Report that rather than hiding it.
"""

from __future__ import annotations

import numpy as np

from signal_control.mdp import HOLD, SWITCH, SignalMDP


def _empty(mdp: SignalMDP) -> np.ndarray:
    return np.zeros(mdp.nS, dtype=int)


def _forced(mdp: SignalMDP, e: int) -> int | None:
    """The action, if the min/max-green masks leave no choice."""
    legal = mdp.legal_actions(e)
    return legal[0] if len(legal) == 1 else None


def always_hold(mdp: SignalMDP) -> np.ndarray:
    pi = _empty(mdp)
    for s in range(mdp.nS):
        _, _, _, e, _ = mdp.unravel(s)
        forced = _forced(mdp, e)
        pi[s] = forced if forced is not None else HOLD
    return pi


def random_legal(mdp: SignalMDP, rng: np.random.Generator) -> np.ndarray:
    pi = _empty(mdp)
    for s in range(mdp.nS):
        _, _, _, e, _ = mdp.unravel(s)
        legal = mdp.legal_actions(e)
        pi[s] = int(rng.choice(legal))
    return pi


def longest_queue_first(mdp: SignalMDP) -> np.ndarray:
    """Switch whenever the red queue exceeds the green one, subject to min green."""
    pi = _empty(mdp)
    for s in range(mdp.nS):
        qa, qb, phi, e, _ = mdp.unravel(s)
        forced = _forced(mdp, e)
        if forced is not None:
            pi[s] = forced
            continue
        q_green, q_red = (qa, qb) if phi == 0 else (qb, qa)
        pi[s] = SWITCH if q_red > q_green else HOLD
    return pi


def myopic_greedy(mdp: SignalMDP) -> np.ndarray:
    """argmax_a R(s, a) over legal actions — the gamma = 0 optimal policy.

    NOTE: in this MDP this provably COLLAPSES ONTO `always_hold`. R(s,HOLD) >= R(s,SWITCH) in
    every state — the delay term is charged on the entering queue and so is action-independent,
    holding discharges the green approach and thus weakly reduces expected spill, and switching
    additionally pays c_switch >= 0. It holds even at c_switch = 0.

    Keep both rows in the results table, but SAY they coincide and say why. The corollary is the
    headline: every voluntary switch in the optimal policy is purely anticipatory, because the
    immediate reward never once argues for switching.
    """
    r = np.where(mdp.legal, mdp.R, -np.inf)
    return np.asarray(r.argmax(axis=1), dtype=int)


def fixed_time(mdp: SignalMDP) -> np.ndarray:
    """Webster-style: green split proportional to demand, blind to the queue.

    A function of (phi, e) only — it never looks at q_A or q_B. That is precisely what is
    actually deployed, and precisely why it can be beaten.
    """
    cfg = mdp.cfg
    lam_a = float(np.mean(cfg.lam_A))
    lam_b = float(np.mean(cfg.lam_B))
    total = lam_a + lam_b
    usable = cfg.E_max  # ticks of green available per phase before the max-green mask fires
    green = {
        0: int(np.clip(round(usable * lam_a / total), cfg.e_min, cfg.E_max)),
        1: int(np.clip(round(usable * lam_b / total), cfg.e_min, cfg.E_max)),
    }

    pi = _empty(mdp)
    for s in range(mdp.nS):
        _, _, phi, e, _ = mdp.unravel(s)
        forced = _forced(mdp, e)
        if forced is not None:
            pi[s] = forced
            continue
        pi[s] = SWITCH if e >= green[phi] else HOLD
    return pi
