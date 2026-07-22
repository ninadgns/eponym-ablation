"""Part B evaluation harness (PROBLEM_STATEMENT.md §B.9).

EVERY policy — Value Iteration's, Q-learning's, certainty-equivalence's, every baseline — is
scored the same way: by *exactly solving* (I - gamma*P_pi) V = R_pi under the TRUE model.

This removes Monte-Carlo noise from the comparison completely. Policies are ranked by exact
expected return, not by which one happened to draw a luckier rollout. Do not replace this with
averaged rollouts; the whole point is that the ranking is not a sampling artefact.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from signal_control.mdp import SignalMDP


def policy_rows(mdp: SignalMDP, policy: np.ndarray) -> np.ndarray:
    """Row indices into mdp.P for the (s, pi(s)) pairs."""
    return np.arange(mdp.nS) * mdp.nA + policy


def exact_policy_value(mdp: SignalMDP, policy: np.ndarray) -> np.ndarray:
    """Solve (I - gamma P_pi) V = R_pi exactly. Returns V^pi over all states."""
    _assert_legal(mdp, policy)
    p_pi = mdp.P[policy_rows(mdp, policy)]
    r_pi = mdp.R[np.arange(mdp.nS), policy]
    a = sparse.eye(mdp.nS, format="csc") - mdp.cfg.gamma * p_pi
    return spsolve(a.tocsc(), r_pi)


def regret(v_star: np.ndarray, v_pi: np.ndarray) -> float:
    """100 * (V* - V^pi) / |V*|, averaged over start states. Zero for the optimal policy."""
    vs, vp = float(v_star.mean()), float(v_pi.mean())
    return 100.0 * (vs - vp) / abs(vs)


def rollout_stats(
    mdp: SignalMDP,
    policy: np.ndarray,
    rng: np.random.Generator,
    n_ticks: int = 100_000,
) -> dict[str, float]:
    """Consequences a traffic engineer would recognise. Reported ALONGSIDE exact regret, never
    instead of it."""
    _assert_legal(mdp, policy)
    cfg = mdp.cfg
    s = mdp.reset(rng)
    total_q = 0
    switches = 0
    spill_events = 0
    served = 0
    queues = np.empty(n_ticks, dtype=int)

    for t in range(n_ticks):
        a = int(policy[s])
        qa, qb, phi, e, p = mdp.unravel(s)
        q_before = qa + qb
        d = 0 if a == 1 else min(qa if phi == 0 else qb, cfg.mu)
        s_next, _ = mdp.step(s, a, rng)
        qa2, qb2, *_ = mdp.unravel(s_next)

        total_q += q_before
        queues[t] = q_before
        switches += int(a == 1)
        served += d
        if qa2 == cfg.Q_max or qb2 == cfg.Q_max:
            spill_events += 1
        s = s_next

    ticks_per_hour = 3600.0 / cfg.dt
    return {
        "mean_queue_veh": total_q / n_ticks,
        "mean_delay_s_per_veh": (total_q * cfg.dt) / max(served, 1),
        "throughput_veh_per_h": served / n_ticks * ticks_per_hour,
        "switches_per_h": switches / n_ticks * ticks_per_hour,
        "spill_ticks_per_h": spill_events / n_ticks * ticks_per_hour,
        "q95_veh": float(np.percentile(queues, 95)),
    }


def _assert_legal(mdp: SignalMDP, policy: np.ndarray) -> None:
    if policy.shape != (mdp.nS,):
        raise ValueError(f"policy must have shape ({mdp.nS},), got {policy.shape}")
    if not mdp.legal[np.arange(mdp.nS), policy].all():
        bad = int(np.flatnonzero(~mdp.legal[np.arange(mdp.nS), policy])[0])
        raise ValueError(
            f"policy picks an illegal action in state {mdp.unravel(bad)} "
            f"(min-green / max-green masks were ignored)"
        )
