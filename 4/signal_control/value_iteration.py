"""Value Iteration — from scratch (PROBLEM_STATEMENT.md §B.7).

Model-based and exact: the sum over s' is only computable because somebody handed you P. That is
what "model-based" means, and it is the distinction the whole of Part B is built to study.
Bellman (1957); the stopping rule is the standard one in Puterman (1994, §6.3).

    V(s) <- max_{a in A(s)} [ R(s,a) + gamma * sum_s' P(s'|s,a) V(s') ]

Stop when ||V_{k+1} - V_k||_inf < eps * (1 - gamma) / (2 * gamma), which bounds
||V_k - V*||_inf < eps. Illegal actions are masked with -inf before the max.

The empirical contraction rate ||dV_k|| / ||dV_{k-1}|| is returned because theory says it converges
to exactly gamma, and checking that it does is a nearly free correctness proof of the Bellman
operator. (Once the greedy policy stabilises, dV_{k+1} = gamma * P_pi dV_k, and the dominant right
eigenvector of a stochastic P_pi is the all-ones vector — so the sup-norm ratio tends to gamma
exactly, not merely to something below it.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from signal_control.mdp import SignalMDP


@dataclass(frozen=True)
class VIResult:
    v: np.ndarray  # V*, shape (nS,)
    policy: np.ndarray  # greedy policy, shape (nS,), legal by construction
    n_sweeps: int
    contraction_rates: np.ndarray  # empirical ||dV_k|| / ||dV_{k-1}|| per sweep


def value_iteration(mdp: SignalMDP, eps: float = 1e-8, max_sweeps: int = 100_000) -> VIResult:
    gamma = mdp.cfg.gamma
    return _solve(mdp.P, mdp.R, mdp.legal, gamma, mdp.nS, mdp.nA, eps, max_sweeps)


def _solve(
    p: sparse.spmatrix,
    r: np.ndarray,
    legal: np.ndarray,
    gamma: float,
    n_s: int,
    n_a: int,
    eps: float,
    max_sweeps: int,
) -> VIResult:
    """The Bellman iteration itself, over any (P, R) — the true model or an estimated one."""
    # Puterman's rule. At gamma = 0 the operator is a projection, not a contraction, so the bound
    # degenerates (it divides by gamma); one sweep is exact there and a plain fixed-point test does
    # the right thing.
    tol = eps * (1.0 - gamma) / (2.0 * gamma) if gamma > 0.0 else eps

    # Illegal (s, a) rows of P are all-zero and their R is 0, which would otherwise look like a
    # perfectly good absorbing action worth 0 — far better than any real one, since every reward
    # here is negative. Mask them out of the max with -inf.
    penalty = np.where(legal, 0.0, -np.inf)

    v = np.zeros(n_s, dtype=float)
    rates: list[float] = []
    prev_delta = None
    n_sweeps = 0

    for _ in range(max_sweeps):
        q = (r.ravel() + gamma * (p @ v)).reshape(n_s, n_a) + penalty
        v_next = q.max(axis=1)

        delta = float(np.abs(v_next - v).max())
        if prev_delta is not None and prev_delta > 0.0:
            rates.append(delta / prev_delta)
        prev_delta = delta

        v = v_next
        n_sweeps += 1
        if delta < tol:
            break

    q = (r.ravel() + gamma * (p @ v)).reshape(n_s, n_a) + penalty
    policy = np.asarray(q.argmax(axis=1), dtype=int)

    return VIResult(
        v=v,
        policy=policy,
        n_sweeps=n_sweeps,
        contraction_rates=np.array(rates, dtype=float),
    )


def certainty_equivalence(
    mdp_true: SignalMDP,
    transitions: list[tuple[int, int, int, float]],
    eps: float = 1e-8,
) -> VIResult:
    """Estimate P-hat, R-hat from (s, a, s', r) samples, then plan on the estimated model.

    This is the arm that can falsify the naive thesis. It sees EXACTLY the data Q-learning saw —
    same transitions, same order, same seed — so any difference between the two is attributable to
    what they do with the data, not to how much of it they got.

    **The fallback for unvisited (s, a) pairs, stated because it is a modelling decision and not an
    implementation detail.** We use a PESSIMISTIC SELF-LOOP: an unvisited action is assumed to
    leave the state unchanged and to earn the worst reward anywhere in the sample. Its value is
    then r_worst / (1 - gamma), a large negative number, so the planner will not reach for an
    action it has no evidence about. The alternatives are materially different and would each be a
    different experiment: a uniform prior over next states invents dynamics out of nothing, and an
    optimistic fallback (R = 0, the best possible reward here) actively lures the planner into
    unvisited pairs — which is exploration policy smuggled in as a modelling choice. Pessimism is
    the honest default when coverage is thin, and coverage IS thin: that is the whole point of
    running this on Q-learning's own budget. Report coverage next to the regret, always.

    Only the transitions are model knowledge. `mdp_true` is used for its state-space shape, its
    action masks (engineering constraints, not dynamics) and gamma — never for its true P or R.
    """
    n_s, n_a = mdp_true.nS, mdp_true.nA
    gamma = mdp_true.cfg.gamma
    legal = mdp_true.legal

    if not transitions:
        raise ValueError("certainty_equivalence needs at least one transition")

    s_arr = np.fromiter((t[0] for t in transitions), dtype=np.int64, count=len(transitions))
    a_arr = np.fromiter((t[1] for t in transitions), dtype=np.int64, count=len(transitions))
    s2_arr = np.fromiter((t[2] for t in transitions), dtype=np.int64, count=len(transitions))
    r_arr = np.fromiter((t[3] for t in transitions), dtype=float, count=len(transitions))

    rows = s_arr * n_a + a_arr  # the (s,a) row index, matching mdp.P's layout
    n_rows = n_s * n_a

    counts = np.bincount(rows, minlength=n_rows).astype(float)
    visited = counts > 0

    # R-hat(s,a) = the sample mean of the realised rewards, which is the right estimator precisely
    # because r is not a function of (s,a,s') — the realised spill lives in r and nowhere else.
    r_sum = np.bincount(rows, weights=r_arr, minlength=n_rows)
    r_hat = np.zeros(n_rows, dtype=float)
    r_hat[visited] = r_sum[visited] / counts[visited]

    # P-hat(s'|s,a) = empirical frequency. Duplicate (row, s') pairs are summed by csr_matrix.
    p_hat = sparse.csr_matrix(
        (np.ones(len(transitions), dtype=float), (rows, s2_arr)),
        shape=(n_rows, n_s),
    )
    p_hat = sparse.diags(np.where(visited, 1.0 / np.maximum(counts, 1.0), 0.0)) @ p_hat

    # Pessimistic self-loop on the legal-but-unvisited pairs.
    r_worst = float(r_arr.min())
    unvisited_legal = np.flatnonzero(legal.ravel() & ~visited)
    if unvisited_legal.size:
        r_hat[unvisited_legal] = r_worst
        self_loop = sparse.csr_matrix(
            (
                np.ones(unvisited_legal.size, dtype=float),
                (unvisited_legal, unvisited_legal // n_a),
            ),
            shape=(n_rows, n_s),
        )
        p_hat = p_hat + self_loop

    return _solve(
        p_hat.tocsr(),
        r_hat.reshape(n_s, n_a),
        legal,
        gamma,
        n_s,
        n_a,
        eps,
        max_sweeps=100_000,
    )
