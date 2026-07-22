"""Part B acceptance tests for the MODEL (PROBLEM_STATEMENT.md, Acceptance tests).

These do not need Value Iteration or Q-learning. They prove the environment is correct before
anybody trusts an algorithm's output on it — a stochastic bug in P is nearly impossible to find
by staring at regret numbers later.
"""

import numpy as np
import pytest

from signal_control.baselines import (
    always_hold,
    fixed_time,
    longest_queue_first,
    myopic_greedy,
    random_legal,
)
from signal_control.evaluation import exact_policy_value
from signal_control.mdp import (
    HOLD,
    SWITCH,
    SignalConfig,
    SignalMDP,
    _truncated_poisson,
)


def test_state_space_size(mdp):
    assert mdp.dims == (13, 13, 2, 7, 3)
    assert mdp.nS == 7098


def test_legal_pair_count(mdp):
    assert int(mdp.legal.sum()) == 11154
    assert mdp.legal.size == 14196


def test_truncated_poisson_is_renormalised_not_clipped():
    """Clipping would dump the tail onto the endpoint and bias the arrival model."""
    pmf = _truncated_poisson(2.0, 8)
    assert pmf.sum() == pytest.approx(1.0)
    assert np.all(pmf > 0)
    # A clipped pmf would put the whole tail on k=8; a renormalised one keeps it decreasing.
    assert pmf[8] < pmf[7] < pmf[6]


def test_action_masks_are_exactly_the_min_and_max_green_rules(mdp):
    cfg = mdp.cfg
    for s in range(0, mdp.nS, 37):  # stride to keep it quick but cover the space
        _, _, _, e, _ = mdp.unravel(s)
        legal = set(np.flatnonzero(mdp.legal[s]).tolist())
        if e < cfg.e_min:
            assert legal == {HOLD}, f"e={e} < e_min: switching must be illegal"
        elif e >= cfg.E_max:
            assert legal == {SWITCH}, f"e={e} = E_max: holding must be illegal"
        else:
            assert legal == {HOLD, SWITCH}


def test_transition_rows_sum_to_one_for_legal_pairs_and_zero_otherwise(mdp):
    sums = np.asarray(mdp.P.sum(axis=1)).ravel()
    legal_flat = mdp.legal.ravel()  # row index is s*nA + a, matching legal.ravel()
    assert np.allclose(sums[legal_flat], 1.0)
    assert np.all(sums[~legal_flat] == 0.0)


def test_switch_burns_a_clearance_tick(mdp):
    """On SWITCH neither approach discharges — that lost time is the whole reason lookahead pays."""
    s = mdp.index(10, 10, 0, 3, 0)  # both queues full-ish, green on A, mid-phase
    rng = np.random.default_rng(0)
    qa_next = []
    for _ in range(200):
        s2, _ = mdp.step(s, SWITCH, rng)
        qa2, _, phi2, e2, _ = mdp.unravel(s2)
        qa_next.append(qa2)
        assert phi2 == 1 and e2 == 0
    assert min(qa_next) >= 10  # queue A never shrinks on a switch tick


def test_hold_discharges_the_green_approach(mdp):
    s = mdp.index(10, 10, 0, 3, 2)  # night regime: few arrivals, so discharge is visible
    rng = np.random.default_rng(0)
    s2, _ = mdp.step(s, HOLD, rng)
    qa2, _, phi2, e2, _ = mdp.unravel(s2)
    assert phi2 == 0 and e2 == 4
    assert qa2 < 10  # A had green and mu=5 > typical night arrivals


def test_expected_reward_matches_the_simulator_on_average(mdp):
    """R(s,a) = E[r]. If these disagree, VI and Q-learning are solving different problems —
    which is exactly the bug that would invalidate the entire Part B comparison."""
    rng = np.random.default_rng(7)
    for s in (mdp.index(11, 9, 0, 3, 0), mdp.index(2, 1, 1, 2, 1), mdp.index(12, 12, 0, 5, 0)):
        for a in np.flatnonzero(mdp.legal[s]):
            rewards = [mdp.step(s, int(a), rng)[1] for _ in range(20000)]
            assert np.mean(rewards) == pytest.approx(mdp.R[s, a], abs=0.15)


def test_reward_is_not_a_function_of_s_a_s_prime(mdp):
    """The realised spill is unrecoverable from s' once the queue truncates. This is the
    'disturbance form' subtlety in §B.5 — assert it is real, because the report claims it is."""
    s = mdp.index(12, 12, 0, 3, 0)  # both queues at Q_max: arrivals will spill
    rng = np.random.default_rng(3)
    seen: dict[int, set[float]] = {}
    for _ in range(3000):
        s2, r = mdp.step(s, HOLD, rng)
        seen.setdefault(s2, set()).add(r)
    assert any(len(rs) > 1 for rs in seen.values()), (
        "same (s,a,s') always yields the same r — the spill is recoverable from s', "
        "so the disturbance-form argument in the report would be false"
    )


def test_lam_scale_builds_a_wrong_model(mdp):
    """Experiment B.10.4 depends on this being the ONLY thing that changes."""
    wrong = SignalMDP(mdp.cfg, lam_scale=0.5)
    assert wrong.nS == mdp.nS
    assert not np.allclose(wrong.R, mdp.R)
    assert np.array_equal(wrong.legal, mdp.legal)  # masks are structural, not model-dependent


@pytest.mark.parametrize("name", ["always_hold", "longest_queue_first", "myopic_greedy", "fixed_time"])
def test_baselines_pick_only_legal_actions(mdp, name):
    pi = {
        "always_hold": always_hold,
        "longest_queue_first": longest_queue_first,
        "myopic_greedy": myopic_greedy,
        "fixed_time": fixed_time,
    }[name](mdp)
    assert mdp.legal[np.arange(mdp.nS), pi].all()


def test_random_baseline_picks_only_legal_actions(mdp):
    pi = random_legal(mdp, np.random.default_rng(0))
    assert mdp.legal[np.arange(mdp.nS), pi].all()


def test_exact_policy_value_satisfies_its_own_bellman_equation(small_mdp):
    """V = R_pi + gamma * P_pi V, to machine precision. If this residual is not ~0 the linear
    solve is wrong, and every regret number in the report is wrong with it."""
    pi = longest_queue_first(small_mdp)
    v = exact_policy_value(small_mdp, pi)
    rows = np.arange(small_mdp.nS) * small_mdp.nA + pi
    r_pi = small_mdp.R[np.arange(small_mdp.nS), pi]
    residual = v - (r_pi + small_mdp.cfg.gamma * (small_mdp.P[rows] @ v))
    assert np.abs(residual).max() < 1e-8


def test_exact_policy_value_rejects_an_illegal_policy(mdp):
    pi = np.zeros(mdp.nS, dtype=int)  # all HOLD, which is illegal at e = E_max
    with pytest.raises(ValueError, match="illegal action"):
        exact_policy_value(mdp, pi)


def test_always_idle_is_worse_than_serving_traffic(mdp):
    """Sanity: a policy that only ever switches when forced must lose to one that watches queues.
    If it does not, the instance is vacuous and there is nothing to learn."""
    v_hold = exact_policy_value(mdp, always_hold(mdp)).mean()
    v_lqf = exact_policy_value(mdp, longest_queue_first(mdp)).mean()
    assert v_lqf > v_hold


def test_myopic_greedy_never_switches_voluntarily(mdp):
    """R(s, HOLD) >= R(s, SWITCH) in EVERY state, so the gamma=0 policy only ever switches
    where the max-green mask forces it.

    Proof: the delay term (q_A + q_B) is charged on the queue entering the tick and is therefore
    identical under both actions. HOLD discharges the green approach, which weakly lowers the
    post-discharge base and hence weakly lowers expected spill. SWITCH discharges nothing AND
    pays c_switch >= 0. So holding dominates on immediate reward, always.

    This is not a defect in the instance — it is the sharpest possible statement of what lookahead
    buys here: EVERY voluntary switch in the optimal policy is purely anticipatory, paid for
    entirely out of future value. Note it holds even at c_switch = 0. Put this in the report.
    """
    r_hold = mdp.R[:, HOLD]
    r_switch = mdp.R[:, SWITCH]
    both = mdp.legal[:, HOLD] & mdp.legal[:, SWITCH]
    assert np.all(r_hold[both] >= r_switch[both])

    free = SignalMDP(SignalConfig(c_switch=0.0))
    both_free = free.legal[:, HOLD] & free.legal[:, SWITCH]
    assert np.all(free.R[both_free, HOLD] >= free.R[both_free, SWITCH])

    # Consequence: the myopic policy collapses onto always-hold.
    pi = myopic_greedy(mdp)
    forced = np.array([mdp.unravel(s)[3] >= mdp.cfg.E_max for s in range(mdp.nS)])
    assert np.array_equal(pi == SWITCH, forced)
    assert np.array_equal(pi, always_hold(mdp))
