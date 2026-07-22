"""Contract tests for the two GRADED algorithms (value iteration and Q-learning).

These are the definition of "done". They skip while a function still raises NotImplementedError,
so the suite stays green as scaffolding and turns into a real gate the moment you start writing.

Do not weaken a test to make an implementation pass. If a test is wrong, fix the test on its
merits and say so — the myopic-switching proof found while building this scaffold came from
taking a failing test seriously.
"""

import numpy as np
import pytest

from signal_control.baselines import longest_queue_first, myopic_greedy
from signal_control.evaluation import exact_policy_value, regret
from signal_control.mdp import SWITCH, SignalConfig, SignalMDP
from signal_control.q_learning import q_learning
from signal_control.value_iteration import value_iteration


def _skip_if_unimplemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")



def test_vi_converges_and_its_policy_is_legal(mdp):
    res = _skip_if_unimplemented(value_iteration, mdp)
    assert res.v.shape == (mdp.nS,)
    assert mdp.legal[np.arange(mdp.nS), res.policy].all()


def test_vi_agrees_with_the_exact_linear_solve(mdp):
    """The cross-check. If V* and the exact value of its own greedy policy disagree, one of the
    two is wrong, and every regret number downstream inherits the error."""
    res = _skip_if_unimplemented(value_iteration, mdp)
    v_pi = exact_policy_value(mdp, res.policy)
    assert np.abs(res.v - v_pi).max() < 1e-6


def test_vi_empirical_contraction_rate_matches_gamma(mdp):
    """Theory says the Bellman operator contracts at exactly gamma. Showing it does is a nearly
    free correctness proof of your operator."""
    res = _skip_if_unimplemented(value_iteration, mdp)
    tail = res.contraction_rates[-20:]
    assert np.mean(tail) == pytest.approx(mdp.cfg.gamma, abs=0.02)


def test_vi_is_optimal_so_every_baseline_has_non_negative_regret(mdp):
    res = _skip_if_unimplemented(value_iteration, mdp)
    for policy in (longest_queue_first(mdp), myopic_greedy(mdp)):
        assert regret(res.v, exact_policy_value(mdp, policy)) >= -1e-9


def test_vi_beats_the_reactive_baselines_by_a_real_margin(mdp):
    """If lookahead buys nothing, the MDP is vacuous. Report the number either way, but a
    near-zero margin means the instance needs rethinking, not a rewrite of the conclusion."""
    res = _skip_if_unimplemented(value_iteration, mdp)
    r_lqf = regret(res.v, exact_policy_value(mdp, longest_queue_first(mdp)))
    assert r_lqf > 1.0


def test_gamma_zero_vi_reproduces_the_myopic_baseline():
    """Known-answer check on the whole Bellman pipeline."""
    m = SignalMDP(SignalConfig(gamma=0.0))
    res = _skip_if_unimplemented(value_iteration, m)
    assert np.array_equal(res.policy, myopic_greedy(m))


def test_optimal_policy_switches_voluntarily(mdp):
    """The corollary of the myopic-collapse proof (see test_signal_mdp.py): the immediate reward
    NEVER argues for switching, so any voluntary switch in V*'s policy is purely anticipatory.
    If the optimal policy never switches except when forced, there is no anticipation to study."""
    res = _skip_if_unimplemented(value_iteration, mdp)
    forced = np.array([mdp.unravel(s)[3] >= mdp.cfg.E_max for s in range(mdp.nS)])
    voluntary = (res.policy == SWITCH) & ~forced
    assert voluntary.sum() > 0


def test_q_learning_never_touches_the_model(mdp):
    """Model-free means model-free. Reaching for mdp.P or mdp.R inside q_learning silently
    destroys the entire comparison Part B exists to make."""
    import inspect

    src = inspect.getsource(q_learning)
    body = src.split('"""')[-1]  # strip the docstring, which legitimately mentions them
    assert ".P" not in body and ".R" not in body, "q_learning must not read mdp.P or mdp.R"


def test_q_learning_covers_the_legal_pairs_and_returns_a_legal_policy(mdp):
    res = _skip_if_unimplemented(
        q_learning, mdp, np.random.default_rng(0), n_episodes=200, episode_len=500
    )
    assert mdp.legal[np.arange(mdp.nS), res.policy].all()
    assert 0.0 <= res.coverage <= 1.0
    assert len(res.transitions) == res.n_steps
