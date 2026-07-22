"""Shared fixtures."""

import pytest

from signal_control.mdp import SignalConfig, SignalMDP


@pytest.fixture(scope="session")
def mdp() -> SignalMDP:
    return SignalMDP(SignalConfig())


@pytest.fixture(scope="session")
def small_mdp() -> SignalMDP:
    """A tiny MDP for tests that would be slow on the full one."""
    return SignalMDP(SignalConfig(Q_max=4, E_max=3, A_max=3, mu=2))
