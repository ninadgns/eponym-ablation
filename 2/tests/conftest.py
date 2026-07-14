"""Shared test fixtures."""

import pytest
from fuel_csp.synthetic import GeneratorConfig, generate_problem


@pytest.fixture
def small_problem():
    return generate_problem(GeneratorConfig(num_vehicles=10, seed=42))


@pytest.fixture
def medium_problem():
    return generate_problem(GeneratorConfig(num_vehicles=20, seed=7))
