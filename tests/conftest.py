import pytest

from pzu.synthetic import generate


@pytest.fixture(scope="session")
def synth():
    return generate("2026-01-01", 40, seed=1)
