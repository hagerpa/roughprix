import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Run slow tests (benchmarks at time_steps=1000, order=6; all reference smiles)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (skipped unless --run-slow)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Pass --run-slow to run benchmark tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)