import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runproviders", action="store_true", default=False, help="run provider tests"
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run GPU integration tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "providers: mark test as requires providers api keys to run"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (skipped unless --run-gpu)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runproviders"):
        skip_providers = pytest.mark.skip(reason="need --runproviders option to run")
        for item in items:
            if "providers" in item.keywords:
                item.add_marker(skip_providers)

    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
