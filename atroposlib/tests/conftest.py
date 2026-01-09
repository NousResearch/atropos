import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runproviders", action="store_true", default=False, help="run provider tests"
    )
    parser.addoption(
        "--runprime", action="store_true", default=False, help="run Prime Hub integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "providers: mark test as requires providers api keys to run"
    )
    config.addinivalue_line(
        "markers", "prime: mark test as requires Prime Hub login to run"
    )


def pytest_collection_modifyitems(config, items):
    # Skip provider tests unless --runproviders is given
    if not config.getoption("--runproviders"):
        skip_providers = pytest.mark.skip(reason="need --runproviders option to run")
        for item in items:
            if "providers" in item.keywords:
                item.add_marker(skip_providers)
    
    # Skip Prime tests unless --runprime is given
    if not config.getoption("--runprime"):
        skip_prime = pytest.mark.skip(reason="need --runprime option to run")
        for item in items:
            if "prime" in item.keywords:
                item.add_marker(skip_prime)

