"""Regression tests for environment module imports.

Ensures every environment module can be imported without errors
(e.g. no stale references to renamed symbols like OpenaiConfig).
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path",
    [
        "environments.sft_loader_server",
        "environments.community.ufc_prediction_env.ufc_server",
        "environments.community.ufc_prediction_env.ufc_image_env",
    ],
)
def test_environment_module_imports(module_path):
    """Each environment module should import without ImportError."""
    importlib.import_module(module_path)
