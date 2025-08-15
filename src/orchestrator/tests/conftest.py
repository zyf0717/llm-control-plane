"""
Shared pytest configuration and fixtures.
"""

import asyncio
import os
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_environment_variables():
    """Mock environment variables for testing."""
    # Clear existing variables and set test values
    test_env = {
        "API_KEY_ID": "test-api-key-id",
        "API_KEY_SECRET": "test-api-secret",
    }

    with patch.dict(os.environ, test_env, clear=False):
        # Mock os.getenv for HeaderManager
        with patch("src.orchestrator.utils.os.getenv") as mock_getenv:

            def getenv_side_effect(key, default=None):
                return test_env.get(key, default)

            mock_getenv.side_effect = getenv_side_effect
            yield


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for network isolation."""
    with patch("httpx.AsyncClient") as mock:
        yield mock
