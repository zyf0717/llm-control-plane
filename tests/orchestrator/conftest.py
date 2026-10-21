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
        "CONVERSATION_DB_PATH": "",
    }

    with patch.dict(os.environ, test_env, clear=False):
        yield


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for network isolation."""
    with patch("httpx.AsyncClient") as mock:
        yield mock


@pytest.fixture
def http_backend(monkeypatch):
    """Install an in-process upstream without replacing httpx response semantics."""
    import httpx
    real_client = httpx.AsyncClient

    def install(handler):
        requests = []

        async def dispatch(request):
            requests.append(request)
            result = handler(request)
            return await result if hasattr(result, "__await__") else result

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
            **kwargs, transport=httpx.MockTransport(dispatch)
        ))
        return requests

    return install
