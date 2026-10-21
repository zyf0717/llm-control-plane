"""
Integration tests for the LLM Control Plane proxy.
These tests require actual network connectivity and may be slower.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.orchestrator.proxy import app


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.mark.integration
class TestProxyIntegration:
    """Integration tests for the proxy functionality."""

    def test_root_endpoint_structure(self, client):
        """Test that root endpoint accepts POST requests."""
        from fastapi import Response
        with patch("src.orchestrator.proxy.proxy_request", AsyncMock(return_value=Response(
            content='{"choices":[{"message":{"content":"test"}}]}', status_code=200,
        ))) as proxy:
            response = client.post("/", json={"messages": [{"role": "user", "content": "test"}]})
        assert response.status_code == 200
        assert proxy.call_args.args[0] == "smart"
