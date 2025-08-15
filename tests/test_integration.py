"""
Integration tests for the LLM Control Plane proxy.
These tests require actual network connectivity and may be slower.
"""

import os
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import app


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.mark.integration
class TestProxyIntegration:
    """Integration tests for the proxy functionality."""

    def test_root_endpoint_structure(self, client):
        """Test that root endpoint accepts POST requests."""
        # This test doesn't actually make upstream calls
        with patch("httpx.AsyncClient") as mock_client_class:
            # Mock the HTTP client to avoid actual network calls
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_response = httpx.Response(
                status_code=200,
                content=b'{"choices": [{"message": {"content": "test"}}]}',
                headers={"content-type": "application/json"},
            )
            mock_client.request.return_value = mock_response

            response = client.post(
                "/",
                json={"messages": [{"role": "user", "content": "test"}]},
                headers={"Content-Type": "application/json"},
            )

            # Should not return error status codes
            assert response.status_code < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
