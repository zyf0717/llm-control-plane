"""
Integration tests for the LLM Control Plane proxy.
These tests require actual network connectivity and may be slower.
"""

import os
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import ENDPOINT_MAP, app, check_endpoint_health


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.mark.integration
class TestRealEndpointHealth:
    """Integration tests for real endpoint health checking."""

    @pytest.mark.asyncio
    async def test_check_google_health(self):
        """Test health check against a known good endpoint."""
        # Use Google as a known good endpoint for testing
        result = await check_endpoint_health("https://www.google.com", timeout=5.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_nonexistent_endpoint(self):
        """Test health check against a non-existent endpoint."""
        result = await check_endpoint_health(
            "https://nonexistent-domain-12345.com", timeout=2.0
        )
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("RUN_ENDPOINT_TESTS"), reason="Set RUN_ENDPOINT_TESTS=1 to run"
    )
    async def test_configured_endpoints(self):
        """Test health of actual configured endpoints (requires environment setup)."""
        results = {}
        for name, url in ENDPOINT_MAP.items():
            try:
                result = await check_endpoint_health(url, timeout=10.0)
                results[name] = result
                print(f"{name}: {'✓' if result else '✗'}")
            except Exception as e:
                results[name] = False
                print(f"{name}: ✗ ({e})")

        # At least log the results
        print(f"Endpoint health results: {results}")


@pytest.mark.integration
class TestProxyIntegration:
    """Integration tests for the proxy functionality."""

    def test_health_endpoint_integration(self, client):
        """Test the health endpoint integration."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "endpoints" in data
        assert data["status"] in ["ok", "degraded"]

        # Verify all configured endpoints are checked
        for endpoint_name in ENDPOINT_MAP.keys():
            assert endpoint_name in data["endpoints"]

    def test_root_endpoint_structure(self, client):
        """Test that root endpoint accepts POST requests."""
        # This test doesn't actually make upstream calls
        with patch("proxy.get_available_endpoint") as mock_get_endpoint:
            mock_get_endpoint.return_value = "https://mock-endpoint.com/test"

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
