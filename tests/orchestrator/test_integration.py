"""
Integration tests for the LLM Control Plane proxy.
These tests require actual network connectivity and may be slower.
"""

from unittest.mock import AsyncMock, Mock, patch

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
        # This test requires proper mocking of the routing system
        with patch(
            "src.orchestrator.proxy.reachable_endpoints", ["test-endpoint"]
        ), patch("src.orchestrator.proxy.get_router") as mock_get_router, patch(
            "src.orchestrator.proxy.proxy_request"
        ) as mock_proxy:

            from fastapi import Response

            from src.orchestrator.llm_router import RouteDecision, WorkloadType

            # Mock router
            mock_router = Mock()
            mock_router.route_request = AsyncMock(
                return_value=RouteDecision(
                    endpoint="test-endpoint",
                    confidence=0.9,
                    reason="Test routing",
                    workload_type=WorkloadType.TTFT_CONTENT,
                )
            )
            mock_router.get_endpoint_by_name = Mock(return_value=None)
            mock_get_router.return_value = mock_router

            # Mock proxy response
            mock_proxy.return_value = Response(
                content='{"choices": [{"message": {"content": "test"}}]}',
                status_code=200,
            )

            response = client.post(
                "/",
                json={"messages": [{"role": "user", "content": "test"}]},
                headers={"Content-Type": "application/json"},
            )

            # Should not return error status codes
            assert response.status_code < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    pytest.main([__file__, "-v"])
