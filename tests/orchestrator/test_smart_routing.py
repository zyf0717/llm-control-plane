"""
Integration tests for smart routing endpoint.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.orchestrator.llm_router import RouteDecision, WorkloadType
from src.orchestrator.proxy import app


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_route_decision():
    """Mock route decision for testing."""
    return RouteDecision(
        endpoint="HRPC-CISR HPC",
        confidence=0.8,
        reason="Complex reasoning task detected",
        workload_type=WorkloadType.REASONING,
    )


class TestSmartRoutingEndpoint:
    """Tests for the /smart endpoint functionality."""

    def test_smart_route_no_messages(self, client):
        """Test smart routing with no messages."""
        payload = {"model": "test-model"}

        response = client.post("/smart", json=payload)

        assert response.status_code == 400
        assert "No messages provided" in response.json()["detail"]

    def test_smart_route_no_user_messages(self, client):
        """Test smart routing with no user messages."""
        payload = {
            "messages": [{"role": "system", "content": "You are a helpful assistant"}]
        }

        response = client.post("/smart", json=payload)

        assert response.status_code == 400
        assert "No user messages found" in response.json()["detail"]

    def test_smart_route_invalid_json(self, client):
        """Test smart routing with invalid JSON."""
        response = client.post(
            "/smart",
            content="invalid json",
            headers={"content-type": "application/json"},
        )

        # FastAPI will raise 422 for invalid JSON, or 500 if it gets to our error handler
        assert response.status_code in [400, 422, 500]

    def test_smart_route_empty_body(self, client):
        """Test smart routing with empty request body."""
        response = client.post("/smart")

        # Should return error status code
        assert response.status_code >= 400

    @patch("src.orchestrator.proxy.get_router")
    def test_smart_route_router_error(self, mock_get_router, client):
        """Test smart routing when router raises an exception."""
        mock_router = Mock()
        mock_router.route_request = AsyncMock(side_effect=Exception("Router error"))
        mock_get_router.return_value = mock_router

        payload = {"messages": [{"role": "user", "content": "Test message"}]}

        response = client.post("/smart", json=payload)

        assert response.status_code == 500
        assert "Smart routing error" in response.json()["detail"]

    @patch("src.orchestrator.proxy.reachable_endpoints", ["HRPC-CISR HPC"])
    @patch("src.orchestrator.proxy.proxy_request")
    @patch("src.orchestrator.proxy.get_router")
    def test_smart_route_basic(
        self, mock_get_router, mock_proxy, client, mock_route_decision
    ):
        """Test basic smart routing functionality."""
        # Setup mocks
        mock_router = Mock()
        mock_router.route_request = AsyncMock(return_value=mock_route_decision)
        mock_router.get_endpoint_by_name = Mock(return_value=None)
        mock_get_router.return_value = mock_router

        # Mock proxy_request to return a simple Response
        from fastapi import Response

        mock_proxy.return_value = Response(
            content='{"test": "response"}', status_code=200
        )

        # Test payload
        payload = {
            "messages": [
                {"role": "user", "content": "Analyze this complex problem step by step"}
            ],
            "model": "test-model",
            "max_tokens": 100,
        }

        # Make request
        response = client.post("/smart", json=payload)

        # Verify router was called with reachable_endpoints
        assert mock_router.route_request.called
        call_args = mock_router.route_request.call_args[0]
        assert "Analyze this complex problem step by step" in call_args

        # Verify proxy was called with correct parameters
        mock_proxy.assert_called_once()
        args, kwargs = mock_proxy.call_args
        assert args[0] == "HRPC-CISR HPC"  # endpoint

        # Check routing headers exist
        routing_headers = args[2] if len(args) > 2 else kwargs.get("extra_headers", {})
        assert "X-Route-Decision" in routing_headers
        assert routing_headers["X-Route-Decision"] == "HRPC-CISR HPC"

        # Should get successful response
        assert response.status_code == 200

    @patch("src.orchestrator.proxy.reachable_endpoints", ["HRPC-CISR HPC"])
    @patch("src.orchestrator.proxy.proxy_request")
    @patch("src.orchestrator.proxy.get_router")
    def test_smart_route_multiple_user_messages(
        self, mock_get_router, mock_proxy, client, mock_route_decision
    ):
        """Test smart routing with multiple user messages uses the latest."""
        mock_router = Mock()
        mock_router.route_request = AsyncMock(return_value=mock_route_decision)
        mock_router.get_endpoint_by_name = Mock(return_value=None)
        mock_get_router.return_value = mock_router

        from fastapi import Response

        mock_proxy.return_value = Response(
            content='{"test": "response"}', status_code=200
        )

        payload = {
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Latest user message for routing"},
            ]
        }

        response = client.post("/smart", json=payload)

        # Should use the latest user message for routing
        assert mock_router.route_request.called
        call_args = mock_router.route_request.call_args[0]
        assert "Latest user message for routing" in call_args
        assert response.status_code == 200

    @patch("src.orchestrator.proxy.reachable_endpoints", ["Mac Mini", "HRPC-CISR HPC"])
    @patch("src.orchestrator.proxy.proxy_request")
    @patch("src.orchestrator.proxy.get_router")
    def test_smart_route_reasoning_detection(self, mock_get_router, mock_proxy, client):
        """Test that different message types trigger appropriate routing decisions."""
        test_cases = [
            {
                "message": "What's 2+2?",
                "expected_endpoint": "Mac Mini",  # Simple task
                "expected_reason": "Simple task, using efficient endpoint",
            },
            {
                "message": "Think step by step about quantum mechanics",
                "expected_endpoint": "HRPC-CISR HPC",  # Complex reasoning
                "expected_reason": "Complex reasoning task detected",
            },
        ]

        from fastapi import Response

        mock_proxy.return_value = Response(
            content='{"test": "response"}', status_code=200
        )

        for case in test_cases:
            # Create appropriate mock decision
            decision = RouteDecision(
                endpoint=case["expected_endpoint"],
                confidence=0.8,
                reason=case["expected_reason"],
                workload_type=WorkloadType.REASONING,
            )

            mock_router = Mock()
            mock_router.route_request = AsyncMock(return_value=decision)
            mock_router.get_endpoint_by_name = Mock(return_value=None)
            mock_get_router.return_value = mock_router

            mock_router = Mock()
            mock_router.route_request = AsyncMock(return_value=decision)
            mock_get_router.return_value = mock_router

            payload = {"messages": [{"role": "user", "content": case["message"]}]}

            response = client.post("/smart", json=payload)

            # Verify the correct endpoint was selected
            mock_proxy.assert_called()
            args, kwargs = mock_proxy.call_args
            assert args[0] == case["expected_endpoint"]

            # Verify routing headers exist
            routing_headers = (
                args[2] if len(args) > 2 else kwargs.get("extra_headers", {})
            )
            assert "X-Route-Decision" in routing_headers

            assert response.status_code == 200


class TestSmartRoutingIntegration:
    """Integration tests for smart routing with existing proxy features."""

    @patch("src.orchestrator.proxy.reachable_endpoints", ["HRPC-CISR HPC"])
    @patch("src.orchestrator.proxy.proxy_request")
    @patch("src.orchestrator.proxy.get_router")
    def test_smart_route_logging(
        self, mock_get_router, mock_proxy, client, mock_route_decision, caplog
    ):
        """Test that smart routing logs decisions appropriately."""
        # Set log level to ensure INFO messages are captured
        import logging

        caplog.set_level(logging.INFO)

        mock_router = Mock()
        mock_router.route_request = AsyncMock(return_value=mock_route_decision)
        mock_router.get_endpoint_by_name = Mock(return_value=None)
        mock_get_router.return_value = mock_router

        from fastapi import Response

        mock_proxy.return_value = Response(
            content='{"test": "response"}', status_code=200
        )

        payload = {"messages": [{"role": "user", "content": "Test logging"}]}

        response = client.post("/smart", json=payload)

        # Check that routing decision was logged
        log_messages = [record.message for record in caplog.records]
        assert any(
            "Smart routing: HRPC-CISR HPC" in message for message in log_messages
        ), f"Expected log message not found. Actual logs: {log_messages}"
        assert response.status_code == 200
