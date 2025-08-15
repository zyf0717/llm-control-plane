"""
Test suite for the LLM Control Plane proxy functionality.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import (
    app,
    convo_history,
    endpoints,
    get_target_endpoint,
    parse_and_inject_history,
)
from utils import HeaderManager


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear conversation history before each test."""
    convo_history.clear()


@pytest.fixture
def mock_request():
    """Mock FastAPI request object."""
    request = Mock()
    request.headers = {
        "content-type": "application/json",
        "user-agent": "test-client",
    }
    return request


class TestEndpointRouting:
    """Tests for endpoint routing functionality."""

    def test_get_target_endpoint_streaming(self):
        """Test endpoint routing for streaming requests."""
        # Mock the endpoints configuration
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]

        with patch("proxy.endpoints", mock_endpoints):
            # Streaming should route to v1/chat/completions
            result = get_target_endpoint("test-endpoint", is_streaming=True)
            assert result == "https://test.example.com/v1/chat/completions"

    def test_get_target_endpoint_non_streaming(self):
        """Test endpoint routing for non-streaming requests."""
        # Mock the endpoints configuration
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]

        with patch("proxy.endpoints", mock_endpoints):
            # Non-streaming should route to api/v0/chat/completions
            result = get_target_endpoint("test-endpoint", is_streaming=False)
            assert result == "https://test.example.com/api/v0/chat/completions"

    def test_get_target_endpoint_unknown(self):
        """Test endpoint routing for unknown endpoints."""
        # Mock empty endpoints configuration
        with patch("proxy.endpoints", []):
            result = get_target_endpoint("unknown-endpoint")
            assert result is None

    def test_get_target_endpoint_with_subpath(self):
        """Test endpoint routing with subpaths."""
        # Mock the endpoints configuration
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]

        with patch("proxy.endpoints", mock_endpoints):
            # Should extract the first part of the path
            result = get_target_endpoint(
                "test-endpoint/some/subpath", is_streaming=True
            )
            assert result == "https://test.example.com/v1/chat/completions"


class TestHeaderPreparation:
    """Tests for header preparation functionality."""

    def test_prepare_headers(self, mock_request):
        """Test header preparation with API keys."""
        with patch("utils.filter_unsafe_headers") as mock_filter:
            mock_filter.return_value = {"content-type": "application/json"}

            result = HeaderManager.prepare_upstream_headers(mock_request)

            expected = {
                "content-type": "application/json",
                "CF-Access-Client-Id": "test-api-key-id",
                "CF-Access-Client-Secret": "test-api-secret",
            }
            assert result == expected
            mock_filter.assert_called_once_with(dict(mock_request.headers))


class TestConversationHistory:
    """Tests for conversation history functionality."""

    def test_parse_and_inject_history_empty_body(self):
        """Test parsing empty request body."""
        result_body, is_streaming = parse_and_inject_history(b"", "test-convo")
        assert result_body is None
        assert is_streaming is False

    def test_parse_and_inject_history_invalid_json(self):
        """Test parsing invalid JSON body."""
        invalid_json = b"not json"
        result_body, is_streaming = parse_and_inject_history(invalid_json, "test-convo")
        assert result_body is None
        assert is_streaming is False

    def test_parse_and_inject_history_no_messages(self):
        """Test parsing JSON without messages."""
        body_json = {"model": "gpt-4", "stream": True}
        body = json.dumps(body_json).encode()

        result_body, is_streaming = parse_and_inject_history(body, "test-convo")

        assert result_body == body_json
        assert is_streaming is True

    def test_parse_and_inject_history_with_messages(self):
        """Test parsing JSON with messages and conversation injection."""
        body_json = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        body = json.dumps(body_json).encode()
        convo_id = "test-convo"

        # Clear any existing history
        convo_history.clear()

        result_body, is_streaming = parse_and_inject_history(body, convo_id)

        assert result_body["messages"] == body_json["messages"]
        assert is_streaming is False
        assert convo_id in convo_history
        assert convo_history[convo_id] == body_json["messages"]

    def test_parse_and_inject_history_append_to_existing(self):
        """Test appending to existing conversation history."""
        convo_id = "test-convo"

        # Set up existing history
        convo_history[convo_id] = [{"role": "assistant", "content": "Hi there!"}]

        body_json = {
            "messages": [{"role": "user", "content": "How are you?"}],
        }
        body = json.dumps(body_json).encode()

        result_body, is_streaming = parse_and_inject_history(body, convo_id)

        expected_messages = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        assert result_body["messages"] == expected_messages
        assert convo_history[convo_id] == expected_messages


class TestModelsEndpoint:
    """Tests for the /models endpoint functionality."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, client):
        """Test successful models endpoint with mock responses."""
        # Mock the endpoints configuration
        mock_endpoints = [
            {
                "name": "test-endpoint-1",
                "url": "https://test1.example.com",
                "gpu": "NVIDIA RTX 4090",
                "vram": "24GB",
                "cpu": "Intel i9",
                "ram": "64GB",
            },
            {
                "name": "test-endpoint-2",
                "url": "https://test2.example.com",
                "gpu": "Apple M2 Max",
                "cpu": "Apple M2 Max",
                "ram": "32GB",
            },
        ]

        # Mock model responses from endpoints
        mock_response_1 = {
            "data": [
                {
                    "id": "gpt-test-1",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "test-org-1",
                }
            ]
        }

        mock_response_2 = {
            "models": [
                {"id": "gpt-test-2", "created": 1234567891, "owned_by": "test-org-2"}
            ]
        }

        with patch("proxy.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock responses for each endpoint
            async def mock_get(url, headers=None):
                mock_response = Mock()
                mock_response.raise_for_status = Mock()

                if "test1.example.com" in url:
                    mock_response.json.return_value = mock_response_1
                elif "test2.example.com" in url:
                    mock_response.json.return_value = mock_response_2
                else:
                    mock_response.json.return_value = {"data": []}

                return mock_response

            mock_client.get = mock_get

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()

            # Check OpenAI-compatible format
            assert data["object"] == "list"
            assert "data" in data
            assert len(data["data"]) == 2

            # Check first model has injected metadata
            model1 = data["data"][0]
            assert model1["id"] == "gpt-test-1"
            assert model1["endpoint"] == "test-endpoint-1"
            assert model1["gpu"] == "NVIDIA RTX 4090"
            assert model1["vram"] == "24GB"
            assert model1["cpu"] == "Intel i9"
            assert model1["ram"] == "64GB"

            # Check second model
            model2 = data["data"][1]
            assert model2["id"] == "gpt-test-2"
            assert model2["endpoint"] == "test-endpoint-2"
            assert model2["gpu"] == "Apple M2 Max"
            assert "vram" not in model2  # Should not have vram since not in config

    @pytest.mark.asyncio
    async def test_list_models_endpoint_failure(self, client):
        """Test models endpoint handles individual endpoint failures gracefully."""
        mock_endpoints = [
            {"name": "failing-endpoint", "url": "https://failing.example.com"},
            {"name": "working-endpoint", "url": "https://working.example.com"},
        ]

        mock_working_response = {
            "data": [{"id": "working-model", "object": "model", "created": 1234567890}]
        }

        with patch("proxy.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            async def mock_get(url, headers=None):
                mock_response = Mock()

                if "failing.example.com" in url:
                    # Simulate HTTP error
                    mock_response.raise_for_status.side_effect = Exception(
                        "Connection failed"
                    )
                elif "working.example.com" in url:
                    mock_response.raise_for_status = Mock()
                    mock_response.json.return_value = mock_working_response

                return mock_response

            mock_client.get = mock_get

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()

            # Should still return successful models despite one endpoint failing
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "working-model"

    def test_list_models_empty_config(self, client):
        """Test models endpoint with empty configuration."""
        with patch("proxy.endpoints", []):
            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert data["data"] == []


if __name__ == "__main__":
    pytest.main([__file__])
