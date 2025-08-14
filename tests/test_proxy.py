"""
Test suite for the LLM Control Plane proxy functionality.
"""

import json
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import (
    DEFAULT_ENDPOINT,
    ENDPOINT_MAP,
    HEALTH_CACHE_TTL,
    app,
    check_endpoint_health,
    convo_history,
    endpoint_health_cache,
    get_available_endpoint,
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
    """Clear health cache before each test."""
    endpoint_health_cache.clear()
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


class TestEndpointHealthCheck:
    """Tests for endpoint health checking functionality."""

    @pytest.mark.asyncio
    async def test_check_endpoint_health_success(self):
        """Test successful endpoint health check."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response

            result = await check_endpoint_health("https://test.example.com")

            assert result is True
            mock_client.get.assert_called_once_with(
                "https://test.example.com",
                headers={
                    "CF-Access-Client-Id": "test-api-key-id",
                    "CF-Access-Client-Secret": "test-api-secret",
                },
            )

    @pytest.mark.asyncio
    async def test_check_endpoint_health_client_error(self):
        """Test endpoint health check with 4xx status (should be considered healthy)."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.status_code = 403  # Forbidden but endpoint is up
            mock_client.get.return_value = mock_response

            result = await check_endpoint_health("https://test.example.com")

            assert result is True

    @pytest.mark.asyncio
    async def test_check_endpoint_health_server_error(self):
        """Test endpoint health check with 5xx status (should be considered unhealthy)."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.status_code = 500  # Server error
            mock_client.get.return_value = mock_response

            result = await check_endpoint_health("https://test.example.com")

            assert result is False

    @pytest.mark.asyncio
    async def test_check_endpoint_health_network_error(self):
        """Test endpoint health check with network error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")

            result = await check_endpoint_health("https://test.example.com")

            assert result is False

    @pytest.mark.asyncio
    async def test_check_endpoint_health_caching(self):
        """Test that health check results are properly cached."""
        endpoint_url = "https://test.example.com"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response

            # First call should hit the network
            result1 = await check_endpoint_health(endpoint_url)
            assert result1 is True
            assert mock_client.get.call_count == 1

            # Second call should use cache
            result2 = await check_endpoint_health(endpoint_url)
            assert result2 is True
            assert mock_client.get.call_count == 1  # No additional network call

            # Check cache contents
            assert endpoint_url in endpoint_health_cache
            is_healthy, timestamp = endpoint_health_cache[endpoint_url]
            assert is_healthy is True
            assert time.time() - timestamp < 1  # Recently cached

    @pytest.mark.asyncio
    async def test_check_endpoint_health_cache_expiry(self):
        """Test that expired cache entries trigger new health checks."""
        endpoint_url = "https://test.example.com"

        # Manually add expired cache entry
        expired_time = time.time() - HEALTH_CACHE_TTL - 1
        endpoint_health_cache[endpoint_url] = (True, expired_time)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response

            result = await check_endpoint_health(endpoint_url)

            assert result is True
            mock_client.get.assert_called_once()  # Should make network call due to expired cache


class TestEndpointRouting:
    """Tests for endpoint routing functionality."""

    def test_get_target_endpoint_mapped(self):
        """Test endpoint mapping for known endpoints."""
        for endpoint_key in ENDPOINT_MAP.keys():
            result = get_target_endpoint(endpoint_key)
            assert result == ENDPOINT_MAP[endpoint_key]

    def test_get_target_endpoint_default(self):
        """Test default endpoint for unknown paths."""
        result = get_target_endpoint("unknown-endpoint")
        assert result == DEFAULT_ENDPOINT

    def test_get_target_endpoint_empty_path(self):
        """Test default endpoint for empty path."""
        result = get_target_endpoint("")
        assert result == DEFAULT_ENDPOINT

    def test_get_target_endpoint_with_subpath(self):
        """Test endpoint mapping with subpaths."""
        result = get_target_endpoint("gpt-oss-20b/some/subpath")
        assert result == ENDPOINT_MAP["gpt-oss-20b"]

    @pytest.mark.asyncio
    async def test_get_available_endpoint_primary_healthy(self):
        """Test that primary endpoint is returned when healthy."""
        with patch("proxy.check_endpoint_health", return_value=True):
            result = await get_available_endpoint("gpt-oss-20b")
            assert result == ENDPOINT_MAP["gpt-oss-20b"]

    @pytest.mark.asyncio
    async def test_get_available_endpoint_fallback(self):
        """Test fallback to alternative endpoint when primary is unhealthy."""

        def mock_health_check(endpoint_url):
            # Primary endpoint is unhealthy, but fallback is healthy
            if endpoint_url == ENDPOINT_MAP["gpt-oss-20b"]:
                return False
            return True

        with patch("proxy.check_endpoint_health", side_effect=mock_health_check):
            result = await get_available_endpoint("gpt-oss-20b")
            # Should return a different endpoint (not the primary)
            assert result != ENDPOINT_MAP["gpt-oss-20b"]
            assert result in ENDPOINT_MAP.values()

    @pytest.mark.asyncio
    async def test_get_available_endpoint_all_unhealthy(self):
        """Test default endpoint when all are unhealthy."""
        with patch("proxy.check_endpoint_health", return_value=False):
            result = await get_available_endpoint("gpt-oss-20b")
            assert result == DEFAULT_ENDPOINT


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
        assert result_body == b""
        assert is_streaming is False

    def test_parse_and_inject_history_invalid_json(self):
        """Test parsing invalid JSON body."""
        invalid_json = b"not json"
        result_body, is_streaming = parse_and_inject_history(invalid_json, "test-convo")
        assert result_body == invalid_json
        assert is_streaming is False

    def test_parse_and_inject_history_no_messages(self):
        """Test parsing JSON without messages."""
        body_json = {"model": "gpt-4", "stream": True}
        body = json.dumps(body_json).encode()

        result_body, is_streaming = parse_and_inject_history(body, "test-convo")

        assert json.loads(result_body) == body_json
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
        result_json = json.loads(result_body)

        assert result_json["messages"] == body_json["messages"]
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
        result_json = json.loads(result_body)

        expected_messages = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        assert result_json["messages"] == expected_messages
        assert convo_history[convo_id] == expected_messages


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_endpoint_all_healthy(self, client):
        """Test health endpoint when all endpoints are healthy."""
        with patch("proxy.check_endpoint_health", return_value=True):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "endpoints" in data

            # All endpoints should be marked as healthy
            for endpoint_name in ENDPOINT_MAP.keys():
                assert data["endpoints"][endpoint_name] is True

    def test_health_endpoint_degraded(self, client):
        """Test health endpoint when all endpoints are unhealthy."""
        with patch("proxy.check_endpoint_health", return_value=False):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert "endpoints" in data

            # All endpoints should be marked as unhealthy
            for endpoint_name in ENDPOINT_MAP.keys():
                assert data["endpoints"][endpoint_name] is False

    def test_health_endpoint_mixed(self, client):
        """Test health endpoint with mixed endpoint health."""

        def mock_health_check(endpoint_url, **kwargs):
            # Only one endpoint is healthy
            return endpoint_url == list(ENDPOINT_MAP.values())[0]

        with patch("proxy.check_endpoint_health", side_effect=mock_health_check):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"  # At least one is healthy
            assert "endpoints" in data


if __name__ == "__main__":
    pytest.main([__file__])
