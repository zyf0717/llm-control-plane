"""
Test suite for the LLM Control Plane proxy functionality.
"""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from proxy import (
    DEFAULT_ENDPOINT,
    ENDPOINT_MAP,
    app,
    convo_history,
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


if __name__ == "__main__":
    pytest.main([__file__])
