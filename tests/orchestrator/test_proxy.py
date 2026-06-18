"""
Test suite for the LLM Control Plane proxy functionality.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.logging_config import LOG_DIR_ENV
from src.orchestrator import proxy_services as proxy_module
from src.orchestrator.conversation_store import MemoryConversationStore
from src.orchestrator.proxy import app
from src.orchestrator.request_processor import RequestProcessor
from src.orchestrator.upstream_proxy import ProxyHandler
from src.orchestrator.utils import HeaderManager
from src.search.safety import EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def memory_conversation_store():
    """Install a fresh in-memory history store for each test."""
    store = MemoryConversationStore()
    proxy_module.set_conversation_store(store)
    yield store
    proxy_module.set_conversation_store(MemoryConversationStore())


@pytest.fixture
def mock_request():
    """Mock FastAPI request object."""
    request = Mock()
    request.headers = {
        "content-type": "application/json",
        "user-agent": "test-client",
    }
    return request


class _MockUpstreamResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.headers = {}
        self.content = json.dumps(payload).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeStreamingResponse:
    headers = {}

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
        yield "data: [DONE]"


class _DelayedStreamingResponse:
    headers = {}

    def __init__(self, release: asyncio.Event):
        self.release = release

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"First"}}]}'
        await self.release.wait()
        yield 'data: {"choices":[{"delta":{"content":"Second"}}]}'
        yield "data: [DONE]"


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class TestEndpointRouting:
    """Tests for endpoint routing functionality."""

    def test_get_target_endpoint_streaming(self):
        """Test endpoint routing for streaming requests."""
        # Mock the endpoints configuration
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]

        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints):
            # Should route to v1/chat/completions
            result = RequestProcessor.get_endpoint_url("/test-endpoint")
            assert result == "https://test.example.com/v1/chat/completions"

    def test_get_target_endpoint_non_streaming(self):
        """Test endpoint routing for non-streaming requests."""
        # Mock the endpoints configuration
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]

        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints):
            # Should route to v1/chat/completions
            result = RequestProcessor.get_endpoint_url("/test-endpoint")
            assert result == "https://test.example.com/v1/chat/completions"

    def test_get_target_endpoint_unknown(self):
        """Test endpoint routing for unknown endpoints."""
        # Mock empty endpoints configuration
        with patch("src.orchestrator.proxy_services.endpoints", []):
            result = RequestProcessor.get_endpoint_url("/unknown-endpoint")
            assert result is None

    def test_get_target_endpoint_with_subpath(self):
        """Test endpoint routing with subpaths."""
        # Mock the endpoints configuration
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]

        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints):
            # Should extract the first part of the path
            result = RequestProcessor.get_endpoint_url("/test-endpoint/some/subpath")
            assert result == "https://test.example.com/v1/chat/completions"

    def test_custom_endpoint_non_stream_includes_trace_header(
        self, client, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]
        payload = {"choices": [{"message": {"content": "Hello"}}]}

        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = _MockUpstreamResponse(payload)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response = client.post(
                "/test-endpoint",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        trace_id = response.headers.get("x-trace-id")
        assert response.status_code == 200
        assert trace_id is not None
        assert len(trace_id) == 32
        trace_events = [
            json.loads(line)
            for line in (tmp_path / "traces.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(trace_events) == 1
        assert trace_events[0]["phase"] == "completed"
        assert trace_events[0]["request_id"] == trace_id
        assert trace_events[0]["endpoint"] == "test-endpoint"
        assert trace_events[0]["status_code"] == 200

    def test_custom_endpoint_stream_includes_trace_header(
        self, client, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]

        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.stream.return_value = _FakeStreamContext(
                _FakeStreamingResponse()
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response = client.post(
                "/test-endpoint",
                json={
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        trace_id = response.headers.get("x-trace-id")
        assert response.status_code == 200
        assert trace_id is not None
        assert len(trace_id) == 32
        trace_events = [
            json.loads(line)
            for line in (tmp_path / "traces.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(trace_events) == 2
        assert [event["phase"] for event in trace_events] == ["started", "completed"]
        assert {event["request_id"] for event in trace_events} == {trace_id}
        assert {event["endpoint"] for event in trace_events} == {"test-endpoint"}
        assert {event["status_code"] for event in trace_events} == {200}

    def test_custom_endpoint_stream_persists_assistant_and_trace_history(
        self, client, memory_conversation_store, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
        mock_endpoints = [{"name": "test-endpoint", "url": "https://test.example.com"}]

        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.stream.return_value = _FakeStreamContext(
                _FakeStreamingResponse()
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response = client.post(
                "/test-endpoint",
                headers={"X-Conversation-ID": "session-stream"},
                json={
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 200
        assert memory_conversation_store.conversations["session-stream"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello"},
        ]
        trace_events = [
            json.loads(line)
            for line in (tmp_path / "traces.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        completed = [event for event in trace_events if event["phase"] == "completed"]
        assert completed[0]["conversation"] == {
            "assistant_chars": 5,
            "assistant_persisted": True,
        }

    @pytest.mark.asyncio
    async def test_stream_history_finalization_survives_caller_cancellation(
        self, memory_conversation_store, monkeypatch
    ):
        started = asyncio.Event()
        release = asyncio.Event()
        original_append = memory_conversation_store.append_messages

        async def delayed_append(conversation_id, messages):
            started.set()
            await release.wait()
            await original_append(conversation_id, messages)

        monkeypatch.setattr(memory_conversation_store, "append_messages", delayed_append)

        waiter = asyncio.create_task(
            RequestProcessor.finalize_stream_response(
                "session-cancel", "Partial assistant", None
            )
        )
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        release.set()
        if proxy_module.conversation_finalization_tasks:
            await asyncio.gather(
                *proxy_module.conversation_finalization_tasks,
                return_exceptions=True,
            )

        assert memory_conversation_store.conversations["session-cancel"] == [
            {"role": "assistant", "content": "Partial assistant"}
        ]

    @pytest.mark.asyncio
    async def test_stream_producer_persists_after_downstream_disconnect(
        self, memory_conversation_store
    ):
        release = asyncio.Event()
        request = Mock()
        request.headers = {"content-type": "application/json"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client.stream.return_value = _FakeStreamContext(
                _DelayedStreamingResponse(release)
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response = await ProxyHandler.stream_response(
                request,
                "https://test.example.com/v1/chat/completions",
                {},
                "session-disconnect",
            )
            stream = response.body_iterator
            first_chunk = await anext(stream)
            await stream.aclose()

        assert b"First" in first_chunk
        release.set()
        if proxy_module.stream_producer_tasks:
            await asyncio.gather(
                *proxy_module.stream_producer_tasks,
                return_exceptions=True,
            )
        if proxy_module.conversation_finalization_tasks:
            await asyncio.gather(
                *proxy_module.conversation_finalization_tasks,
                return_exceptions=True,
            )

        assert memory_conversation_store.conversations["session-disconnect"] == [
            {"role": "assistant", "content": "FirstSecond"}
        ]


class TestHeaderPreparation:
    """Tests for header preparation functionality."""

    def test_prepare_headers(self):
        """Test header preparation with API keys."""
        request = Mock()
        request.headers = {
            "content-type": "application/json",
            "user-agent": "test-client",
        }

        with patch("src.orchestrator.utils.filter_unsafe_headers") as mock_filter:
            mock_filter.return_value = {"content-type": "application/json"}

            result = HeaderManager.prepare_upstream_headers(request)

            expected = {
                "content-type": "application/json",
                "CF-Access-Client-Id": "test-api-key-id",
                "CF-Access-Client-Secret": "test-api-secret",
            }
            assert result == expected
            mock_filter.assert_called_once_with(dict(request.headers))


class TestConversationHistory:
    """Tests for conversation history functionality."""

    @pytest.mark.asyncio
    async def test_conversation_history_basic(self, memory_conversation_store):
        """Test basic conversation history tracking."""
        conversation_id = "test-conversation"
        messages = [{"role": "user", "content": "Hello"}]

        await memory_conversation_store.append_messages(conversation_id, messages)

        assert await memory_conversation_store.get_conversation(conversation_id) == messages

    @pytest.mark.asyncio
    async def test_conversation_history_append(self, memory_conversation_store):
        """Test appending to existing conversation history."""
        conversation_id = "test-conversation"
        await memory_conversation_store.append_messages(
            conversation_id, [{"role": "assistant", "content": "Hi there!"}]
        )

        await memory_conversation_store.append_messages(
            conversation_id, [{"role": "user", "content": "How are you?"}]
        )

        expected_messages = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        assert (
            await memory_conversation_store.get_conversation(conversation_id) == expected_messages
        )


class TestRequestPreparation:
    """Tests for proxy request preparation logic."""

    @pytest.mark.asyncio
    async def test_prepare_request_rewrites_latest_user_turn_with_retrieval_context(self):
        request = Mock()
        request.headers = {"X-Retrieval-Endpoint": "http://localhost:8100/api/retrieve"}
        request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "Base system"},
                        {"role": "assistant", "content": "Prior answer"},
                        {"role": "user", "content": "Need context"},
                    ]
                }
            ).encode("utf-8")
        )

        retrieval_user_content = """Retrieved evidence excerpts:

Source: doc-1
Excerpt:
Retrieved context

Current user question:
Need context"""
        retrieval_headers = {
            "X-Retrieval-Injected": "true",
            "X-Retrieval-Hits": "1",
        }

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(retrieval_user_content, retrieval_headers)),
        ):
            body, response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"] == [
            {"role": "system", "content": "Base system"},
            {"role": "assistant", "content": "Prior answer"},
            {"role": "user", "content": retrieval_user_content},
        ]
        assert response_headers == retrieval_headers

    @pytest.mark.asyncio
    async def test_prepare_request_does_not_persist_retrieval_in_history(self):
        await proxy_module.conversation_store.append_messages(
            "session-1", [{"role": "assistant", "content": "Earlier reply"}]
        )

        request = Mock()
        request.headers = {
            "X-Conversation-ID": "session-1",
            "X-Retrieval-Endpoint": "http://localhost:8100/api/retrieve",
        }
        request.body = AsyncMock(
            return_value=json.dumps(
                {"messages": [{"role": "user", "content": "Current question"}]}
            ).encode("utf-8")
        )

        retrieval_user_content = """Retrieved evidence excerpts:

Source: doc-1
Excerpt:
Retrieved context

Current user question:
Current question"""
        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(retrieval_user_content, {"X-Retrieval-Injected": "true"})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"] == [
            {"role": "assistant", "content": "Earlier reply"},
            {"role": "user", "content": retrieval_user_content},
        ]
        assert await proxy_module.conversation_store.get_conversation("session-1") == [
            {"role": "assistant", "content": "Earlier reply"},
            {"role": "user", "content": "Current question"},
        ]

    @pytest.mark.asyncio
    async def test_prepare_request_does_not_persist_reasoning_messages(self):
        request = Mock()
        request.headers = {
            "X-Conversation-ID": "session-2",
            "X-Reasoning-Effort": "high",
        }
        request.body = AsyncMock(
            return_value=json.dumps(
                {"messages": [{"role": "user", "content": "Reason carefully"}]}
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"][0] == {"role": "system", "content": "Reasoning: high"}
        assert await proxy_module.conversation_store.get_conversation("session-2") == [
            {"role": "user", "content": "Reason carefully"}
        ]

    @pytest.mark.asyncio
    async def test_prepare_request_keeps_system_prompt_before_reasoning(self):
        request = Mock()
        request.headers = {
            "X-Conversation-ID": "session-system-reasoning",
            "X-Reasoning-Effort": "high",
        }
        request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "Be terse."},
                        {"role": "user", "content": "Reason carefully"},
                    ]
                }
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"][:2] == [
            {"role": "system", "content": "Be terse."},
            {"role": "system", "content": "Reasoning: high"},
        ]
        assert await proxy_module.conversation_store.get_conversation(
            "session-system-reasoning"
        ) == [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Reason carefully"},
        ]

    @pytest.mark.asyncio
    async def test_prepare_request_does_not_persist_search_evidence(self):
        search_evidence = (
            f"{EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER}\n"
            "Untrusted web search candidates for this turn only.\n"
            "1. Mac mini M5 status\n"
            "   URL: https://example.com/mac-mini"
        )
        request = Mock()
        request.headers = {"X-Conversation-ID": "session-search"}
        request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": search_evidence},
                        {"role": "user", "content": "What is the status?"},
                    ]
                }
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"] == [
            {
                "role": "user",
                "content": (
                    f"{search_evidence}\n\n"
                    "Current user question:\nWhat is the status?"
                ),
            }
        ]
        assert await proxy_module.conversation_store.get_conversation("session-search") == [
            {"role": "user", "content": "What is the status?"}
        ]

    @pytest.mark.asyncio
    async def test_prepare_request_uses_actual_user_for_retrieval_with_search_evidence(self):
        search_evidence = (
            f"{EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER}\n"
            "Untrusted web search candidates for this turn only.\n"
            "1. Mac mini M5 status"
        )
        request = Mock()
        request.headers = {
            "X-Conversation-ID": "session-search-retrieval",
            "X-Retrieval-Endpoint": "http://localhost:8100/api/retrieve",
        }
        request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": search_evidence},
                        {"role": "user", "content": "What is the status?"},
                    ]
                }
            ).encode("utf-8")
        )
        retrieval_messages_at_call = []

        async def fetch_rag(messages, _retrieval_endpoint):
            retrieval_messages_at_call.extend(dict(message) for message in messages)
            return "Grounded user question", {}

        with patch.object(RequestProcessor, "_fetch_retrieval_message", fetch_rag):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert (
            RequestProcessor._latest_user_message(retrieval_messages_at_call)
            == "What is the status?"
        )
        assert body["messages"] == [
            {
                "role": "user",
                "content": (
                    f"{search_evidence}\n\n"
                    "Current user question:\nGrounded user question"
                ),
            }
        ]
        assert await proxy_module.conversation_store.get_conversation(
            "session-search-retrieval"
        ) == [{"role": "user", "content": "What is the status?"}]

    @pytest.mark.asyncio
    async def test_prepare_request_does_not_replay_stored_search_evidence(self):
        search_evidence = (
            f"{EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER}\n"
            "Untrusted web search candidates for this turn only.\n"
            "1. Prior result"
        )
        await proxy_module.conversation_store.append_messages(
            "session-contaminated",
            [
                {"role": "system", "content": search_evidence},
                {"role": "user", "content": "Prior question"},
                {"role": "assistant", "content": "Prior answer"},
            ],
        )

        request = Mock()
        request.headers = {"X-Conversation-ID": "session-contaminated"}
        request.body = AsyncMock(
            return_value=json.dumps(
                {"messages": [{"role": "user", "content": "Follow-up"}]}
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"] == [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
            {"role": "user", "content": "Follow-up"},
        ]

    @pytest.mark.asyncio
    async def test_prepare_request_history_mode_conversation_replays_raw_messages(self):
        await proxy_module.conversation_store.append_messages(
            "session-conversation",
            [
                {"role": "user", "content": "Prior question"},
                {"role": "assistant", "content": "Prior answer"},
            ],
        )
        request = Mock()
        request.headers = {
            "X-Conversation-ID": "session-conversation",
            "X-History-Mode": "conversation",
        }
        request.body = AsyncMock(
            return_value=json.dumps(
                {"messages": [{"role": "user", "content": "Current question"}]}
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"] == [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
            {"role": "user", "content": "Current question"},
        ]

    @pytest.mark.asyncio
    async def test_prepare_request_history_mode_none_uses_only_current_payload(self):
        await proxy_module.conversation_store.append_messages(
            "session-none",
            [
                {"role": "user", "content": "Prior question"},
                {"role": "assistant", "content": "Prior answer"},
            ],
        )
        request = Mock()
        request.headers = {"X-Conversation-ID": "session-none", "X-History-Mode": "none"}
        request.body = AsyncMock(
            return_value=json.dumps(
                {"messages": [{"role": "user", "content": "Current question"}]}
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"] == [
            {"role": "user", "content": "Current question"}
        ]
        assert await proxy_module.conversation_store.get_conversation("session-none") == [
            {"role": "user", "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
            {"role": "user", "content": "Current question"},
        ]

    @pytest.mark.asyncio
    async def test_prepare_request_rejects_old_history_mode_value(self):
        request = Mock()
        request.headers = {
            "X-Conversation-ID": "session-old-mode",
            "X-History-Mode": "compacted",
        }
        request.body = AsyncMock(
            return_value=json.dumps(
                {"messages": [{"role": "user", "content": "Current question"}]}
            ).encode("utf-8")
        )

        with pytest.raises(HTTPException) as exc:
            await RequestProcessor.prepare_request(request)

        assert exc.value.status_code == 400
        assert "unsupported history mode: compacted" in exc.value.detail

    @pytest.mark.asyncio
    async def test_prepare_request_history_mode_thread_uses_bounded_context(self):
        await proxy_module.conversation_store.append_messages(
            "session-thread",
            [
                {"role": "system", "content": "Base system"},
                {"role": "user", "content": "Old question"},
                {"role": "assistant", "content": "Old answer"},
                {"role": "user", "content": "Recent question"},
                {"role": "assistant", "content": "Recent answer"},
            ],
        )
        records = await proxy_module.conversation_store.get_conversation_message_records(
            "session-thread"
        )
        await proxy_module.conversation_store.upsert_thread_state(
            "session-thread",
            covered_message_id=records[2]["id"],
            state_text="prior thread state",
        )
        request = Mock()
        request.headers = {
            "X-Conversation-ID": "session-thread",
            "X-History-Mode": "thread",
        }
        request.body = AsyncMock(
            return_value=json.dumps(
                {"messages": [{"role": "user", "content": "Current question"}]}
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"][0] == {"role": "system", "content": "Base system"}
        assert body["messages"][1]["role"] == "system"
        assert "prior thread state" in body["messages"][1]["content"]
        assert "Recent answer" in body["messages"][1]["content"]
        assert "Current question" not in body["messages"][1]["content"]
        assert body["messages"][-1] == {
            "role": "user",
            "content": "Current question",
        }
        assert {"role": "user", "content": "Old question"} not in body["messages"]
        assert await proxy_module.conversation_store.get_conversation(
            "session-thread"
        ) == [
            {"role": "system", "content": "Base system"},
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Old answer"},
            {"role": "user", "content": "Recent question"},
            {"role": "assistant", "content": "Recent answer"},
            {"role": "user", "content": "Current question"},
        ]

    def test_normalize_retrieval_endpoint_targets_external_context_route(self):
        assert (
            RequestProcessor._normalize_retrieval_endpoint(
                "http://localhost:8100/api/retrieve"
            )
            == "http://localhost:8100/api/retrieve/context"
        )
        assert (
            RequestProcessor._normalize_retrieval_endpoint("localhost:8100")
            == "http://localhost:8100/api/retrieve/context"
        )

    def test_normalize_retrieval_endpoint_rejects_internal_evidence_route(self):
        with pytest.raises(ValueError, match="/api/retrieve/context"):
            RequestProcessor._normalize_retrieval_endpoint(
                "http://localhost:8100/api/retrieve/evidence"
            )

    @pytest.mark.asyncio
    async def test_prepare_request_rewrites_only_latest_user_turn_when_retrieval_is_present(
        self,
    ):
        request = Mock()
        request.headers = {"X-Retrieval-Endpoint": "http://localhost:8100/api/retrieve"}
        request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Need context"},
                    ]
                }
            ).encode("utf-8")
        )

        retrieval_user_content = """[Retrieved evidence excerpts]

Source: doc-1
Excerpt:
Definition text

[Current user question]
Need context"""
        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(retrieval_user_content, {"X-Retrieval-Injected": "true"})),
        ):
            body, _response_headers = await RequestProcessor.prepare_request(request)

        assert body["messages"][0] == {
            "role": "user",
            "content": retrieval_user_content,
        }
        assert all(isinstance(message, dict) for message in body["messages"])

    @pytest.mark.asyncio
    async def test_fetch_retrieval_message_returns_none_without_grounded_user_message(self):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "context_blocks": [{"chunk_id": "doc-1"}],
            "grounded_user_message": "   ",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            retrieval_user_content, retrieval_headers = await RequestProcessor._fetch_retrieval_message(
                [{"role": "user", "content": "Need context"}],
                "http://localhost:8100/api/retrieve",
            )

        assert retrieval_user_content is None
        assert retrieval_headers["X-Retrieval-Injected"] == "false"
        assert retrieval_headers["X-Retrieval-Reason"] == "empty-grounded-user-message"

    @pytest.mark.asyncio
    async def test_fetch_retrieval_message_uses_grounded_user_message_from_external_response(
        self,
    ):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "context_blocks": [
                {"chunk_id": "chunk-1", "content": "Strong semantic match"},
                {"chunk_id": "chunk-2", "content": "Another match"},
            ],
            "grounded_user_message": """[Retrieved evidence excerpts]

Source: chunk-1
Excerpt:
Strong semantic match

[Current user question]
Need context""",
            "mode": "hybrid",
            "truncated": True,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            retrieval_user_content, retrieval_headers = await RequestProcessor._fetch_retrieval_message(
                [{"role": "user", "content": "Need context"}],
                "http://localhost:8100/api/retrieve",
            )

        mock_client.post.assert_awaited_once()
        post_args = mock_client.post.await_args.args
        post_kwargs = mock_client.post.await_args.kwargs
        assert post_args[0] == "http://localhost:8100/api/retrieve/context"
        assert post_kwargs["json"] == {"query": "Need context", "limit": 10}
        assert retrieval_user_content is not None
        assert "Strong semantic match" in retrieval_user_content
        assert retrieval_headers == {
            "X-Retrieval-Endpoint": "http://localhost:8100/api/retrieve/context",
            "X-Retrieval-Hits": "2",
            "X-Retrieval-Injected": "true",
            "X-Retrieval-Mode": "hybrid",
            "X-Retrieval-Truncated": "true",
        }

    @pytest.mark.asyncio
    async def test_fetch_retrieval_message_logs_evidence_summary(self, caplog):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "context_blocks": [
                {
                    "chunk_id": "chunk-1",
                    "content": "Strong semantic match from handbook section",
                },
                {
                    "chunk_id": "chunk-2",
                    "content": "Another useful match from release notes",
                },
            ],
            "grounded_user_message": "grounded",
            "mode": "hybrid",
            "truncated": False,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            with caplog.at_level("INFO"):
                await RequestProcessor._fetch_retrieval_message(
                    [{"role": "user", "content": "Need context"}],
                    "http://localhost:8100/api/retrieve",
                )

        assert (
            "Retrieval evidence retrieved 2 blocks via http://localhost:8100/api/retrieve/context"
            in caplog.text
        )
        assert "mode=hybrid" in caplog.text
        assert "truncated=False" in caplog.text


class TestConversationHistoryEndpoint:

    def test_list_conversations_returns_most_recent_first(
        self, client, memory_conversation_store
    ):
        memory_conversation_store.updated_at = {
            "older": "2026-05-20T10:00:00+00:00",
            "newer": "2026-05-21T10:00:00+00:00",
        }
        memory_conversation_store.conversations = {
            "older": [{"role": "user", "content": "first"}],
            "newer": [
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "reply"},
            ],
        }

        response = client.get("/conversations")

        assert response.status_code == 200
        assert response.json() == [
            {
                "conversation_id": "newer",
                "last_updated": "2026-05-21T10:00:00+00:00",
                "message_count": 2,
            },
            {
                "conversation_id": "older",
                "last_updated": "2026-05-20T10:00:00+00:00",
                "message_count": 1,
            },
        ]

    def test_retrieve_conversation_returns_404_for_unknown_id(self, client):
        response = client.post("/conversations/retrieve", json={"conversation_id": "missing"})

        assert response.status_code == 404
        assert response.json()["detail"] == "Conversation 'missing' not found"

    def test_retrieve_conversation_omits_search_evidence(
        self, client, memory_conversation_store
    ):
        search_evidence = (
            f"{EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER}\n"
            "Untrusted web search candidates for this turn only.\n"
            "1. Prior result"
        )
        memory_conversation_store.conversations = {
            "session-contaminated": [
                {"role": "system", "content": search_evidence},
                {"role": "user", "content": "Prior question"},
            ]
        }
        memory_conversation_store.updated_at = {
            "session-contaminated": "2026-06-07T00:00:00+00:00"
        }

        response = client.post(
            "/conversations/retrieve", json={"conversation_id": "session-contaminated"}
        )

        assert response.status_code == 200
        assert response.json() == [{"role": "user", "content": "Prior question"}]

    def test_retrieve_conversation_returns_public_transcript_only(
        self, client, memory_conversation_store
    ):
        memory_conversation_store.conversations = {
            "session-raw": [
                {"role": "system", "content": "Reasoning: high"},
                {"role": "user", "content": "Question", "reasoning": "hidden"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call-1"}],
                },
                {
                    "role": "assistant",
                    "content": "tool planning",
                    "function_call": {"name": "search"},
                },
                {"role": "developer", "content": "internal instruction"},
                {"role": "tool", "content": "tool output"},
                {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning": "private chain",
                    "metadata": {"workflow_id": "wf"},
                },
            ]
        }
        memory_conversation_store.updated_at = {
            "session-raw": "2026-06-07T00:00:00+00:00"
        }

        response = client.post(
            "/conversations/retrieve", json={"conversation_id": "session-raw"}
        )

        assert response.status_code == 200
        assert response.json() == [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Final answer"},
        ]
        assert asyncio.run(
            memory_conversation_store.get_conversation("session-raw")
        ) == memory_conversation_store.conversations["session-raw"]

    def test_append_conversation_accepts_public_messages_only(
        self, client, memory_conversation_store
    ):
        response = client.post(
            "/conversations/append",
            json={
                "conversation_id": "session-append",
                "messages": [
                    {"role": "system", "content": "Reasoning: high"},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Final", "reasoning": "hidden"},
                    {"role": "tool", "content": "tool output"},
                ],
            },
        )

        assert response.status_code == 200
        assert response.json() == {"appended": 2}
        assert asyncio.run(
            memory_conversation_store.get_conversation("session-append")
        ) == [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Final"},
        ]

    def test_append_conversation_rejects_non_transcript_messages(self, client):
        response = client.post(
            "/conversations/append",
            json={
                "conversation_id": "session-empty",
                "messages": [
                    {"role": "system", "content": "Reasoning: high"},
                    {"role": "tool", "content": "tool output"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                ],
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "No public messages to append"

    def test_append_then_retrieve_returns_user_and_final_assistant_only(self, client):
        append_response = client.post(
            "/conversations/append",
            json={
                "conversation_id": "session-public",
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "Final answer",
                        "reasoning": "hidden",
                        "metadata": {"workflow_id": "wf_123"},
                    },
                ],
            },
        )
        retrieve_response = client.post(
            "/conversations/retrieve", json={"conversation_id": "session-public"}
        )

        assert append_response.status_code == 200
        assert retrieve_response.status_code == 200
        assert retrieve_response.json() == [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Final answer"},
        ]

    def test_retrieve_conversation_control_state_returns_reasoning_and_route(
        self, client, memory_conversation_store
    ):
        memory_conversation_store.conversation_control_states = {
            "session-state": {
                "conversation_id": "session-state",
                "route_endpoint": "primary",
                "reasoning_effort": "high",
                "slots": {},
                "updated_at": "2026-06-09T00:00:00+00:00",
            }
        }

        response = client.post(
            "/conversations/state", json={"conversation_id": "session-state"}
        )

        assert response.status_code == 200
        assert response.json()["route_endpoint"] == "primary"
        assert response.json()["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_update_conversation_persists_assistant_reply(self):
        await RequestProcessor.update_conversation("session-3", "Assistant reply")

        assert await proxy_module.conversation_store.get_conversation("session-3") == [
            {"role": "assistant", "content": "Assistant reply"}
        ]


class TestCanonicalConversationState:
    @pytest.mark.asyncio
    async def test_replay_payload_rejected_before_append(self):
        await proxy_module.conversation_store.append_messages(
            "session-replay",
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Answer"},
            ],
        )
        request = Mock()
        request.headers = {"X-Conversation-ID": "session-replay"}
        request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "First"},
                        {"role": "assistant", "content": "Answer"},
                        {"role": "user", "content": "Follow-up"},
                    ]
                }
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            with pytest.raises(Exception) as exc_info:
                await RequestProcessor.prepare_request(request)

        assert getattr(exc_info.value, "status_code", None) == 400
        assert await proxy_module.conversation_store.get_conversation("session-replay") == [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
        ]

    @pytest.mark.asyncio
    async def test_ephemeral_search_and_retrieval_messages_do_not_create_replay_prefix(self):
        await proxy_module.conversation_store.append_messages(
            "session-ephemeral", [{"role": "user", "content": "Prior"}]
        )
        request = Mock()
        request.headers = {"X-Conversation-ID": "session-ephemeral"}
        request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER}\nturn-local search",
                        },
                        {
                            "role": "user",
                            "content": "[Retrieved reference excerpts]\nturn-local retrieval",
                        },
                        {"role": "user", "content": "Follow-up"},
                    ]
                }
            ).encode("utf-8")
        )

        with patch.object(
            RequestProcessor,
            "_fetch_retrieval_message",
            AsyncMock(return_value=(None, {})),
        ):
            body, _headers = await RequestProcessor.prepare_request(request)

        assert body["messages"][-1] == {"role": "user", "content": "Follow-up"}
        assert await proxy_module.conversation_store.get_conversation("session-ephemeral") == [
            {"role": "user", "content": "Prior"},
            {"role": "user", "content": "Follow-up"},
        ]

    def test_direct_endpoint_conflict_returns_409(self, client):
        mock_endpoints = [
            {"name": "primary", "url": "https://primary.example.com"},
            {"name": "secondary", "url": "https://secondary.example.com"},
        ]
        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            first = client.post(
                "/primary",
                headers={"X-Conversation-ID": "session-route"},
                json={"messages": [{"role": "user", "content": "one"}]},
            )
            second = client.post(
                "/secondary",
                headers={"X-Conversation-ID": "session-route"},
                json={"messages": [{"role": "user", "content": "two"}]},
            )

        assert first.status_code == 200
        assert second.status_code == 409

    def test_direct_endpoint_switch_with_opt_in_replays_history(
        self, client, memory_conversation_store
    ):
        mock_endpoints = [
            {"name": "primary", "url": "https://primary.example.com"},
            {"name": "secondary", "url": "https://secondary.example.com"},
        ]
        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            first = client.post(
                "/primary",
                headers={"X-Conversation-ID": "session-route-switch"},
                json={"messages": [{"role": "user", "content": "one"}]},
            )
            second = client.post(
                "/secondary",
                headers={
                    "X-Conversation-ID": "session-route-switch",
                    "X-Allow-Route-Switch": "true",
                },
                json={"messages": [{"role": "user", "content": "two"}]},
            )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.headers["x-route-switched"] == "true"
        assert second.headers["x-route-previous"] == "primary"
        assert second.headers["x-route-decision"] == "secondary"
        assert second.headers["x-warning"] == proxy_module.SWITCH_WARNING_MESSAGE
        assert memory_conversation_store.conversation_control_states["session-route-switch"][
            "route_endpoint"
        ] == "secondary"
        persisted = memory_conversation_store.conversations["session-route-switch"]
        assert persisted == [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "ok"},
        ]
        assert all("Warning:" not in message["content"] for message in persisted)
        assert mock_client.post.call_args.args[0] == (
            "https://secondary.example.com/v1/chat/completions"
        )
        second_body = mock_client.post.call_args.kwargs["json"]
        assert second_body["messages"] == [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "two"},
        ]

    def test_reasoning_conflict_without_opt_in_returns_409(self, client):
        mock_endpoints = [{"name": "primary", "url": "https://primary.example.com"}]
        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            first = client.post(
                "/primary",
                headers={
                    "X-Conversation-ID": "session-reasoning-conflict",
                    "X-Reasoning-Effort": "low",
                },
                json={"messages": [{"role": "user", "content": "one"}]},
            )
            second = client.post(
                "/primary",
                headers={
                    "X-Conversation-ID": "session-reasoning-conflict",
                    "X-Reasoning-Effort": "high",
                },
                json={"messages": [{"role": "user", "content": "two"}]},
            )

        assert first.status_code == 200
        assert second.status_code == 409

    def test_reasoning_switch_with_opt_in_updates_state(
        self, client, memory_conversation_store
    ):
        mock_endpoints = [{"name": "primary", "url": "https://primary.example.com"}]
        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            first = client.post(
                "/primary",
                headers={
                    "X-Conversation-ID": "session-reasoning-switch",
                    "X-Reasoning-Effort": "low",
                },
                json={"messages": [{"role": "user", "content": "one"}]},
            )
            second = client.post(
                "/primary",
                headers={
                    "X-Conversation-ID": "session-reasoning-switch",
                    "X-Reasoning-Effort": "high",
                    "X-Allow-Reasoning-Switch": "true",
                },
                json={"messages": [{"role": "user", "content": "two"}]},
            )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.headers["x-reasoning-switched"] == "true"
        assert second.headers["x-reasoning-previous"] == "low"
        assert second.headers["x-reasoning-effort"] == "high"
        assert second.headers["x-warning"] == proxy_module.SWITCH_WARNING_MESSAGE
        assert memory_conversation_store.conversation_control_states["session-reasoning-switch"][
            "reasoning_effort"
        ] == "high"
        second_body = mock_client.post.call_args.kwargs["json"]
        assert second_body["reasoning_effort"] == "high"
        assert second_body["messages"][0] == {
            "role": "system",
            "content": "Reasoning: high",
        }

    def test_pinned_reasoning_is_reused_when_later_turn_omits_header(self, client):
        mock_endpoints = [{"name": "primary", "url": "https://primary.example.com"}]
        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            first = client.post(
                "/primary",
                headers={
                    "X-Conversation-ID": "session-reasoning",
                    "X-Reasoning-Effort": "high",
                },
                json={"messages": [{"role": "user", "content": "one"}]},
            )
            second = client.post(
                "/primary",
                headers={"X-Conversation-ID": "session-reasoning"},
                json={"messages": [{"role": "user", "content": "two"}]},
            )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.headers["x-reasoning-effort"] == "high"
        second_body = mock_client.post.call_args.kwargs["json"]
        assert second_body["reasoning_effort"] == "high"
        assert second_body["messages"][0] == {
            "role": "system",
            "content": "Reasoning: high",
        }

    def test_no_conversation_id_keeps_stateless_behavior(self, client):
        mock_endpoints = [{"name": "primary", "url": "https://primary.example.com"}]
        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response = client.post(
                "/primary",
                headers={"X-Reasoning-Effort": "low"},
                json={"messages": [{"role": "user", "content": "stateless"}]},
            )

        assert response.status_code == 200
        assert proxy_module.conversation_store.conversations == {}
        assert proxy_module.conversation_store.conversation_control_states == {}

    def test_skip_conversation_header_bypasses_state_and_message_persistence(self, client):
        mock_endpoints = [{"name": "primary", "url": "https://primary.example.com"}]
        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response = client.post(
                "/primary",
                headers={
                    "X-Conversation-ID": "session-skip",
                    "X-LLMCP-Skip-Conversation": "true",
                    "X-Reasoning-Effort": "high",
                },
                json={"messages": [{"role": "user", "content": "transient"}]},
            )

        assert response.status_code == 200
        assert response.headers["x-conversation-skipped"] == "true"
        assert "x-conversation-id" not in response.headers
        assert proxy_module.conversation_store.conversations == {}
        assert proxy_module.conversation_store.conversation_control_states == {}
        upstream_body = mock_client.post.call_args.kwargs["json"]
        assert upstream_body["messages"] == [
            {"role": "system", "content": "Reasoning: high"},
            {"role": "user", "content": "transient"},
        ]

    def test_slot_affinity_success_injects_slot_fields(self, client):
        mock_endpoints = [
            {
                "name": "llama",
                "url": "https://llama.example.com",
                "slot_affinity": True,
            }
        ]
        slots_response = Mock()
        slots_response.raise_for_status = Mock()
        slots_response.json.return_value = [{"id": 2, "is_processing": False}]

        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = slots_response
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response = client.post(
                "/llama",
                headers={"X-Conversation-ID": "session-slot"},
                json={"messages": [{"role": "user", "content": "slot me"}]},
            )

        assert response.status_code == 200
        assert response.headers["x-upstream-slot-id"] == "2"
        assert response.headers["x-upstream-slot-status"] == "affinity-applied"
        upstream_body = mock_client.post.call_args.kwargs["json"]
        assert upstream_body["id_slot"] == 2
        assert upstream_body["cache_prompt"] is True

    def test_slot_affinity_non_llama_upstream_does_not_fail_request(self, client):
        mock_endpoints = [
            {
                "name": "not-llama",
                "url": "https://not-llama.example.com",
                "slot_affinity": True,
            }
        ]
        slots_response = Mock()
        slots_response.raise_for_status.side_effect = Exception("not found")

        with patch("src.orchestrator.proxy_services.endpoints", mock_endpoints), patch(
            "httpx.AsyncClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = slots_response
            mock_client.post.return_value = _MockUpstreamResponse(
                {"choices": [{"message": {"content": "ok"}}]}
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response = client.post(
                "/not-llama",
                headers={"X-Conversation-ID": "session-non-llama"},
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 200
        assert response.headers["x-upstream-slot-status"] == "unavailable"
        upstream_body = mock_client.post.call_args.kwargs["json"]
        assert "id_slot" not in upstream_body
        assert "cache_prompt" not in upstream_body


class TestModelsEndpoint:
    """Tests for the /models endpoint functionality."""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Test successful models endpoint with mock responses."""
        test_client = TestClient(app)
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

        with (
            patch("src.orchestrator.proxy_services.endpoints", mock_endpoints),
            patch("httpx.AsyncClient") as mock_client_class,
        ):

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock responses for each endpoint
            async def mock_get(url, **_kwargs):
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

            response = test_client.get("/models")

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
    async def test_list_models_keeps_evo_x2_primary_utility_distinct(self):
        """Test /models exposes the Evo X2 servers as distinct endpoints."""
        test_client = TestClient(app)
        mock_endpoints = [
            {
                "name": "gmktec-evo-x2-primary",
                "url": "https://llm-evo-x2.paperclips.dev",
                "soc": "AMD Ryzen AI Max+ 395",
                "ram": "128GB",
            },
            {
                "name": "gmktec-evo-x2-utility",
                "url": "https://llm-evo-x2-utility.paperclips.dev",
                "soc": "AMD Ryzen AI Max+ 395",
                "ram": "128GB",
            },
        ]

        with (
            patch("src.orchestrator.proxy_services.endpoints", mock_endpoints),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            async def mock_get(url, **_kwargs):
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                if "evo-x2-utility" in url:
                    mock_response.json.return_value = {
                        "data": [{"id": "utility-model", "object": "model"}]
                    }
                else:
                    mock_response.json.return_value = {
                        "data": [{"id": "primary-model", "object": "model"}]
                    }
                return mock_response

            mock_client.get = mock_get

            response = test_client.get("/models")

        assert response.status_code == 200
        models = response.json()["data"]
        assert {
            (model["endpoint"], model["id"])
            for model in models
        } == {
            ("gmktec-evo-x2-primary", "primary-model"),
            ("gmktec-evo-x2-utility", "utility-model"),
        }
        assert all(model["soc"] == "AMD Ryzen AI Max+ 395" for model in models)
        assert all(model["ram"] == "128GB" for model in models)

    @pytest.mark.asyncio
    async def test_list_models_endpoint_failure(self):
        """Test models endpoint handles individual endpoint failures gracefully."""
        test_client = TestClient(app)
        mock_endpoints = [
            {"name": "failing-endpoint", "url": "https://failing.example.com"},
            {"name": "working-endpoint", "url": "https://working.example.com"},
        ]

        mock_working_response = {
            "data": [{"id": "working-model", "object": "model", "created": 1234567890}]
        }

        with (
            patch("src.orchestrator.proxy_services.endpoints", mock_endpoints),
            patch("httpx.AsyncClient") as mock_client_class,
        ):

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            async def mock_get(url, **_kwargs):
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

            response = test_client.get("/models")

            assert response.status_code == 200
            data = response.json()

            # Should still return successful models despite one endpoint failing
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "working-model"

    def test_list_models_empty_config(self):
        """Test models endpoint with empty configuration."""
        test_client = TestClient(app)
        with patch("src.orchestrator.proxy_services.endpoints", []):
            response = test_client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert data["data"] == []


if __name__ == "__main__":
    pytest.main([__file__])
