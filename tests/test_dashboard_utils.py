import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.dashboard import utils


def test_load_rag_endpoint_config_reads_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
rag:
  default_endpoint: "http://localhost:8100/api/retrieve/context"
  endpoints:
    - name: "localhost:8100"
      retrieve_url: "http://localhost:8100/api/retrieve/context"
      health_url: "http://localhost:8100/api/health"
    - name: "rag.internal:8200"
      retrieve_url: "http://rag.internal:8200/api/retrieve/context"
      health_url: "http://rag.internal:8200/api/health"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "CONFIG_PATH", config_path)

    endpoints, default_endpoint = utils.load_rag_endpoint_config()

    assert endpoints == [
        {
            "name": "localhost:8100",
            "retrieve_url": "http://localhost:8100/api/retrieve/context",
            "health_url": "http://localhost:8100/api/health",
        },
        {
            "name": "rag.internal:8200",
            "retrieve_url": "http://rag.internal:8200/api/retrieve/context",
            "health_url": "http://rag.internal:8200/api/health",
        },
    ]
    assert default_endpoint == "http://localhost:8100/api/retrieve/context"


def test_read_trace_events_handles_missing_file(tmp_path):
    assert utils.read_trace_events(trace_path=tmp_path / "missing.jsonl") == []


def test_read_trace_events_filters_and_returns_newest_first(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    events = [
        {
            "timestamp": "2026-06-09T00:00:00+00:00",
            "request_id": "trace-old",
            "convo_id": "convo-a",
            "endpoint": "primary",
            "status_code": 200,
        },
        {
            "timestamp": "2026-06-09T00:01:00+00:00",
            "request_id": "trace-mid",
            "convo_id": "convo-b",
            "endpoint": "secondary",
            "status_code": 200,
        },
        {
            "timestamp": "2026-06-09T00:02:00+00:00",
            "request_id": "trace-new",
            "convo_id": "convo-a",
            "endpoint": "primary",
            "status_code": 500,
        },
    ]
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(events[0]),
                "{bad-json",
                json.dumps(events[1]),
                json.dumps(events[2]),
                '{"partial":',
            ]
        ),
        encoding="utf-8",
    )

    assert utils.read_trace_events(trace_path=trace_path, max_events=2) == [
        events[2],
        events[1],
    ]
    assert utils.read_trace_events(
        trace_path=trace_path,
        convo_id="convo-a",
        endpoint="primary",
        max_events=10,
    ) == [events[2], events[0]]
    assert utils.read_trace_events(
        trace_path=trace_path,
        trace_id="mid",
        max_events=10,
    ) == [events[1]]


def test_load_rag_endpoint_config_falls_back_to_default(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("rag: {}", encoding="utf-8")
    monkeypatch.setattr(utils, "CONFIG_PATH", config_path)

    endpoints, default_endpoint = utils.load_rag_endpoint_config()

    assert endpoints == [
        {
            "name": "localhost:8100",
            "retrieve_url": "http://localhost:8100/api/retrieve/context",
            "health_url": "http://localhost:8100/api/health",
        }
    ]
    assert default_endpoint == "http://localhost:8100/api/retrieve/context"


def test_load_search_provider_config_reads_enabled_providers_in_priority_order(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
search:
  enabled: true
  providers:
    mojeek_html:
      enabled: true
      priority: 30
    duckduckgo_html:
      enabled: true
      priority: 10
    marginalia_html:
      enabled: false
      priority: 20
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "CONFIG_PATH", config_path)

    providers = utils.load_search_provider_config()

    assert providers == [
        {"id": "duckduckgo_html", "name": "DuckDuckGo", "priority": 10},
        {"id": "mojeek_html", "name": "Mojeek", "priority": 30},
    ]


def test_load_search_planner_max_context_chars_reads_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "search: {planner_max_context_chars: 24}",
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "CONFIG_PATH", config_path)

    assert utils.load_search_planner_max_context_chars() == 24


def test_trim_search_context_preserves_tail():
    trimmed = utils.trim_search_context("abcdef", 4)

    assert trimmed == "cdef"


def test_fetch_available_search_providers_returns_none_when_search_disabled(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("search: {enabled: false}", encoding="utf-8")
    monkeypatch.setattr(utils, "CONFIG_PATH", config_path)

    choices, selected = utils.fetch_available_search_providers()

    assert choices == {
        utils.NONE_SEARCH_PROVIDER_VALUE: utils.NONE_SEARCH_PROVIDER_LABEL
    }
    assert selected == utils.NONE_SEARCH_PROVIDER_VALUE


@pytest.mark.asyncio
async def test_fetch_search_results_posts_to_proxy(monkeypatch):
    monkeypatch.setattr(utils, "PROXY_BASE_URL", "http://proxy.local")
    monkeypatch.setattr(utils, "load_search_planner_max_context_chars", lambda: 12000)

    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {"provider": "duckduckgo_html", "results": []}

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = response

        payload = await utils.fetch_search_results(
            query="Ada Lovelace",
            provider="duckduckgo_html",
            count=5,
        )

    assert payload == {"provider": "duckduckgo_html", "results": []}
    mock_client.post.assert_awaited_once_with(
        "http://proxy.local/search/web",
        json={
            "query": "Ada Lovelace",
            "provider": "duckduckgo_html",
            "count": 5,
        },
    )


@pytest.mark.asyncio
async def test_fetch_search_results_posts_trimmed_context(monkeypatch):
    monkeypatch.setattr(utils, "PROXY_BASE_URL", "http://proxy.local")
    monkeypatch.setattr(utils, "load_search_planner_max_context_chars", lambda: 12)

    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {"provider": "duckduckgo_html", "results": []}

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.return_value = response

        await utils.fetch_search_results(
            query="Ada Lovelace",
            provider="duckduckgo_html",
            count=5,
            context="0123456789abcdef",
        )

    sent = mock_client.post.await_args.kwargs["json"]
    assert sent["context"] == "456789abcdef"


@pytest.mark.asyncio
async def test_fetch_available_rag_endpoints_keeps_unhealthy_configured_options(
    monkeypatch,
):
    monkeypatch.setattr(
        utils,
        "load_rag_endpoint_config",
        lambda: (
            [
                {
                    "name": "healthy",
                    "retrieve_url": "http://healthy/api/retrieve/context",
                    "health_url": "http://healthy/api/health",
                },
                {
                    "name": "down",
                    "retrieve_url": "http://down/api/retrieve/context",
                    "health_url": "http://down/api/health",
                },
            ],
            "http://healthy/api/retrieve/context",
        ),
    )

    healthy_response = Mock()
    healthy_response.raise_for_status = Mock()

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        async def mock_get(url, **_kwargs):
            if url == "http://healthy/api/health":
                return healthy_response
            raise RuntimeError("unreachable")

        mock_client.get.side_effect = mock_get

        choices, selected = await utils.fetch_available_rag_endpoints()

    assert choices == {
        utils.NONE_RAG_OPTION_VALUE: "None",
        "http://healthy/api/retrieve/context": "healthy (http://healthy/api/retrieve/context)",
        "http://down/api/retrieve/context": "down (http://down/api/retrieve/context)",
    }
    assert selected == utils.NONE_RAG_OPTION_VALUE


@pytest.mark.asyncio
async def test_fetch_available_rag_endpoints_does_not_auto_select_configured_default(
    monkeypatch,
):
    monkeypatch.setattr(
        utils,
        "load_rag_endpoint_config",
        lambda: (
            [
                {
                    "name": "down",
                    "retrieve_url": "http://down/api/retrieve/context",
                    "health_url": "http://down/api/health",
                }
            ],
            "http://down/api/retrieve/context",
        ),
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = RuntimeError("unreachable")

        choices, selected = await utils.fetch_available_rag_endpoints()

    assert choices == {
        utils.NONE_RAG_OPTION_VALUE: "None",
        "http://down/api/retrieve/context": "down (http://down/api/retrieve/context)",
    }
    assert selected == utils.NONE_RAG_OPTION_VALUE


def test_create_endpoint_display_choices_keeps_evo_x2_routes_distinct():
    endpoints_data = {
        "gmktec-evo-x2-primary": {
            "endpoint_url": "https://llm-evo-x2.paperclips.dev",
            "models": [{"id": "primary-model"}],
        },
        "gmktec-evo-x2-secondary": {
            "endpoint_url": "https://llm-evo-x2-2.paperclips.dev",
            "models": [{"id": "secondary-model"}],
        },
    }

    choices, mapping = utils.create_endpoint_display_choices(endpoints_data)

    assert choices == {
        "gmktec-evo-x2-primary (primary-model)": "gmktec-evo-x2-primary",
        "gmktec-evo-x2-secondary (secondary-model)": "gmktec-evo-x2-secondary",
    }
    assert mapping == choices


def test_create_history_select_choices_uses_convo_id_as_value():
    conversations = [
        {
            "convo_id": "c3630b242ac7",
            "last_updated": "2026-05-21T05:48:40+00:00",
        }
    ]

    assert utils.create_history_select_choices(conversations) == {
        "c3630b242ac7": "2026-05-21 13:48:40 | c3630b242ac7"
    }
