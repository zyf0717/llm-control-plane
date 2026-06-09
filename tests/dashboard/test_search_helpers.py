from src.dashboard.app_server import (
    _format_trace_summary,
    _format_trace_timestamp,
    build_search_failure_state,
    build_search_preface,
    build_search_planner_context,
    build_search_success_state,
    build_search_turn_messages,
    merge_run_info,
    pin_auto_route_decision,
    resolve_auto_endpoint_key,
)
from src.search.safety import EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER


def test_build_search_planner_context_includes_prompt_history_and_current_request():
    context = build_search_planner_context(
        system_prompt="Prefer primary sources.",
        history=[
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "ignored", "content": ""},
        ],
        user_input="Current question",
    )

    assert "System prompt:\nPrefer primary sources." in context
    assert "Conversation history:\nuser: Earlier question" in context
    assert "assistant: Earlier answer" in context
    assert "Current user request:\nCurrent question" in context


def test_format_trace_timestamp_displays_gmt_plus_8():
    assert (
        _format_trace_timestamp("2026-06-09T02:16:33.957094+00:00")
        == "2026-06-09 10:16:33 GMT+8"
    )
    assert _format_trace_timestamp("bad-time") == "bad-time"


def test_format_trace_summary_uses_display_timezone():
    summary = _format_trace_summary(
        {
            "timestamp": "2026-06-09T02:16:33.957094+00:00",
            "phase": "started",
            "status_code": 200,
            "endpoint": "gmktec-evo-x2",
            "convo_id": "abc123",
            "request_id": "trace123",
            "timing": {"elapsed_ms": 42},
        }
    )

    assert summary == (
        "2026-06-09 10:16:33 GMT+8 | started | 200 | "
        "gmktec-evo-x2 | abc123 | trace123 | 42ms"
    )


def test_build_search_success_state_normalizes_proxy_payload():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "results": [
                {
                    "title": "Ada Lovelace",
                    "url": "https://example.com/ada",
                    "snippet": "Computing pioneer",
                }
            ],
            "degraded": False,
            "warnings": ["provider warning"],
            "wrapped_results": '{"source":"web_search"}',
        }
    )

    assert state["provider_label"] == "DuckDuckGo"
    assert state["result_count"] == 1
    assert state["wrapped_results"] == '{"source":"web_search"}'
    assert state["show_preface"] is True


def test_build_search_turn_messages_uses_wrapped_results_as_ephemeral_user_context():
    wrapped_results = '{"source":"web_search","untrusted":true}'
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "query": "Ada Lovelace",
            "results": [
                {
                    "title": "Ada Lovelace",
                    "url": "https://example.com/ada",
                    "snippet": "Computing pioneer",
                }
            ],
            "wrapped_results": wrapped_results,
        }
    )

    messages = build_search_turn_messages(state)

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == (
        f"{EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER}\n{wrapped_results}"
    )
    assert "URL:" not in messages[0]["content"]
    assert "Snippet:" not in messages[0]["content"]


def test_build_search_turn_messages_requires_wrapped_results():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "query": "Ada Lovelace",
            "results": [
                {
                    "title": "Ada Lovelace",
                    "url": "https://example.com/ada",
                    "snippet": "Computing pioneer",
                }
            ],
        }
    )

    assert build_search_turn_messages(state) == []


def test_build_search_failure_state_disables_preface_and_injection():
    state = build_search_failure_state("duckduckgo_html", RuntimeError("boom"))

    assert state["provider_label"] == "DuckDuckGo"
    assert state["show_preface"] is False
    assert build_search_turn_messages(state) == []
    assert build_search_preface(state) is None


def test_build_search_preface_renders_results_and_warnings():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "results": [
                {
                    "title": "Ada Lovelace",
                    "url": "https://example.com/ada",
                    "snippet": "Computing pioneer",
                }
            ],
            "degraded": True,
            "warnings": ["selector drift"],
            "wrapped_results": '{"source":"web_search"}',
        }
    )

    preface = build_search_preface(state)

    assert preface is not None
    assert "**Search candidates** via DuckDuckGo" in preface
    assert "https://example.com/ada" in preface
    assert "Computing pioneer" in preface
    assert "selector drift" in preface
    assert "degraded" in preface.lower()


def test_build_search_preface_shows_optimized_query_when_planner_used():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "query": "llama.cpp server KV cache metrics cached_tokens",
            "planner": {
                "used": True,
                "effective_query": "llama.cpp server KV cache metrics cached_tokens",
            },
            "results": [],
            "wrapped_results": '{"source":"web_search"}',
        }
    )

    preface = build_search_preface(state)

    assert preface is not None
    assert (
        '**Search candidates** via DuckDuckGo for "llama.cpp server KV cache metrics cached_tokens"'
        in preface
    )

    merged = merge_run_info({}, state)
    assert merged["search"]["query"] == "llama.cpp server KV cache metrics cached_tokens"


def test_build_search_preface_shows_all_fanout_queries_when_planner_used():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "query": "q1",
            "planner": {"used": True, "queries": ["q1", "q2", "q1"]},
            "results": [],
            "wrapped_results": '{"source":"web_search"}',
        }
    )

    preface = build_search_preface(state)

    assert preface is not None
    assert '**Search candidates** via DuckDuckGo for "q1"; "q2"' in preface

    merged = merge_run_info({}, state)
    assert merged["search"]["query"] == "q1"
    assert merged["search"]["queries"] == ["q1", "q2"]


def test_build_search_preface_keeps_legacy_heading_without_planner_use():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "query": "Ada Lovelace",
            "planner": {"used": False},
            "results": [],
            "wrapped_results": '{"source":"web_search"}',
        }
    )

    preface = build_search_preface(state)

    assert preface is not None
    assert '**Search candidates** via DuckDuckGo\n' in preface
    assert 'for "Ada Lovelace"' not in preface


def test_build_search_preface_handles_empty_successful_search():
    state = build_search_success_state(
        {
            "provider": "wikipedia_opensearch",
            "results": [],
            "degraded": False,
            "warnings": [],
            "wrapped_results": '{"source":"web_search"}',
        }
    )

    assert build_search_turn_messages(state) == []
    assert "No candidates found." in str(build_search_preface(state))


def test_merge_run_info_adds_search_block_without_dropping_existing_metadata():
    merged = merge_run_info(
        {"routing": {"decision": "Auto"}},
        build_search_success_state(
            {
                "provider": "duckduckgo_html",
                "results": [],
                "degraded": True,
                "warnings": ["timeout"],
                "wrapped_results": '{"source":"web_search"}',
            }
        ),
    )

    assert merged == {
        "routing": {"decision": "Auto"},
        "search": {
            "provider": "DuckDuckGo",
            "provider_id": "duckduckgo_html",
            "result_count": 0,
            "degraded": True,
            "warnings": ["timeout"],
        },
    }


def test_resolve_auto_endpoint_key_uses_first_pinned_decision_for_convo():
    pins = {"convo-a": "gmktec-evo-x2-primary"}

    assert resolve_auto_endpoint_key("Auto", "convo-a", pins) == "gmktec-evo-x2-primary"
    assert resolve_auto_endpoint_key("Auto", "convo-b", pins) == "Auto"
    assert resolve_auto_endpoint_key("mac-mini", "convo-a", pins) == "mac-mini"


def test_pin_auto_route_decision_records_only_first_auto_decision():
    pins = pin_auto_route_decision(
        {},
        "convo-a",
        "Auto",
        {"routing": {"decision": "gmktec-evo-x2-primary", "strategy": "programming"}},
    )

    assert pins == {"convo-a": "gmktec-evo-x2-primary"}
    assert pin_auto_route_decision(
        pins,
        "convo-a",
        "Auto",
        {"routing": {"decision": "mac-mini", "strategy": "ttft_content"}},
    ) == {"convo-a": "gmktec-evo-x2-primary"}


def test_pin_auto_route_decision_ignores_static_endpoint_and_missing_convo():
    metadata = {"routing": {"decision": "gmktec-evo-x2-primary"}}

    assert pin_auto_route_decision({}, "convo-a", "mac-mini", metadata) == {}
    assert pin_auto_route_decision({}, "", "Auto", metadata) == {}
