from src.dashboard.app_server import (
    _format_trace_summary,
    _format_trace_timestamp,
    build_fork_notice,
    build_search_failure_state,
    build_search_preface,
    build_search_planner_context,
    build_search_success_state,
    build_search_turn_messages,
    conversation_control_change_reasons,
    merge_run_info,
    normalize_reasoning_effort,
    resolve_endpoint_display_selection,
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
        {"routing": {"decision": "smart"}},
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
        "routing": {"decision": "smart"},
        "search": {
            "provider": "DuckDuckGo",
            "provider_id": "duckduckgo_html",
            "result_count": 0,
            "degraded": True,
            "warnings": ["timeout"],
        },
    }


def test_resolve_endpoint_display_selection_uses_persisted_endpoint_key():
    choices = {
        "node-a (model-a)": "node-a",
        "node-b (model-b)": "node-b",
    }
    mapping = dict(choices)

    assert (
        resolve_endpoint_display_selection(
            choices,
            mapping,
            preferred_endpoint_key="node-a",
        )
        == "node-a (model-a)"
    )


def test_resolve_endpoint_display_selection_uses_stored_endpoint_after_rebuild():
    choices = {
        "node-a (new-model-a)": "node-a",
        "node-b (new-model-b)": "node-b",
    }
    mapping = dict(choices)

    assert (
        resolve_endpoint_display_selection(
            choices,
            mapping,
            preferred_endpoint_key="node-b",
        )
        == "node-b (new-model-b)"
    )


def test_resolve_endpoint_display_selection_falls_back_when_endpoint_disappears():
    choices = {
        "node-a (model-a)": "node-a",
        "node-c (model-c)": "node-c",
    }
    mapping = dict(choices)

    assert (
        resolve_endpoint_display_selection(
            choices,
            mapping,
            preferred_endpoint_key="node-b",
        )
        == "node-a (model-a)"
    )


def test_normalize_reasoning_effort_defaults_missing_to_requested_default():
    assert normalize_reasoning_effort(None) == "none"
    assert normalize_reasoning_effort(None, default="medium") == "medium"
    assert normalize_reasoning_effort("", default="medium") == "medium"


def test_normalize_reasoning_effort_preserves_explicit_none():
    assert normalize_reasoning_effort("none", default="medium") == "none"


def test_conversation_control_change_reasons_detects_mid_convo_prompt_change():
    state = {
        "started": True,
        "committed_prompt": "Be concise.",
        "reasoning_effort": "medium",
    }

    assert conversation_control_change_reasons(
        state,
        current_prompt="Be precise.",
        current_reasoning="medium",
    ) == ["system prompt"]


def test_conversation_control_change_reasons_detects_mid_convo_reasoning_change():
    state = {
        "started": True,
        "committed_prompt": "Be concise.",
        "reasoning_effort": "medium",
    }

    assert conversation_control_change_reasons(
        state,
        current_prompt="Be concise.",
        current_reasoning="high",
    ) == ["reasoning"]


def test_conversation_control_change_reasons_ignores_unstarted_convo():
    state = {
        "started": False,
        "committed_prompt": "Be concise.",
        "reasoning_effort": "medium",
    }

    assert (
        conversation_control_change_reasons(
            state,
            current_prompt="Be precise.",
            current_reasoning="high",
        )
        == []
    )


def test_conversation_control_change_reasons_defaults_missing_reasoning_to_medium():
    state = {
        "started": True,
        "committed_prompt": "Be concise.",
    }

    assert (
        conversation_control_change_reasons(
            state,
            current_prompt="Be concise.",
            current_reasoning="medium",
        )
        == []
    )


def test_build_fork_notice_names_old_new_ids_and_reasons():
    notice = build_fork_notice(
        old_convo_id="old123",
        new_convo_id="new456",
        reasons=["system prompt", "reasoning"],
    )

    assert "`old123`" in notice
    assert "`new456`" in notice
    assert "system prompt and reasoning changed" in notice
