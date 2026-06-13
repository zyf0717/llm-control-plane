import pytest

from src.dashboard.app_server import (
    build_workflow_routing_choices,
    build_workflow_retry_step_choices,
    build_fork_notice,
    conversation_control_change_reasons,
    normalize_reasoning_effort,
    resolve_endpoint_display_selection,
    resolve_first_search_provider_selection,
    resolve_workflow_retry_step_selection,
    resolve_workflow_routing_selection,
    workflow_run_ended,
    workflow_run_in_progress,
    workflow_snapshot_with_next_pending_running,
)
from src.dashboard.search_flow import (
    build_query_refiner_context,
    build_search_failure_state,
    build_search_preface,
    build_search_success_state,
    build_search_turn_messages,
    merge_run_info,
)
from src.dashboard.trace_formatters import (
    format_trace_summary,
    format_trace_timestamp,
)
from src.dashboard.workflow_server_helpers import (
    advance_workflow_to_terminal,
    build_uploaded_file_context,
    build_workflow_chat_params,
    build_workflow_chat_run_payload,
    build_workflow_params_template,
    format_workflow_intermediate_content,
    format_workflow_conversation_context,
    merge_uploaded_context,
    workflow_chat_response_text,
)
from src.search.safety import EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER


def test_build_query_refiner_context_includes_prompt_history_and_current_request():
    context = build_query_refiner_context(
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


def test_build_workflow_params_template_uses_selected_schema_fields():
    rendered = build_workflow_params_template(
        {
            "params_schema": {
                "required": ["latest_user_prompt"],
                "properties": {
                    "latest_user_prompt": {"type": "string"},
                    "conversation_context": {"type": "string"},
                    "context": {"type": "string"},
                },
            }
        }
    )

    assert rendered == (
        '{\n'
        '  "latest_user_prompt": "",\n'
        '  "conversation_context": "",\n'
        '  "context": ""\n'
        '}'
    )


def test_build_uploaded_file_context_reads_utf8_files(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello workflow", encoding="utf-8")

    context = build_uploaded_file_context(
        [{"name": "notes.txt", "datapath": str(path)}]
    )

    assert "--- File: notes.txt ---" in context
    assert "hello workflow" in context


def test_merge_uploaded_context_appends_to_existing_param():
    merged = merge_uploaded_context(
        {"goal": "ship", "uploaded_context": "manual upload"},
        "file upload",
    )

    assert merged == {
        "goal": "ship",
        "uploaded_context": "manual upload\n\nfile upload",
    }


def test_build_workflow_chat_params_maps_contextual_search_schema():
    params = build_workflow_chat_params(
        {
            "params_schema": {
                "required": ["latest_user_prompt"],
                "properties": {
                    "latest_user_prompt": {"type": "string"},
                    "conversation_context": {"type": "string"},
                    "context": {"type": "string"},
                    "uploaded_context": {"type": "string"},
                },
            }
        },
        latest_user_prompt="find sources",
        conversation_context="user: previous",
        context="system prompt",
        uploaded_context="file contents",
    )

    assert params == {
        "latest_user_prompt": "find sources",
        "conversation_context": "user: previous",
        "context": "system prompt",
        "uploaded_context": "file contents",
    }


def test_build_workflow_chat_params_maps_goal_and_question_schemas():
    assert build_workflow_chat_params(
        {
            "params_schema": {
                "required": ["goal"],
                "properties": {"goal": {"type": "string"}},
            }
        },
        latest_user_prompt="plan this",
    ) == {"goal": "plan this"}
    assert build_workflow_chat_params(
        {
            "params_schema": {
                "required": ["question"],
                "properties": {"question": {"type": "string"}},
            }
        },
        latest_user_prompt="research this",
    ) == {"question": "research this"}


def test_build_workflow_chat_params_omits_unknown_required_param():
    params = build_workflow_chat_params(
        {"params_schema": {"required": ["custom"], "properties": {}}},
        latest_user_prompt="cannot infer",
    )

    assert params == {}


def test_build_workflow_chat_run_payload_excludes_single_node_rag_and_search():
    payload = build_workflow_chat_run_payload(
        {
            "params_schema": {
                "required": ["latest_user_prompt"],
                "properties": {
                    "latest_user_prompt": {"type": "string"},
                    "uploaded_context": {"type": "string"},
                },
            }
        },
        latest_user_prompt="latest only",
        uploaded_context="--- File: notes.txt ---\nfile contents",
        endpoint="node-a",
        reasoning_effort="high",
        convo_id="convo-1",
    )

    assert payload == {
        "params": {
            "latest_user_prompt": "latest only",
            "uploaded_context": "--- File: notes.txt ---\nfile contents",
        },
        "endpoint": "node-a",
        "reasoning_effort": "high",
        "convo_id": "convo-1",
    }
    assert "rag_endpoint" not in payload
    assert "search_provider" not in payload
    assert "file contents" not in payload["params"]["latest_user_prompt"]


def test_build_workflow_chat_run_payload_includes_workflow_search_provider():
    payload = build_workflow_chat_run_payload(
        {
            "params_schema": {
                "required": ["latest_user_prompt"],
                "properties": {"latest_user_prompt": {"type": "string"}},
            }
        },
        latest_user_prompt="latest only",
        endpoint="node-a",
        search_provider="duckduckgo_html",
    )

    assert payload["search_provider"] == "duckduckgo_html"


def test_format_workflow_conversation_context_skips_system_messages():
    rendered = format_workflow_conversation_context(
        [
            {"role": "system", "content": "hidden"},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "", "content": "skip"},
        ]
    )

    assert rendered == "user: Earlier question\n\nassistant: Earlier answer"


def test_workflow_chat_response_text_uses_last_completed_step_text():
    text = workflow_chat_response_text(
        {
            "run": {"run_id": "wf_1", "workflow_id": "sample", "status": "completed"},
            "steps": [
                {
                    "step_id": "first",
                    "status": "completed",
                    "output_json": {"text": "first answer"},
                },
                {
                    "step_id": "second",
                    "status": "completed",
                    "output_json": {"text": "final answer"},
                },
            ],
        }
    )

    assert text == "final answer"


def test_workflow_chat_response_text_falls_back_to_artifact_text():
    text = workflow_chat_response_text(
        {
            "run": {"run_id": "wf_1", "workflow_id": "sample", "status": "completed"},
            "steps": [
                {"step_id": "first", "status": "completed", "output_json": {}},
            ],
            "artifacts": [{"content_text": "artifact answer"}],
        }
    )

    assert text == "artifact answer"


def test_workflow_chat_response_text_reports_failed_step_error():
    text = workflow_chat_response_text(
        {
            "run": {"run_id": "wf_1", "workflow_id": "sample", "status": "failed"},
            "steps": [
                {"step_id": "search", "status": "failed", "error": "bad query"},
            ],
        }
    )

    assert "Workflow `sample` run `wf_1` ended with status `failed`." in text
    assert "Failed step `search`: bad query" in text


def test_workflow_chat_response_text_reports_non_terminal_status():
    text = workflow_chat_response_text(
        {
            "run": {"run_id": "wf_1", "workflow_id": "sample", "status": "running"},
            "steps": [],
        }
    )

    assert text == "Workflow `sample` run `wf_1` is still in progress with status `running`."


def test_format_workflow_intermediate_content_renders_json_generically():
    rendered = format_workflow_intermediate_content(
        """
{
  "queries": ["World War II overview", "World War II causes timeline"],
  "reason": "The user asks for a definition of World War II.",
  "source_preferences": ["encyclopedia", "educational history sites"]
}
"""
    )

    assert rendered == (
        '- queries: ["World War II overview", "World War II causes timeline"]\n'
        "- reason: The user asks for a definition of World War II.\n"
        '- source_preferences: ["encyclopedia", "educational history sites"]'
    )


def test_format_workflow_intermediate_content_pretty_prints_unknown_json():
    rendered = format_workflow_intermediate_content('{"foo": {"bar": 1}}')

    assert rendered == '- foo: {"bar": 1}'


def test_format_workflow_intermediate_content_leaves_text_as_is():
    assert format_workflow_intermediate_content("plain update") == "plain update"


def test_build_workflow_routing_choices_prepends_none_option():
    choices = build_workflow_routing_choices(
        {"contextual_search": "Contextual Search"}
    )

    assert choices == {
        "": "None",
        "contextual_search": "Contextual Search",
    }


def test_resolve_workflow_routing_selection_defaults_to_none():
    selected = resolve_workflow_routing_selection(
        {
            "": "None",
            "implementation_plan": "Implementation Plan",
            "contextual_search": "Contextual Search",
        },
        current_selection="",
    )

    assert selected == ""


def test_resolve_workflow_routing_selection_preserves_valid_current_selection():
    selected = resolve_workflow_routing_selection(
        {
            "": "None",
            "implementation_plan": "Implementation Plan",
            "contextual_search": "Contextual Search",
        },
        current_selection="implementation_plan",
    )

    assert selected == "implementation_plan"


def test_resolve_workflow_routing_selection_falls_back_to_none_option():
    selected = resolve_workflow_routing_selection(
        {
            "": "None",
            "research_brief": "Research Brief",
            "implementation_plan": "Implementation Plan",
        },
        current_selection="missing",
    )

    assert selected == ""


def test_resolve_workflow_routing_selection_falls_back_to_first_without_none_option():
    selected = resolve_workflow_routing_selection(
        {
            "research_brief": "Research Brief",
            "implementation_plan": "Implementation Plan",
        },
        current_selection="missing",
    )

    assert selected == "research_brief"


def test_resolve_workflow_routing_selection_handles_empty_choices():
    selected = resolve_workflow_routing_selection({}, current_selection="missing")

    assert selected is None


def test_resolve_first_search_provider_selection_skips_none_option():
    selected = resolve_first_search_provider_selection(
        {
            "": "None",
            "duckduckgo_html": "DuckDuckGo",
            "wikipedia_opensearch": "Wikipedia",
        }
    )

    assert selected == "duckduckgo_html"


def test_resolve_first_search_provider_selection_returns_empty_without_provider():
    assert resolve_first_search_provider_selection({"": "None"}) == ""


def test_build_workflow_retry_step_choices_uses_snapshot_steps():
    choices = build_workflow_retry_step_choices(
        {
            "steps": [
                {
                    "step_id": "plan",
                    "status": "completed",
                    "input_json": {"current_step": {"name": "Plan"}},
                },
                {"step_id": "search", "status": "failed"},
            ]
        }
    )

    assert choices == {
        "": "Select step",
        "plan": "Plan (completed)",
        "search": "search (failed)",
    }


def test_resolve_workflow_retry_step_selection_defaults_blank_when_all_completed():
    selected = resolve_workflow_retry_step_selection(
        {
            "run": {"status": "completed"},
            "steps": [{"step_id": "plan", "status": "completed"}],
        },
        current_selection="",
    )

    assert selected == ""


def test_resolve_workflow_retry_step_selection_defaults_to_first_non_green_step():
    selected = resolve_workflow_retry_step_selection(
        {
            "run": {"status": "failed"},
            "steps": [
                {"step_id": "plan", "status": "completed"},
                {"step_id": "search", "status": "failed"},
                {"step_id": "reply", "status": "pending"},
            ]
        },
        current_selection="",
    )

    assert selected == "search"


def test_resolve_workflow_retry_step_selection_stays_blank_until_run_ends():
    selected = resolve_workflow_retry_step_selection(
        {
            "run": {"status": "pending"},
            "steps": [
                {"step_id": "plan", "status": "completed"},
                {"step_id": "search", "status": "pending"},
            ],
        },
        current_selection="",
    )

    assert selected == ""


def test_resolve_workflow_retry_step_selection_preserves_valid_current_selection():
    selected = resolve_workflow_retry_step_selection(
        {
            "run": {"status": "failed"},
            "steps": [
                {"step_id": "plan", "status": "completed"},
                {"step_id": "search", "status": "failed"},
            ]
        },
        current_selection="plan",
    )

    assert selected == "plan"


def test_workflow_run_in_progress_checks_run_and_step_status():
    assert workflow_run_in_progress({"run": {"status": "running"}, "steps": []})
    assert workflow_run_in_progress(
        {"run": {"status": "pending"}, "steps": [{"status": "running"}]}
    )
    assert not workflow_run_in_progress(
        {"run": {"status": "failed"}, "steps": [{"status": "failed"}]}
    )


def test_workflow_run_ended_requires_terminal_status_without_running_step():
    assert workflow_run_ended({"run": {"status": "completed"}, "steps": []})
    assert workflow_run_ended(
        {"run": {"status": "failed"}, "steps": [{"status": "failed"}]}
    )
    assert not workflow_run_ended(
        {"run": {"status": "completed"}, "steps": [{"status": "running"}]}
    )
    assert not workflow_run_ended({"run": {"status": "pending"}, "steps": []})


def test_workflow_snapshot_with_next_pending_running_marks_first_pending_step():
    snapshot = {
        "run": {"status": "pending", "current_step_id": None},
        "steps": [
            {"step_id": "first", "status": "completed"},
            {"step_id": "second", "status": "pending"},
            {"step_id": "third", "status": "pending"},
        ],
    }

    updated = workflow_snapshot_with_next_pending_running(snapshot)

    assert updated is not None
    assert updated["run"]["status"] == "running"
    assert updated["run"]["current_step_id"] == "second"
    assert [step["status"] for step in updated["steps"]] == [
        "completed",
        "running",
        "pending",
    ]
    assert snapshot["steps"][1]["status"] == "pending"


def test_workflow_snapshot_with_next_pending_running_preserves_running_step():
    snapshot = {
        "run": {"status": "running", "current_step_id": "first"},
        "steps": [
            {"step_id": "first", "status": "running"},
            {"step_id": "second", "status": "pending"},
        ],
    }

    updated = workflow_snapshot_with_next_pending_running(snapshot)

    assert updated is not None
    assert [step["status"] for step in updated["steps"]] == ["running", "pending"]
    assert updated["run"]["current_step_id"] == "first"


@pytest.mark.asyncio
async def test_advance_workflow_to_terminal_stops_on_running_step():
    calls = 0

    async def advance(_run_id):
        nonlocal calls
        calls += 1
        return {
            "run": {"status": "running"},
            "steps": [
                {"step_id": "first", "status": "running"},
                {"step_id": "second", "status": "pending"},
            ],
        }

    snapshot = await advance_workflow_to_terminal("wf_123", advance, max_steps=100)

    assert calls == 1
    assert snapshot["steps"][0]["status"] == "running"


@pytest.mark.asyncio
async def test_advance_workflow_to_terminal_stops_on_completed():
    snapshots = [
        {"run": {"status": "running"}},
        {"run": {"status": "completed"}},
    ]
    seen = []

    async def advance(run_id):
        assert run_id == "wf_123"
        return snapshots.pop(0)

    async def after_step(snapshot, step_number):
        seen.append((step_number, snapshot["run"]["status"]))

    snapshot = await advance_workflow_to_terminal(
        "wf_123", advance, after_step=after_step
    )

    assert snapshot["run"]["status"] == "completed"
    assert seen == [(1, "running"), (2, "completed")]


@pytest.mark.asyncio
async def test_advance_workflow_to_terminal_stops_on_failed():
    calls = 0

    async def advance(_run_id):
        nonlocal calls
        calls += 1
        return {"run": {"status": "failed"}}

    snapshot = await advance_workflow_to_terminal("wf_123", advance)

    assert snapshot["run"]["status"] == "failed"
    assert calls == 1


@pytest.mark.asyncio
async def test_advance_workflow_to_terminal_stops_at_step_budget():
    calls = 0

    async def advance(_run_id):
        nonlocal calls
        calls += 1
        return {"run": {"status": "running", "call": calls}}

    snapshot = await advance_workflow_to_terminal("wf_123", advance, max_steps=3)

    assert calls == 3
    assert snapshot["run"]["call"] == 3


def test_format_trace_timestamp_displays_gmt_plus_8():
    assert (
        format_trace_timestamp("2026-06-09T02:16:33.957094+00:00")
        == "2026-06-09 10:16:33 GMT+8"
    )
    assert format_trace_timestamp("bad-time") == "bad-time"


def test_format_trace_summary_uses_display_timezone():
    summary = format_trace_summary(
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


def test_build_search_preface_shows_refined_query_when_query_refiner_used():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "query": "llama.cpp server KV cache metrics cached_tokens",
            "query_refinement": {
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


def test_build_search_preface_shows_all_fanout_queries_when_query_refiner_used():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "query": "q1",
            "query_refinement": {"used": True, "queries": ["q1", "q2", "q1"]},
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


def test_build_search_preface_keeps_legacy_heading_without_query_refiner_use():
    state = build_search_success_state(
        {
            "provider": "duckduckgo_html",
            "query": "Ada Lovelace",
            "query_refinement": {"used": False},
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
