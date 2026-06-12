import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

import shinyswatch
from dotenv import load_dotenv
from shiny import reactive, render, ui

from src.search.safety import EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER

load_dotenv()

from .chat_client import stream_chat_response
from .formatters import (
    format_all_available_models,
    format_cache_info,
    format_hardware_info,
    format_history_json,
    format_model_details,
    format_response_info,
    format_timings_info,
)
from .prompt_state import (
    append_managed_rag_suffix,
    build_system_prompt_state,
    extract_first_system_prompt,
    first_turn_system_prompt_to_send,
    normalize_system_prompt,
)
from .utils import (
    HISTORY_DISPLAY_TIMEZONE,
    create_history_select_choices,
    create_endpoint_display_choices,
    fetch_conversation_summaries,
    fetch_available_endpoints,
    fetch_available_rag_endpoints,
    fetch_available_search_providers,
    fetch_convo_history,
    fetch_convo_state,
    fetch_search_results,
    format_search_provider_label,
    find_model_by_endpoint,
    read_trace_events,
)
from .workflow_client import (
    advance_workflow_run,
    create_workflow_run,
    fetch_workflow,
    fetch_workflow_run,
    fetch_workflow_runs,
    fetch_workflows,
    retry_workflow_step,
)
from .workflow_formatters import (
    format_artifacts,
    format_step_timeline,
    format_workflow_choices,
    format_workflow_run_choices,
    format_workflow_spec,
)

TRACE_DISPLAY_TIMEZONE_LABEL = "GMT+8"
TERMINAL_WORKFLOW_STATUSES = {"completed", "failed", "cancelled"}
WORKFLOW_RUN_MAX_STEPS = 100


def build_search_success_state(search_response: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a successful proxy search response for UI and request shaping."""
    provider_id = str(search_response.get("provider") or "").strip()
    warnings = [
        str(warning).strip()
        for warning in search_response.get("warnings", [])
        if str(warning).strip()
    ]
    results = search_response.get("results", [])
    if not isinstance(results, list):
        results = []

    query_refinement = search_response.get("query_refinement")
    if not isinstance(query_refinement, dict):
        query_refinement = search_response.get("planner")
    if not isinstance(query_refinement, dict):
        query_refinement = {}

    return {
        "provider": provider_id,
        "provider_label": format_search_provider_label(provider_id),
        "query": str(search_response.get("query") or "").strip(),
        "query_refinement": query_refinement,
        "degraded": bool(search_response.get("degraded", False)),
        "warnings": warnings,
        "results": results,
        "result_count": len(results),
        "wrapped_results": (
            search_response.get("wrapped_results")
            if isinstance(search_response.get("wrapped_results"), str)
            else None
        ),
        "show_preface": True,
    }


def build_query_refiner_context(
    *,
    system_prompt: Optional[str],
    history: Any,
    user_input: str,
) -> str:
    """Build compact source context for search query refinement."""
    sections = []
    prompt = normalize_system_prompt(system_prompt)
    if prompt:
        sections.append(f"System prompt:\n{prompt}")

    history_lines = []
    if isinstance(history, list):
        for message in history:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if role and content:
                history_lines.append(f"{role}: {content}")
    if history_lines:
        sections.append("Conversation history:\n" + "\n".join(history_lines))

    request = str(user_input or "").strip()
    if request:
        sections.append(f"Current user request:\n{request}")

    return "\n\n".join(sections)


def build_search_failure_state(provider_id: str, error: Any) -> Dict[str, Any]:
    """Build a non-blocking search failure state for metadata only."""
    return {
        "provider": str(provider_id or "").strip(),
        "provider_label": format_search_provider_label(provider_id),
        "degraded": True,
        "warnings": [f"search request failed: {str(error)}"],
        "results": [],
        "result_count": 0,
        "wrapped_results": None,
        "show_preface": False,
    }


def _format_search_context(search_state: Dict[str, Any]) -> Optional[str]:
    """Render turn-local search context for model consumption, not history."""
    results = search_state.get("results")
    if not isinstance(results, list) or not results:
        return None

    wrapped_results = search_state.get("wrapped_results")
    if not isinstance(wrapped_results, str) or not wrapped_results.strip():
        return None

    return "\n".join([EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER, wrapped_results.strip()])


def build_search_turn_messages(
    search_state: Optional[Dict[str, Any]],
) -> list[Dict[str, str]]:
    """Convert successful search state into turn-local injected messages."""
    if not isinstance(search_state, dict):
        return []

    content = _format_search_context(search_state)
    if not content:
        return []

    return [{"role": "user", "content": content}]


def _query_refinement_queries(
    query_refinement: Dict[str, Any], fallback_query: str
) -> list[str]:
    raw_queries = query_refinement.get("queries")
    queries = raw_queries if isinstance(raw_queries, list) else []
    cleaned = []
    seen = set()
    for item in [*queries, fallback_query]:
        query = " ".join(str(item or "").split())
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
    return cleaned


def build_search_preface(search_state: Optional[Dict[str, Any]]) -> Optional[str]:
    """Render a markdown transcript preface for successful search calls."""
    if not isinstance(search_state, dict) or not search_state.get("show_preface"):
        return None

    provider_label = str(search_state.get("provider_label") or "Search").strip()
    query_refinement = (
        search_state.get("query_refinement")
        if isinstance(search_state.get("query_refinement"), dict)
        else {}
    )
    query = " ".join(str(search_state.get("query") or "").split())
    refined_queries = _query_refinement_queries(query_refinement, query)
    results = search_state.get("results")
    warnings = search_state.get("warnings") or []
    degraded = bool(search_state.get("degraded", False))
    heading = f"**Search candidates** via {provider_label}"
    if query_refinement.get("used") is True and refined_queries:
        quoted_queries = "; ".join(f'"{item}"' for item in refined_queries)
        heading = f"{heading} for {quoted_queries}"
    lines = [heading]

    if degraded:
        lines.append("_Search returned degraded results._")

    if isinstance(results, list) and results:
        for index, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or result.get("url") or f"Result {index}")
            url = str(result.get("url") or "").strip()
            snippet = str(result.get("snippet") or "").strip()
            line = f"{index}. [{title}]({url})" if url else f"{index}. {title}"
            if snippet:
                line = f"{line} - {snippet}"
            lines.append(line)
    else:
        lines.append("No candidates found.")

    if warnings:
        lines.append(f"Warnings: {'; '.join(str(warning) for warning in warnings)}")

    return "\n".join(lines)


def merge_run_info(
    metadata: Optional[Dict[str, Any]], search_state: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge search metadata into the runtime panel payload."""
    merged = dict(metadata or {})

    if isinstance(search_state, dict):
        merged["search"] = {
            "provider": search_state.get("provider_label")
            or search_state.get("provider")
            or "Unknown",
            "provider_id": search_state.get("provider") or "",
            "result_count": search_state.get("result_count", 0),
            "degraded": bool(search_state.get("degraded", False)),
            "warnings": list(search_state.get("warnings") or []),
        }
        query_refinement = search_state.get("query_refinement")
        query = str(search_state.get("query") or "").strip()
        if (
            isinstance(query_refinement, dict)
            and query_refinement.get("used") is True
            and query
        ):
            merged["search"]["query"] = query
            queries = _query_refinement_queries(query_refinement, query)
            if len(queries) > 1:
                merged["search"]["queries"] = queries

    return merged or None


def workflow_snapshot_status(snapshot: dict[str, Any] | None) -> str:
    run = snapshot.get("run") if isinstance(snapshot, dict) else {}
    if not isinstance(run, dict):
        return ""
    return str(run.get("status") or "").strip()


def workflow_snapshot_has_running_step(snapshot: dict[str, Any] | None) -> bool:
    steps = snapshot.get("steps") if isinstance(snapshot, dict) else []
    if not isinstance(steps, list):
        return False
    return any(
        isinstance(step, dict) and step.get("status") == "running" for step in steps
    )


def build_workflow_params_template(spec: dict[str, Any] | None) -> str:
    if not isinstance(spec, dict):
        return "{}"
    schema = spec.get("params_schema")
    if not isinstance(schema, dict):
        return "{}"
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    required_names = [str(item) for item in required] if isinstance(required, list) else []

    keys: list[str] = []
    for key in [*required_names, *properties.keys()]:
        text = str(key or "").strip()
        if text and text not in keys:
            keys.append(text)

    return json.dumps({key: "" for key in keys}, indent=2)


async def advance_workflow_to_terminal(
    run_id: str,
    advance: Callable[[str], Awaitable[dict[str, Any]]],
    *,
    after_step: Callable[[dict[str, Any], int], Awaitable[None]] | None = None,
    max_steps: int = WORKFLOW_RUN_MAX_STEPS,
) -> dict[str, Any]:
    budget = max(1, int(max_steps))
    snapshot: dict[str, Any] = {}
    for step_number in range(1, budget + 1):
        snapshot = await advance(run_id)
        if after_step is not None:
            await after_step(snapshot, step_number)
        if (
            workflow_snapshot_status(snapshot) in TERMINAL_WORKFLOW_STATUSES
            or workflow_snapshot_has_running_step(snapshot)
        ):
            return snapshot
    return snapshot


def build_uploaded_file_context(uploaded_files: Any) -> str:
    if not uploaded_files:
        return ""

    file_contents = []
    for file_info in uploaded_files:
        try:
            datapath = file_info.get("datapath") if isinstance(file_info, dict) else None
            if not datapath or not os.path.exists(datapath):
                continue
            with open(datapath, "r", encoding="utf-8") as handle:
                content = handle.read()
            filename = (
                file_info.get("name", "uploaded")
                if isinstance(file_info, dict)
                else "uploaded"
            )
            file_contents.append(f"--- File: {filename} ---\n{content}")
        except Exception as exc:
            filename = (
                file_info.get("name", "unknown")
                if isinstance(file_info, dict)
                else "unknown"
            )
            file_contents.append(f"--- Error reading {filename}: {str(exc)} ---")
    return "\n\n".join(file_contents)


def merge_uploaded_context(
    params: dict[str, Any], uploaded_context: str
) -> dict[str, Any]:
    uploaded_context = str(uploaded_context or "").strip()
    if not uploaded_context:
        return params

    merged = dict(params)
    existing = str(merged.get("uploaded_context") or "").strip()
    merged["uploaded_context"] = (
        f"{existing}\n\n{uploaded_context}" if existing else uploaded_context
    )
    return merged


def resolve_endpoint_display_selection(
    choices: Dict[str, str],
    mapping: Dict[str, str],
    *,
    preferred_endpoint_key: Optional[str],
) -> Optional[str]:
    """Resolve a persisted endpoint key against the current machine snapshot."""
    preferred_endpoint = str(preferred_endpoint_key or "").strip()
    if preferred_endpoint:
        for display_value, endpoint_key in mapping.items():
            if endpoint_key == preferred_endpoint:
                return display_value

    return next(iter(choices), None)


def normalize_reasoning_effort(
    reasoning_effort: Optional[str],
    *,
    default: str = "none",
) -> str:
    normalized = str(reasoning_effort or "").strip().lower()
    valid_efforts = {"none", "low", "medium", "high"}
    if normalized in valid_efforts:
        return normalized
    return default if default in valid_efforts else "none"


def conversation_control_change_reasons(
    state: Dict[str, Any],
    *,
    current_prompt: Optional[str],
    current_reasoning: Optional[str],
) -> list[str]:
    """Return control changes that require a fork before the next turn."""
    if not bool(state.get("started")):
        return []

    reasons: list[str] = []
    committed_prompt = normalize_system_prompt(
        state.get("committed_prompt")
        if state.get("committed_prompt") is not None
        else state.get("prompt")
    )
    if normalize_system_prompt(current_prompt) != committed_prompt:
        reasons.append("system prompt")

    committed_reasoning = normalize_reasoning_effort(
        state.get("reasoning_effort"),
        default="medium",
    )
    if (
        normalize_reasoning_effort(current_reasoning, default="medium")
        != committed_reasoning
    ):
        reasons.append("reasoning")

    return reasons


def build_fork_notice(
    *,
    old_convo_id: str,
    new_convo_id: str,
    reasons: list[str],
) -> str:
    reason_text = " and ".join(reasons) if reasons else "conversation controls"
    return (
        f"Conversation forked from `{old_convo_id}` to `{new_convo_id}` because "
        f"{reason_text} changed. Prior history was copied and the new settings "
        "were applied at the start of the fork."
    )


def _format_trace_summary(event: Dict[str, Any]) -> str:
    timestamp = _format_trace_timestamp(event.get("timestamp"))
    trace_id = str(event.get("request_id") or "unknown-trace")
    endpoint = str(event.get("endpoint") or "unknown-endpoint")
    phase = str(event.get("phase") or "completed")
    status = event.get("status_code")
    convo_id = str(event.get("convo_id") or "no-convo")
    elapsed = ""
    timing = event.get("timing")
    if isinstance(timing, dict) and timing.get("elapsed_ms") is not None:
        elapsed = f" | {timing['elapsed_ms']}ms"
    status_label = status if status is not None else "?"
    return (
        f"{timestamp} | {phase} | {status_label} | "
        f"{endpoint} | {convo_id} | {trace_id}{elapsed}"
    )


def _format_trace_timestamp(raw_timestamp: Any) -> str:
    timestamp = str(raw_timestamp or "").strip()
    if not timestamp:
        return f"unknown-time {TRACE_DISPLAY_TIMEZONE_LABEL}"

    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        parsed.astimezone(HISTORY_DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
        + f" {TRACE_DISPLAY_TIMEZONE_LABEL}"
    )


def server(input, output, session):
    available_endpoints = reactive.Value({})
    endpoint_info = reactive.Value({})
    endpoint_display_mapping = reactive.Value({})
    selected_endpoint_key_state = reactive.Value("")
    last_runtime = reactive.Value(None)
    run_info = reactive.Value(None)
    send_button_state = reactive.Value("ready")
    info_accordion_open = reactive.Value(True)
    file_upload_key = reactive.Value(0)
    current_files = reactive.Value({"key": 0, "files": None})
    workflow_file_upload_key = reactive.Value(0)
    current_workflow_files = reactive.Value({"key": 0, "files": None})
    system_prompt_states = reactive.Value({})
    system_prompt_seed = reactive.Value("")
    history_refresh_trigger = reactive.Value(0)
    history_selected_convo_id = reactive.Value("")
    trace_snapshot = reactive.Value(None)
    workflow_specs = reactive.Value([])
    workflow_spec = reactive.Value(None)
    workflow_runs = reactive.Value([])
    selected_workflow_id = reactive.Value("")
    selected_workflow_run_id = reactive.Value("")
    workflow_run_snapshot = reactive.Value(None)
    workflow_status_message = reactive.Value("")

    def current_active_convo_id() -> str:
        return str(input.convoID() or "").strip()

    def current_history_convo_id() -> str:
        return str(history_selected_convo_id.get() or "").strip()

    def current_endpoint_display_value() -> str:
        return str(input.endpoint() or "").strip()

    def current_endpoint_key() -> Optional[str]:
        display_value = current_endpoint_display_value()
        mapping = endpoint_display_mapping.get()
        endpoint_key = mapping.get(display_value)
        if endpoint_key:
            return endpoint_key
        stored_endpoint_key = str(selected_endpoint_key_state.get() or "").strip()
        return stored_endpoint_key or None

    def get_system_prompt_state(convo_id: str) -> Dict[str, Any]:
        state = system_prompt_states.get().get(convo_id)
        if isinstance(state, dict):
            return state
        return build_system_prompt_state()

    def set_system_prompt_state(
        convo_id: str,
        *,
        prompt: Optional[str] = None,
        started: Optional[bool] = None,
        locked: Optional[bool] = None,
        committed_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = dict(get_system_prompt_state(convo_id))
        if prompt is not None:
            state["prompt"] = normalize_system_prompt(prompt)
        if committed_prompt is not None:
            state["committed_prompt"] = normalize_system_prompt(committed_prompt)
        if reasoning_effort is not None:
            state["reasoning_effort"] = normalize_reasoning_effort(
                reasoning_effort,
                default="medium",
            )
        if started is not None:
            state["started"] = bool(started)
        if locked is not None:
            state["locked"] = bool(locked)

        states = dict(system_prompt_states.get())
        states[convo_id] = state
        system_prompt_states.set(states)
        return state

    def apply_system_prompt_view(state: Dict[str, Any]) -> None:
        system_prompt_seed.set(str(state.get("prompt") or ""))
        reasoning_effort = normalize_reasoning_effort(
            state.get("reasoning_effort"),
            default="medium",
        )
        ui.update_select("reasoningEffort", selected=reasoning_effort, session=session)

    async def load_system_prompt_state(convo_id: str) -> Dict[str, Any]:
        if not convo_id:
            state = build_system_prompt_state()
            apply_system_prompt_view(state)
            return state

        cached_state = system_prompt_states.get().get(convo_id)
        if isinstance(cached_state, dict):
            apply_system_prompt_view(cached_state)
            return cached_state

        convo_history = await fetch_convo_history(convo_id)
        convo_state = await fetch_convo_state(convo_id)
        persisted_reasoning = str(convo_state.get("reasoning_effort") or "").strip()
        reasoning_effort = normalize_reasoning_effort(
            persisted_reasoning or input.reasoningEffort(),
            default="medium",
        )
        if isinstance(convo_history, list) and convo_history:
            restored_prompt = extract_first_system_prompt(convo_history)
            state = build_system_prompt_state(
                restored_prompt,
                started=True,
                locked=False,
                committed_prompt=restored_prompt,
                reasoning_effort=reasoning_effort,
            )
        else:
            state = build_system_prompt_state(reasoning_effort=reasoning_effort)

        states = dict(system_prompt_states.get())
        states[convo_id] = state
        system_prompt_states.set(states)
        apply_system_prompt_view(state)
        return state

    async def update_endpoints_and_data() -> None:
        endpoints, data = await fetch_available_endpoints()
        available_endpoints.set(endpoints)
        endpoint_info.set(data)

        choices, mapping = create_endpoint_display_choices(endpoints)
        endpoint_display_mapping.set(mapping)
        with reactive.isolate():
            preferred_endpoint_key = selected_endpoint_key_state.get()
        selected = resolve_endpoint_display_selection(
            choices,
            mapping,
            preferred_endpoint_key=preferred_endpoint_key,
        )
        if selected:
            selected_endpoint_key_state.set(mapping.get(selected, ""))
        ui.update_select(
            "endpoint",
            choices=choices,
            selected=selected,
            session=session,
        )
        workflow_choices = choices
        workflow_selected = (
            selected if selected in workflow_choices else next(iter(workflow_choices), None)
        )
        ui.update_select(
            "workflowEndpoint",
            choices=workflow_choices,
            selected=workflow_selected,
            session=session,
        )

    async def update_rag_endpoints(current_selection: Optional[str] = None) -> None:
        rag_choices, default_selection = await fetch_available_rag_endpoints()
        selected = (
            current_selection if current_selection in rag_choices else default_selection
        )
        ui.update_select(
            "ragEndpoint",
            choices=rag_choices,
            selected=selected,
            session=session,
        )
        with reactive.isolate():
            workflow_current = str(input.workflowRagEndpoint() or "")
        workflow_selected = (
            workflow_current if workflow_current in rag_choices else default_selection
        )
        ui.update_select(
            "workflowRagEndpoint",
            choices=rag_choices,
            selected=workflow_selected,
            session=session,
        )

    async def update_search_providers(
        current_selection: Optional[str] = None,
    ) -> None:
        search_choices, default_selection = fetch_available_search_providers()
        selected = (
            current_selection
            if current_selection in search_choices
            else default_selection
        )
        ui.update_select(
            "searchProvider",
            choices=search_choices,
            selected=selected,
            session=session,
        )
        with reactive.isolate():
            workflow_current = str(input.workflowSearchProvider() or "")
        workflow_selected = (
            workflow_current
            if workflow_current in search_choices
            else default_selection
        )
        ui.update_select(
            "workflowSearchProvider",
            choices=search_choices,
            selected=workflow_selected,
            session=session,
        )

    async def update_history_selector() -> None:
        with reactive.isolate():
            current_selection = current_history_convo_id()
        conversations = await fetch_conversation_summaries()
        history_choices = create_history_select_choices(conversations)
        selected = current_selection if current_selection in history_choices else None
        if not selected and conversations:
            selected = str(conversations[0]["convo_id"])
        ui.update_select(
            "historyConvoSelector",
            choices=history_choices,
            selected=selected,
            session=session,
        )

    async def update_workflow_selector() -> None:
        with reactive.isolate():
            current_selection = str(selected_workflow_id.get() or "").strip()
        workflows = await fetch_workflows()
        workflow_specs.set(workflows)
        choices = format_workflow_choices(workflows)
        selected = current_selection if current_selection in choices else None
        if selected is None and choices:
            selected = next(iter(choices))
        ui.update_select(
            "workflowSelector",
            choices=choices,
            selected=selected,
            session=session,
        )
        if selected:
            selected_workflow_id.set(selected)
            try:
                spec = await fetch_workflow(selected)
                workflow_spec.set(spec)
                ui.update_text_area(
                    "workflowParams",
                    value=build_workflow_params_template(spec),
                    session=session,
                )
            except Exception as exc:
                workflow_status_message.set(f"Failed to load workflow: {exc}")

    async def update_workflow_run_selector() -> None:
        with reactive.isolate():
            current_selection = str(selected_workflow_run_id.get() or "").strip()
        runs = await fetch_workflow_runs()
        workflow_runs.set(runs)
        choices = format_workflow_run_choices(runs)
        selected = current_selection if current_selection in choices else None
        if selected is None and choices:
            selected = next(iter(choices))
        ui.update_select(
            "workflowRunSelector",
            choices=choices,
            selected=selected,
            session=session,
        )
        if selected:
            selected_workflow_run_id.set(selected)
            try:
                workflow_run_snapshot.set(await fetch_workflow_run(selected))
            except Exception as exc:
                workflow_status_message.set(f"Failed to load workflow run: {exc}")

    shinyswatch.theme_picker_server()

    @reactive.Effect
    async def _initialize_endpoints():
        await update_endpoints_and_data()
        await update_rag_endpoints()
        await update_search_providers()
        await update_history_selector()
        await update_workflow_selector()
        await update_workflow_run_selector()

    @reactive.Effect
    def _initialize_convo_id():
        ui.update_text("convoID", value=str(uuid.uuid4().hex[:12]), session=session)

    @reactive.Effect
    @reactive.event(input.refreshEndpoints)
    async def _refresh_endpoints():
        await update_endpoints_and_data()

    @reactive.Effect
    @reactive.event(input.refreshRagEndpoints)
    async def _refresh_rag_endpoints():
        await update_rag_endpoints(str(input.ragEndpoint() or ""))
        await update_search_providers(str(input.searchProvider() or ""))

    @reactive.Effect
    @reactive.event(input.refreshWorkflows)
    async def _refresh_workflows():
        await update_workflow_selector()

    @reactive.Effect
    @reactive.event(input.refreshWorkflowRuns)
    async def _refresh_workflow_runs():
        await update_workflow_run_selector()

    @reactive.Effect
    @reactive.event(input.workflowSelector)
    async def _sync_selected_workflow():
        workflow_id = str(input.workflowSelector() or "").strip()
        selected_workflow_id.set(workflow_id)
        if not workflow_id:
            workflow_spec.set(None)
            return
        try:
            spec = await fetch_workflow(workflow_id)
            workflow_spec.set(spec)
            ui.update_text_area(
                "workflowParams",
                value=build_workflow_params_template(spec),
                session=session,
            )
            workflow_status_message.set("")
        except Exception as exc:
            workflow_status_message.set(f"Failed to load workflow: {exc}")

    @reactive.Effect
    @reactive.event(input.workflowRunSelector)
    async def _sync_selected_workflow_run():
        run_id = str(input.workflowRunSelector() or "").strip()
        selected_workflow_run_id.set(run_id)
        if not run_id:
            workflow_run_snapshot.set(None)
            return
        try:
            workflow_run_snapshot.set(await fetch_workflow_run(run_id))
            workflow_status_message.set("")
        except Exception as exc:
            workflow_status_message.set(f"Failed to load workflow run: {exc}")

    def workflow_endpoint_key() -> str:
        raw = str(input.workflowEndpoint() or "").strip()
        endpoint = str(endpoint_display_mapping.get().get(raw, raw) or "").strip()
        if not endpoint:
            raise ValueError("select a workflow endpoint")
        if endpoint.lower() == "smart":
            raise ValueError("workflow endpoint must be a concrete endpoint")
        return endpoint

    def workflow_params_payload() -> dict[str, Any]:
        raw = str(input.workflowParams() or "").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("workflow params must be a JSON object")
        return data

    async def refresh_selected_workflow_run(run_id: str) -> None:
        workflow_run_snapshot.set(await fetch_workflow_run(run_id))
        selected_workflow_run_id.set(run_id)
        await update_workflow_run_selector()

    @reactive.Effect
    @reactive.event(input.createWorkflowRun)
    async def _create_workflow_run():
        workflow_id = str(selected_workflow_id.get() or input.workflowSelector() or "").strip()
        if not workflow_id:
            workflow_status_message.set("Select a workflow first.")
            return
        try:
            workflow_files_data = current_workflow_files.get()
            uploaded_files = (
                workflow_files_data.get("files")
                if workflow_files_data.get("key") == workflow_file_upload_key.get()
                else None
            )
            params = merge_uploaded_context(
                workflow_params_payload(),
                build_uploaded_file_context(uploaded_files),
            )
            payload = {
                "params": params,
                "endpoint": workflow_endpoint_key(),
            }
            reasoning = str(input.workflowReasoning() or "").strip()
            if reasoning:
                payload["reasoning_effort"] = reasoning
            rag_endpoint = str(input.workflowRagEndpoint() or "").strip()
            if rag_endpoint:
                payload["rag_endpoint"] = rag_endpoint
            search_provider = str(input.workflowSearchProvider() or "").strip()
            if search_provider:
                payload["search_provider"] = search_provider
            convo_id = str(input.workflowConvoID() or "").strip()
            if convo_id:
                payload["convo_id"] = convo_id
            created = await create_workflow_run(workflow_id, payload)
            run_id = str(created.get("run_id") or "").strip()
            if not run_id:
                raise ValueError("workflow API did not return run_id")
            workflow_status_message.set(f"Created run {run_id}.")
            await refresh_selected_workflow_run(run_id)
        except Exception as exc:
            workflow_status_message.set(f"Create run failed: {exc}")

    @reactive.Effect
    @reactive.event(input.advanceWorkflowRun)
    async def _advance_workflow_run():
        run_id = str(selected_workflow_run_id.get() or input.workflowRunSelector() or "").strip()
        if not run_id:
            workflow_status_message.set("Select a run first.")
            return
        try:
            workflow_run_snapshot.set(await advance_workflow_run(run_id))
            workflow_status_message.set(f"Advanced run {run_id}.")
            await update_workflow_run_selector()
        except Exception as exc:
            workflow_status_message.set(f"Advance failed: {exc}")

    @reactive.Effect
    @reactive.event(input.runWorkflowToCompletion)
    async def _run_workflow_to_completion():
        run_id = str(selected_workflow_run_id.get() or input.workflowRunSelector() or "").strip()
        if not run_id:
            workflow_status_message.set("Select a run first.")
            return
        try:
            async def after_step(snapshot: dict[str, Any], step_number: int) -> None:
                workflow_run_snapshot.set(snapshot)
                status = workflow_snapshot_status(snapshot) or "unknown"
                workflow_status_message.set(
                    f"Advanced {run_id}: step {step_number}, status {status}."
                )
                await update_workflow_run_selector()
                await reactive.flush()

            snapshot = await advance_workflow_to_terminal(
                run_id,
                advance_workflow_run,
                after_step=after_step,
                max_steps=WORKFLOW_RUN_MAX_STEPS,
            )
            status = workflow_snapshot_status(snapshot) or "unknown"
            workflow_status_message.set(f"Ran {run_id}: status {status}.")
            await update_workflow_run_selector()
        except Exception as exc:
            workflow_status_message.set(f"Run failed: {exc}")

    @reactive.Effect
    @reactive.event(input.retryWorkflowStep)
    async def _retry_workflow_step():
        run_id = str(selected_workflow_run_id.get() or input.workflowRunSelector() or "").strip()
        step_id = str(input.workflowRetryStepID() or "").strip()
        if not run_id or not step_id:
            workflow_status_message.set("Select a run and enter a failed step id.")
            return
        try:
            workflow_run_snapshot.set(await retry_workflow_step(run_id, step_id))
            workflow_status_message.set(f"Retried step {step_id}.")
            await update_workflow_run_selector()
        except Exception as exc:
            workflow_status_message.set(f"Retry failed: {exc}")

    @reactive.Effect
    @reactive.event(input.endpoint)
    def _clear_info_on_endpoint_change():
        endpoint_key = current_endpoint_key()
        if endpoint_key:
            selected_endpoint_key_state.set(endpoint_key)
        run_info.set(None)
        last_runtime.set(None)

    @reactive.Effect
    @reactive.event(input.convoID)
    async def _sync_system_prompt_for_conversation():
        await load_system_prompt_state(current_active_convo_id())

    @reactive.Effect
    @reactive.event(input.outputJSON)
    def _handle_json_toggle():
        if input.outputJSON():
            ui.update_switch("stream", value=False)
            ui.update_switch("autoScroll", value=False)

    @reactive.Effect
    @reactive.event(input.autoScroll)
    def _handle_autoscroll_toggle():
        if input.autoScroll():
            ui.update_switch("outputJSON", value=False)
            ui.update_switch("stream", value=True)

    @reactive.Effect
    @reactive.event(input.stream)
    def _handle_stream_toggle():
        if input.stream():
            ui.update_switch("outputJSON", value=False)
        else:
            ui.update_switch("autoScroll", value=False)

    @reactive.Effect
    @reactive.event(input.generateConvoID)
    def _generate_convo_id():
        new_uuid = str(uuid.uuid4().hex[:12])
        apply_system_prompt_view(
            set_system_prompt_state(
                new_uuid,
                prompt="",
                reasoning_effort=input.reasoningEffort(),
                started=False,
                locked=False,
            )
        )
        ui.update_text("convoID", value=new_uuid, session=session)

    @render.ui
    def system_prompt_ui():
        prompt = system_prompt_seed.get()
        return ui.input_text_area(
            "systemPrompt",
            "System Prompt",
            value=prompt,
            placeholder="System prompt",
            width="100%",
            rows=5,
        )

    @reactive.Effect
    @reactive.event(input.systemPrompt)
    def _store_system_prompt_input():
        convo_id = current_active_convo_id()
        if not convo_id:
            return
        set_system_prompt_state(convo_id, prompt=input.systemPrompt() or "")

    @reactive.Effect
    @reactive.event(input.ragEndpoint)
    async def _append_rag_suffix_for_selected_endpoint():
        convo_id = current_active_convo_id()
        if not convo_id:
            return

        rag_endpoint = str(input.ragEndpoint() or "").strip()
        rag_choices, _default_selection = await fetch_available_rag_endpoints()
        configured_rag_endpoints = {
            str(value).strip() for value in rag_choices if value
        }
        if not rag_endpoint or rag_endpoint not in configured_rag_endpoints:
            return

        current_prompt = input.systemPrompt()
        if current_prompt is None:
            current_prompt = get_system_prompt_state(convo_id).get("prompt", "")

        updated_prompt = append_managed_rag_suffix(current_prompt)
        if updated_prompt == normalize_system_prompt(current_prompt):
            return

        apply_system_prompt_view(
            set_system_prompt_state(convo_id, prompt=updated_prompt)
        )

    @render.ui
    @reactive.event(file_upload_key)
    def file_upload_ui():
        return ui.layout_columns(
            ui.input_file(
                "uploadFile",
                "",
                button_label="Upload (UTF-8)",
                placeholder="No files uploaded",
                multiple=True,
                width="100%",
            ),
            ui.input_action_button("clearUpload", "❌"),
            col_widths=[10, 2],
        )

    @render.ui
    @reactive.event(workflow_file_upload_key)
    def workflow_file_upload_ui():
        return ui.div(
            ui.tags.label(
                "Workflow Files",
                class_="form-label",
                **{"for": "workflowUploadFile"},
            ),
            ui.div(
                ui.div(
                    ui.input_file(
                        "workflowUploadFile",
                        "",
                        button_label="Upload (UTF-8)",
                        placeholder="No files uploaded",
                        multiple=True,
                        width="100%",
                    ),
                    class_="dashboard-workflow-upload-file",
                    style="flex: 1 1 auto; min-width: 0;",
                ),
                ui.div(
                    ui.input_action_button(
                        "clearWorkflowUpload",
                        "Clear",
                        class_="dashboard-full-width-action",
                    ),
                    style="flex: 0 0 5.5rem;",
                ),
                style=(
                    "display: flex; align-items: flex-start; "
                    "gap: 0.5rem; width: 100%;"
                ),
            ),
            style="margin-top: 0.25rem;",
        )

    @reactive.Effect
    @reactive.event(input.uploadFile)
    def _track_uploaded_files():
        uploaded = input.uploadFile()
        if uploaded and len(uploaded) > 0:
            current_files.set({"key": file_upload_key.get(), "files": uploaded})

    @reactive.Effect
    @reactive.event(input.workflowUploadFile)
    def _track_workflow_uploaded_files():
        uploaded = input.workflowUploadFile()
        if uploaded and len(uploaded) > 0:
            current_workflow_files.set(
                {"key": workflow_file_upload_key.get(), "files": uploaded}
            )

    @reactive.Effect
    @reactive.event(input.clearUpload)
    def _clear_upload():
        current_files.set({"key": file_upload_key.get() + 1, "files": None})
        file_upload_key.set(file_upload_key.get() + 1)

    @reactive.Effect
    @reactive.event(input.clearWorkflowUpload)
    def _clear_workflow_upload():
        current_workflow_files.set(
            {"key": workflow_file_upload_key.get() + 1, "files": None}
        )
        workflow_file_upload_key.set(workflow_file_upload_key.get() + 1)

    @reactive.Effect
    @reactive.event(input.logout)
    async def _logout():
        await session.send_custom_message("logout", {})

    @reactive.Effect
    @reactive.event(input.info_accordion)
    def _track_accordion_state():
        info_accordion_open.set("info_panel" in (input.info_accordion() or []))

    @render.ui
    def workflowSpecDetails():
        return format_workflow_spec(workflow_spec.get())

    @render.ui
    def workflowRunDetails():
        snapshot = workflow_run_snapshot.get()
        return ui.div(
            format_step_timeline(snapshot),
            format_artifacts(snapshot),
        )

    @render.ui
    def outputRunInfo():
        return ui.accordion(
            ui.accordion_panel(
                "Runtime & System Information",
                ui.output_ui("info_content"),
                value="info_panel",
            ),
            id="info_accordion",
            open=False,
        )

    @render.ui
    @reactive.event(run_info, last_runtime, endpoint_info, input.endpoint)
    def info_content():
        info = run_info.get()
        runtime = last_runtime.get()
        endpoint_data = endpoint_info.get()
        current_endpoint = current_endpoint_key()

        def fmt(value: Any) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.2f}" if value != int(value) else str(int(value))
            return str(value)

        sections = []
        if runtime is not None:
            sections.append(f"**Elapsed Time**: {runtime:.2f}s")

        if info and "trace" in info:
            trace = info["trace"]
            if isinstance(trace, dict) and trace.get("id"):
                sections.append(f"**Trace**<br>Trace ID: {trace['id']}")

        routing_info = None
        if info and "routing" in info:
            routing_info = info["routing"]
            routing_lines = []
            routing_fields = {
                "decision": lambda value: f"Selected Node: {value}",
                "strategy": lambda value: f"Workload: {value}",
            }
            for field, formatter in routing_fields.items():
                if field in routing_info:
                    routing_lines.append(formatter(routing_info[field]))
            if routing_lines:
                sections.append(
                    "**Smart Routing Decision**<br>" + "<br>".join(routing_lines)
                )

        if info and "rag" in info:
            rag = info["rag"]
            rag_lines = []
            if rag.get("endpoint"):
                rag_lines.append(f"Endpoint: {rag['endpoint']}")
            if rag.get("injected"):
                rag_lines.append(f"Injected: {rag['injected']}")
            if rag.get("confidence"):
                rag_lines.append(f"Confidence: {rag['confidence']}")
            if rag.get("threshold"):
                rag_lines.append(f"Threshold: {rag['threshold']}")
            if rag.get("hits"):
                rag_lines.append(f"Hits: {rag['hits']}")
            if rag.get("method"):
                rag_lines.append(f"Method: {rag['method']}")
            if rag.get("reason") and rag.get("injected") == "false":
                rag_lines.append(f"Skipped: {rag['reason']}")
            if rag_lines:
                sections.append("**RAG**<br>" + "<br>".join(rag_lines))

        if info and "search" in info:
            search = info["search"]
            search_lines = []
            if search.get("provider"):
                search_lines.append(f"Provider: {search['provider']}")
            queries = search.get("queries")
            if isinstance(queries, list) and queries:
                search_lines.append(
                    "Queries: " + "; ".join(str(query) for query in queries)
                )
            elif search.get("query"):
                search_lines.append(f"Query: {search['query']}")
            if "result_count" in search:
                search_lines.append(f"Results: {search['result_count']}")
            if "degraded" in search:
                search_lines.append(f"Degraded: {search['degraded']}")
            warnings = search.get("warnings") or []
            if warnings:
                search_lines.append(
                    f"Warnings: {'; '.join(str(item) for item in warnings)}"
                )
            if search_lines:
                sections.append("**Search**<br>" + "<br>".join(search_lines))

        if info:
            sections.extend(format_cache_info(info, fmt))
            sections.extend(format_response_info(info, fmt))
            sections.extend(format_timings_info(info, fmt))

        endpoint_for_model_info = current_endpoint
        if routing_info and "decision" in routing_info:
            endpoint_for_model_info = routing_info["decision"]

        model = find_model_by_endpoint(endpoint_data, endpoint_for_model_info)
        if not model and routing_info and "decision" in routing_info:
            model = find_model_by_endpoint(endpoint_data, routing_info["decision"])

        if current_endpoint == "smart" and not routing_info:
            all_models = format_all_available_models(endpoint_data)
            if all_models:
                sections.append(
                    "**Available Models (smart routing)**<br><br>"
                    + "<br>".join(all_models)
                )
        elif model:
            hardware_info = format_hardware_info(model, routing_info)
            if hardware_info:
                sections.append("**Hardware**<br>" + "<br>".join(hardware_info))
            model_details = format_model_details(model)
            if model_details:
                sections.append("**Model Info**<br>" + "<br>".join(model_details))
        elif not model and not routing_info:
            all_models = format_all_available_models(endpoint_data)
            if all_models:
                sections.append(
                    "**Available Models**<br><br>" + "<br>".join(all_models)
                )

        if not sections:
            return ui.div()
        return ui.markdown("<br><br>".join(sections))

    chat = ui.Chat(id="chat")

    @chat.on_user_submit
    async def _handle_chat_input(user_input: str):
        current_endpoints = available_endpoints.get()
        current_run_info = run_info.get() or {}
        selected_endpoint_key = current_endpoint_key()
        convo_id = current_active_convo_id() or None
        actual_endpoint_key = selected_endpoint_key
        search_query = str(user_input or "").strip()
        prompt_state = (
            get_system_prompt_state(convo_id)
            if convo_id
            else build_system_prompt_state()
        )

        files_data = current_files.get()
        uploaded_files = (
            files_data.get("files")
            if files_data.get("key") == file_upload_key.get()
            else None
        )
        if uploaded_files and len(uploaded_files) > 0:
            file_contents = []
            for file_info in uploaded_files:
                try:
                    if not os.path.exists(file_info["datapath"]):
                        continue
                    with open(file_info["datapath"], "r", encoding="utf-8") as handle:
                        content = handle.read()
                    file_contents.append(
                        f"--- File: {file_info['name']} ---\n{content}"
                    )
                except Exception as exc:
                    filename = file_info.get("name", "unknown")
                    file_contents.append(
                        f"--- Error reading {filename}: {str(exc)} ---"
                    )
            if file_contents:
                user_input = f"{user_input}\n\n{'\n\n'.join(file_contents)}"

        current_prompt = normalize_system_prompt(
            input.systemPrompt()
            if input.systemPrompt() is not None
            else prompt_state.get("prompt")
        )
        current_reasoning = normalize_reasoning_effort(
            input.reasoningEffort(),
            default="medium",
        )
        forked_history: list[dict[str, Any]] = []
        fork_reasons = conversation_control_change_reasons(
            prompt_state,
            current_prompt=current_prompt,
            current_reasoning=current_reasoning,
        )
        if convo_id and fork_reasons:
            old_convo_id = convo_id
            loaded_history = await fetch_convo_history(old_convo_id)
            if isinstance(loaded_history, list):
                forked_history = [
                    dict(message)
                    for message in loaded_history
                    if isinstance(message, dict) and message.get("role") != "system"
                ]
            convo_id = str(uuid.uuid4().hex[:12])
            prompt_state = set_system_prompt_state(
                convo_id,
                prompt=current_prompt,
                committed_prompt=current_prompt,
                reasoning_effort=current_reasoning,
                started=False,
                locked=False,
            )
            apply_system_prompt_view(prompt_state)
            ui.update_text("convoID", value=convo_id, session=session)
            await chat.append_message(
                {
                    "role": "assistant",
                    "content": build_fork_notice(
                        old_convo_id=old_convo_id,
                        new_convo_id=convo_id,
                        reasons=fork_reasons,
                    ),
                }
            )

        system_prompt_to_send = first_turn_system_prompt_to_send(
            current_prompt,
            bool(prompt_state.get("started")),
        )

        search_state = None
        selected_search_provider = str(input.searchProvider() or "").strip()
        if selected_search_provider and search_query:
            try:
                if fork_reasons:
                    query_refiner_history = forked_history
                elif convo_id:
                    query_refiner_history = await fetch_convo_history(convo_id)
                else:
                    query_refiner_history = []
                query_refiner_context = build_query_refiner_context(
                    system_prompt=current_prompt,
                    history=query_refiner_history,
                    user_input=user_input,
                )
                search_response = await fetch_search_results(
                    query=search_query,
                    provider=selected_search_provider,
                    count=5,
                    context=query_refiner_context,
                )
                search_state = build_search_success_state(search_response)
            except Exception as exc:
                search_state = build_search_failure_state(selected_search_provider, exc)

        search_preface = build_search_preface(search_state)
        if search_preface:
            await chat.append_message({"role": "assistant", "content": search_preface})

        extra_turn_messages = [
            *forked_history,
            *build_search_turn_messages(search_state),
        ]
        if search_state:
            run_info.set(merge_run_info({}, search_state))

        def publish_run_info(metadata: Optional[Dict[str, Any]]) -> None:
            merged_info = merge_run_info(metadata, search_state)
            run_info.set(merged_info)

        response_stream = stream_chat_response(
            endpoint_key=actual_endpoint_key,
            text=user_input,
            endpoints_dict=current_endpoints,
            stream=input.stream(),
            output_json=input.outputJSON(),
            reasoning_effort=current_reasoning,
            output_reasoning=input.outputReasoning(),
            convo_id=convo_id,
            current_routing_info=current_run_info.get("routing", {}),
            system_prompt=system_prompt_to_send,
            extra_turn_messages=extra_turn_messages,
            rag_endpoint=input.ragEndpoint() if input.ragEndpoint() else None,
            on_metadata=publish_run_info,
            on_send_button_state=send_button_state.set,
            on_runtime=last_runtime.set,
        )
        await chat.append_message_stream(response_stream)

        if convo_id and not bool(prompt_state.get("started")):
            apply_system_prompt_view(
                set_system_prompt_state(
                    convo_id,
                    prompt=current_prompt,
                    committed_prompt=current_prompt,
                    reasoning_effort=current_reasoning,
                    started=True,
                    locked=False,
                )
            )

    @reactive.Effect
    @reactive.event(input.send)
    def _trigger_history_refresh():
        history_refresh_trigger.set(history_refresh_trigger.get() + 1)

    @reactive.Effect
    @reactive.event(input.refreshHistory)
    def _manual_history_refresh():
        history_refresh_trigger.set(history_refresh_trigger.get() + 1)

    @reactive.Effect
    @reactive.event(input.refreshTraces)
    def _manual_trace_refresh():
        with reactive.isolate():
            events = read_trace_events(
                convo_id=str(input.traceConvoFilter() or ""),
                trace_id=str(input.traceIDFilter() or ""),
                endpoint=str(input.traceEndpointFilter() or ""),
                max_events=int(input.traceMaxRows() or 200),
            )
        trace_snapshot.set(
            {
                "events": events,
                "refreshed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    @reactive.Effect
    @reactive.event(input.historyConvoSelector)
    def _sync_history_selector_state():
        history_selected_convo_id.set(str(input.historyConvoSelector() or "").strip())

    @reactive.Effect
    @reactive.event(history_refresh_trigger)
    async def _refresh_history_selector():
        await update_history_selector()

    @render.ui
    @reactive.event(input.historyConvoSelector, history_refresh_trigger)
    async def historyBox():
        convo_id = current_history_convo_id()
        if not convo_id:
            return ui.card(
                ui.markdown(
                    "**No conversation ID provided**\n\nEnter a conversation ID to view history."
                ),
            )

        convo_history = await fetch_convo_history(convo_id)
        if not convo_history:
            return ui.card(
                ui.markdown(
                    f"**No history found for conversation: {convo_id}**\n\nThis conversation may not exist or has no messages."
                ),
            )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_count = len(convo_history) if isinstance(convo_history, list) else None
        return ui.card(
            ui.markdown(f"**Conversation History** *(refreshed at {timestamp})*"),
            (
                ui.markdown(f"`{message_count} messages`")
                if message_count is not None
                else None
            ),
            ui.tags.pre(
                format_history_json(convo_history),
                style=(
                    "max-height: 36rem; overflow: auto; white-space: pre-wrap; "
                    "overflow-wrap: anywhere; margin-bottom: 0;"
                ),
            ),
        )

    @render.ui
    def traceBox():
        snapshot = trace_snapshot.get()
        if not isinstance(snapshot, dict):
            return ui.card(
                ui.markdown(
                    "**Trace snapshot not loaded**\n\nClick Refresh to read traces."
                )
            )

        events = snapshot.get("events")
        if not isinstance(events, list):
            events = []
        timestamp = str(snapshot.get("refreshed_at") or "unknown time")
        if not events:
            return ui.card(
                ui.markdown(f"**No trace events found** *(refreshed at {timestamp})*")
            )

        panels = [
            ui.accordion_panel(
                _format_trace_summary(event),
                ui.tags.pre(
                    json.dumps(event, indent=2, ensure_ascii=False),
                    class_="dashboard-trace-json",
                ),
            )
            for event in events
        ]
        return ui.card(
            ui.markdown(
                f"**Trace Events** *(refreshed at {timestamp}; {len(events)} shown)*"
            ),
            ui.accordion(*panels, multiple=True, open=False),
        )
