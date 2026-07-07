import json
import time
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, AsyncIterable, AsyncIterator, Callable, Dict, Optional

import shinyswatch
from dotenv import load_dotenv
from shiny import reactive, render, ui
from shiny.types import SilentException

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
    append_managed_retrieval_suffix,
    build_system_prompt_state,
    extract_first_system_prompt,
    first_turn_system_prompt_to_send,
    normalize_system_prompt,
)
from .search_flow import (
    build_query_refiner_source_text,
    build_search_failure_state,
    build_search_preface,
    build_search_success_state,
    build_search_turn_messages,
    merge_run_info,
)
from .trace_formatters import format_trace_summary
from .utils import (
    append_conversation_messages,
    create_history_select_choices,
    create_endpoint_display_choices,
    fetch_conversation_summaries,
    fetch_available_endpoints,
    fetch_available_repo_context_repositories,
    fetch_available_retrieval_endpoints,
    fetch_available_search_providers,
    fetch_conversation_history,
    fetch_conversation_control_state,
    fetch_search_results,
    find_model_by_endpoint,
    read_trace_events,
)
from .graph_client import (
    create_graph_run,
    fetch_graph,
    fetch_graph_run,
    fetch_graph_runs,
    fetch_graphs,
    run_graph_to_completion,
    stream_graph_run_events,
)
from .graph_formatters import (
    format_graph_choices,
    format_graph_run_choices,
    format_graph_run_details,
    format_graph_spec,
    graph_input_template,
)
from .workflow_server_helpers import (
    WORKFLOW_RUN_MAX_STEPS,
    advance_workflow_to_terminal,
    build_uploaded_file_source_text,
    build_workflow_chat_run_payload,
    build_workflow_params_template,
    format_workflow_intermediate_content,
    format_workflow_thread_briefing,
    merge_repo_context_repo_name,
    merge_uploaded_source_text,
    workflow_chat_response_text,
    workflow_chat_run_info,
    workflow_snapshot_status,
)
from .workflow_client import (
    advance_workflow_run,
    create_workflow_run,
    fetch_workflow,
    fetch_workflow_run,
    fetch_workflow_runs,
    fetch_workflows,
    retry_workflow_step,
    stream_workflow_run_events,
)
from .workflow_formatters import (
    format_artifacts,
    format_step_timeline,
    format_workflow_choices,
    format_workflow_run_choices,
    format_workflow_spec,
)

DEFAULT_WORKFLOW_DISPATCH_ID = ""
WORKFLOW_DISPATCH_NONE_LABEL = "None"
THREADED_SEARCH_WORKFLOW_ID = "threaded_search"
THREADED_RAG_WORKFLOW_ID = "threaded_rag"
REPO_CONTEXT_WORKFLOW_ID = "repo_context"
ENDED_WORKFLOW_RUN_STATUSES = {"completed", "failed", "cancelled"}


def elapsed_seconds_since(started_at: float, *, now: float | None = None) -> float:
    current = time.perf_counter() if now is None else float(now)
    return max(0.0, current - float(started_at))


async def stream_with_finalizer(
    stream: AsyncIterable[str],
    finalizer: Callable[[], None],
) -> AsyncIterator[str]:
    try:
        async for chunk in stream:
            yield chunk
    finally:
        finalizer()


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


def resolve_workflow_dispatch_selection(
    choices: Dict[str, str],
    *,
    current_selection: Optional[str],
    default_workflow_id: str = DEFAULT_WORKFLOW_DISPATCH_ID,
) -> Optional[str]:
    current = str(current_selection or "").strip()
    if current in choices:
        return current

    default = str(default_workflow_id or "").strip()
    if default in choices:
        return default

    return next(iter(choices), None)


def resolve_workflow_dispatch_selection_for_repo_context(
    choices: Dict[str, str],
    *,
    current_selection: Optional[str],
    repo_name: Optional[str],
    repo_context_workflow_id: str = REPO_CONTEXT_WORKFLOW_ID,
) -> Optional[str]:
    if str(repo_name or "").strip() and repo_context_workflow_id in choices:
        return repo_context_workflow_id
    return resolve_workflow_dispatch_selection(
        choices,
        current_selection=current_selection,
    )


def build_workflow_dispatch_choices(choices: Dict[str, str]) -> Dict[str, str]:
    return {"": WORKFLOW_DISPATCH_NONE_LABEL, **choices}


def resolve_first_search_provider_selection(choices: Dict[str, str]) -> str:
    for provider_id in choices:
        provider = str(provider_id or "").strip()
        if provider:
            return provider
    return ""


def resolve_first_retrieval_endpoint_selection(choices: Dict[str, str]) -> str:
    for endpoint in choices:
        normalized = str(endpoint or "").strip()
        if normalized:
            return normalized
    return ""


def workflow_requires_search_provider(workflow_id: Optional[str]) -> bool:
    return str(workflow_id or "").strip() == THREADED_SEARCH_WORKFLOW_ID


def workflow_requires_retrieval_endpoint(workflow_id: Optional[str]) -> bool:
    return str(workflow_id or "").strip() == THREADED_RAG_WORKFLOW_ID


def workflow_step_kinds(spec: dict[str, Any] | None) -> set[str]:
    steps = spec.get("steps") if isinstance(spec, dict) else []
    if not isinstance(steps, list):
        return set()
    return {
        str(step.get("kind") or "").strip()
        for step in steps
        if isinstance(step, dict) and str(step.get("kind") or "").strip()
    }


def workflow_param_names(spec: dict[str, Any] | None) -> set[str]:
    schema = spec.get("params_schema") if isinstance(spec, dict) else {}
    if not isinstance(schema, dict):
        return set()
    properties = schema.get("properties")
    names = set(str(key) for key in properties) if isinstance(properties, dict) else set()
    required = schema.get("required")
    if isinstance(required, list):
        names.update(str(item) for item in required if str(item or "").strip())
    return names


def workflow_uses_search_provider(spec: dict[str, Any] | None) -> bool:
    return "search" in workflow_step_kinds(spec)


def workflow_uses_retrieval_endpoint(spec: dict[str, Any] | None) -> bool:
    return "retrieval" in workflow_step_kinds(spec)


def workflow_uses_repo_context_repo(spec: dict[str, Any] | None) -> bool:
    return "repo_context" in workflow_step_kinds(
        spec
    ) or "repo_name" in workflow_param_names(spec)


def workflow_accepts_uploaded_source(spec: dict[str, Any] | None) -> bool:
    return "uploaded_source_text" in workflow_param_names(spec)


def single_node_right_panel_state(
    workflow_id: Optional[str],
    spec: dict[str, Any] | None,
) -> dict[str, bool]:
    active = bool(str(workflow_id or "").strip())
    return {
        "active": active,
        "search_provider_enabled": (not active) or workflow_uses_search_provider(spec),
        "retrieval_endpoint_enabled": (not active)
        or workflow_uses_retrieval_endpoint(spec),
        "repo_context_repo_enabled": (not active)
        or workflow_uses_repo_context_repo(spec),
        "upload_enabled": (not active) or workflow_accepts_uploaded_source(spec),
    }


def build_search_provider_choices(
    choices: Dict[str, str],
    *,
    require_provider: bool,
) -> Dict[str, str]:
    if require_provider:
        required_choices = {
            provider_id: label
            for provider_id, label in choices.items()
            if str(provider_id or "").strip()
        }
        return required_choices or choices
    return choices


def build_retrieval_endpoint_choices(
    choices: Dict[str, str],
    *,
    require_endpoint: bool,
) -> Dict[str, str]:
    if require_endpoint:
        required_choices = {
            endpoint: label
            for endpoint, label in choices.items()
            if str(endpoint or "").strip()
        }
        return required_choices or choices
    return choices


def resolve_search_provider_selection(
    choices: Dict[str, str],
    *,
    current_selection: Optional[str] = None,
    default_selection: str = "",
    require_provider: bool = False,
    force_default: bool = False,
) -> str:
    current = str(current_selection or "").strip()
    default = str(default_selection or "")

    if require_provider:
        if current and current in choices:
            return current
        return resolve_first_search_provider_selection(choices) or default

    if force_default:
        if default in choices:
            return default
        if "" in choices:
            return ""
        return next(iter(choices), "")

    if current and current in choices:
        return current
    if default in choices:
        return default
    if "" in choices:
        return ""
    return next(iter(choices), "")


def resolve_retrieval_endpoint_selection(
    choices: Dict[str, str],
    *,
    current_selection: Optional[str] = None,
    default_selection: str = "",
    require_endpoint: bool = False,
) -> str:
    current = str(current_selection or "").strip()
    default = str(default_selection or "")

    if require_endpoint:
        if current and current in choices:
            return current
        if default and default in choices:
            return default
        return resolve_first_retrieval_endpoint_selection(choices)

    if current and current in choices:
        return current
    if default in choices:
        return default
    if "" in choices:
        return ""
    return next(iter(choices), "")


def workflow_run_in_progress(snapshot: dict[str, Any] | None) -> bool:
    if not isinstance(snapshot, dict):
        return False
    run = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
    if str(run.get("status") or "").strip().lower() == "running":
        return True
    return any(
        isinstance(step, dict)
        and str(step.get("status") or "").strip().lower() == "running"
        for step in snapshot.get("steps", [])
        if isinstance(step, dict)
    )


def workflow_run_ended(snapshot: dict[str, Any] | None) -> bool:
    if workflow_run_in_progress(snapshot):
        return False
    if not isinstance(snapshot, dict):
        return False
    run = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
    return str(run.get("status") or "").strip().lower() in ENDED_WORKFLOW_RUN_STATUSES


def build_workflow_retry_step_choices(
    snapshot: dict[str, Any] | None,
) -> Dict[str, str]:
    choices: Dict[str, str] = {"": "Select step"}
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("steps"), list):
        return choices
    for step in snapshot["steps"]:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id") or "").strip()
        if not step_id:
            continue
        status = str(step.get("status") or "unknown").strip() or "unknown"
        name = _workflow_step_display_name(step)
        label = f"{name} ({status})" if name != step_id else f"{step_id} ({status})"
        choices[step_id] = label
    return choices


def resolve_workflow_retry_step_selection(
    snapshot: dict[str, Any] | None,
    *,
    current_selection: Optional[str],
) -> str:
    choices = build_workflow_retry_step_choices(snapshot)
    current = str(current_selection or "").strip()
    if current and current in choices:
        return current
    if not workflow_run_ended(snapshot):
        return ""
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("steps"), list):
        return ""
    for step in snapshot["steps"]:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id") or "").strip()
        status = str(step.get("status") or "").strip().lower()
        if step_id and status and status != "completed":
            return step_id
    return ""


def _workflow_step_display_name(step: dict[str, Any]) -> str:
    step_id = str(step.get("step_id") or "").strip()
    input_json = step.get("input_json")
    current_step = (
        input_json.get("current_step")
        if isinstance(input_json, dict)
        and isinstance(input_json.get("current_step"), dict)
        else {}
    )
    return str(current_step.get("name") or step_id or "step").strip()


def workflow_snapshot_with_next_pending_running(
    snapshot: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    updated = deepcopy(snapshot)
    steps = updated.get("steps")
    if not isinstance(steps, list):
        return updated
    if any(isinstance(step, dict) and step.get("status") == "running" for step in steps):
        return updated

    for step in steps:
        if not isinstance(step, dict) or step.get("status") != "pending":
            continue
        step["status"] = "running"
        run = updated.get("run")
        if isinstance(run, dict):
            run["status"] = "running"
            run["current_step_id"] = step.get("step_id")
        return updated
    return updated


def workflow_dispatch_event_updates_run_details(event_type: str) -> bool:
    """Low-frequency dispatch state boundaries should re-render run details."""
    return str(event_type or "").strip() in {
        "step_completed",
        "run_completed",
        "error",
    }


def _input_value(input_obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(input_obj, name)()
    except SilentException:
        return default


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
    old_conversation_id: str,
    new_conversation_id: str,
    reasons: list[str],
) -> str:
    reason_text = " and ".join(reasons) if reasons else "conversation controls"
    return (
        f"Conversation forked from `{old_conversation_id}` to `{new_conversation_id}` because "
        f"{reason_text} changed. Prior history was copied and the new settings "
        "were applied at the start of the fork."
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
    history_selected_conversation_id = reactive.Value("")
    trace_snapshot = reactive.Value(None)
    workflow_specs = reactive.Value([])
    workflow_spec = reactive.Value(None)
    workflow_runs = reactive.Value([])
    selected_workflow_id = reactive.Value("")
    selected_workflow_dispatch_id = reactive.Value(DEFAULT_WORKFLOW_DISPATCH_ID)
    selected_workflow_run_id = reactive.Value("")
    workflow_run_snapshot = reactive.Value(None)
    workflow_status_message = reactive.Value("")
    graph_specs = reactive.Value([])
    graph_spec = reactive.Value(None)
    graph_runs = reactive.Value([])
    selected_graph_id = reactive.Value("")
    selected_graph_run_id = reactive.Value("")
    graph_run_snapshot = reactive.Value(None)
    graph_status_message = reactive.Value("")

    def current_active_conversation_id() -> str:
        return str(input.conversationID() or "").strip()

    def current_history_conversation_id() -> str:
        return str(history_selected_conversation_id.get() or "").strip()

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

    def get_system_prompt_state(conversation_id: str) -> Dict[str, Any]:
        state = system_prompt_states.get().get(conversation_id)
        if isinstance(state, dict):
            return state
        return build_system_prompt_state()

    def set_system_prompt_state(
        conversation_id: str,
        *,
        prompt: Optional[str] = None,
        started: Optional[bool] = None,
        locked: Optional[bool] = None,
        committed_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = dict(get_system_prompt_state(conversation_id))
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
        states[conversation_id] = state
        system_prompt_states.set(states)
        return state

    def apply_system_prompt_view(state: Dict[str, Any]) -> None:
        system_prompt_seed.set(str(state.get("prompt") or ""))
        reasoning_effort = normalize_reasoning_effort(
            state.get("reasoning_effort"),
            default="medium",
        )
        ui.update_select("reasoningEffort", selected=reasoning_effort, session=session)

    async def load_system_prompt_state(conversation_id: str) -> Dict[str, Any]:
        if not conversation_id:
            state = build_system_prompt_state()
            apply_system_prompt_view(state)
            return state

        cached_state = system_prompt_states.get().get(conversation_id)
        if isinstance(cached_state, dict):
            apply_system_prompt_view(cached_state)
            return cached_state

        conversation_history = await fetch_conversation_history(conversation_id)
        conversation_control_state = await fetch_conversation_control_state(conversation_id)
        persisted_reasoning = str(conversation_control_state.get("reasoning_effort") or "").strip()
        reasoning_effort = normalize_reasoning_effort(
            persisted_reasoning or input.reasoningEffort(),
            default="medium",
        )
        if isinstance(conversation_history, list) and conversation_history:
            restored_prompt = extract_first_system_prompt(conversation_history)
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
        states[conversation_id] = state
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

    async def single_node_dispatch_spec(
        workflow_dispatch_id: str,
    ) -> dict[str, Any] | None:
        workflow_id = str(workflow_dispatch_id or "").strip()
        if not workflow_id:
            return None
        try:
            return await fetch_workflow(workflow_id)
        except Exception:
            return None

    async def update_retrieval_endpoints(current_selection: Optional[str] = None) -> None:
        retrieval_choices, default_selection = await fetch_available_retrieval_endpoints()
        with reactive.isolate():
            workflow_dispatch_id = str(
                input.workflowDispatch() or selected_workflow_dispatch_id.get() or ""
            ).strip()
        dispatch_spec = await single_node_dispatch_spec(workflow_dispatch_id)
        panel_state = single_node_right_panel_state(
            workflow_dispatch_id,
            dispatch_spec,
        )
        requires_endpoint = panel_state["retrieval_endpoint_enabled"] and bool(
            workflow_dispatch_id
        )
        single_node_choices = build_retrieval_endpoint_choices(
            retrieval_choices,
            require_endpoint=requires_endpoint,
        )
        if not panel_state["retrieval_endpoint_enabled"]:
            current_selection = ""
        selected = (
            resolve_retrieval_endpoint_selection(
                single_node_choices,
                current_selection=current_selection,
                default_selection=default_selection,
                require_endpoint=requires_endpoint,
            )
        )
        ui.update_select(
            "retrievalEndpoint",
            choices=single_node_choices,
            selected=selected,
            session=session,
        )
        with reactive.isolate():
            workflow_current = str(input.workflowRetrievalEndpoint() or "")
        workflow_selected = (
            workflow_current if workflow_current in retrieval_choices else default_selection
        )
        ui.update_select(
            "workflowRetrievalEndpoint",
            choices=retrieval_choices,
            selected=workflow_selected,
            session=session,
        )

    async def update_single_node_retrieval_endpoint_select(
        workflow_dispatch_id: str,
        *,
        spec: dict[str, Any] | None = None,
        current_selection: Optional[str] = None,
    ) -> None:
        retrieval_choices, default_selection = await fetch_available_retrieval_endpoints()
        panel_state = single_node_right_panel_state(workflow_dispatch_id, spec)
        requires_endpoint = panel_state["retrieval_endpoint_enabled"] and bool(
            workflow_dispatch_id
        )
        choices = build_retrieval_endpoint_choices(
            retrieval_choices,
            require_endpoint=requires_endpoint,
        )
        if not panel_state["retrieval_endpoint_enabled"]:
            current_selection = ""
        selected = resolve_retrieval_endpoint_selection(
            choices,
            current_selection=current_selection,
            default_selection=default_selection,
            require_endpoint=requires_endpoint,
        )
        ui.update_select(
            "retrievalEndpoint",
            choices=choices,
            selected=selected,
            session=session,
        )

    def update_single_node_search_provider_select(
        workflow_dispatch_id: str,
        *,
        spec: dict[str, Any] | None = None,
        current_selection: Optional[str] = None,
    ) -> None:
        search_choices, default_selection = fetch_available_search_providers()
        requires_provider = workflow_requires_search_provider(workflow_dispatch_id)
        panel_state = single_node_right_panel_state(workflow_dispatch_id, spec)
        choices = build_search_provider_choices(
            search_choices,
            require_provider=requires_provider,
        )
        force_default = bool(workflow_dispatch_id) and not panel_state[
            "search_provider_enabled"
        ]
        selected = resolve_search_provider_selection(
            choices,
            current_selection=current_selection,
            default_selection=default_selection,
            require_provider=requires_provider,
            force_default=force_default,
        )
        ui.update_select(
            "searchProvider",
            choices=choices,
            selected=selected,
            session=session,
        )

    def update_workflow_search_provider_select(
        workflow_id: str,
        *,
        current_selection: Optional[str] = None,
    ) -> None:
        search_choices, default_selection = fetch_available_search_providers()
        requires_provider = workflow_requires_search_provider(workflow_id)
        choices = build_search_provider_choices(
            search_choices,
            require_provider=requires_provider,
        )
        selected = resolve_search_provider_selection(
            choices,
            current_selection=current_selection,
            default_selection=default_selection,
            require_provider=requires_provider,
            force_default=not requires_provider,
        )
        ui.update_select(
            "workflowSearchProvider",
            choices=choices,
            selected=selected,
            session=session,
        )

    async def update_search_providers(
        current_selection: Optional[str] = None,
    ) -> None:
        with reactive.isolate():
            workflow_dispatch_id = str(
                input.workflowDispatch() or selected_workflow_dispatch_id.get() or ""
            ).strip()
            workflow_current = str(input.workflowSearchProvider() or "")
            workflow_id = str(
                input.workflowSelector() or selected_workflow_id.get() or ""
            )
        dispatch_spec = await single_node_dispatch_spec(workflow_dispatch_id)
        update_single_node_search_provider_select(
            workflow_dispatch_id,
            spec=dispatch_spec,
            current_selection=current_selection,
        )
        update_workflow_search_provider_select(
            workflow_id,
            current_selection=workflow_current,
        )

    async def apply_single_node_workflow_dispatch_state(
        workflow_dispatch_id: str,
        *,
        current_search_provider: Optional[str] = None,
    ) -> None:
        selected_workflow_dispatch_id.set(workflow_dispatch_id)
        dispatch_spec = await single_node_dispatch_spec(workflow_dispatch_id)
        panel_state = single_node_right_panel_state(
            workflow_dispatch_id,
            dispatch_spec,
        )
        update_single_node_search_provider_select(
            workflow_dispatch_id,
            spec=dispatch_spec,
            current_selection=current_search_provider,
        )
        await update_single_node_retrieval_endpoint_select(
            workflow_dispatch_id,
            spec=dispatch_spec,
            current_selection=str(input.retrievalEndpoint() or ""),
        )
        if not panel_state["repo_context_repo_enabled"]:
            ui.update_select("repoContextRepo", selected="", session=session)
        await session.send_custom_message(
            "workflowDispatchState",
            {
                "active": panel_state["active"],
                "retrievalEndpointEnabled": panel_state[
                    "retrieval_endpoint_enabled"
                ],
                "searchProviderEnabled": panel_state["search_provider_enabled"],
                "repoContextRepoEnabled": panel_state["repo_context_repo_enabled"],
                "uploadEnabled": panel_state["upload_enabled"],
            },
        )

    async def update_repo_context_repositories() -> None:
        choices, default_selection = await fetch_available_repo_context_repositories()
        with reactive.isolate():
            current_single_node = str(input.repoContextRepo() or "")
            current_workflow = str(input.workflowRepoContextRepo() or "")
        ui.update_select(
            "repoContextRepo",
            choices=choices,
            selected=current_single_node
            if current_single_node in choices
            else default_selection,
            session=session,
        )
        ui.update_select(
            "workflowRepoContextRepo",
            choices=choices,
            selected=(
                current_workflow if current_workflow in choices else default_selection
            ),
            session=session,
        )

    async def update_history_selector() -> None:
        with reactive.isolate():
            current_selection = current_history_conversation_id()
        conversations = await fetch_conversation_summaries()
        history_choices = create_history_select_choices(conversations)
        selected = current_selection if current_selection in history_choices else None
        if not selected and conversations:
            selected = str(conversations[0]["conversation_id"])
        ui.update_select(
            "historyConversationSelector",
            choices=history_choices,
            selected=selected,
            session=session,
        )

    async def update_workflow_selector() -> None:
        with reactive.isolate():
            current_selection = str(selected_workflow_id.get() or "").strip()
            workflow_dispatch_input = input.workflowDispatch()
            current_dispatch_selection = str(
                workflow_dispatch_input
                if workflow_dispatch_input is not None
                else selected_workflow_dispatch_id.get()
            ).strip()
            current_repo_context_repo = str(input.repoContextRepo() or "")
            current_search_provider = str(input.searchProvider() or "")
        workflows = await fetch_workflows()
        workflow_specs.set(workflows)
        choices = format_workflow_choices(workflows)
        dispatch_choices = build_workflow_dispatch_choices(choices)

        dispatch_selected = resolve_workflow_dispatch_selection_for_repo_context(
            dispatch_choices,
            current_selection=current_dispatch_selection,
            repo_name=current_repo_context_repo,
        )
        ui.update_select(
            "workflowDispatch",
            choices=dispatch_choices,
            selected=dispatch_selected,
            session=session,
        )
        await apply_single_node_workflow_dispatch_state(
            dispatch_selected or "",
            current_search_provider=current_search_provider,
        )

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
                with reactive.isolate():
                    workflow_search_current = str(input.workflowSearchProvider() or "")
                update_workflow_search_provider_select(
                    selected,
                    current_selection=workflow_search_current,
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

    async def update_graph_selector() -> None:
        with reactive.isolate():
            current_selection = str(selected_graph_id.get() or "").strip()
        graphs = await fetch_graphs()
        graph_specs.set(graphs)
        choices = format_graph_choices(graphs)
        selected = current_selection if current_selection in choices else None
        if selected is None and choices:
            selected = next(iter(choices))
        ui.update_select(
            "graphSelector",
            choices=choices,
            selected=selected,
            session=session,
        )
        if selected:
            selected_graph_id.set(selected)
            try:
                spec = await fetch_graph(selected)
                graph_spec.set(spec)
                ui.update_text_area(
                    "graphInput",
                    value=graph_input_template(spec),
                    session=session,
                )
                graph_status_message.set("")
            except Exception as exc:
                graph_status_message.set(f"Failed to load graph: {exc}")

    async def update_graph_run_selector() -> None:
        with reactive.isolate():
            current_selection = str(selected_graph_run_id.get() or "").strip()
        runs = await fetch_graph_runs()
        graph_runs.set(runs)
        choices = format_graph_run_choices(runs)
        selected = current_selection if current_selection in choices else None
        if selected is None and choices:
            selected = next(iter(choices))
        ui.update_select(
            "graphRunSelector",
            choices=choices,
            selected=selected,
            session=session,
        )
        if selected:
            selected_graph_run_id.set(selected)
            try:
                graph_run_snapshot.set(await fetch_graph_run(selected))
            except Exception as exc:
                graph_status_message.set(f"Failed to load graph run: {exc}")

    shinyswatch.theme_picker_server()

    @reactive.Effect
    async def _initialize_endpoints():
        await update_endpoints_and_data()
        await update_retrieval_endpoints()
        await update_search_providers()
        await update_repo_context_repositories()
        await update_history_selector()
        await update_workflow_selector()
        await update_workflow_run_selector()
        await update_graph_selector()
        await update_graph_run_selector()

    @reactive.Effect
    def _initialize_conversation_id():
        ui.update_text("conversationID", value=str(uuid.uuid4().hex[:12]), session=session)

    @reactive.Effect
    @reactive.event(input.refreshEndpoints)
    async def _refresh_endpoints():
        await update_endpoints_and_data()

    @reactive.Effect
    @reactive.event(input.refreshRetrievalEndpoints)
    async def _refresh_retrieval_endpoints():
        await update_retrieval_endpoints(str(input.retrievalEndpoint() or ""))
        await update_search_providers(str(input.searchProvider() or ""))

    @reactive.Effect
    @reactive.event(input.refreshWorkflows)
    async def _refresh_workflows():
        await update_workflow_selector()
        await update_repo_context_repositories()

    @reactive.Effect
    @reactive.event(input.refreshWorkflowRuns)
    async def _refresh_workflow_runs():
        await update_workflow_run_selector()

    @reactive.Effect
    @reactive.event(input.refreshGraphs)
    async def _refresh_graphs():
        await update_graph_selector()

    @reactive.Effect
    @reactive.event(input.refreshGraphRuns)
    async def _refresh_graph_runs():
        await update_graph_run_selector()

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
            update_workflow_search_provider_select(
                workflow_id,
                current_selection=input.workflowSearchProvider(),
            )
        except Exception as exc:
            workflow_status_message.set(f"Failed to load workflow: {exc}")

    @reactive.Effect
    @reactive.event(input.workflowDispatch)
    async def _sync_selected_workflow_dispatch():
        workflow_dispatch_id = str(input.workflowDispatch() or "").strip()
        await apply_single_node_workflow_dispatch_state(
            workflow_dispatch_id,
            current_search_provider=input.searchProvider(),
        )

    @reactive.Effect
    @reactive.event(input.repoContextRepo)
    async def _sync_repo_context_workflow_dispatch():
        repo_name = str(input.repoContextRepo() or "").strip()
        if not repo_name:
            return
        with reactive.isolate():
            workflow_choices = format_workflow_choices(workflow_specs.get())
            dispatch_choices = build_workflow_dispatch_choices(workflow_choices)
            workflow_dispatch_input = input.workflowDispatch()
            current_dispatch_selection = str(
                workflow_dispatch_input
                if workflow_dispatch_input is not None
                else selected_workflow_dispatch_id.get()
            ).strip()
            current_search_provider = str(input.searchProvider() or "")
        dispatch_selected = resolve_workflow_dispatch_selection_for_repo_context(
            dispatch_choices,
            current_selection=current_dispatch_selection,
            repo_name=repo_name,
        )
        if not dispatch_selected or dispatch_selected == current_dispatch_selection:
            return
        ui.update_select(
            "workflowDispatch",
            selected=dispatch_selected,
            session=session,
        )
        await apply_single_node_workflow_dispatch_state(
            dispatch_selected,
            current_search_provider=current_search_provider,
        )

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

    @reactive.Effect
    @reactive.event(input.graphSelector)
    async def _sync_selected_graph():
        graph_id = str(input.graphSelector() or "").strip()
        selected_graph_id.set(graph_id)
        if not graph_id:
            graph_spec.set(None)
            return
        try:
            spec = await fetch_graph(graph_id)
            graph_spec.set(spec)
            ui.update_text_area(
                "graphInput",
                value=graph_input_template(spec),
                session=session,
            )
            graph_status_message.set("")
        except Exception as exc:
            graph_status_message.set(f"Failed to load graph: {exc}")

    @reactive.Effect
    @reactive.event(input.graphRunSelector)
    async def _sync_selected_graph_run():
        run_id = str(input.graphRunSelector() or "").strip()
        selected_graph_run_id.set(run_id)
        if not run_id:
            graph_run_snapshot.set(None)
            return
        try:
            graph_run_snapshot.set(await fetch_graph_run(run_id))
            graph_status_message.set("")
        except Exception as exc:
            graph_status_message.set(f"Failed to load graph run: {exc}")

    @reactive.Effect
    async def _sync_workflow_run_controls():
        snapshot = workflow_run_snapshot.get()
        in_progress = workflow_run_in_progress(snapshot)
        await session.send_custom_message(
            "workflowRunControlState",
            {"disabled": in_progress},
        )
        if in_progress:
            return
        with reactive.isolate():
            current_retry_step = str(input.workflowRetryStepID() or "").strip()
        ui.update_select(
            "workflowRetryStepID",
            choices=build_workflow_retry_step_choices(snapshot),
            selected=resolve_workflow_retry_step_selection(
                snapshot,
                current_selection=current_retry_step,
            ),
            session=session,
        )

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

    def graph_json_payload(input_id: str, *, label: str) -> dict[str, Any]:
        raw = str(getattr(input, input_id)() or "").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{label} must be a JSON object")
        return data

    async def refresh_selected_workflow_run(run_id: str) -> None:
        workflow_run_snapshot.set(await fetch_workflow_run(run_id))
        selected_workflow_run_id.set(run_id)
        await update_workflow_run_selector()

    async def refresh_selected_graph_run(run_id: str) -> None:
        graph_run_snapshot.set(await fetch_graph_run(run_id))
        selected_graph_run_id.set(run_id)
        await update_graph_run_selector()

    @reactive.Effect
    @reactive.event(input.createWorkflowRun)
    async def _create_workflow_run():
        workflow_id = str(
            selected_workflow_id.get() or input.workflowSelector() or ""
        ).strip()
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
            params = merge_uploaded_source_text(
                workflow_params_payload(),
                build_uploaded_file_source_text(uploaded_files),
            )
            selected_spec = workflow_spec.get()
            if (
                not isinstance(selected_spec, dict)
                or str(selected_spec.get("id") or "").strip() != workflow_id
            ):
                selected_spec = await fetch_workflow(workflow_id)
            params = merge_repo_context_repo_name(
                params,
                selected_spec,
                str(input.workflowRepoContextRepo() or ""),
            )
            payload = {
                "params": params,
                "endpoint": workflow_endpoint_key(),
            }
            reasoning = str(input.workflowReasoning() or "").strip()
            if reasoning:
                payload["reasoning_effort"] = reasoning
            retrieval_endpoint = str(input.workflowRetrievalEndpoint() or "").strip()
            if retrieval_endpoint:
                payload["retrieval_endpoint"] = retrieval_endpoint
            search_provider = str(input.workflowSearchProvider() or "").strip()
            if search_provider:
                payload["search_provider"] = search_provider
            conversation_id = str(input.workflowConversationID() or "").strip()
            if conversation_id:
                payload["conversation_id"] = conversation_id
            created = await create_workflow_run(workflow_id, payload)
            run_id = str(created.get("run_id") or "").strip()
            if not run_id:
                raise ValueError("workflow API did not return run_id")
            workflow_status_message.set(f"Created run {run_id}.")
            await refresh_selected_workflow_run(run_id)
        except Exception as exc:
            workflow_status_message.set(f"Create run failed: {exc}")

    @reactive.Effect
    @reactive.event(input.createGraphRun)
    async def _create_graph_run():
        graph_id = str(selected_graph_id.get() or input.graphSelector() or "").strip()
        if not graph_id:
            graph_status_message.set("Select a graph first.")
            return
        try:
            created = await create_graph_run(
                graph_id,
                {
                    "input": graph_json_payload("graphInput", label="graph input"),
                    "config": graph_json_payload("graphConfig", label="graph config"),
                },
            )
            run_id = str(created.get("run_id") or "").strip()
            if not run_id:
                raise ValueError("graph API did not return run_id")
            graph_status_message.set(f"Created graph run {run_id}.")
            await refresh_selected_graph_run(run_id)
        except Exception as exc:
            graph_status_message.set(f"Create graph run failed: {exc}")

    @reactive.Effect
    @reactive.event(input.advanceWorkflowRun)
    async def _advance_workflow_run():
        run_id = str(selected_workflow_run_id.get() or input.workflowRunSelector() or "").strip()
        if not run_id:
            workflow_status_message.set("Select a run first.")
            return
        if workflow_run_in_progress(workflow_run_snapshot.get()):
            workflow_status_message.set(f"Run {run_id} is already in progress.")
            return
        try:
            optimistic = workflow_snapshot_with_next_pending_running(
                workflow_run_snapshot.get()
            )
            if optimistic is not None:
                workflow_run_snapshot.set(optimistic)
                workflow_status_message.set(f"Advancing run {run_id}.")
                await reactive.flush()
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
        if workflow_run_in_progress(workflow_run_snapshot.get()):
            workflow_status_message.set(f"Run {run_id} is already in progress.")
            return
        try:
            async def advance_with_running_snapshot(run_id: str) -> dict[str, Any]:
                optimistic = workflow_snapshot_with_next_pending_running(
                    workflow_run_snapshot.get()
                )
                if optimistic is not None:
                    workflow_run_snapshot.set(optimistic)
                    workflow_status_message.set(f"Advancing run {run_id}.")
                    await reactive.flush()
                return await advance_workflow_run(run_id)

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
                advance_with_running_snapshot,
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
            workflow_status_message.set("Select a run and retry step.")
            return
        if workflow_run_in_progress(workflow_run_snapshot.get()):
            workflow_status_message.set(f"Run {run_id} is already in progress.")
            return
        try:
            workflow_run_snapshot.set(await retry_workflow_step(run_id, step_id))
            workflow_status_message.set(f"Retried step {step_id}.")
            await update_workflow_run_selector()
        except Exception as exc:
            workflow_status_message.set(f"Retry failed: {exc}")

    @reactive.Effect
    @reactive.event(input.runGraphToCompletion)
    async def _run_graph_to_completion():
        run_id = str(selected_graph_run_id.get() or input.graphRunSelector() or "").strip()
        if not run_id:
            graph_status_message.set("Select a graph run first.")
            return
        try:
            snapshot = await run_graph_to_completion(run_id)
            graph_run_snapshot.set(snapshot)
            status = str(snapshot.get("run", {}).get("status") or "unknown")
            graph_status_message.set(f"Ran graph run {run_id}: status {status}.")
            await update_graph_run_selector()
        except Exception as exc:
            graph_status_message.set(f"Run graph failed: {exc}")

    @reactive.Effect
    @reactive.event(input.streamGraphRun)
    async def _stream_graph_run():
        run_id = str(selected_graph_run_id.get() or input.graphRunSelector() or "").strip()
        if not run_id:
            graph_status_message.set("Select a graph run first.")
            return
        try:
            final_snapshot = None
            async for event in stream_graph_run_events(run_id):
                snapshot = event.get("snapshot") if isinstance(event, dict) else None
                if isinstance(snapshot, dict):
                    final_snapshot = snapshot
                    graph_run_snapshot.set(snapshot)
            if final_snapshot is None:
                final_snapshot = await fetch_graph_run(run_id)
                graph_run_snapshot.set(final_snapshot)
            status = str(final_snapshot.get("run", {}).get("status") or "unknown")
            graph_status_message.set(f"Streamed graph run {run_id}: status {status}.")
            await update_graph_run_selector()
        except Exception as exc:
            graph_status_message.set(f"Stream graph failed: {exc}")

    @reactive.Effect
    @reactive.event(input.endpoint)
    def _clear_info_on_endpoint_change():
        endpoint_key = current_endpoint_key()
        if endpoint_key:
            selected_endpoint_key_state.set(endpoint_key)
        run_info.set(None)
        last_runtime.set(None)

    @reactive.Effect
    @reactive.event(input.conversationID)
    async def _sync_system_prompt_for_conversation():
        await load_system_prompt_state(current_active_conversation_id())

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
    @reactive.event(input.generateConversationID)
    def _generate_conversation_id():
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
        ui.update_text("conversationID", value=new_uuid, session=session)

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
        conversation_id = current_active_conversation_id()
        if not conversation_id:
            return
        set_system_prompt_state(conversation_id, prompt=input.systemPrompt() or "")

    @reactive.Effect
    @reactive.event(input.retrievalEndpoint)
    async def _append_retrieval_suffix_for_selected_endpoint():
        conversation_id = current_active_conversation_id()
        if not conversation_id:
            return

        retrieval_endpoint = str(input.retrievalEndpoint() or "").strip()
        retrieval_choices, _default_selection = await fetch_available_retrieval_endpoints()
        configured_retrieval_endpoints = {
            str(value).strip() for value in retrieval_choices if value
        }
        if not retrieval_endpoint or retrieval_endpoint not in configured_retrieval_endpoints:
            return

        current_prompt = input.systemPrompt()
        if current_prompt is None:
            current_prompt = get_system_prompt_state(conversation_id).get("prompt", "")

        updated_prompt = append_managed_retrieval_suffix(current_prompt)
        if updated_prompt == normalize_system_prompt(current_prompt):
            return

        apply_system_prompt_view(
            set_system_prompt_state(conversation_id, prompt=updated_prompt)
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
    def graphSpecDetails():
        return format_graph_spec(graph_spec.get())

    @render.ui
    def graphRunDetails():
        return format_graph_run_details(graph_run_snapshot.get())

    @render.text
    def graphStatusMessage():
        return graph_status_message.get()

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

        if info and "workflow" in info:
            workflow = info["workflow"]
            if isinstance(workflow, dict):
                workflow_lines = []
                if workflow.get("workflow_id"):
                    workflow_lines.append(f"Workflow: {workflow['workflow_id']}")
                if workflow.get("run_id"):
                    workflow_lines.append(f"Run: {workflow['run_id']}")
                if workflow.get("status"):
                    workflow_lines.append(f"Status: {workflow['status']}")
                if workflow_lines:
                    sections.append("**Workflow**<br>" + "<br>".join(workflow_lines))

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

        if info and "retrieval" in info:
            retrieval = info["retrieval"]
            retrieval_lines = []
            if retrieval.get("endpoint"):
                retrieval_lines.append(f"Endpoint: {retrieval['endpoint']}")
            if retrieval.get("injected"):
                retrieval_lines.append(f"Injected: {retrieval['injected']}")
            if retrieval.get("confidence"):
                retrieval_lines.append(f"Confidence: {retrieval['confidence']}")
            if retrieval.get("threshold"):
                retrieval_lines.append(f"Threshold: {retrieval['threshold']}")
            if retrieval.get("hits"):
                retrieval_lines.append(f"Hits: {retrieval['hits']}")
            if retrieval.get("method"):
                retrieval_lines.append(f"Method: {retrieval['method']}")
            if retrieval.get("reason") and retrieval.get("injected") == "false":
                retrieval_lines.append(f"Skipped: {retrieval['reason']}")
            if retrieval_lines:
                sections.append("**Retrieval**<br>" + "<br>".join(retrieval_lines))

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

    async def _handle_chat_input_impl(
        user_input: str,
        finalize_runtime: Callable[[], None],
    ) -> bool:
        current_endpoints = available_endpoints.get()
        current_run_info = run_info.get() or {}
        selected_endpoint_key = current_endpoint_key()
        conversation_id = current_active_conversation_id() or None
        actual_endpoint_key = selected_endpoint_key
        latest_user_prompt = str(user_input or "").strip()
        search_query = latest_user_prompt
        workflow_dispatch_input = _input_value(input, "workflowDispatch")
        workflow_dispatch_id = str(
            workflow_dispatch_input
            if workflow_dispatch_input is not None
            else selected_workflow_dispatch_id.get()
        ).strip()
        prompt_state = (
            get_system_prompt_state(conversation_id)
            if conversation_id
            else build_system_prompt_state()
        )

        files_data = current_files.get()
        uploaded_files = (
            files_data.get("files")
            if files_data.get("key") == file_upload_key.get()
            else None
        )
        uploaded_source_text = build_uploaded_file_source_text(uploaded_files)
        if uploaded_source_text and not workflow_dispatch_id:
            user_input = f"{user_input}\n\n{uploaded_source_text}"

        system_prompt_input = _input_value(input, "systemPrompt")
        current_prompt = normalize_system_prompt(
            system_prompt_input
            if system_prompt_input is not None
            else prompt_state.get("prompt")
        )
        current_reasoning = normalize_reasoning_effort(
            _input_value(input, "reasoningEffort"),
            default="medium",
        )
        forked_history: list[dict[str, Any]] = []
        fork_reasons = conversation_control_change_reasons(
            prompt_state,
            current_prompt=current_prompt,
            current_reasoning=current_reasoning,
        )
        if conversation_id and fork_reasons:
            old_conversation_id = conversation_id
            loaded_history = await fetch_conversation_history(old_conversation_id)
            if isinstance(loaded_history, list):
                forked_history = [
                    dict(message)
                    for message in loaded_history
                    if isinstance(message, dict) and message.get("role") != "system"
                ]
            conversation_id = str(uuid.uuid4().hex[:12])
            prompt_state = set_system_prompt_state(
                conversation_id,
                prompt=current_prompt,
                committed_prompt=current_prompt,
                reasoning_effort=current_reasoning,
                started=False,
                locked=False,
            )
            apply_system_prompt_view(prompt_state)
            ui.update_text("conversationID", value=conversation_id, session=session)
            await chat.append_message(
                {
                    "role": "assistant",
                    "content": build_fork_notice(
                        old_conversation_id=old_conversation_id,
                        new_conversation_id=conversation_id,
                        reasons=fork_reasons,
                    ),
                }
            )

        system_prompt_to_send = first_turn_system_prompt_to_send(
            current_prompt,
            bool(prompt_state.get("started")),
        )

        if workflow_dispatch_id:
            send_button_state.set("busy")
            runtime_deferred = False
            try:
                endpoint_for_workflow = str(actual_endpoint_key or "").strip()
                if not endpoint_for_workflow:
                    raise ValueError("select a machine endpoint")
                if endpoint_for_workflow.lower() == "smart":
                    raise ValueError(
                        "workflow dispatch requires a concrete Machine Endpoint"
                    )
                workflow_retrieval_endpoint = str(
                    _input_value(input, "retrievalEndpoint") or ""
                ).strip()
                spec = await fetch_workflow(workflow_dispatch_id)
                panel_state = single_node_right_panel_state(
                    workflow_dispatch_id,
                    spec,
                )
                if panel_state["retrieval_endpoint_enabled"]:
                    if not workflow_retrieval_endpoint:
                        raise ValueError(
                            "workflow dispatch requires a Retrieval Endpoint"
                        )

                if fork_reasons:
                    workflow_history = forked_history
                elif conversation_id:
                    loaded_history = await fetch_conversation_history(conversation_id)
                    workflow_history = (
                        loaded_history if isinstance(loaded_history, list) else []
                    )
                else:
                    workflow_history = []

                workflow_uploaded_source_text = (
                    uploaded_source_text if panel_state["upload_enabled"] else ""
                )
                workflow_repo_name = (
                    str(_input_value(input, "repoContextRepo") or "").strip()
                    if panel_state["repo_context_repo_enabled"]
                    else ""
                )
                workflow_search_provider = (
                    str(_input_value(input, "searchProvider") or "").strip()
                    if panel_state["search_provider_enabled"]
                    else ""
                )

                payload = build_workflow_chat_run_payload(
                    spec,
                    latest_user_prompt=latest_user_prompt,
                    thread_briefing=format_workflow_thread_briefing(
                        workflow_history
                    ),
                    manual_source_text=current_prompt,
                    uploaded_source_text=workflow_uploaded_source_text,
                    repo_name=workflow_repo_name,
                    endpoint=endpoint_for_workflow,
                    reasoning_effort=current_reasoning,
                    conversation_id=conversation_id or "",
                    retrieval_endpoint=(
                        workflow_retrieval_endpoint
                        if panel_state["retrieval_endpoint_enabled"]
                        else ""
                    ),
                    search_provider=workflow_search_provider,
                )
                created = await create_workflow_run(workflow_dispatch_id, payload)
                run_id = str(created.get("run_id") or "").strip()
                if not run_id:
                    raise ValueError("workflow API did not return run_id")

                if conversation_id:
                    await append_conversation_messages(
                        conversation_id,
                        [{"role": "user", "content": latest_user_prompt}],
                    )

                workflow_status_message.set(f"Created run {run_id}.")
                selected_workflow_run_id.set(run_id)
                await update_workflow_run_selector()
                await refresh_selected_workflow_run(run_id)
                await reactive.flush()
                workflow_stream_enabled = bool(_input_value(input, "stream", False))
                workflow_output_reasoning = bool(
                    _input_value(input, "outputReasoning", False)
                )

                async def apply_workflow_snapshot(
                    snapshot: dict[str, Any],
                    *,
                    step_number: int | None = None,
                    refresh_runs: bool = True,
                ) -> None:
                    workflow_run_snapshot.set(snapshot)
                    selected_workflow_run_id.set(run_id)
                    status = workflow_snapshot_status(snapshot) or "unknown"
                    if step_number is not None:
                        workflow_status_message.set(
                            f"Advanced {run_id}: step {step_number}, status {status}."
                        )
                    if refresh_runs:
                        await update_workflow_run_selector()
                    await reactive.flush()

                final_text_parts: list[str] = []
                final_text_persisted = False

                async def workflow_response_stream():
                    nonlocal final_text_persisted
                    rendered_final = False
                    open_intermediate_step = ""
                    final_reasoning_open = False
                    final_snapshot: dict[str, Any] | None = None

                    async def close_intermediate():
                        nonlocal open_intermediate_step
                        if open_intermediate_step:
                            open_intermediate_step = ""
                            return "</em>\n\n---\n\n"
                        return ""

                    async def close_final_reasoning():
                        nonlocal final_reasoning_open
                        if final_reasoning_open:
                            final_reasoning_open = False
                            return "</em>\n\n---\n\n"
                        return ""

                    async for event in stream_workflow_run_events(
                        run_id,
                        stream=workflow_stream_enabled,
                        max_steps=WORKFLOW_RUN_MAX_STEPS,
                    ):
                        event_type = str(event.get("type") or "")
                        snapshot = event.get("snapshot")
                        if isinstance(snapshot, dict):
                            final_snapshot = snapshot

                        visibility = str(event.get("chat_visibility") or "hidden")
                        step_id = str(event.get("step_id") or "")
                        step_name = str(
                            event.get("step_name") or step_id or "Workflow step"
                        )

                        if event_type == "step_delta":
                            content = str(event.get("content") or "")
                            channel = str(event.get("channel") or "content")
                            if not content:
                                continue
                            if visibility == "intermediate" and channel == "content":
                                if open_intermediate_step != step_id:
                                    closing = await close_intermediate()
                                    if closing:
                                        yield closing
                                    open_intermediate_step = step_id
                                    yield f"<em>**{step_name}**\n\n"
                                yield content
                            elif visibility == "final":
                                if open_intermediate_step:
                                    yield await close_intermediate()
                                if channel == "reasoning":
                                    if workflow_output_reasoning:
                                        if not final_reasoning_open:
                                            final_reasoning_open = True
                                            yield "<em>"
                                        yield content
                                elif channel == "content":
                                    if final_reasoning_open:
                                        yield await close_final_reasoning()
                                    rendered_final = True
                                    final_text_parts.append(content)
                                    yield content
                            continue

                        if event_type == "step_completed":
                            content = str(event.get("content") or "")
                            streamed = bool(event.get("streamed"))
                            if visibility == "intermediate":
                                if streamed:
                                    if open_intermediate_step == step_id:
                                        yield await close_intermediate()
                                elif content:
                                    yield (
                                        f"<em>**{step_name}**\n\n"
                                        f"{format_workflow_intermediate_content(content)}"
                                        "</em>\n\n---\n\n"
                                    )
                            elif visibility == "final" and content and not streamed:
                                if open_intermediate_step:
                                    yield await close_intermediate()
                                if final_reasoning_open:
                                    yield await close_final_reasoning()
                                rendered_final = True
                                final_text_parts.append(content)
                                yield content
                            if (
                                workflow_dispatch_event_updates_run_details(event_type)
                                and isinstance(snapshot, dict)
                            ):
                                await apply_workflow_snapshot(
                                    snapshot,
                                    refresh_runs=False,
                                )
                            continue

                        if event_type == "error":
                            if open_intermediate_step:
                                yield await close_intermediate()
                            if final_reasoning_open:
                                yield await close_final_reasoning()
                            if (
                                workflow_dispatch_event_updates_run_details(event_type)
                                and final_snapshot is not None
                            ):
                                await apply_workflow_snapshot(final_snapshot)
                            yield f"Error: {event.get('error') or 'workflow failed'}"
                            return

                        if event_type == "run_completed":
                            if open_intermediate_step:
                                yield await close_intermediate()
                            if final_reasoning_open:
                                yield await close_final_reasoning()
                            if isinstance(snapshot, dict):
                                final_snapshot = snapshot
                            if final_snapshot is not None:
                                status = (
                                    workflow_snapshot_status(final_snapshot)
                                    or "unknown"
                                )
                                workflow_status_message.set(
                                    f"Ran {run_id}: status {status}."
                                )
                                if workflow_dispatch_event_updates_run_details(event_type):
                                    await apply_workflow_snapshot(final_snapshot)
                                run_info.set(workflow_chat_run_info(final_snapshot))
                                if not rendered_final:
                                    rendered_final = True
                                    final_text = workflow_chat_response_text(final_snapshot)
                                    final_text_parts.append(final_text)
                                    yield final_text
                                if (
                                    conversation_id
                                    and final_text_parts
                                    and not final_text_persisted
                                ):
                                    await append_conversation_messages(
                                        conversation_id,
                                        [
                                            {
                                                "role": "assistant",
                                                "content": "".join(final_text_parts),
                                            }
                                        ],
                                    )
                                    final_text_persisted = True
                            return

                await chat.append_message_stream(
                    stream_with_finalizer(workflow_response_stream(), finalize_runtime)
                )
                runtime_deferred = True
                if conversation_id and not bool(prompt_state.get("started")):
                    apply_system_prompt_view(
                        set_system_prompt_state(
                            conversation_id,
                            prompt=current_prompt,
                            committed_prompt=current_prompt,
                            reasoning_effort=current_reasoning,
                            started=True,
                            locked=False,
                        )
                    )
            except Exception as exc:
                message = f"Workflow dispatch failed: {exc}"
                workflow_status_message.set(message)
                await chat.append_message(
                    {"role": "assistant", "content": f"Error: {message}"}
                )
            finally:
                send_button_state.set("ready")
            return runtime_deferred

        search_state = None
        selected_search_provider = str(
            _input_value(input, "searchProvider") or ""
        ).strip()
        if selected_search_provider and search_query:
            try:
                if fork_reasons:
                    query_refiner_history = forked_history
                elif conversation_id:
                    query_refiner_history = await fetch_conversation_history(conversation_id)
                else:
                    query_refiner_history = []
                query_refiner_source_text = build_query_refiner_source_text(
                    system_prompt=current_prompt,
                    history=query_refiner_history,
                    user_input=user_input,
                )
                search_response = await fetch_search_results(
                    query=search_query,
                    provider=selected_search_provider,
                    count=5,
                    source_text=query_refiner_source_text,
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

        retrieval_endpoint = _input_value(input, "retrievalEndpoint")
        response_stream = stream_chat_response(
            endpoint_key=actual_endpoint_key,
            text=user_input,
            endpoints_dict=current_endpoints,
            stream=bool(_input_value(input, "stream", False)),
            output_json=bool(_input_value(input, "outputJSON", False)),
            reasoning_effort=current_reasoning,
            output_reasoning=bool(_input_value(input, "outputReasoning", False)),
            conversation_id=conversation_id,
            current_routing_info=current_run_info.get("routing", {}),
            system_prompt=system_prompt_to_send,
            extra_turn_messages=extra_turn_messages,
            retrieval_endpoint=retrieval_endpoint if retrieval_endpoint else None,
            on_metadata=publish_run_info,
            on_send_button_state=send_button_state.set,
        )
        await chat.append_message_stream(
            stream_with_finalizer(response_stream, finalize_runtime)
        )

        if conversation_id and not bool(prompt_state.get("started")):
            apply_system_prompt_view(
                set_system_prompt_state(
                    conversation_id,
                    prompt=current_prompt,
                    committed_prompt=current_prompt,
                    reasoning_effort=current_reasoning,
                    started=True,
                    locked=False,
                )
            )
        return True

    @chat.on_user_submit
    async def _handle_chat_input(user_input: str):
        task_started_at = time.perf_counter()
        last_runtime.set(None)
        runtime_finalized = False

        def finalize_runtime() -> None:
            nonlocal runtime_finalized
            if runtime_finalized:
                return
            runtime_finalized = True
            last_runtime.set(elapsed_seconds_since(task_started_at))

        runtime_deferred = False
        try:
            runtime_deferred = await _handle_chat_input_impl(user_input, finalize_runtime)
        finally:
            if not runtime_deferred:
                finalize_runtime()

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
                conversation_id=str(input.traceConversationFilter() or ""),
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
    @reactive.event(input.historyConversationSelector)
    def _sync_history_selector_state():
        history_selected_conversation_id.set(str(input.historyConversationSelector() or "").strip())

    @reactive.Effect
    @reactive.event(history_refresh_trigger)
    async def _refresh_history_selector():
        await update_history_selector()

    @render.ui
    @reactive.event(input.historyConversationSelector, history_refresh_trigger)
    async def historyBox():
        conversation_id = current_history_conversation_id()
        if not conversation_id:
            return ui.card(
                ui.markdown(
                    "**No conversation ID provided**\n\nEnter a conversation ID to view history."
                ),
            )

        conversation_history = await fetch_conversation_history(conversation_id)
        if not conversation_history:
            return ui.card(
                ui.markdown(
                    f"**No history found for conversation: {conversation_id}**\n\nThis conversation may not exist or has no messages."
                ),
            )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_count = len(conversation_history) if isinstance(conversation_history, list) else None
        return ui.card(
            ui.markdown(f"**Conversation History** *(refreshed at {timestamp})*"),
            (
                ui.markdown(f"`{message_count} messages`")
                if message_count is not None
                else None
            ),
            ui.tags.pre(
                format_history_json(conversation_history),
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
                format_trace_summary(event),
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
