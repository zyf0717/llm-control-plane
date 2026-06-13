import json
import time
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

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
    append_managed_rag_suffix,
    build_system_prompt_state,
    extract_first_system_prompt,
    first_turn_system_prompt_to_send,
    normalize_system_prompt,
)
from .search_flow import (
    build_query_refiner_context,
    build_search_failure_state,
    build_search_preface,
    build_search_success_state,
    build_search_turn_messages,
    merge_run_info,
)
from .trace_formatters import format_trace_summary
from .utils import (
    create_history_select_choices,
    create_endpoint_display_choices,
    fetch_conversation_summaries,
    fetch_available_endpoints,
    fetch_available_rag_endpoints,
    fetch_available_search_providers,
    fetch_convo_history,
    fetch_convo_state,
    fetch_search_results,
    find_model_by_endpoint,
    read_trace_events,
)
from .workflow_server_helpers import (
    WORKFLOW_RUN_MAX_STEPS,
    advance_workflow_to_terminal,
    build_uploaded_file_context,
    build_workflow_chat_run_payload,
    build_workflow_params_template,
    format_workflow_intermediate_content,
    format_workflow_conversation_context,
    merge_uploaded_context,
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

DEFAULT_WORKFLOW_ROUTING_ID = ""
WORKFLOW_ROUTING_NONE_LABEL = "None"
CONTEXTUAL_SEARCH_WORKFLOW_ID = "contextual_search"
ENDED_WORKFLOW_RUN_STATUSES = {"completed", "failed", "cancelled"}


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


def resolve_workflow_routing_selection(
    choices: Dict[str, str],
    *,
    current_selection: Optional[str],
    default_workflow_id: str = DEFAULT_WORKFLOW_ROUTING_ID,
) -> Optional[str]:
    current = str(current_selection or "").strip()
    if current in choices:
        return current

    default = str(default_workflow_id or "").strip()
    if default in choices:
        return default

    return next(iter(choices), None)


def build_workflow_routing_choices(choices: Dict[str, str]) -> Dict[str, str]:
    return {"": WORKFLOW_ROUTING_NONE_LABEL, **choices}


def resolve_first_search_provider_selection(choices: Dict[str, str]) -> str:
    for provider_id in choices:
        provider = str(provider_id or "").strip()
        if provider:
            return provider
    return ""


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
    selected_workflow_routing_id = reactive.Value(DEFAULT_WORKFLOW_ROUTING_ID)
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
            workflow_routing_input = input.workflowRouting()
            current_routing_selection = str(
                workflow_routing_input
                if workflow_routing_input is not None
                else selected_workflow_routing_id.get()
            ).strip()
        workflows = await fetch_workflows()
        workflow_specs.set(workflows)
        choices = format_workflow_choices(workflows)
        routing_choices = build_workflow_routing_choices(choices)

        routing_selected = resolve_workflow_routing_selection(
            routing_choices,
            current_selection=current_routing_selection,
        )
        ui.update_select(
            "workflowRouting",
            choices=routing_choices,
            selected=routing_selected,
            session=session,
        )
        selected_workflow_routing_id.set(routing_selected or "")
        await session.send_custom_message(
            "workflowRoutingState", {"active": bool(routing_selected)}
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
            if workflow_id == CONTEXTUAL_SEARCH_WORKFLOW_ID:
                search_choices, _ = fetch_available_search_providers()
                search_selected = resolve_first_search_provider_selection(search_choices)
                ui.update_select(
                    "workflowSearchProvider",
                    choices=search_choices,
                    selected=search_selected,
                    session=session,
                )
        except Exception as exc:
            workflow_status_message.set(f"Failed to load workflow: {exc}")

    @reactive.Effect
    @reactive.event(input.workflowRouting)
    async def _sync_selected_workflow_routing():
        workflow_routing_id = str(input.workflowRouting() or "").strip()
        selected_workflow_routing_id.set(workflow_routing_id)
        if workflow_routing_id == CONTEXTUAL_SEARCH_WORKFLOW_ID:
            search_choices, _ = fetch_available_search_providers()
            search_selected = resolve_first_search_provider_selection(search_choices)
            ui.update_select(
                "searchProvider",
                choices=search_choices,
                selected=search_selected,
                session=session,
            )
        await session.send_custom_message(
            "workflowRoutingState", {"active": bool(workflow_routing_id)}
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
        latest_user_prompt = str(user_input or "").strip()
        search_query = latest_user_prompt
        workflow_routing_input = _input_value(input, "workflowRouting")
        workflow_routing_id = str(
            workflow_routing_input
            if workflow_routing_input is not None
            else selected_workflow_routing_id.get()
        ).strip()
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
        uploaded_context = build_uploaded_file_context(uploaded_files)
        if uploaded_context and not workflow_routing_id:
            user_input = f"{user_input}\n\n{uploaded_context}"

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

        if workflow_routing_id:
            started_at = time.time()
            send_button_state.set("busy")
            try:
                endpoint_for_workflow = str(actual_endpoint_key or "").strip()
                if not endpoint_for_workflow:
                    raise ValueError("select a machine endpoint")
                if endpoint_for_workflow.lower() == "smart":
                    raise ValueError(
                        "workflow routing requires a concrete Machine Endpoint"
                    )

                if fork_reasons:
                    workflow_history = forked_history
                elif convo_id:
                    loaded_history = await fetch_convo_history(convo_id)
                    workflow_history = (
                        loaded_history if isinstance(loaded_history, list) else []
                    )
                else:
                    workflow_history = []

                spec = await fetch_workflow(workflow_routing_id)
                payload = build_workflow_chat_run_payload(
                    spec,
                    latest_user_prompt=latest_user_prompt,
                    conversation_context=format_workflow_conversation_context(
                        workflow_history
                    ),
                    context=current_prompt,
                    uploaded_context=uploaded_context,
                    endpoint=endpoint_for_workflow,
                    reasoning_effort=current_reasoning,
                    convo_id=convo_id or "",
                    search_provider=(
                        str(_input_value(input, "searchProvider") or "").strip()
                        if workflow_routing_id == CONTEXTUAL_SEARCH_WORKFLOW_ID
                        else ""
                    ),
                )
                created = await create_workflow_run(workflow_routing_id, payload)
                run_id = str(created.get("run_id") or "").strip()
                if not run_id:
                    raise ValueError("workflow API did not return run_id")

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
                ) -> None:
                    workflow_run_snapshot.set(snapshot)
                    selected_workflow_run_id.set(run_id)
                    status = workflow_snapshot_status(snapshot) or "unknown"
                    if step_number is not None:
                        workflow_status_message.set(
                            f"Advanced {run_id}: step {step_number}, status {status}."
                        )
                    await update_workflow_run_selector()
                    await reactive.flush()

                async def workflow_response_stream():
                    rendered_final = False
                    open_intermediate_step = ""
                    final_reasoning_open = False
                    step_count = 0
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
                            if event_type in {"step_completed", "snapshot"}:
                                step_count += 1 if event_type == "step_completed" else 0
                                await apply_workflow_snapshot(
                                    snapshot,
                                    step_number=(
                                        step_count
                                        if event_type == "step_completed"
                                        else None
                                    ),
                                )

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
                                yield content
                            continue

                        if event_type == "error":
                            if open_intermediate_step:
                                yield await close_intermediate()
                            if final_reasoning_open:
                                yield await close_final_reasoning()
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
                                run_info.set(workflow_chat_run_info(final_snapshot))
                                if not rendered_final:
                                    rendered_final = True
                                    yield workflow_chat_response_text(final_snapshot)
                            return

                await chat.append_message_stream(workflow_response_stream())
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
            except Exception as exc:
                message = f"Workflow routing failed: {exc}"
                workflow_status_message.set(message)
                await chat.append_message(
                    {"role": "assistant", "content": f"Error: {message}"}
                )
            finally:
                last_runtime.set(time.time() - started_at)
                send_button_state.set("ready")
            return

        search_state = None
        selected_search_provider = str(
            _input_value(input, "searchProvider") or ""
        ).strip()
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

        rag_endpoint = _input_value(input, "ragEndpoint")
        response_stream = stream_chat_response(
            endpoint_key=actual_endpoint_key,
            text=user_input,
            endpoints_dict=current_endpoints,
            stream=bool(_input_value(input, "stream", False)),
            output_json=bool(_input_value(input, "outputJSON", False)),
            reasoning_effort=current_reasoning,
            output_reasoning=bool(_input_value(input, "outputReasoning", False)),
            convo_id=convo_id,
            current_routing_info=current_run_info.get("routing", {}),
            system_prompt=system_prompt_to_send,
            extra_turn_messages=extra_turn_messages,
            rag_endpoint=rag_endpoint if rag_endpoint else None,
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
