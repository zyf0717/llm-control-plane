import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

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
    create_history_select_choices,
    create_endpoint_display_choices,
    fetch_conversation_summaries,
    fetch_available_endpoints,
    fetch_available_rag_endpoints,
    fetch_available_search_providers,
    fetch_convo_history,
    fetch_search_results,
    format_search_provider_label,
    find_model_by_endpoint,
)


AUTO_ENDPOINT_KEY = "Auto"


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

    planner = search_response.get("planner")
    if not isinstance(planner, dict):
        planner = {}

    return {
        "provider": provider_id,
        "provider_label": format_search_provider_label(provider_id),
        "query": str(search_response.get("query") or "").strip(),
        "planner": planner,
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


def build_search_planner_context(
    *,
    system_prompt: Optional[str],
    history: Any,
    user_input: str,
) -> str:
    """Build compact source context for search planning."""
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

    return "\n".join(
        [EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER, wrapped_results.strip()]
    )


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


def _search_planner_queries(planner: Dict[str, Any], fallback_query: str) -> list[str]:
    raw_queries = planner.get("queries")
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
    planner = (
        search_state.get("planner")
        if isinstance(search_state.get("planner"), dict)
        else {}
    )
    query = " ".join(str(search_state.get("query") or "").split())
    planner_queries = _search_planner_queries(planner, query)
    results = search_state.get("results")
    warnings = search_state.get("warnings") or []
    degraded = bool(search_state.get("degraded", False))
    heading = f"**Search candidates** via {provider_label}"
    if planner.get("used") is True and planner_queries:
        quoted_queries = "; ".join(f'"{item}"' for item in planner_queries)
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
        planner = search_state.get("planner")
        query = str(search_state.get("query") or "").strip()
        if isinstance(planner, dict) and planner.get("used") is True and query:
            merged["search"]["query"] = query
            queries = _search_planner_queries(planner, query)
            if len(queries) > 1:
                merged["search"]["queries"] = queries

    return merged or None


def resolve_auto_endpoint_key(
    endpoint_key: Optional[str],
    convo_id: Optional[str],
    auto_route_pins: Optional[Dict[str, str]],
) -> Optional[str]:
    """Resolve Auto to a previously pinned endpoint for this conversation."""
    if endpoint_key != AUTO_ENDPOINT_KEY:
        return endpoint_key

    normalized_convo_id = str(convo_id or "").strip()
    if not normalized_convo_id:
        return endpoint_key

    pinned_endpoint = str((auto_route_pins or {}).get(normalized_convo_id) or "").strip()
    if pinned_endpoint and pinned_endpoint != AUTO_ENDPOINT_KEY:
        return pinned_endpoint

    return endpoint_key


def pin_auto_route_decision(
    auto_route_pins: Optional[Dict[str, str]],
    convo_id: Optional[str],
    selected_endpoint_key: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Pin the first Auto routing decision for a conversation."""
    pins = dict(auto_route_pins or {})
    normalized_convo_id = str(convo_id or "").strip()
    if (
        selected_endpoint_key != AUTO_ENDPOINT_KEY
        or not normalized_convo_id
        or normalized_convo_id in pins
        or not isinstance(metadata, dict)
    ):
        return pins

    routing_info = metadata.get("routing")
    if not isinstance(routing_info, dict):
        return pins

    decision = str(routing_info.get("decision") or "").strip()
    if decision and decision != AUTO_ENDPOINT_KEY:
        pins[normalized_convo_id] = decision

    return pins


def server(input, output, session):
    available_endpoints = reactive.Value({})
    endpoint_info = reactive.Value({})
    endpoint_display_mapping = reactive.Value({})
    last_runtime = reactive.Value(None)
    run_info = reactive.Value(None)
    send_button_state = reactive.Value("ready")
    info_accordion_open = reactive.Value(True)
    file_upload_key = reactive.Value(0)
    current_files = reactive.Value({"key": 0, "files": None})
    system_prompt_states = reactive.Value({})
    system_prompt_locked = reactive.Value(False)
    system_prompt_seed = reactive.Value("")
    history_refresh_trigger = reactive.Value(0)
    history_selected_convo_id = reactive.Value("")
    auto_route_pins = reactive.Value({})

    def current_active_convo_id() -> str:
        return str(input.convoID() or "").strip()

    def current_history_convo_id() -> str:
        return str(history_selected_convo_id.get() or "").strip()

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
    ) -> Dict[str, Any]:
        state = dict(get_system_prompt_state(convo_id))
        if prompt is not None:
            state["prompt"] = normalize_system_prompt(prompt)
        if started is not None:
            state["started"] = bool(started)
        if locked is not None:
            state["locked"] = bool(locked)

        states = dict(system_prompt_states.get())
        states[convo_id] = state
        system_prompt_states.set(states)
        return state

    def apply_system_prompt_view(state: Dict[str, Any]) -> None:
        system_prompt_locked.set(bool(state.get("locked")))
        system_prompt_seed.set(str(state.get("prompt") or ""))

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
        if isinstance(convo_history, list) and convo_history:
            restored_prompt = extract_first_system_prompt(convo_history)
            state = build_system_prompt_state(
                restored_prompt,
                started=True,
                locked=True,
            )
        else:
            state = build_system_prompt_state()

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
        choice_list = list(choices.keys())
        ui.update_select(
            "endpoint",
            choices=choices,
            selected=choice_list[0] if choice_list else None,
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

    async def update_history_selector() -> None:
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

    shinyswatch.theme_picker_server()

    @reactive.Effect
    async def _initialize_endpoints():
        await update_endpoints_and_data()
        await update_rag_endpoints()
        await update_search_providers()
        await update_history_selector()

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
    @reactive.event(input.endpoint)
    def _clear_info_on_endpoint_change():
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
            set_system_prompt_state(new_uuid, prompt="", started=False, locked=False)
        )
        ui.update_text("convoID", value=new_uuid, session=session)

    @render.ui
    def system_prompt_ui():
        prompt = system_prompt_seed.get()
        locked = system_prompt_locked.get()
        prompt_input = ui.input_text_area(
            "systemPrompt",
            "System Prompt" + " (locked after turn 1)" if locked else "System Prompt",
            value=prompt,
            placeholder="System prompt (optional, locked after turn 1)",
            width="100%",
            rows=4,
        )

        if not locked:
            return prompt_input

        return ui.div(
            ui.tags.fieldset(
                prompt_input,
                disabled="disabled",
                style="border: 0; margin: 0; padding: 0; min-inline-size: 0;",
            ),
        )

    @reactive.Effect
    @reactive.event(input.systemPrompt)
    def _store_system_prompt_input():
        convo_id = current_active_convo_id()
        if not convo_id or system_prompt_locked.get():
            return
        set_system_prompt_state(convo_id, prompt=input.systemPrompt() or "")

    @reactive.Effect
    @reactive.event(input.ragEndpoint)
    async def _append_rag_suffix_for_selected_endpoint():
        convo_id = current_active_convo_id()
        if not convo_id or system_prompt_locked.get():
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

    @reactive.Effect
    @reactive.event(input.uploadFile)
    def _track_uploaded_files():
        uploaded = input.uploadFile()
        if uploaded and len(uploaded) > 0:
            current_files.set({"key": file_upload_key.get(), "files": uploaded})

    @reactive.Effect
    @reactive.event(input.clearUpload)
    def _clear_upload():
        current_files.set({"key": file_upload_key.get() + 1, "files": None})
        file_upload_key.set(file_upload_key.get() + 1)

    @reactive.Effect
    @reactive.event(input.logout)
    async def _logout():
        await session.send_custom_message("logout", {})

    @reactive.Effect
    @reactive.event(input.info_accordion)
    def _track_accordion_state():
        info_accordion_open.set("info_panel" in (input.info_accordion() or []))

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
        display_mapping = endpoint_display_mapping.get()
        current_endpoint = display_mapping.get(input.endpoint())

        def fmt(value: Any) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.2f}" if value != int(value) else str(int(value))
            return str(value)

        sections = []
        if runtime is not None:
            sections.append(f"**Elapsed Time**: {runtime:.2f}s")

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
                    "**Auto-Routing Decision**<br>" + "<br>".join(routing_lines)
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

        if current_endpoint in ["Auto", "smart"] and not routing_info:
            all_models = format_all_available_models(endpoint_data)
            if all_models:
                sections.append(
                    "**Available Models (Auto-mode)**<br><br>" + "<br>".join(all_models)
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
        display_mapping = endpoint_display_mapping.get()
        current_run_info = run_info.get() or {}
        selected_endpoint_key = display_mapping.get(input.endpoint())
        convo_id = current_active_convo_id() or None
        auto_route_pins_snapshot = dict(auto_route_pins.get())
        actual_endpoint_key = resolve_auto_endpoint_key(
            selected_endpoint_key, convo_id, auto_route_pins_snapshot
        )
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
        system_prompt_to_send = first_turn_system_prompt_to_send(
            current_prompt,
            bool(prompt_state.get("started")),
        )

        search_state = None
        selected_search_provider = str(input.searchProvider() or "").strip()
        if selected_search_provider and search_query:
            try:
                planner_history = await fetch_convo_history(convo_id) if convo_id else []
                planner_context = build_search_planner_context(
                    system_prompt=current_prompt,
                    history=planner_history,
                    user_input=user_input,
                )
                search_response = await fetch_search_results(
                    query=search_query,
                    provider=selected_search_provider,
                    count=5,
                    context=planner_context,
                )
                search_state = build_search_success_state(search_response)
            except Exception as exc:
                search_state = build_search_failure_state(selected_search_provider, exc)

        search_preface = build_search_preface(search_state)
        if search_preface:
            await chat.append_message({"role": "assistant", "content": search_preface})

        extra_turn_messages = build_search_turn_messages(search_state)
        if search_state:
            run_info.set(merge_run_info({}, search_state))

        local_auto_route_pins = auto_route_pins_snapshot

        def publish_run_info(metadata: Optional[Dict[str, Any]]) -> None:
            nonlocal local_auto_route_pins

            merged_info = merge_run_info(metadata, search_state)
            run_info.set(merged_info)

            updated_pins = pin_auto_route_decision(
                local_auto_route_pins, convo_id, selected_endpoint_key, merged_info
            )
            if updated_pins != local_auto_route_pins:
                local_auto_route_pins = updated_pins
                auto_route_pins.set(updated_pins)

        response_stream = stream_chat_response(
            endpoint_key=actual_endpoint_key,
            text=user_input,
            endpoints_dict=current_endpoints,
            stream=input.stream(),
            output_json=input.outputJSON(),
            reasoning_effort=input.reasoningEffort(),
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
                    started=True,
                    locked=True,
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
