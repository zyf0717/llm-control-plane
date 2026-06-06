import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import shinyswatch
from dotenv import load_dotenv
from shiny import reactive, render, ui

load_dotenv()

from .chat_client import stream_chat_response
from .formatters import (
    format_all_available_models,
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
    fetch_convo_history,
    find_model_by_endpoint,
)


def server(input, output, session):
    available_endpoints = reactive.Value({})
    endpoint_info = reactive.Value({})
    endpoint_display_mapping = reactive.Value({})
    last_runtime = reactive.Value(None)
    run_info = reactive.Value(None)
    send_button_state = reactive.Value("ready")
    active_stop_event = reactive.Value(None)
    active_request_streaming = reactive.Value(False)
    info_accordion_open = reactive.Value(True)
    file_upload_key = reactive.Value(0)
    current_files = reactive.Value({"key": 0, "files": None})
    system_prompt_states = reactive.Value({})
    system_prompt_locked = reactive.Value(False)
    system_prompt_seed = reactive.Value("")
    history_refresh_trigger = reactive.Value(0)
    history_selected_convo_id = reactive.Value("")

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
    def stop_control_ui():
        is_enabled = (
            send_button_state.get() == "busy" and active_request_streaming.get()
        )
        button = ui.input_action_button(
            "stopResponse",
            "Stop",
            class_="btn btn-outline-danger w-100",
        )
        if is_enabled:
            return button

        return ui.tags.fieldset(
            button,
            disabled="disabled",
            style="border: 0; margin: 0; padding: 0; min-inline-size: 0;",
        )

    @reactive.Effect
    @reactive.event(input.stopResponse)
    def _stop_active_stream():
        stop_event = active_stop_event.get()
        if stop_event is not None and not stop_event.is_set():
            stop_event.set()

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

        if info:
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
        if send_button_state.get() == "busy":
            return

        current_endpoints = available_endpoints.get()
        display_mapping = endpoint_display_mapping.get()
        current_run_info = run_info.get() or {}
        actual_endpoint_key = display_mapping.get(input.endpoint())
        convo_id = current_active_convo_id() or None
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

        stop_event = asyncio.Event()
        active_stop_event.set(stop_event)
        active_request_streaming.set(bool(input.stream()))

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
            rag_endpoint=input.ragEndpoint() if input.ragEndpoint() else None,
            stop_event=stop_event,
            on_metadata=run_info.set,
            on_send_button_state=send_button_state.set,
            on_runtime=last_runtime.set,
        )
        try:
            await chat.append_message_stream(response_stream)
        finally:
            if active_stop_event.get() is stop_event:
                active_stop_event.set(None)
                active_request_streaming.set(False)

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
