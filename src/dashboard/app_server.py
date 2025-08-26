import json
import os
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator

import httpx
import shinyswatch
from dotenv import load_dotenv
from shiny import reactive, render, ui

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL")

from .utils import (
    create_endpoint_display_choices,
    fetch_available_endpoints,
    fetch_convo_history,
    find_model_by_endpoint,
)


def server(input, output, session):

    # Reactive values
    available_endpoints = reactive.Value({})
    endpoint_info = reactive.Value({})
    endpoint_display_mapping = reactive.Value({})
    last_runtime = reactive.Value(None)
    run_info = reactive.Value(None)
    send_button_state = reactive.Value("ready")
    info_accordion_open = reactive.Value(True)  # Track accordion state

    async def update_endpoints_and_data():
        """Helper function to update both endpoints and models data."""
        # Fetch data only once and get both endpoints and raw data
        endpoints, data = await fetch_available_endpoints()
        available_endpoints.set(endpoints)
        endpoint_info.set(data)

        # Create endpoint choices and mapping
        choices, mapping = create_endpoint_display_choices(endpoints)
        endpoint_display_mapping.set(mapping)

        choice_list = list(choices.keys())
        ui.update_select(
            "endpoint",
            choices=choices,
            selected=choice_list[0] if choice_list else None,
        )

    # Initialize endpoints on startup
    @reactive.Effect
    async def _initialize_endpoints():
        await update_endpoints_and_data()

    # Refresh endpoints when button is clicked
    @reactive.Effect
    @reactive.event(input.refreshEndpoints)
    async def _refresh_endpoints():
        await update_endpoints_and_data()

    # Clear info and runtime when endpoint changes
    @reactive.Effect
    @reactive.event(input.endpoint)
    def _clear_info_on_endpoint_change():
        run_info.set(None)
        last_runtime.set(None)

    # Enable dynamic theme switching
    shinyswatch.theme_picker_server()

    # Handle UI state changes for switches
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

    # Utility actions
    @reactive.Effect
    @reactive.event(input.generateConvoID)
    def _generate_convo_id():
        new_uuid = str(uuid.uuid4().hex[:12])
        ui.update_text("convoID", value=new_uuid, session=session)

    @reactive.Effect
    @reactive.event(input.logout)
    async def _logout():
        await session.send_custom_message("logout", {})

    @reactive.effect
    @reactive.event(input.send)
    def _on_send():
        ui.update_text_area("userTextInput", value="")

    # Update send button state
    @reactive.effect
    def _update_send_button():
        status = send_button_state.get()
        ui.update_task_button("send", state=status)

    # Track accordion state changes
    @reactive.Effect
    @reactive.event(input.info_accordion)
    def _track_accordion_state():
        # Update our reactive value when user manually toggles accordion
        is_open = "info_panel" in (input.info_accordion() or [])
        info_accordion_open.set(is_open)

    async def llm_stream_generator(
        endpoint_key: str,
        text: str,
        endpoints_dict: dict,
        stream: bool = True,
        output_json: bool = False,
        reasoning_effort: str = "medium",
        output_reasoning: bool = False,
        convo_id: str = None,
        current_routing_info: dict = None,
    ) -> AsyncGenerator[str, None]:
        text = (text or "Hello! What model are you?").strip()
        if not text:
            yield ""
            return

        now = time.time()
        send_button_state.set("busy")

        try:
            # Handle Auto routing specially - route to smart endpoint
            if endpoint_key == "Auto":
                url = f"{PROXY_BASE_URL}/smart"
            else:
                if not endpoints_dict or endpoint_key not in endpoints_dict:
                    yield f"Error: Endpoint '{endpoint_key}' not available"
                    return
                url = f"{PROXY_BASE_URL}/{endpoint_key}"

            # Prepare request

            payload = {"messages": [{"role": "user", "content": text}]}
            headers = {
                "CF-Access-Client-Id": API_KEY_ID,
                "CF-Access-Client-Secret": API_KEY_SECRET,
                "Content-Type": "application/json",
                "X-Reasoning-Effort": reasoning_effort,
            }
            if convo_id:
                headers["X-Convo-ID"] = convo_id
            timeout = httpx.Timeout(connect=5, read=None, write=5, pool=10)

            # Use passed routing info instead of reading reactive source
            routing_info_holder = {"routing": current_routing_info or {}}

            def extract_metadata(
                obj, response_headers=None, preserve_routing_from=None
            ):
                """Extract metadata from response object and headers."""
                metadata_keys = ("stats", "usage", "model_info", "runtime")
                combined = {}

                # Preserve existing routing info if provided
                if preserve_routing_from and "routing" in preserve_routing_from:
                    combined["routing"] = preserve_routing_from["routing"]

                # Extract standard metadata
                for key in metadata_keys:
                    if key in obj and isinstance(obj[key], dict):
                        combined[key] = obj[key]

                # Extract routing information from headers (for smart routing)
                # Check if this is an auto-routing request
                is_auto_routing = (
                    endpoint_key == "smart"
                    or endpoint_key == "Auto"
                    or (
                        response_headers
                        and any(
                            h.lower().startswith("x-route-")
                            for h in response_headers.keys()
                        )
                    )
                )

                if response_headers and is_auto_routing:
                    routing_info = {}
                    for header_name, header_value in response_headers.items():
                        if header_name.lower().startswith("x-route-"):
                            key = header_name.lower().replace("x-route-", "")
                            routing_info[key] = header_value

                    if routing_info:
                        combined["routing"] = routing_info
                        # Store routing info for later use
                        routing_info_holder["routing"] = routing_info

                run_info.set(combined if combined else None)

            async with httpx.AsyncClient(timeout=timeout) as client:
                if stream:
                    payload["stream"] = True
                    payload["stream_options"] = {"include_usage": True}
                    async with client.stream(
                        "POST", url, headers=headers, json=payload
                    ) as r:
                        r.raise_for_status()

                        # Extract routing info from headers once
                        extract_metadata({}, dict(r.headers))

                        reasoning_chunk_found = False

                        async for line in r.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break

                            obj = json.loads(data)

                            choices = obj.get("choices", [])
                            if choices:
                                if output_reasoning:
                                    reasoning_chunk = (
                                        choices[0].get("delta", {}).get("reasoning", "")
                                    )
                                    if reasoning_chunk:
                                        if reasoning_chunk_found is False:
                                            reasoning_chunk_found = True
                                            yield "<em>"
                                        yield reasoning_chunk

                                content_chunk = (
                                    choices[0].get("delta", {}).get("content", "")
                                )
                                if content_chunk:
                                    if reasoning_chunk_found:
                                        yield "</em>\n\n---\n\n"
                                        reasoning_chunk_found = False

                                    # Legacy: <think> tags for reasoning/CoT
                                    content_chunk = content_chunk.replace(
                                        "<think>", "<em>"
                                    )
                                    content_chunk = content_chunk.replace(
                                        "</think>", "</em>\n\n---\n\n"
                                    )
                                    yield content_chunk

                            # Use the stored routing info
                            extract_metadata(
                                obj, preserve_routing_from=routing_info_holder
                            )

                else:
                    # Non-streaming request
                    resp = await client.post(
                        url, headers=headers, json=payload, timeout=timeout
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    extract_metadata(data, dict(resp.headers))

                    if output_json:
                        yield f"```json\n{json.dumps(data, indent=2)}\n```"
                    else:
                        if output_reasoning:
                            # GPT-style
                            reasoning = (
                                data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("reasoning", "")
                            )
                            if reasoning:
                                yield f"<em>{str(reasoning)}</em>\n\n---\n\n"
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        # Legacy: <think> tags for reasoning/CoT
                        content = content.replace("<think>", "<em>")
                        content = content.replace("</think>", "</em>\n\n---\n\n")
                        yield str(content)

        except httpx.HTTPStatusError as e:
            # For streaming responses, we can't access response.text directly
            # So we'll use the status code and reason phrase instead
            reason = getattr(e.response, "reason_phrase", "Unknown Error")
            yield f"HTTP Error {e.response.status_code}: {reason}"
        except httpx.RequestError as e:
            yield f"Request Error: {str(e)}"
        except Exception as e:
            # Log the full error for debugging
            print(
                f"Unexpected error in llm_stream_generator: {type(e).__name__}: {str(e)}"
            )
            import traceback

            traceback.print_exc()
            yield f"Error: {str(e)}"
        finally:
            last_runtime.set(time.time() - now)
            send_button_state.set("ready")

    md = ui.MarkdownStream("streamOutput")

    @reactive.effect
    @reactive.event(input.send)
    async def _():
        # Read reactive values outside the extended task
        current_endpoints = available_endpoints.get()
        display_mapping = endpoint_display_mapping.get()
        current_run_info = run_info.get() or {}

        # Get the actual endpoint key from the display name
        actual_endpoint_key = display_mapping.get(input.endpoint())
        await md.stream(
            llm_stream_generator(
                endpoint_key=actual_endpoint_key,
                text=input.userTextInput(),
                endpoints_dict=current_endpoints,
                stream=input.stream(),
                output_json=input.outputJSON(),
                # reasoning_effort=input.reasoningEffort(),
                output_reasoning=input.outputReasoning(),
                convo_id=input.convoID() if input.convoID() else None,
                current_routing_info=current_run_info.get("routing", {}),
            )
        )

    def format_hardware_info(model, routing_info=None):
        """Format hardware information for display."""
        hardware_info = []

        # Use routing hardware info if available (from smart routing)
        if routing_info:
            for key in ["gpu", "vram", "soc", "cpu", "ram"]:
                value = routing_info.get(key)
                if value:
                    hardware_info.append(f"{key.upper()}: {value}")

        # If no routing info or routing info was incomplete, fallback to model hardware info
        if not hardware_info and model:
            for key in ["gpu", "vram", "soc", "cpu", "ram"]:
                value = model.get(key)
                if value:
                    hardware_info.append(f"{key.upper()}: {value}")

        return hardware_info

    def format_model_details(model):
        """Format model details for display."""
        model_details = []

        # Add the full model name first
        model_name = model.get("id")
        if model_name:
            model_details.append(f"Model: {model_name}")

        # Add other model properties
        for key in ["arch", "quantization", "compatibility_type", "state"]:
            value = model.get(key)
            if value:
                key_name = key.replace("_", " ").title()
                model_details.append(f"{key_name}: {value}")

        # Add context information
        max_ctx = model.get("max_context_length")
        loaded_ctx = model.get("loaded_context_length")
        if max_ctx:
            model_details.append(f"Max Context: {max_ctx:,}")
        if loaded_ctx:
            model_details.append(f"Loaded Context: {loaded_ctx:,}")

        return model_details

    def format_all_available_models(endpoint_data):
        """Format all available models for display when Auto is selected."""
        models_list = []

        if not endpoint_data or "data" not in endpoint_data:
            return []

        # Get all models (excluding auto-router)
        for model in endpoint_data["data"]:
            model_details = format_model_details(model)
            if model_details:
                models_list.extend(model_details)
                models_list.append("")  # Add spacing between models

        return models_list

    def format_response_info(info, fmt_func):
        """Format response information (usage, stats, runtime) for display."""
        sections = []
        for section_name in ("usage", "stats", "runtime"):
            section_data = info.get(section_name)
            if section_data and isinstance(section_data, dict):
                title = section_name.replace("_", " ").title()
                lines = [f"**{title}**"]
                for k, v in section_data.items():
                    key_name = k.replace("_", " ").title()
                    lines.append(f"{key_name}: {fmt_func(v)}")
                sections.append("<br>".join(lines))
        return sections

    @render.ui
    def outputRunInfo():
        # Create static accordion structure - this only renders once
        return ui.accordion(
            ui.accordion_panel(
                "Runtime & System Information",
                ui.output_ui("info_content"),  # Dynamic content goes here
                value="info_panel",
            ),
            id="info_accordion",
            open=[],  # Start collapsed, user can expand as needed
        )

    @render.ui
    @reactive.event(run_info, last_runtime, endpoint_info, input.endpoint)
    def info_content():
        info = run_info.get()
        runtime = last_runtime.get()
        endpoint_data = endpoint_info.get()
        display_mapping = endpoint_display_mapping.get()

        # Get the actual endpoint key from the display name
        current_endpoint = display_mapping.get(input.endpoint())

        def fmt(v):
            """Helper to format numbers and values."""
            if isinstance(v, (int, float)):
                return f"{v:.2f}" if v != int(v) else str(int(v))
            return str(v)

        sections = []

        # Elapsed time section
        if runtime is not None:
            sections.append(f"**Elapsed Time**: {runtime:.2f}s")

        # Routing information (for smart routing)
        routing_info = None
        if info and "routing" in info:
            routing = info["routing"]
            routing_info = routing  # Store for hardware info use

            # Build routing display lines if we have meaningful data
            routing_lines = []
            routing_fields = {
                "decision": lambda v: f"Selected Node: {v}",
                # "confidence": lambda v: f"Confidence: {float(v):.1%}",
                # "reason": lambda v: f"Reason: {v}",
                "strategy": lambda v: f"Workload: {v}",
            }

            for field, formatter in routing_fields.items():
                if field in routing:
                    routing_lines.append(formatter(routing[field]))

            # Only show routing section if we have actual routing data
            if routing_lines:
                sections.append(
                    "**Auto-Routing Decision**<br>" + "<br>".join(routing_lines)
                )  # Request info (usage, stats, model_info, runtime from response)
        if info:
            sections.extend(format_response_info(info, fmt))

        # Current endpoint info from /models endpoint
        # For Auto/smart routing: show all pre-run, use the routed endpoint post-run
        # For manual selection: use the selected endpoint
        endpoint_for_model_info = current_endpoint

        # If we have routing info, prioritize the routed endpoint
        if routing_info and "decision" in routing_info:
            endpoint_for_model_info = routing_info["decision"]

        model = find_model_by_endpoint(endpoint_data, endpoint_for_model_info)

        # Fallback: if no model found and we have routing info, try the routed endpoint directly
        if not model and routing_info and "decision" in routing_info:
            model = find_model_by_endpoint(endpoint_data, routing_info["decision"])

        # Special handling for Auto/smart endpoints when no routing decision has been made
        if current_endpoint in ["Auto", "smart"] and not routing_info:
            # For Auto/smart without routing decision, always show all available models
            all_models = format_all_available_models(endpoint_data)
            if all_models:
                sections.append(
                    "**Available Models (Auto-mode)**<br><br>" + "<br>".join(all_models)
                )
        elif model:
            # Hardware info - use routing info if available, otherwise use model info
            hardware_info = format_hardware_info(model, routing_info)
            if hardware_info:
                sections.append("**Hardware**<br>" + "<br>".join(hardware_info))

            # Model details
            model_details = format_model_details(model)
            if model_details:
                sections.append("**Model Info**<br>" + "<br>".join(model_details))

        # Fallback: when no specific model is found and no routing decision, show all available models
        elif not model and not routing_info:
            all_models = format_all_available_models(endpoint_data)
            if all_models:
                sections.append(
                    "**Available Models**<br><br>" + "<br>".join(all_models)
                )

        if not sections:
            return ui.div()

        # Join sections with double line breaks for spacing
        markdown_content = "<br><br>".join(sections)
        return ui.markdown(markdown_content)

    @render.ui
    def responseBox():
        return ui.card(
            ui.output_markdown_stream("streamOutput", auto_scroll=input.autoScroll()),
            height="88vh",
        )

    # Add a reactive value to track history refresh needs
    history_refresh_trigger = reactive.Value(0)

    # Trigger history refresh when messages are sent
    @reactive.Effect
    @reactive.event(input.send)
    def _trigger_history_refresh():
        # Increment the trigger to force history refresh after sending messages
        history_refresh_trigger.set(history_refresh_trigger.get() + 1)

    # Trigger history refresh when refresh button is clicked
    @reactive.Effect
    @reactive.event(input.refreshHistory)
    def _manual_history_refresh():
        # Increment the trigger to force history refresh
        history_refresh_trigger.set(history_refresh_trigger.get() + 1)

    @render.ui
    @reactive.event(input.convoID, history_refresh_trigger)
    async def historyBox():
        # Only fetch if conversation ID is provided
        if not input.convoID():
            return ui.card(
                ui.markdown(
                    "**No conversation ID provided**\n\nEnter a conversation ID to view history."
                ),
            )

        convo_history = await fetch_convo_history(input.convoID())

        # Handle empty or error responses
        if not convo_history:
            return ui.card(
                ui.markdown(
                    f"**No history found for conversation: {input.convoID()}**\n\nThis conversation may not exist or has no messages."
                ),
            )

        # Add timestamp to show when history was last refreshed
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return ui.card(
            ui.markdown(f"**Conversation History** *(refreshed at {timestamp})*"),
            ui.markdown(f"```json\n{json.dumps(convo_history, indent=2)}\n```"),
        )
