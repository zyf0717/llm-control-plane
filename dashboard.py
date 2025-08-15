import json
import os
import time
import uuid
from typing import AsyncGenerator

import httpx
import shinyswatch
from dotenv import load_dotenv
from shiny import App, reactive, render, ui

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")

# URL for the proxy service
PROXY_BASE_URL = "http://localhost:12340"


async def fetch_models_data():
    """Fetch raw models data from the proxy /models endpoint for telemetry."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{PROXY_BASE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Failed to fetch models data: {e}")
        return {}


async def fetch_available_endpoints():
    """Fetch available endpoints grouped by endpoint name."""
    try:
        data = await fetch_models_data()

        # Group models by endpoint
        endpoints = {}
        for model in data.get("data", []):
            endpoint_name = model.get("endpoint")
            endpoint_url = model.get("endpoint_url")
            model_id = model.get("id")

            if endpoint_name and endpoint_url and model_id:
                if endpoint_name not in endpoints:
                    endpoints[endpoint_name] = {
                        "endpoint_url": endpoint_url,
                        "models": [],
                    }
                endpoints[endpoint_name]["models"].append(model)

        return endpoints
    except Exception as e:
        print(f"Failed to fetch endpoints: {e}")
        return {}


app_ui = ui.page_fluid(
    ui.tags.script(
        """
        document.addEventListener('DOMContentLoaded', () => {
            const ta = document.getElementById('userTextInput');
            const btn = document.getElementById('send');
            if (!ta || !btn) return;

            ta.addEventListener('keydown', (e) => {
                if (e.isComposing) return; // IME
                if (e.key === 'Enter' && e.shiftKey) {
                    e.preventDefault();
                    btn.click();
                }
            });
        });
        """
    ),
    ui.tags.style(
        """
        .sidebar {
        height: 95dvh;           /* use dynamic vh; falls back to 100vh if unsupported */
        box-sizing: border-box;   /* include padding/border in the 100dvh */
        overflow-y: auto;
        position: sticky;
        top: 0;
        background: inherit;
        display: flow-root;       /* creates a new block formatting context -> no margin collapse */
        }
        """
    ),
    ui.page_sidebar(
        ui.sidebar(
            ui.tags.script(
                """
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://llm-dashboard.paperclips.dev/cdn-cgi/access/logout";
                });
                """
            ),
            ui.input_select("endpoint", "Endpoint", choices=[]),
            ui.input_action_button("refreshEndpoints", "Refresh Endpoints"),
            ui.input_switch("stream", "Streaming", True),
            ui.input_switch("autoScroll", "Auto-scroll", True),
            ui.input_switch("outputJSON", "JSON", False),
            shinyswatch.theme_picker_ui(),
            ui.hr(),
            ui.input_action_button("logout", "Logout"),
        ),
        ui.layout_columns(
            ui.card(
                ui.layout_columns(
                    ui.input_text(
                        "convoID",
                        "",
                        placeholder="Conversation ID",
                        width="100%",
                    ),
                    ui.input_action_button("generateConvoID", "New"),
                    col_widths=[7, 5],
                ),
                ui.input_text_area(
                    "userTextInput",
                    "",
                    rows=6,
                    placeholder="Ask anything",
                    width="100%",
                ),
                ui.input_task_button("send", "Send (Shift + Enter)", auto_reset=False),
                ui.output_ui("outputRunInfo"),
            ),
            ui.output_ui("responseBox"),
            col_widths=[3, 9],
            fillable=False,
        ),
    ),
    theme=shinyswatch.theme.flatly,
)


def server(input, output, session):

    # Reactive values
    available_endpoints = reactive.Value({})
    endpoint_info = reactive.Value({})
    last_runtime = reactive.Value(None)
    run_info = reactive.Value(None)
    send_button_state = reactive.Value("ready")

    async def update_endpoints_and_data():
        """Helper function to update both endpoints and models data."""
        endpoints = await fetch_available_endpoints()
        data = await fetch_models_data()
        available_endpoints.set(endpoints)
        endpoint_info.set(data)

        # Update the endpoint choices
        endpoint_choices = list(endpoints.keys()) if endpoints else []
        ui.update_select(
            "endpoint",
            choices=endpoint_choices,
            selected=endpoint_choices[0] if endpoint_choices else None,
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

    async def llm_stream_generator(
        endpoint_key: str,
        text: str,
        endpoints_dict: dict,
        stream: bool = True,
        output_json: bool = False,
        convo_id: str = None,
    ) -> AsyncGenerator[str, None]:
        text = (text or "Hello! What model are you?").strip()
        if not text:
            yield ""
            return

        now = time.time()
        send_button_state.set("busy")

        try:
            # Validate endpoint
            if not endpoints_dict or endpoint_key not in endpoints_dict:
                yield f"Error: Endpoint '{endpoint_key}' not available"
                return

            # Prepare request
            url = f"{PROXY_BASE_URL}/{endpoint_key}"
            payload = {"messages": [{"role": "user", "content": text}]}
            headers = {
                "CF-Access-Client-Id": API_KEY_ID,
                "CF-Access-Client-Secret": API_KEY_SECRET,
                "Content-Type": "application/json",
            }
            if convo_id:
                headers["X-Convo-ID"] = convo_id
            timeout = httpx.Timeout(connect=5, read=None, write=5, pool=10)

            def extract_metadata(obj):
                """Extract metadata from response object."""
                metadata_keys = ("stats", "usage", "model_info", "runtime")
                if any(k in obj for k in metadata_keys):
                    combined = {}
                    for key in metadata_keys:
                        if key in obj and isinstance(obj[key], dict):
                            combined[key] = obj[key]
                    run_info.set(combined if combined else None)

            async with httpx.AsyncClient(timeout=timeout) as client:
                if stream:
                    payload["stream"] = True
                    async with client.stream(
                        "POST", url, headers=headers, json=payload
                    ) as r:
                        r.raise_for_status()
                        first = True

                        async for line in r.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break

                            obj = json.loads(data)

                            if output_json:
                                if first:
                                    yield "```json\n["
                                    first = False
                                yield ("" if first else ",\n") + json.dumps(
                                    obj, indent=2
                                )
                            else:
                                choices = obj.get("choices", [])
                                if choices:
                                    chunk = (
                                        choices[0].get("delta", {}).get("content", "")
                                    )
                                    if chunk:
                                        chunk = chunk.replace(
                                            "<think>", "*Reasoning*:\n"
                                        )
                                        chunk = chunk.replace("</think>", "\n\n---\n\n")
                                        yield chunk

                            extract_metadata(obj)

                        if output_json and not first:
                            yield "]\n```"
                else:
                    # Non-streaming request
                    resp = await client.post(
                        url, headers=headers, json=payload, timeout=timeout
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    extract_metadata(data)

                    if output_json:
                        yield f"```json\n{json.dumps(data, indent=2)}\n```"
                    else:
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        content = content.replace("<think>", "*Reasoning*:\n")
                        content = content.replace("</think>", "\n\n---\n\n")
                        yield str(content)

        except Exception as e:
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

        await md.stream(
            llm_stream_generator(
                endpoint_key=input.endpoint(),
                text=input.userTextInput(),
                endpoints_dict=current_endpoints,
                stream=input.stream(),
                output_json=input.outputJSON(),
                convo_id=input.convoID() if input.convoID() else None,
            )
        )

    @render.ui
    @reactive.event(run_info, last_runtime, endpoint_info, input.endpoint)
    def outputRunInfo():
        info = run_info.get()
        runtime = last_runtime.get()
        endpoint_data = endpoint_info.get()
        current_endpoint = input.endpoint()

        def fmt(v):
            """Helper to format numbers and values."""
            if isinstance(v, (int, float)):
                return f"{v:.2f}" if v != int(v) else str(int(v))
            return str(v)

        sections = []

        # Runtime section
        if runtime is not None:
            sections.append(f"**Runtime**: {runtime:.2f}s")

        # Request info (usage, stats, model_info, runtime from response)
        if info:
            for section_name in ("usage", "stats", "runtime"):
                section_data = info.get(section_name)
                if section_data and isinstance(section_data, dict):
                    title = section_name.replace("_", " ").title()
                    lines = [f"**{title}**"]
                    for k, v in section_data.items():
                        key_name = k.replace("_", " ").title()
                        lines.append(f"{key_name}: {fmt(v)}")
                    sections.append("<br>".join(lines))

        # Current endpoint info from /models endpoint
        if endpoint_data and current_endpoint:
            endpoint_models = []
            for model in endpoint_data.get("data", []):
                if model.get("endpoint") == current_endpoint:
                    endpoint_models.append(model)

            if endpoint_models:
                # Show endpoint info for the current endpoint
                model = endpoint_models[0]  # Use first model for endpoint info

                # Hardware info
                hardware_info = []
                for key in [
                    "gpu",
                    "vram",
                    "soc",
                    "cpu",
                    "ram",
                ]:
                    value = model.get(key)
                    if value:
                        hardware_info.append(f"{key.upper()}: {value}")

                if hardware_info:
                    sections.append("**Hardware**<br>" + "<br>".join(hardware_info))

                # Model details
                model_details = []
                for key in ["arch", "quantization", "compatibility_type", "state"]:
                    value = model.get(key)
                    if value:
                        key_name = key.replace("_", " ").title()
                        model_details.append(f"{key_name}: {value}")

                # Context info
                max_ctx = model.get("max_context_length")
                loaded_ctx = model.get("loaded_context_length")
                if max_ctx:
                    model_details.append(f"Max Context: {max_ctx:,}")
                if loaded_ctx:
                    model_details.append(f"Loaded Context: {loaded_ctx:,}")

                if model_details:
                    sections.append("**Model Info**<br>" + "<br>".join(model_details))

                # # Capabilities
                # capabilities = model.get("capabilities", [])
                # if capabilities:
                #     sections.append(f"**Capabilities**: {', '.join(capabilities)}")

        if not sections:
            return ui.div()

        # Join sections with double line breaks for spacing
        markdown_content = "<br><br>".join(sections)
        return ui.markdown(markdown_content)

    @render.ui
    @reactive.event(input.send)
    def responseBox():
        return ui.card(
            ui.div(
                ui.output_markdown_stream(
                    "streamOutput", auto_scroll=input.autoScroll()
                ),
            ),
            style="height:93dvh; overflow:auto;",
        )


app = App(app_ui, server)
