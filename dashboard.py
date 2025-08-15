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


async def fetch_available_models():
    """Fetch available models from the proxy /models endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{PROXY_BASE_URL}/models")
            response.raise_for_status()
            data = response.json()

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
        print(f"Failed to fetch models: {e}")
        # Fallback to empty dict if proxy is not available
        return {}


# Global reactive value to store endpoints
available_endpoints = reactive.Value({})

app_ui = ui.page_fluid(
    ui.tags.script(
        """
        document.addEventListener('DOMContentLoaded', () => {
            const ta = document.getElementById('userTextInput');
            const btn = document.getElementById('send');
            if (!ta || !btn) return;

            ta.addEventListener('keydown', (e) => {
                if (e.isComposing) return; // IME
                if (e.key === 'Enter' && !e.shiftKey) {
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
                ui.input_task_button("send", "Send", auto_reset=False),
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
    last_runtime = reactive.Value(None)
    run_info = reactive.Value(None)
    send_button_state = reactive.Value("ready")

    # Initialize endpoints on startup
    @reactive.Effect
    async def _initialize_endpoints():
        endpoints = await fetch_available_models()
        available_endpoints.set(endpoints)

        # Update the endpoint choices
        endpoint_choices = list(endpoints.keys()) if endpoints else []
        ui.update_select(
            "endpoint",
            choices=endpoint_choices,
            selected=endpoint_choices[0] if endpoint_choices else None,
        )

    # Refresh endpoints when button is clicked
    @reactive.Effect
    @reactive.event(input.refreshEndpoints)
    async def _refresh_endpoints():
        endpoints = await fetch_available_models()
        available_endpoints.set(endpoints)

        # Update the endpoint choices
        endpoint_choices = list(endpoints.keys()) if endpoints else []
        ui.update_select(
            "endpoint",
            choices=endpoint_choices,
            selected=endpoint_choices[0] if endpoint_choices else None,
        )

    # Enable dynamic theme switching
    shinyswatch.theme_picker_server()

    @reactive.Effect
    @reactive.event(input.outputJSON)
    def _():
        if input.outputJSON():
            ui.update_switch("stream", value=False)
            ui.update_switch("autoScroll", value=False)

    @reactive.Effect
    @reactive.event(input.stream, input.autoScroll)
    def _():
        if input.stream() or input.autoScroll():
            ui.update_switch("outputJSON", value=False)

    @reactive.Effect
    @reactive.event(input.autoScroll)
    def _():
        if input.autoScroll():
            ui.update_switch("stream", value=True)

    @reactive.Effect
    @reactive.event(input.stream)
    def _():
        if not input.stream():
            ui.update_switch("autoScroll", value=False)

    @reactive.Effect
    @reactive.event(input.generateConvoID)
    def _generate_convo_id():
        new_uuid = str(uuid.uuid4().hex[:12])
        ui.update_text("convoID", value=new_uuid, session=session)

    # Logout
    @reactive.Effect
    @reactive.event(input.logout)
    async def _():
        await session.send_custom_message("logout", {})

    @reactive.effect
    @reactive.event(input.send)
    def _on_send():
        ui.update_text_area("userTextInput", value="")

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
        now = time.time()
        send_button_state.set("busy")

        # Get the selected endpoint info
        if not endpoints_dict or endpoint_key not in endpoints_dict:
            yield f"Error: Endpoint '{endpoint_key}' not available"
            send_button_state.set("ready")
            return

        # Use the proxy URL with the endpoint name as path
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

        if stream:
            payload["stream"] = True
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as r:
                    r.raise_for_status()
                    first = True
                    first_obj = True
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
                                first_obj = True
                            else:
                                first_obj = False
                            pretty = json.dumps(obj, indent=2)
                            if not first_obj:
                                yield ",\n" + pretty
                            else:
                                yield pretty
                        else:
                            choices = obj.get("choices")
                            if isinstance(choices, list) and choices:
                                delta = choices[0].get("delta", {})
                                chunk = delta.get("content", "")
                                if chunk:
                                    yield chunk
                    # Optionally, handle the case where choices is missing or empty
                    if output_json and not first:
                        yield "]\n```"
                    if any(
                        k in obj
                        for k in (
                            "stats",
                            "usage",
                            "model_info",
                            "runtime",
                        )
                    ):
                        combined = {}
                        for key in ("stats", "usage", "model_info", "runtime"):
                            if key in obj and isinstance(obj[key], dict):
                                combined[key] = obj[key]
                        run_info.set(combined if combined else None)
                    else:
                        run_info.set(None)

        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if any(k in data for k in ("stats", "usage", "model_info", "runtime")):
                    combined = {}
                    for key in ("stats", "usage", "model_info", "runtime"):
                        if key in data and isinstance(data[key], dict):
                            combined[key] = data[key]
                    run_info.set(combined if combined else None)
                else:
                    run_info.set(None)
                if output_json:
                    pretty = json.dumps(data, indent=2)
                    yield f"```json\n{pretty}\n```"
                else:
                    # Non-stream schema uses message.content, not delta.content
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if not isinstance(content, str):
                        content = str(content)
                    yield content

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

    @reactive.effect
    def _():
        status = send_button_state.get()
        if status == "busy":
            ui.update_task_button("send", state="busy")
        else:
            ui.update_task_button("send", state="ready")

    @render.ui
    @reactive.event(run_info, last_runtime)
    def outputRunInfo():
        info = run_info.get()
        runtime = last_runtime.get()

        # Helper to format numbers
        def fmt(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

        sections = []
        sections.append(
            f"total runtime (sec): {runtime:.2f}<br>" if runtime is not None else ""
        )
        for section_name in ("usage", "stats", "model_info", "runtime"):
            section_data = info.get(section_name)
            if section_data and isinstance(section_data, dict):
                # Heading for section
                lines = [f"**{section_name.replace('_', ' ').title()}**<br>"]
                # Each key/value with <br> for clean break
                for k, v in section_data.items():
                    lines.append(f"{k.replace('_', ' ')}: {fmt(v)}<br>")
                sections.append("".join(lines))

        md = "<br>".join(sections)  # extra spacing between sections
        return ui.markdown(md)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
