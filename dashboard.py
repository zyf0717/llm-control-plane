import json
import os
import time
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from shiny import App, reactive, render, ui

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")

ENDPOINTS = {
    "default": "https://llm.paperclips.dev/gpt-oss-20b-api",
    "gpt-oss-20b": "https://llm.paperclips.dev/gpt-oss-20b-api",
    "qwen3-4b": "https://llm.paperclips.dev/qwen3-4b-api",
}

app_ui = ui.page_fluid(
    ui.page_sidebar(
        ui.sidebar(
            ui.input_select("endpoint", "Endpoint", choices=list(ENDPOINTS.keys())),
            ui.input_checkbox("stream", "Streaming", True),
            ui.input_checkbox("autoScroll", "Auto-scroll", True),
            ui.input_checkbox("outputJSON", "JSON", False),
        ),
        ui.layout_columns(
            ui.card(
                ui.input_text_area(
                    "text", "Input text", rows=8, placeholder="Type here…"
                ),
                ui.input_task_button("send", "Send", auto_reset=False),
                ui.output_ui("outputStats"),
            ),
            ui.output_ui("responseBox"),
            col_widths=[3, 9],
        ),
    ),
)


def server(input, output, session):
    last_runtime = reactive.Value(None)
    run_info = reactive.Value(None)
    send_button_state = reactive.Value("ready")

    async def llm_stream_generator(
        endpoint_key: str,
        text: str,
        stream: bool = True,
        output_json: bool = False,
    ) -> AsyncGenerator[str, None]:
        text = (text or "Hello! What model are you?").strip()
        if not text:
            yield ""
        now = time.time()
        send_button_state.set("busy")

        url = ENDPOINTS.get(endpoint_key, ENDPOINTS["default"])
        payload = {"messages": [{"role": "user", "content": text}]}
        headers = {
            "CF-Access-Client-Id": API_KEY_ID,
            "CF-Access-Client-Secret": API_KEY_SECRET,
            "Content-Type": "application/json",
        }
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
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            chunk = delta.get("content")
                            if chunk:
                                yield chunk
                    if output_json and not first:
                        yield "]\n```"
                    if any(
                        k in obj for k in ("stats", "usage", "model_info", "runtime")
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
        await md.stream(
            llm_stream_generator(
                endpoint_key=input.endpoint(),
                text=input.text(),
                stream=input.stream(),
                output_json=input.outputJSON(),
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
    def outputStats():
        info = run_info.get()
        if not info:
            rt = last_runtime.get()
            return ui.markdown(
                f"total runtime (sec): {rt:.2f}" if rt is not None else ""
            )

        # Helper to format numbers
        def fmt(v):
            return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

        sections = []
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
            ui.output_markdown_stream("streamOutput", auto_scroll=input.autoScroll()),
            height="92vh",
        )


app = App(app_ui, server)
