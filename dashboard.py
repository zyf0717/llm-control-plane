import json
import os
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from shiny import App, reactive, ui

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
            ui.input_checkbox("outputJSON", "JSON", False),
        ),
        ui.layout_columns(
            ui.card(
                ui.input_text_area(
                    "text", "Input text", rows=8, placeholder="Type here…"
                ),
                ui.input_task_button("send", "Send"),
            ),
            ui.card(ui.output_markdown_stream("streamOutput", auto_scroll=False)),
            col_widths=[3, 9],
        ),
    ),
)


def server(input, output, session):

    async def llm_stream_generator(
        endpoint_key: str,
        text: str = "test",
        stream: bool = True,
        output_json: bool = False,
    ) -> AsyncGenerator[str, None]:
        text = (text or "").strip()
        if not text:
            yield "Please enter some text."

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
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break

                        # Parse vendor schema (OpenAI: choices[0].delta.content)
                        obj = json.loads(data)
                        if output_json:
                            yield json.dumps(obj, indent=2)
                            yield f"```json\n{pretty}\n```"
                        else:
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            chunk = delta.get("content")
                            if chunk:
                                yield chunk

        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

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

    md = ui.MarkdownStream("streamOutput")

    @reactive.effect
    @reactive.event(input.send)
    async def _():
        md.clear()
        await md.stream(
            llm_stream_generator(
                endpoint_key=input.endpoint(),
                text=input.text(),
                stream=input.stream(),
                output_json=input.outputJSON(),
            )
        )


app = App(app_ui, server)
