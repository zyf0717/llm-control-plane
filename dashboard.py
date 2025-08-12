import json
import os

import httpx
import markdown
from dotenv import load_dotenv
from shiny import App, reactive, render, ui

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")

ENDPOINTS = {
    "default": "https://llm.paperclips.dev",
    "gpt-oss-20b": "https://llm.paperclips.dev/gpt-oss-20b-api",
    "qwen3-4b": "https://llm.paperclips.dev/qwen3-4b-api",
}

app_ui = ui.page_fluid(
    ui.layout_columns(
        ui.card(
            ui.input_select("endpoint", "Endpoint", choices=list(ENDPOINTS.keys())),
            ui.input_text_area("text", "Input text", rows=8, placeholder="Type here…"),
            ui.input_task_button("send", "Send"),
            ui.input_checkbox("pretty", "Pretty-print JSON", False),
        ),
        ui.card(ui.output_ui("responseBox")),
        col_widths=[2, 10],
    ),
)


def server(input, output, session):

    @ui.bind_task_button(button_id="send")
    @reactive.extended_task
    async def _call_llm(endpoint_key: str, text: str, pretty: bool) -> str:
        text = (text or "").strip()
        if not text:
            return "Please enter some text."

        url = ENDPOINTS[endpoint_key]
        payload = {"messages": [{"role": "user", "content": text}]}
        headers = {
            "CF-Access-Client-Id": API_KEY_ID,
            "CF-Access-Client-Secret": API_KEY_SECRET,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"Request failed: {e!r}"

        if pretty:
            return json.dumps(data, indent=2)

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "<no content>")
        )
        return content

    @reactive.effect
    @reactive.event(input.send)
    def _run_task():
        _call_llm(input.endpoint(), input.text(), input.pretty())

    @output
    @render.ui
    def responseBox():
        status = _call_llm.status()
        if status == "running":
            return ui.markdown("_Processing…_")
        if status == "error":
            # Optional: show the exception detail
            err = _call_llm.error()
            return ui.markdown(f"**Request failed.**\n\n```\n{err}\n```")

        result = _call_llm.result()
        if result is None:
            return ui.markdown("_Ready._")

        # If pretty was selected, result is JSON text; render as code block.
        if input.pretty():
            return ui.markdown(f"```json\n{result}\n```")

        # Otherwise assume Markdown/plain text from the LLM.
        return ui.markdown(result)


app = App(app_ui, server)
