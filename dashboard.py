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
    "gpt-oss-20b": "https://llm.paperclips.dev/gpt-oss-20b",
    "qwen3-4b": "https://llm.paperclips.dev/qwen3-4b",
}

app_ui = ui.page_fluid(
    ui.layout_columns(
        ui.card(
            ui.input_select("endpoint", "Endpoint", choices=list(ENDPOINTS.keys())),
            ui.input_text_area("text", "Input text", rows=8, placeholder="Type here…"),
            ui.input_task_button("send", "Send"),
            ui.input_checkbox("pretty", "Pretty-print JSON", False),
        ),
        ui.card(ui.output_ui("response_box")),
        col_widths=[2, 10],
    ),
)


def server(input, output, session):

    @ui.bind_task_button(button_id="send")
    @reactive.extended_task
    async def call_llm(endpoint_key: str, text: str, pretty: bool) -> str:
        text = (text or "").strip()
        if not text:
            return "<em>Please enter some text.</em>"

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
            return f"<pre>Request failed: {e!r}</pre>"

        if pretty:
            return f"<pre>{json.dumps(data, indent=2)}</pre>"

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "<no content>")
        )
        return markdown.markdown(
            content,
            extensions=[
                "fenced_code",
                "codehilite",
                "tables",
                "toc",
                "abbr",
                "attr_list",
                "def_list",
                "footnotes",
                "md_in_html",
                "meta",
                "nl2br",
                "sane_lists",
                "smarty",
                "wikilinks",
            ],
        )

    @reactive.effect
    @reactive.event(input.send)
    def _run_task():
        call_llm(input.endpoint(), input.text(), input.pretty())

    @render.ui
    def response_box():
        status = call_llm.status()
        if status == "running":
            return ui.HTML("<em>Processing…</em>")
        if status == "error":
            return ui.HTML("<em>Request failed.</em>")
        return ui.HTML(call_llm.result() or "<em>Ready.</em>")


app = App(app_ui, server)
