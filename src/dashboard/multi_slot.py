"""Independent session-local chats for exercising one endpoint concurrently."""

import asyncio
import math
import time
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from shiny import reactive, render, ui
from shinychat import Chat, chat_ui

from .chat_controls import (
    REASONING_LEVELS, endpoint_models_text, endpoint_selector_ui, resolve_endpoint_selection,
)
from .chat_client import stream_chat_response
from .response_metrics import response_metadata, response_metrics_ui

MAX_PANES = 8


def parse_generation_settings(temperature: str, max_tokens: str) -> dict[str, float | int]:
    """Blank controls preserve upstream defaults; zero temperature is explicit."""
    settings = {}
    if temperature.strip():
        try:
            value = float(temperature)
        except ValueError:
            raise ValueError("Temperature must be a finite, nonnegative number.") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError("Temperature must be a finite, nonnegative number.")
        settings["temperature"] = value
    if max_tokens.strip():
        try:
            value = int(max_tokens)
        except ValueError:
            raise ValueError("Maximum output tokens must be a positive integer.") from None
        if value < 1:
            raise ValueError("Maximum output tokens must be a positive integer.")
        settings["max_tokens"] = value
    return settings


@dataclass(slots=True)
class SlotConversation:
    """One pane's completed turns and the lifecycle of its current request."""

    history: list[dict[str, str]] = field(default_factory=list)
    busy: bool = False
    status: str = "Ready"
    started_at: float | None = None
    elapsed: float | None = None
    ttft: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def prepare(self, text: str, settings: dict[str, Any]) -> dict[str, Any]:
        if self.busy:
            raise ValueError("This pane already has a request in progress.")
        text = text.strip()
        if not text:
            raise ValueError("Enter a message before sending.")
        request = {
            **deepcopy(settings),
            "text": text,
            "extra_turn_messages": deepcopy(self.history),
        }
        self.busy = True
        self.status = "Running"
        self.started_at = time.perf_counter()
        self.elapsed = None
        self.ttft = None
        self.metadata = {}
        return request

    async def response(self, request: dict[str, Any]) -> AsyncIterator[str]:
        chunks = []
        failed = False

        def on_error(_message):
            nonlocal failed
            failed = True

        def on_metadata(metadata):
            self.metadata = response_metadata(
                metadata, endpoints=request.get("endpoints_dict"), endpoint_key=request.get("endpoint_key")
            )

        def on_first_token():
            if self.ttft is None and self.started_at is not None:
                self.ttft = max(0.0, time.perf_counter() - self.started_at)

        try:
            async for chunk in stream_chat_response(
                **request, on_error=on_error, on_metadata=on_metadata, on_first_token=on_first_token
            ):
                chunks.append(chunk)
                yield chunk
            self.status = "Failed" if failed else "Completed"
            if not failed:
                self.history.extend(
                    [
                        {"role": "user", "content": request["text"]},
                        {"role": "assistant", "content": "".join(chunks)},
                    ]
                )
        except asyncio.CancelledError:
            self.status = "Stopped"
            raise
        except Exception as exc:
            self.status = "Failed"
            yield f"Error: {exc}"
        finally:
            if self.started_at is not None:
                self.elapsed = max(0.0, time.perf_counter() - self.started_at)

    def finish(self, *, cancelled: bool = False, error: bool = False) -> None:
        """Finalize after the UI stream task ends, even if it never yielded a chunk."""
        if cancelled:
            self.status = "Stopped"
        elif error:
            self.status = "Failed"
        if self.elapsed is None and self.started_at is not None:
            self.elapsed = max(0.0, time.perf_counter() - self.started_at)
        self.busy = False


async def response_with_metrics(pane: SlotConversation, request: dict[str, Any]) -> AsyncIterator[Any]:
    """Append UI telemetry within the same message, outside the model's history."""
    try:
        async for chunk in pane.response(request):
            yield chunk
    except asyncio.CancelledError:
        yield response_metrics_ui(pane.metadata, elapsed=pane.elapsed, ttft=pane.ttft, status=pane.status)
        raise
    else:
        yield response_metrics_ui(pane.metadata, elapsed=pane.elapsed, ttft=pane.ttft, status=pane.status)


def multi_slot_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            endpoint_selector_ui("multiSlotEndpoint", "multiSlotModels", refresh_id="multiSlotRefresh"),
            ui.input_select(
                "multiSlotCount", "Chat panes", choices=[str(n) for n in range(1, MAX_PANES + 1)],
                selected="4",
            ),
            ui.tags.details(
                ui.tags.summary("Generation settings"),
                ui.input_select(
                    "multiSlotReasoning", "Reasoning Level",
                    choices=REASONING_LEVELS,
                    selected="medium",
                ),
                ui.input_text("multiSlotTemperature", "Temperature", placeholder="Server default"),
                ui.input_text("multiSlotMaxTokens", "Maximum output tokens", placeholder="Server default"),
                ui.p("Generation settings apply to new requests; active responses keep their settings."),
                ui.input_switch("multiSlotShowMetrics", "Show response metrics", True),
            ),
            ui.tags.details(
                ui.tags.summary("Burst test"),
                ui.input_text_area("multiSlotBroadcast", "Shared prompt", rows=4),
                ui.input_action_button("multiSlotSendAll", "Send all", width="100%"),
            ),
            ui.input_action_button("multiSlotStop", "Stop all"),
            ui.input_action_button("multiSlotReset", "Reset chats"),
            ui.output_text("multiSlotMessage"),
            ui.p("Independent conversations, kept for this dashboard session. The server assigns slots."),
            width="300px",
        ),
        ui.div(
            *[
                ui.panel_conditional(
                    f"Number(input.multiSlotCount) >= {index + 1}",
                    ui.card(
                        ui.card_header(
                            f"Pane {index + 1}",
                            ui.output_text(f"multiSlotStatus{index}"),
                        ),
                        chat_ui(
                            f"multiSlotChat{index}", width="100%", height="26rem",
                            show_history=False, drawer=False, icon_assistant=True,
                        ),
                    ),
                )
                for index in range(MAX_PANES)
            ],
            class_="multi-slot-grid",
        ),
    )


def multi_slot_server(input, output, session, available_endpoints, refresh_endpoints, endpoint_load_status):
    panes = [SlotConversation() for _ in range(MAX_PANES)]
    chats = [Chat(id=f"multiSlotChat{i}", history=False) for i in range(MAX_PANES)]
    tasks = [reactive.Value(None) for _ in panes]
    revision = reactive.Value(0)
    selected_endpoint = reactive.Value("")
    message = reactive.Value("")

    def touch():
        with reactive.isolate():
            revision.set(revision.get() + 1)

    def any_busy():
        return any(pane.busy for pane in panes)

    def request_settings():
        endpoint = selected_endpoint.get()
        endpoints = available_endpoints.get()
        if endpoint not in endpoints:
            raise ValueError("Select an available endpoint first.")
        return {
            "endpoint_key": endpoint,
            "endpoints_dict": deepcopy(endpoints),
            "reasoning_effort": input.multiSlotReasoning(),
            **parse_generation_settings(
                input.multiSlotTemperature() or "", input.multiSlotMaxTokens() or ""
            ),
        }

    async def reset_chats():
        for i, chat in enumerate(chats):
            panes[i] = SlotConversation()
            tasks[i].set(None)
            await chat.clear_messages()
            chat.update_user_input(value="")
        message.set("")
        touch()

    @reactive.effect
    async def update_endpoints():
        endpoints = available_endpoints.get()
        choices = {key: key for key in endpoints}
        with reactive.isolate():
            previous = selected_endpoint.get()
        if any_busy() and previous:
            choices.setdefault(previous, previous)
        selected = resolve_endpoint_selection(choices, preferred_endpoint_key=previous) or ""
        if selected != previous:
            await reset_chats()
            selected_endpoint.set(selected)
        ui.update_select("multiSlotEndpoint", choices=choices, selected=selected, session=session)

    @output
    @render.text
    def multiSlotModels():
        endpoint = available_endpoints.get().get(selected_endpoint.get(), {})
        return endpoint_models_text(endpoint, loading=endpoint_load_status() == "running")

    @reactive.effect
    @reactive.event(input.multiSlotEndpoint)
    async def change_endpoint():
        endpoint = input.multiSlotEndpoint() or ""
        if endpoint == selected_endpoint.get():
            return
        if any_busy():
            ui.update_select("multiSlotEndpoint", selected=selected_endpoint.get(), session=session)
            return
        await reset_chats()
        selected_endpoint.set(endpoint)

    async def launch(index, text, settings, *, append_user):
        pane = panes[index]
        request = pane.prepare(text, settings)
        touch()
        try:
            if append_user:
                await chats[index].append_message({"role": "user", "content": request["text"]})
            task = await chats[index].append_message_stream(response_with_metrics(pane, request))
            tasks[index].set(task)
        except BaseException:
            pane.finish(error=True)
            touch()
            raise

    def bind_pane(index):
        @chats[index].on_user_submit
        async def submit(text):
            try:
                await launch(index, text, request_settings(), append_user=False)
                message.set("")
            except ValueError as exc:
                message.set(str(exc))
                if not panes[index].busy:
                    await chats[index].append_message({"role": "assistant", "content": f"Error: {exc}"})
                    touch()

        @output(id=f"multiSlotStatus{index}")
        @render.text
        def status():
            revision.get()
            pane = panes[index]
            elapsed = pane.elapsed
            if pane.busy:
                reactive.invalidate_later(0.25)
                elapsed = time.perf_counter() - pane.started_at
            return pane.status if elapsed is None else f"{pane.status} · {elapsed:.1f} s"

        @reactive.effect
        def complete():
            task = tasks[index].get()
            if task is None:
                return
            status = task.status()
            if status in {"success", "error", "cancelled"} and panes[index].busy:
                panes[index].finish(cancelled=status == "cancelled", error=status == "error")
                touch()

    for index in range(MAX_PANES):
        bind_pane(index)

    @output
    @render.text
    def multiSlotMessage():
        return message.get()

    @reactive.effect
    async def control_state():
        revision.get()
        await session.send_custom_message(
            "multiSlotControls",
            {"busy": any_busy(), "showMetrics": input.multiSlotShowMetrics()},
        )

    @reactive.effect
    @reactive.event(input.multiSlotSendAll)
    async def send_all():
        try:
            text = (input.multiSlotBroadcast() or "").strip()
            if not text:
                raise ValueError("Enter a shared prompt before sending.")
            settings = request_settings()
            count = min(MAX_PANES, max(1, int(input.multiSlotCount())))
            idle = [i for i in range(count) if not panes[i].busy]
            for index in idle:
                await launch(index, text, settings, append_user=True)
            message.set(f"Started {len(idle)} requests; skipped {count - len(idle)} busy panes.")
        except ValueError as exc:
            message.set(str(exc))

    def cancel_streams():
        with reactive.isolate():
            for task in tasks:
                stream = task.get()
                if stream is not None:
                    stream.cancel()

    @reactive.effect
    @reactive.event(input.multiSlotStop)
    def stop_all():
        cancel_streams()

    session.on_ended(cancel_streams)

    @reactive.effect
    @reactive.event(input.multiSlotReset)
    async def reset():
        if not any_busy():
            await reset_chats()

    @reactive.effect
    @reactive.event(input.multiSlotRefresh)
    def refresh():
        if not any_busy():
            refresh_endpoints()
