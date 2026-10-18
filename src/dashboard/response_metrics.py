"""Render per-response telemetry without adding it to model context."""

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from shiny import ui


@dataclass(slots=True)
class ResponseMetrics:
    """Collect one turn's telemetry separately from its text and durable history."""

    started_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    ttft: float | None = None
    status: str = "Completed"

    def on_first_token(self) -> None:
        if self.ttft is None:
            self.ttft = max(0.0, time.perf_counter() - self.started_at)

    def on_error(self, _message: str) -> None:
        self.status = "Failed"

    def footer(self):
        return response_metrics_ui(
            self.metadata,
            elapsed=max(0.0, time.perf_counter() - self.started_at),
            ttft=self.ttft,
            status=self.status,
        )

    async def stream(self, source: AsyncIterable[str]) -> AsyncIterator[Any]:
        try:
            async for chunk in source:
                yield chunk
        except asyncio.CancelledError:
            self.status = "Stopped"
            yield self.footer()
            raise
        except Exception as exc:
            self.on_error(str(exc))
            logging.getLogger(__name__).exception("Chat response stream failed")
            yield f"Error: {exc}"
        yield self.footer()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _count(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def response_metrics_ui(metadata: dict, *, elapsed: float | None, ttft: float | None, status: str):
    usage = _mapping(metadata.get("usage"))
    timings = _mapping(metadata.get("timings"))
    cache = _mapping(metadata.get("cache"))
    slot = _mapping(metadata.get("slot"))

    input_tokens = _count(usage.get("prompt_tokens"))
    if input_tokens is None:
        input_tokens = _count(usage.get("input_tokens"))
    if input_tokens is None:
        processed, cached = _count(timings.get("prompt_n")), _count(timings.get("cache_n"))
        if processed is not None and cached is not None:
            input_tokens = processed + cached
    output_tokens = _count(usage.get("completion_tokens"))
    if output_tokens is None:
        output_tokens = _count(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _count(timings.get("predicted_n"))
    rate = _number(timings.get("predicted_per_second"))

    def tokens(value):
        return "—" if value is None else f"{value:,}"

    def seconds(value):
        return "—" if value is None else f"{value:.2f}s"

    details = []
    reasoning = _count(_mapping(usage.get("completion_tokens_details")).get("reasoning_tokens"))
    if reasoning is None:
        reasoning = _count(_mapping(usage.get("output_tokens_details")).get("reasoning_tokens"))
    if reasoning is not None:
        details.append(ui.tags.div(f"Reasoning tokens: {reasoning:,}"))
    cached_tokens = _count(cache.get("cache_tokens"))
    if cached_tokens is not None:
        details.append(ui.tags.div(f"Cached input tokens: {cached_tokens:,}"))
    slot_id = _count(slot.get("id"))
    if slot_id is not None:
        details.append(ui.tags.div(f"Upstream slot: {slot_id}"))

    return ui.div(
        ui.div(
            *([ui.span(status)] if status != "Completed" else []),
            ui.span(
                f"TTFT {seconds(ttft)}",
                title="Time from submission to the first observed model token, including reasoning.",
            ),
            ui.span(
                f"Total {seconds(elapsed)}",
                title="Time from submission to the end of the response, including network and queue time.",
            ),
            ui.span(f"Tokens {tokens(input_tokens)} in / {tokens(output_tokens)} out"),
            ui.span(
                "— tok/s" if rate is None else f"{rate:.1f} tok/s",
                title="Generation throughput reported by the endpoint; — means unavailable.",
            ),
            class_="response-metrics-summary",
        ),
        ui.tags.details(ui.tags.summary("Details"), *details) if details else None,
        class_="response-metrics",
    )
