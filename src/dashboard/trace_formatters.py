from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from .utils import HISTORY_DISPLAY_TIMEZONE

TRACE_DISPLAY_TIMEZONE_LABEL = "GMT+8"


def format_trace_summary(event: Dict[str, Any]) -> str:
    timestamp = format_trace_timestamp(event.get("timestamp"))
    trace_id = str(event.get("request_id") or "unknown-trace")
    endpoint = str(event.get("endpoint") or "unknown-endpoint")
    phase = str(event.get("phase") or "completed")
    status = event.get("status_code")
    convo_id = str(event.get("convo_id") or "no-convo")
    elapsed = ""
    timing = event.get("timing")
    if isinstance(timing, dict) and timing.get("elapsed_ms") is not None:
        elapsed = f" | {timing['elapsed_ms']}ms"
    status_label = status if status is not None else "?"
    return (
        f"{timestamp} | {phase} | {status_label} | "
        f"{endpoint} | {convo_id} | {trace_id}{elapsed}"
    )


def format_trace_timestamp(raw_timestamp: Any) -> str:
    text = str(raw_timestamp or "").strip()
    if not text:
        return f"unknown-time {TRACE_DISPLAY_TIMEZONE_LABEL}"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        parsed.astimezone(HISTORY_DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
        + f" {TRACE_DISPLAY_TIMEZONE_LABEL}"
    )
