"""Per-call LLM telemetry; independent of the dashboard and model prompts."""

import asyncio
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


_collectors: ContextVar[tuple[list[dict], ...]] = ContextVar("llm_metrics", default=())


@contextmanager
def capture_llm_calls(calls: list[dict] | None = None):
    """Collect calls in this task, including nested and parallel work."""
    calls = [] if calls is None else calls
    token = _collectors.set((*_collectors.get(), calls))
    try:
        yield calls
    finally:
        _collectors.reset(token)


async def capture_llm_stream(source: AsyncIterable, calls: list[dict]) -> AsyncIterator:
    """Collect streaming work without leaving a collection scope open across yields."""
    iterator = aiter(source)
    try:
        while True:
            with capture_llm_calls(calls):
                try:
                    item = await anext(iterator)
                except StopAsyncIteration:
                    return
            yield item
    finally:
        with capture_llm_calls(calls):
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()


def llm_response_metadata(payload: dict) -> dict:
    metadata = {
        key: deepcopy(payload[key])
        for key in ("usage", "stats", "timings", "model_info", "runtime")
        if isinstance(payload.get(key), dict)
    }
    if isinstance(payload.get("model"), str) and payload["model"].strip():
        metadata["model"] = payload["model"]
    for key in ("id_slot", "slot_id"):
        if payload.get(key) is not None:
            metadata["slot"] = {"id": payload[key]}
            break
    return metadata


@dataclass(slots=True)
class CallMetrics:
    started_at: float = field(default_factory=time.perf_counter)
    metadata: dict[str, Any] = field(default_factory=dict)
    ttft: float | None = None
    status: str = "Completed"
    elapsed: float | None = None
    label: str = "LLM"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _targets: tuple[list[dict], ...] = field(default=(), init=False, repr=False)

    def on_first_token(self) -> None:
        if self.ttft is None:
            self.ttft = max(0.0, time.perf_counter() - self.started_at)

    def on_error(self, _message: str) -> None:
        self.status = "Failed"

    def finish(self) -> None:
        if self.elapsed is None:
            self.elapsed = max(0.0, time.perf_counter() - self.started_at)

    def record(self) -> dict:
        self.finish()
        return {
            "id": self.id, "label": self.label, "elapsed": self.elapsed,
            "ttft": self.ttft, "status": self.status, "metadata": deepcopy(self.metadata),
        }

    def __enter__(self):
        self._targets = _collectors.get()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.status = (
                "Stopped" if issubclass(exc_type, (asyncio.CancelledError, GeneratorExit)) else "Failed"
            )
        record = self.record()
        for calls in self._targets:
            calls.append(record)


def without_call_metrics(value: Any) -> Any:
    """Remove telemetry records when constructing downstream model inputs."""
    if isinstance(value, list):
        return [without_call_metrics(item) for item in value]
    if isinstance(value, dict):
        return {
            key: without_call_metrics(item) for key, item in value.items()
            if not (key == "llm_calls" and isinstance(item, list) and item
                    and all(isinstance(call, dict) and "elapsed" in call and "metadata" in call for call in item))
        }
    return value
