import asyncio

import pytest

from src.dashboard import response_metrics
from src.dashboard.response_metrics import ResponseMetrics, response_metrics_ui
from src.dashboard.app_server import stream_with_finalizer


def render(metadata):
    return str(response_metrics_ui(metadata, elapsed=5.0, ttft=1.0, status="Completed"))


def test_metrics_keep_zero_values_and_use_upstream_rate():
    html = render({
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                  "completion_tokens_details": {"reasoning_tokens": 0}},
        "timings": {"predicted_per_second": 0},
        "cache": {"cache_tokens": 0}, "slot": {"id": 0},
    })
    assert "Tokens 0 in / 0 out" in html
    assert "0.0 tok/s" in html
    assert "Reasoning tokens: 0" in html
    assert "Cached input tokens: 0" in html
    assert "Upstream slot: 0" in html
    assert "<details>" in html


def test_missing_or_invalid_metrics_are_not_estimated():
    html = render({
        "usage": {"prompt_tokens": -1, "completion_tokens": True},
        "timings": {"prompt_n": 100, "predicted_per_second": float("nan")},
        "cache": [], "slot": {"id": -1},
    })
    assert "Tokens — in / — out" in html
    assert "— tok/s" in html
    assert "<details>" not in html


def test_timing_fallback_includes_cached_input_and_usage_aliases():
    html = render({"timings": {"prompt_n": 10, "cache_n": 90, "predicted_n": 12}})
    assert "Tokens 100 in / 12 out" in html
    html = render({"usage": {"input_tokens": 100, "output_tokens": 12}})
    assert "Tokens 100 in / 12 out" in html


@pytest.mark.asyncio
async def test_response_metrics_are_frozen_per_turn_and_finalize_before_footer(monkeypatch):
    clock = iter([11.0, 14.0])
    monkeypatch.setattr(response_metrics.time, "perf_counter", lambda: next(clock))
    metrics = ResponseMetrics(started_at=10.0)
    finalized = []

    async def source():
        metrics.on_first_token()
        metrics.on_first_token()
        yield "answer"
        metrics.metadata = {"usage": {"prompt_tokens": 12, "completion_tokens": 3}}

    chunks = [chunk async for chunk in metrics.stream(
        stream_with_finalizer(source(), lambda: finalized.append(True))
    )]
    assert chunks[0] == "answer"
    assert finalized == [True]
    footer = str(chunks[-1])
    assert "TTFT 1.00s" in footer
    assert "Total 4.00s" in footer
    assert "Tokens 12 in / 3 out" in footer
    metrics.metadata["usage"]["completion_tokens"] = 99
    assert str(chunks[-1]) == footer


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_response_failure_footer_does_not_invent_ttft(raises):
    metrics = ResponseMetrics(started_at=response_metrics.time.perf_counter())

    async def source():
        if raises:
            raise RuntimeError("connection lost")
        metrics.on_error("connection lost")
        yield "Error: connection lost"

    chunks = [chunk async for chunk in metrics.stream(source())]
    assert chunks[0] == "Error: connection lost"
    assert "Failed" in str(chunks[-1])
    assert "TTFT —" in str(chunks[-1])


@pytest.mark.asyncio
async def test_cancelled_response_keeps_footer_and_propagates_cancellation():
    metrics = ResponseMetrics(started_at=response_metrics.time.perf_counter())
    started = asyncio.Event()
    chunks = []
    finalized = []

    async def source():
        metrics.on_first_token()
        yield "partial"
        started.set()
        await asyncio.Event().wait()

    async def collect():
        async for chunk in metrics.stream(stream_with_finalizer(
            source(), lambda: finalized.append(True)
        )):
            chunks.append(chunk)

    task = asyncio.create_task(collect())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert chunks[0] == "partial"
    assert "Stopped" in str(chunks[-1])
    assert finalized == [True]
