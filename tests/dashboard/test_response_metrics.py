import asyncio

import pytest

from src.dashboard import response_metrics
from src.dashboard.response_metrics import ResponseMetrics, response_metadata, response_metrics_ui
from src.dashboard.search_flow import merge_run_info


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
    assert "Prefill — tok/s" in html
    assert "<details>" not in html


def test_timing_fallback_includes_cached_input_and_usage_aliases():
    html = render({"timings": {"prompt_n": 10, "cache_n": 90, "predicted_n": 12}})
    assert "Tokens 100 in / 12 out" in html
    html = render({"usage": {"input_tokens": 100, "output_tokens": 12}})
    assert "Tokens 100 in / 12 out" in html


def test_prefill_precedes_generation_and_uses_server_rate():
    html = render({"timings": {"prompt_per_second": 1234.5, "predicted_per_second": 45.6}})
    assert html.index("Prefill 1234.5 tok/s") < html.index("Generation 45.6 tok/s")
    html = render({"timings": {"prompt_per_second": 0, "predicted_per_second": 0}})
    assert "Prefill 0.0 tok/s" in html
    assert "Generation 0.0 tok/s" in html


@pytest.mark.parametrize("drafted,accepted,expected", [
    (184, 139, "75.54% (139 / 184)"),
    (10, 0, "0.00% (0 / 10)"),
    (10, 10, "100.00% (10 / 10)"),
    (0, 0, "— (0 / 0)"),
    (0, 1, "—"),
    (10, 11, "—"),
    (-1, 0, "—"),
    (True, 1, "—"),
    (10, None, "—"),
    (float("nan"), 1, "—"),
])
def test_mtp_acceptance_uses_head_counts_instead_of_aggregate(drafted, accepted, expected):
    html = render({"timings": {
        "draft_n": 1000, "draft_n_accepted": 900,
        "speculation": {"head": {"drafted": drafted, "accepted": accepted},
                        "lookup": {"drafted": 1000, "accepted": 900}},
    }})
    assert f"MTP draft acceptance: {expected}" in html
    assert "90.00%" not in html
    if expected == "—":
        assert "MTP draft acceptance: —</div>" in html


@pytest.mark.parametrize("section", ["timings", "stats"])
def test_aggregate_draft_acceptance_is_labelled_without_claiming_mtp(section):
    html = render({section: {"draft_n": 184, "draft_n_accepted": 139}})
    assert "Draft acceptance: 75.54% (139 / 184)" in html
    assert "MTP draft acceptance" not in html


def test_model_information_is_displayed_as_escaped_text():
    html = render({
        "model": "served-model<script>",
        "model_info": {"id": "served-model<script>", "quantization": "Q4_K_M",
                       "max_context_length": 32768, "architecture": {"layers": 32}},
    })
    assert html.count("served-model&lt;script&gt;") == 1
    assert "<script>" not in html
    assert "Quantization: Q4_K_M" in html
    assert "Max context length: 32768" in html
    assert "Architecture:" in html
    assert "Model: info-model" in render({"model_info": {"id": "info-model"}})


@pytest.mark.asyncio
async def test_response_metrics_are_frozen_per_turn_after_source_finishes(monkeypatch):
    clock = iter([11.0, 14.0])
    monkeypatch.setattr(response_metrics.time, "perf_counter", lambda: next(clock))
    metrics = ResponseMetrics(started_at=10.0)
    finalized = []

    async def source():
        metrics.on_first_token()
        metrics.on_first_token()
        yield "answer"
        metrics.metadata = {"usage": {"prompt_tokens": 12, "completion_tokens": 3}}

        finalized.append(True)

    chunks = [chunk async for chunk in metrics.stream(source())]
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
        try:
            metrics.on_first_token()
            yield "partial"
            started.set()
            await asyncio.Event().wait()
        finally:
            finalized.append(True)

    async def collect():
        async for chunk in metrics.stream(source()):
            chunks.append(chunk)

    task = asyncio.create_task(collect())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert chunks[0] == "partial"
    assert "Stopped" in str(chunks[-1])
    assert finalized == [True]


def test_turn_diagnostics_keep_zeroes_and_are_escaped_without_raw_metric_dumps():
    html = render({
        "trace": {"id": "trace<script>"},
        "endpoint": "requested-node",
        "routing": {"decision": "actual-node", "strategy": "code"},
        "workflow": {"workflow_id": "repo_context", "run_id": "wf_1", "status": "completed"},
        "search": {"provider": "test", "result_count": 0, "degraded": False,
                   "query": "duplicate-query", "queries": ["first", "second"], "warnings": []},
        "retrieval": {"hits": 0, "injected": False, "threshold": 0, "reason": "no evidence"},
        "stats": {"internal_counter": 999}, "runtime": {"raw_dump": "unused"},
    })
    assert "Trace ID: trace&lt;script&gt;" in html
    assert "<script>" not in html
    for text in ["Endpoint: actual-node", "Workload: code", "ID: repo_context", "Run: wf_1",
                 "Status: completed", "Provider: test", "Results: 0", "Queries: first; second",
                 "Hits: 0", "Injected: false", "Threshold: 0", "Reason: no evidence"]:
        assert text in html
    for text in ["requested-node", "duplicate-query", "Warnings:", "internal_counter", "raw_dump"]:
        assert text not in html
    assert "Trace ID: runtime-trace" in render({"trace_id": "runtime-trace"})


def test_response_metadata_snapshots_routed_model_and_cache_configuration():
    endpoints = {
        "a": {"models": [{"id": "model-a", "quantization": "Q8"}]},
        "b": {"models": [{"id": "model-b", "meta": {"ftype": "Q4_K_M", "n_ctx": 16384,
                                                     "n_ctx_train": 262144}}]},
    }
    result = response_metadata({
        "model": "model-b", "routing": {"decision": "b"},
        "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}},
    }, endpoints=endpoints, endpoint_key="a")
    assert result["endpoint"] == "b"
    assert result["model_info"]["quantization"] == "Q4_K_M"
    assert result["model_info"]["loaded_context_length"] == 16384
    assert result["model_info"]["training_context_length"] == 262144
    endpoints["b"]["models"][0]["meta"]["n_ctx"] = 8192
    assert result["model_info"]["loaded_context_length"] == 16384
    html = render(result)
    assert "Cached input tokens: 80 (80.0% reused)" in html
    assert "Q8" not in html
    unmatched = response_metadata({"model": "different-model"}, endpoints=endpoints, endpoint_key="a")
    assert unmatched["model_info"] == {}
    assert "Model: different-model" in render(unmatched)


def test_search_failure_is_frozen_in_footer_without_carrying_to_next_turn():
    failed = response_metadata(merge_run_info({}, {
        "provider": "test", "degraded": True, "warnings": ["search failed"], "result_count": 0,
    }))
    html = render(failed)
    assert "Degraded: true" in html
    assert "Warnings: search failed" in html
    assert "<strong>Search</strong>" not in render(response_metadata({}))
    assert render(failed) == html


def test_equal_context_limits_are_not_repeated():
    endpoints = {"a": {"models": [{"id": "model-a", "max_model_len": 32768,
                                   "meta": {"n_ctx": 32768, "n_ctx_train": 32768}}]}}
    info = response_metadata({}, endpoints=endpoints, endpoint_key="a")["model_info"]
    assert info == {"id": "model-a", "loaded_context_length": 32768}
    endpoints["a"]["models"][0]["meta"]["n_ctx"] = 16384
    info = response_metadata({}, endpoints=endpoints, endpoint_key="a")["model_info"]
    assert info["loaded_context_length"] == 16384
    assert info["max_context_length"] == 32768
    assert "training_context_length" not in info
