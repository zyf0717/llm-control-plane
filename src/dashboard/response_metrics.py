"""Render per-response telemetry without adding it to model context."""

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterable, AsyncIterator
from copy import deepcopy
from typing import Any

from shiny import ui
from src.llm_metrics import CallMetrics

from .cache_telemetry import normalize_cache_telemetry


class ResponseMetrics(CallMetrics):
    """Collect one turn's telemetry separately from its text and durable history."""

    def footer(self):
        self.finish()
        return response_metrics_ui(
            self.metadata,
            elapsed=self.elapsed,
            ttft=self.ttft,
            status=self.status,
        )

    async def stream(self, source: AsyncIterable[Any], *, append_footer: bool = True) -> AsyncIterator[Any]:
        try:
            async for chunk in source:
                yield chunk
        except asyncio.CancelledError:
            self.status = "Stopped"
            if append_footer:
                yield self.footer()
            raise
        except Exception as exc:
            self.on_error(str(exc))
            logging.getLogger(__name__).exception("Chat response stream failed")
            yield f"Error: {exc}"
        if append_footer:
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


def response_metadata(
    metadata: dict | None, *, endpoints: dict | None = None, endpoint_key: str | None = None,
) -> dict:
    """Freeze response diagnostics with the matching endpoint's model configuration."""
    info = deepcopy(metadata or {})
    endpoint = _mapping(info.get("routing")).get("decision") or info.get("endpoint") or endpoint_key
    if endpoint:
        info["endpoint"] = endpoint
    if endpoint not in (endpoints or {}) and isinstance(endpoint, str):
        base_url = endpoint.removesuffix("/v1/chat/completions").rstrip("/")
        endpoint = next((key for key, value in (endpoints or {}).items()
                         if str(value.get("endpoint_url", "")).rstrip("/") == base_url), endpoint)
        info["endpoint"] = endpoint
    candidates = _mapping((endpoints or {}).get(endpoint)).get("models") or []
    reported = _mapping(info.get("model_info"))
    name = info.get("model") or reported.get("id")
    model = next((item for item in candidates if item.get("id") == name), {}) if name else (
        candidates[0] if candidates else {}
    )
    meta = _mapping(model.get("meta"))
    fields = {
        "id": model.get("id"),
        "arch": model.get("arch"),
        "quantization": model.get("quantization", meta.get("ftype")),
        "max_context_length": model.get("max_context_length", model.get("max_model_len")),
        "loaded_context_length": model.get("loaded_context_length", model.get("context_length", meta.get("n_ctx"))),
        "training_context_length": meta.get("n_ctx_train"),
        "state": model.get("state"),
    }
    info["model_info"] = {key: value for key, value in fields.items() if value is not None}
    info["model_info"].update(reported)
    model_info = info["model_info"]
    for key in ("max_context_length", "training_context_length"):
        if key in model_info and model_info[key] == model_info.get("loaded_context_length"):
            model_info.pop(key)
    if "training_context_length" in model_info and model_info["training_context_length"] == model_info.get("max_context_length"):
        model_info.pop("training_context_length")
    cache = normalize_cache_telemetry(info)
    if cache["status"] != "unknown":
        info["cache"] = cache
    return info


def _detail_value(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).lower() if isinstance(value, bool) else str(value)


def _detail_section(title: str, info: dict, fields: dict[str, str]):
    rows = [
        ui.div(f"{label}: {_detail_value(info[key])}")
        for key, label in fields.items()
        if key in info and info[key] is not None and info[key] != "" and info[key] != []
    ]
    return ui.div(ui.strong(title), *rows, class_="response-details-section") if rows else None


def call_metrics_ui(call: dict, *, endpoints: dict | None = None, label: str | None = None):
    """Use the same footer for live chat and recorded internal calls."""
    return response_metrics_ui(
        response_metadata(call.get("metadata"), endpoints=endpoints),
        elapsed=call.get("elapsed"), ttft=call.get("ttft"),
        status=call.get("status", "Completed"), label=label or call.get("label"),
    )


def workflow_metrics_ui(snapshot: dict | None, *, elapsed: float, status: str,
                        rendered_ids: set[str], endpoints: dict | None = None):
    """Whole-workflow elapsed time, with remaining calls under Details → Steps."""
    steps = []
    for step in (snapshot or {}).get("steps", []):
        output = _mapping(step.get("output_json"))
        calls = _mapping(output.get("metadata")).get("llm_calls", [])
        for index, call in enumerate(calls):
            if call.get("id") not in rendered_ids:
                steps.append(call_metrics_ui(
                    call, endpoints=endpoints,
                    label=f"{step['step_id']} · {call.get('label', 'LLM')} {index + 1}",
                ))
    run = _mapping((snapshot or {}).get("run"))
    workflow = _detail_section("Workflow", run, {"workflow_id": "ID", "run_id": "Run", "status": "Status"})
    return ui.div(
        ui.div(ui.span(f"Workflow total {elapsed:.2f}s"),
               *([ui.span(status)] if status != "Completed" else []),
               class_="response-metrics-summary"),
        ui.tags.details(ui.tags.summary("Details"), workflow,
                        ui.tags.details(ui.tags.summary("Steps"), *steps) if steps else None)
        if workflow is not None or steps else None,
        class_="response-metrics",
    )


def response_metrics_ui(metadata: dict, *, elapsed: float | None, ttft: float | None, status: str,
                        label: str | None = None):
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
    prefill_rate = _number(timings.get("prompt_per_second"))
    generation_rate = _number(timings.get("predicted_per_second"))

    def tokens(value):
        return "—" if value is None else f"{value:,}"

    def seconds(value):
        return "—" if value is None else f"{value:.2f}s"

    def rate(value):
        return "—" if value is None else f"{value:.1f}"

    details = []
    reasoning = _count(_mapping(usage.get("completion_tokens_details")).get("reasoning_tokens"))
    if reasoning is None:
        reasoning = _count(_mapping(usage.get("output_tokens_details")).get("reasoning_tokens"))
    if reasoning is not None:
        details.append(ui.tags.div(f"Reasoning tokens: {reasoning:,}"))
    cached_tokens = _count(cache.get("cache_tokens"))
    if cached_tokens is not None:
        ratio = _number(cache.get("cache_ratio"))
        reused = f" ({ratio:.1%} reused)" if ratio is not None and ratio <= 1 else ""
        details.append(ui.tags.div(f"Cached input tokens: {cached_tokens:,}{reused}"))
    if cache.get("fields_disagree"):
        details.append(ui.div("Cache counts differ between usage and timings."))

    head = _mapping(_mapping(timings.get("speculation")).get("head"))
    if head:
        drafted, accepted = _count(head.get("drafted")), _count(head.get("accepted"))
        draft_label = "MTP draft acceptance"
        has_draft_counts = True
    else:
        counts = (
            timings if any(key in timings for key in ("draft_n", "draft_n_accepted"))
            else _mapping(metadata.get("stats"))
        )
        drafted, accepted = _count(counts.get("draft_n")), _count(counts.get("draft_n_accepted"))
        draft_label = "Draft acceptance"
        has_draft_counts = "draft_n" in counts or "draft_n_accepted" in counts
    if has_draft_counts:
        valid = drafted is not None and accepted is not None and accepted <= drafted
        acceptance = f"{100 * accepted / drafted:.2f}%" if valid and drafted else "—"
        counts_text = f" ({accepted:,} / {drafted:,})" if valid else ""
        details.append(ui.tags.div(
            f"{draft_label}: {acceptance}{counts_text}",
            title="Accepted / proposed draft tokens. MTP uses head-only counts; Draft acceptance includes all reported draft sources.",
        ))

    slot_id = _count(slot.get("id"))
    if slot_id is not None:
        details.append(ui.tags.div(f"Upstream slot: {slot_id}"))

    model_info = _mapping(metadata.get("model_info"))
    model = metadata.get("model") or model_info.get("id") or model_info.get("name")
    if model:
        details.append(ui.tags.div(f"Model: {model}"))
    for key, value in model_info.items():
        if value is None or value == "" or (key in {"id", "name"} and value == model):
            continue
        details.append(ui.tags.div(f"{key.replace('_', ' ').capitalize()}: {_detail_value(value)}"))

    trace_id = _mapping(metadata.get("trace")).get("id") or metadata.get("trace_id")
    if trace_id:
        details.append(ui.div(f"Trace ID: {trace_id}"))
    routing = _mapping(metadata.get("routing"))
    endpoint = routing.get("decision") or metadata.get("endpoint")
    if endpoint:
        details.append(ui.div(f"Endpoint: {endpoint}"))
    if routing.get("strategy"):
        details.append(ui.div(f"Workload: {routing['strategy']}"))

    search = dict(_mapping(metadata.get("search")))
    if search.get("queries"):
        search.pop("query", None)
    for title, info, fields in (
        ("Workflow", _mapping(metadata.get("workflow")), {
            "workflow_id": "ID", "run_id": "Run", "status": "Status",
        }),
        ("Search", search, {
            "provider": "Provider", "result_count": "Results", "query": "Query", "queries": "Queries",
            "degraded": "Degraded", "warnings": "Warnings",
        }),
        ("RAG", _mapping(metadata.get("retrieval")), {
            "endpoint": "Endpoint", "hits": "Hits", "injected": "Injected",
            "mode": "Mode", "truncated": "Truncated", "confidence": "Confidence",
            "threshold": "Threshold", "method": "Method", "reason": "Reason",
        }),
    ):
        section = _detail_section(title, info, fields)
        if section is not None:
            details.append(section)

    return ui.div(
        ui.div(
            *([ui.strong(label)] if label else []),
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
                f"Prefill {rate(prefill_rate)} tok/s",
                title="Prompt-processing throughput reported by the endpoint; cached tokens are excluded when the endpoint reports processed prompt tokens.",
            ),
            ui.span(
                f"Generation {rate(generation_rate)} tok/s",
                title="Generation throughput reported by the endpoint; — means unavailable.",
            ),
            class_="response-metrics-summary",
        ),
        ui.tags.details(ui.tags.summary("Details"), *details) if details else None,
        class_="response-metrics",
    )
