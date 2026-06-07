from __future__ import annotations

from typing import Any, Dict, Optional


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _nested_get(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def normalize_cache_telemetry(info: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize llama.cpp/OpenAI-compatible cache metrics.

    Source priority:
    1. timings.cache_n
    2. usage.prompt_tokens_details.cached_tokens
    3. usage.input_tokens_details.cached_tokens for Responses API-style payloads
    """
    usage = info.get("usage") if isinstance(info.get("usage"), dict) else {}
    timings = info.get("timings") if isinstance(info.get("timings"), dict) else {}

    cache_from_timings = _to_int(timings.get("cache_n"))
    cache_from_usage = _to_int(
        _nested_get(usage, "prompt_tokens_details", "cached_tokens")
    )
    cache_from_responses_usage = _to_int(
        _nested_get(usage, "input_tokens_details", "cached_tokens")
    )

    cache_tokens = next(
        (
            value
            for value in (
                cache_from_timings,
                cache_from_usage,
                cache_from_responses_usage,
            )
            if value is not None
        ),
        None,
    )

    prompt_tokens = _to_int(usage.get("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _to_int(usage.get("input_tokens"))

    processed_prompt_tokens = _to_int(timings.get("prompt_n"))
    generated_tokens = _to_int(timings.get("predicted_n"))
    if generated_tokens is None:
        generated_tokens = _to_int(usage.get("completion_tokens"))
    if generated_tokens is None:
        generated_tokens = _to_int(usage.get("output_tokens"))

    approx_context_tokens = None
    if cache_tokens is not None and processed_prompt_tokens is not None:
        approx_context_tokens = cache_tokens + processed_prompt_tokens
        if generated_tokens is not None:
            approx_context_tokens += generated_tokens

    cache_ratio = None
    if cache_tokens is not None and prompt_tokens:
        cache_ratio = cache_tokens / prompt_tokens

    fields_disagree = (
        cache_from_timings is not None
        and cache_from_usage is not None
        and cache_from_timings != cache_from_usage
    )

    if cache_tokens is None:
        status = "unknown"
        reason = "No llama.cpp cache fields found in usage/timings."
    elif cache_tokens > 0:
        status = "hit"
        reason = "Prompt/KV cache reuse detected."
    else:
        status = "miss"
        reason = "No prompt/KV cache reuse reported for this request."

    return {
        "status": status,
        "reason": reason,
        "cache_tokens": cache_tokens,
        "prompt_tokens": prompt_tokens,
        "processed_prompt_tokens": processed_prompt_tokens,
        "generated_tokens": generated_tokens,
        "approx_context_tokens": approx_context_tokens,
        "cache_ratio": cache_ratio,
        "source": (
            "timings.cache_n"
            if cache_from_timings is not None
            else (
                "usage.prompt_tokens_details.cached_tokens"
                if cache_from_usage is not None
                else (
                    "usage.input_tokens_details.cached_tokens"
                    if cache_from_responses_usage is not None
                    else None
                )
            )
        ),
        "fields_disagree": fields_disagree,
        "raw": {
            "timings_cache_n": cache_from_timings,
            "usage_cached_tokens": cache_from_usage,
            "responses_usage_cached_tokens": cache_from_responses_usage,
        },
    }
