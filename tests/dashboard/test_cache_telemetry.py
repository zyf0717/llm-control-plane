import pytest
from src.dashboard.cache_telemetry import (
    normalize_cache_telemetry,
    _to_int,
    _nested_get,
)


class TestToInt:
    def test_int(self):
        assert _to_int(42) == 42

    def test_float_integer(self):
        assert _to_int(42.0) == 42

    def test_float_non_integer(self):
        assert _to_int(42.5) is None

    def test_bool(self):
        assert _to_int(True) is None
        assert _to_int(False) is None

    def test_string_valid(self):
        assert _to_int("42") == 42

    def test_string_invalid(self):
        assert _to_int("abc") is None

    def test_none(self):
        assert _to_int(None) is None


class TestNestedGet:
    def test_simple(self):
        obj = {"a": {"b": 1}}
        assert _nested_get(obj, "a", "b") == 1

    def test_missing_intermediate(self):
        obj = {"a": 1}
        assert _nested_get(obj, "a", "b", "c") is None

    def test_non_dict_intermediate(self):
        obj = {"a": 1}
        assert _nested_get(obj, "a", "b") is None


class TestNormalizeCacheTelemetry:
    # 1. timings.cache_n present, usage absent
    def test_prefers_timings_cache_n(self):
        info = {
            "timings": {"cache_n": 91, "prompt_n": 9, "predicted_n": 20},
        }
        cache = normalize_cache_telemetry(info)
        assert cache["cache_tokens"] == 91
        assert cache["source"] == "timings.cache_n"
        assert cache["status"] == "hit"
        assert cache["approx_context_tokens"] == 120
        assert cache["processed_prompt_tokens"] == 9
        assert cache["generated_tokens"] == 20
        assert cache["prompt_tokens"] is None
        assert cache["cache_ratio"] is None

    # 2. usage.prompt_tokens_details.cached_tokens present, timings absent
    def test_falls_back_to_usage_cached_tokens(self):
        info = {
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        }
        cache = normalize_cache_telemetry(info)
        assert cache["cache_tokens"] == 80
        assert cache["source"] == "usage.prompt_tokens_details.cached_tokens"
        assert cache["status"] == "hit"
        assert cache["cache_ratio"] == 0.8

    # 3. Both present and equal
    def test_both_present_and_equal(self):
        info = {
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 90},
            },
            "timings": {"cache_n": 90, "prompt_n": 10, "predicted_n": 30},
        }
        cache = normalize_cache_telemetry(info)
        assert cache["cache_tokens"] == 90
        assert cache["source"] == "timings.cache_n"
        assert cache["fields_disagree"] is False
        assert cache["approx_context_tokens"] == 130

    # 4. Both present and disagree
    def test_both_present_and_disagree(self):
        info = {
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 90},
            },
            "timings": {"cache_n": 91, "prompt_n": 9, "predicted_n": 20},
        }
        cache = normalize_cache_telemetry(info)
        assert cache["cache_tokens"] == 91
        assert cache["source"] == "timings.cache_n"
        assert cache["fields_disagree"] is True
        assert cache["approx_context_tokens"] == 120

    # 5. Responses API-style input_tokens_details.cached_tokens
    def test_responses_api_style(self):
        info = {
            "usage": {
                "input_tokens": 200,
                "input_tokens_details": {"cached_tokens": 150},
            },
        }
        cache = normalize_cache_telemetry(info)
        assert cache["cache_tokens"] == 150
        assert cache["source"] == "usage.input_tokens_details.cached_tokens"
        assert cache["status"] == "hit"
        assert cache["prompt_tokens"] == 200
        assert cache["cache_ratio"] == 0.75

    # 6. No cache fields -> status == unknown
    def test_no_cache_fields_unknown(self):
        info = {}
        cache = normalize_cache_telemetry(info)
        assert cache["status"] == "unknown"
        assert cache["cache_tokens"] is None
        assert cache["source"] is None
        assert cache["reason"] == "No llama.cpp cache fields found in usage/timings."

    # 7. cache_tokens == 0 -> status == miss
    def test_cache_zero_is_miss(self):
        info = {"timings": {"cache_n": 0, "prompt_n": 50, "predicted_n": 10}}
        cache = normalize_cache_telemetry(info)
        assert cache["status"] == "miss"
        assert cache["cache_tokens"] == 0
        assert cache["approx_context_tokens"] == 60

    # 8. cache_tokens > 0 -> status == hit
    def test_cache_positive_is_hit(self):
        info = {"timings": {"cache_n": 1, "prompt_n": 49, "predicted_n": 10}}
        cache = normalize_cache_telemetry(info)
        assert cache["status"] == "hit"
        assert cache["cache_tokens"] == 1

    # 9. approx_context_tokens == cache_n + prompt_n + predicted_n
    def test_approx_context_tokens_with_generated(self):
        info = {"timings": {"cache_n": 100, "prompt_n": 50}}
        cache = normalize_cache_telemetry(info)
        assert cache["approx_context_tokens"] == 150
        assert cache["generated_tokens"] is None

    def test_approx_context_tokens_without_generated(self):
        info = {
            "timings": {"cache_n": 100, "prompt_n": 50, "predicted_n": 30},
            "usage": {"completion_tokens": 30},
        }
        cache = normalize_cache_telemetry(info)
        assert cache["approx_context_tokens"] == 180
        assert cache["generated_tokens"] == 30

    # 10. cache_ratio == cache_tokens / prompt_tokens
    def test_cache_ratio(self):
        info = {
            "usage": {"prompt_tokens": 100},
            "timings": {"cache_n": 75},
        }
        cache = normalize_cache_telemetry(info)
        assert cache["cache_ratio"] == 0.75

    def test_cache_ratio_none_when_no_prompt_tokens(self):
        info = {"timings": {"cache_n": 75}}
        cache = normalize_cache_telemetry(info)
        assert cache["cache_ratio"] is None

    def test_usage_not_dict(self):
        info = {"usage": "not a dict", "timings": {"cache_n": 10}}
        cache = normalize_cache_telemetry(info)
        assert cache["cache_tokens"] == 10
        assert cache["status"] == "hit"

    def test_timings_not_dict(self):
        info = {
            "timings": "not a dict",
            "usage": {"prompt_tokens_details": {"cached_tokens": 5}},
        }
        cache = normalize_cache_telemetry(info)
        assert cache["cache_tokens"] == 5
        assert cache["source"] == "usage.prompt_tokens_details.cached_tokens"
