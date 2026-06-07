import json
from typing import Any, Callable, Dict, List, Optional


def format_history_json(history_payload: Any) -> str:
    """Render history payload as readable JSON for the History tab."""
    return json.dumps(history_payload, indent=2, ensure_ascii=False)


def format_hardware_info(
    model: Optional[Dict[str, Any]], routing_info: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Format hardware information for display."""
    hardware_info: List[str] = []

    if routing_info:
        for key in ["gpu", "vram", "soc", "cpu", "ram"]:
            value = routing_info.get(key)
            if value:
                hardware_info.append(f"{key}: {value}")

    if not hardware_info and model:
        for key in ["gpu", "vram", "soc", "cpu", "ram"]:
            value = model.get(key)
            if value:
                hardware_info.append(f"{key}: {value}")

    return hardware_info


def format_model_details(model: Dict[str, Any]) -> List[str]:
    """Format model details for display."""
    model_details: List[str] = []

    model_name = model.get("id")
    if model_name:
        model_details.append(f"model: {model_name}")

    for key in ["arch", "quantization", "compatibility_type", "state"]:
        value = model.get(key)
        if value:
            model_details.append(f"{key}: {value}")

    max_ctx = model.get("max_context_length")
    loaded_ctx = model.get("loaded_context_length")
    if max_ctx:
        model_details.append(f"max_context: {max_ctx}")
    if loaded_ctx:
        model_details.append(f"loaded_context: {loaded_ctx}")

    return model_details


def format_all_available_models(endpoint_data: Optional[Dict[str, Any]]) -> List[str]:
    """Format all available models for display when Auto is selected."""
    if not endpoint_data or "data" not in endpoint_data:
        return []

    models_list: List[str] = []
    for model in endpoint_data["data"]:
        model_details = format_model_details(model)
        if model_details:
            models_list.extend(model_details)
            models_list.append("")

    return models_list


def format_response_info(
    info: Dict[str, Any], fmt_func: Callable[[Any], str]
) -> List[str]:
    """Format response information (usage, stats, runtime) for display."""
    sections: List[str] = []
    for section_name in ("usage", "stats", "runtime"):
        section_data = info.get(section_name)
        if section_data and isinstance(section_data, dict):
            title = section_name.replace("_", " ").title()
            lines = [f"**{title}**"]
            _skip_detail_keys = {"prompt_tokens_details", "input_tokens_details"}
            for key, value in section_data.items():
                if section_name == "usage" and key in _skip_detail_keys:
                    continue
                lines.append(f"{key}: {fmt_func(value)}")
            sections.append("<br>".join(lines))
    return sections


def format_timings_info(
    info: Dict[str, Any], fmt_func: Callable[[Any], str]
) -> List[str]:
    timings = info.get("timings", {})
    if not timings:
        return []

    timings_info = [
        f"{key}: {fmt_func(value)}"
        for key, value in timings.items()
        if value is not None
    ]
    if not timings_info:
        return []
    return ["**Timings**<br>" + "<br>".join(timings_info)]


def format_cache_info(
    info: Dict[str, Any], fmt_func: Callable[[Any], str]
) -> List[str]:
    cache = info.get("cache")
    if not isinstance(cache, dict):
        return []

    lines = ["**KV Cache**"]

    status = cache.get("status")
    if status:
        lines.append(f"status: {status}")

    if cache.get("cache_tokens") is not None:
        lines.append(f"reused_prompt_tokens: {fmt_func(cache['cache_tokens'])}")

    if cache.get("processed_prompt_tokens") is not None:
        lines.append(
            f"new_prompt_tokens_processed: {fmt_func(cache['processed_prompt_tokens'])}"
        )

    if cache.get("prompt_tokens") is not None:
        lines.append(f"logical_prompt_tokens: {fmt_func(cache['prompt_tokens'])}")

    if cache.get("cache_ratio") is not None:
        lines.append(f"cache_reuse_ratio: {cache['cache_ratio']:.1%}")

    if cache.get("generated_tokens") is not None:
        lines.append(f"generated_tokens: {fmt_func(cache['generated_tokens'])}")

    if cache.get("approx_context_tokens") is not None:
        lines.append(
            f"approx_context_tokens: {fmt_func(cache['approx_context_tokens'])}"
        )

    if cache.get("source"):
        lines.append(f"source: {cache['source']}")

    if cache.get("fields_disagree"):
        lines.append("warning: usage/timings cache fields disagree")

    if cache.get("reason"):
        lines.append(f"interpretation: {cache['reason']}")

    return ["<br>".join(lines)]
