import json
from typing import Any


def format_history_json(history_payload: Any) -> str:
    """Render history payload as readable JSON for the History tab."""
    return json.dumps(history_payload, indent=2, ensure_ascii=False)


def format_hardware_info(model: dict[str, Any] | None) -> list[str]:
    """Describe the selected endpoint's hardware independently of response telemetry."""
    if not model:
        return []
    return [f"{key}: {model[key]}" for key in ("gpu", "vram", "soc", "cpu", "ram")
            if model.get(key)]
