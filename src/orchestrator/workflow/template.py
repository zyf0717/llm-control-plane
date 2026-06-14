from __future__ import annotations

import json
import re
from typing import Any


_TEMPLATE_PATTERN = re.compile(
    r"{{\s*(?:(json)\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*}}"
)


def render_template(template: str, data: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value: Any = data
        for part in match.group(2).split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return ""
        if match.group(1) == "json":
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        return "" if value is None else str(value)

    return _TEMPLATE_PATTERN.sub(replace, template)


def parse_json_text(text: str) -> Any:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    fenced = strip_json_code_fence(stripped)
    if fenced != stripped:
        stripped = fenced
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def strip_json_code_fence(text: str) -> str:
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else text
