import json
from typing import Any


def format_history_json(history_payload: Any) -> str:
    """Render history payload as readable JSON for the History tab."""
    return json.dumps(history_payload, indent=2, ensure_ascii=False)
