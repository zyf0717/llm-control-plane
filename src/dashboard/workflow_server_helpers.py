from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable

TERMINAL_WORKFLOW_STATUSES = {"completed", "failed", "cancelled"}
WORKFLOW_RUN_MAX_STEPS = 100


def workflow_snapshot_status(snapshot: dict[str, Any] | None) -> str:
    run = snapshot.get("run") if isinstance(snapshot, dict) else {}
    if not isinstance(run, dict):
        return ""
    return str(run.get("status") or "").strip()


def workflow_snapshot_has_running_step(snapshot: dict[str, Any] | None) -> bool:
    steps = snapshot.get("steps") if isinstance(snapshot, dict) else []
    if not isinstance(steps, list):
        return False
    return any(
        isinstance(step, dict) and step.get("status") == "running" for step in steps
    )


def build_workflow_params_template(spec: dict[str, Any] | None) -> str:
    if not isinstance(spec, dict):
        return "{}"
    schema = spec.get("params_schema")
    if not isinstance(schema, dict):
        return "{}"
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    required_names = [str(item) for item in required] if isinstance(required, list) else []

    keys: list[str] = []
    for key in [*required_names, *properties.keys()]:
        text = str(key or "").strip()
        if text and text not in keys:
            keys.append(text)

    return json.dumps({key: "" for key in keys}, indent=2)


async def advance_workflow_to_terminal(
    run_id: str,
    advance_once: Callable[[str], Awaitable[dict[str, Any]]],
    *,
    after_step: Callable[[dict[str, Any], int], Awaitable[None]] | None = None,
    max_steps: int = WORKFLOW_RUN_MAX_STEPS,
) -> dict[str, Any]:
    budget = max(1, int(max_steps))
    snapshot: dict[str, Any] = {}
    for step_number in range(1, budget + 1):
        snapshot = await advance_once(run_id)
        if after_step is not None:
            await after_step(snapshot, step_number)
        status = workflow_snapshot_status(snapshot)
        if (
            status in TERMINAL_WORKFLOW_STATUSES
            or workflow_snapshot_has_running_step(snapshot)
        ):
            return snapshot
    return snapshot


def build_uploaded_file_context(uploaded_files: Any) -> str:
    if not uploaded_files:
        return ""

    file_contents = []
    for file_info in uploaded_files:
        try:
            path = file_info.get("datapath")
            if not path or not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
            file_contents.append(
                f"--- File: {file_info.get('name', 'upload')} ---\n{content}"
            )
        except Exception as exc:
            filename = file_info.get("name", "unknown")
            file_contents.append(f"--- Error reading {filename}: {str(exc)} ---")
    return "\n\n".join(file_contents)


def merge_uploaded_context(params: dict[str, Any], uploaded_context: str) -> dict[str, Any]:
    context = str(uploaded_context or "").strip()
    if not context:
        return params

    merged = dict(params)
    existing = str(merged.get("uploaded_context") or "").strip()
    merged["uploaded_context"] = "\n\n".join(
        item for item in [existing, context] if item
    )
    return merged
