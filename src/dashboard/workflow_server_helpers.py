from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable

TERMINAL_WORKFLOW_STATUSES = {"completed", "failed", "cancelled"}
WORKFLOW_RUN_MAX_STEPS = 100
WORKFLOW_CHAT_PROMPT_PARAM_NAMES = ("latest_user_prompt", "goal", "question")


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


def build_workflow_chat_params(
    spec: dict[str, Any] | None,
    *,
    latest_user_prompt: str,
    conversation_context: str = "",
    context: str = "",
    uploaded_context: str = "",
) -> dict[str, str]:
    keys = _workflow_param_keys(spec)
    params: dict[str, str] = {}
    for key in keys:
        if key in WORKFLOW_CHAT_PROMPT_PARAM_NAMES:
            params[key] = str(latest_user_prompt or "")
        elif key == "conversation_context":
            params[key] = str(conversation_context or "")
        elif key == "context":
            params[key] = str(context or "")
        elif key == "uploaded_context":
            params[key] = str(uploaded_context or "")
    return params


def build_workflow_chat_run_payload(
    spec: dict[str, Any] | None,
    *,
    latest_user_prompt: str,
    conversation_context: str = "",
    context: str = "",
    uploaded_context: str = "",
    endpoint: str,
    reasoning_effort: str = "",
    convo_id: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "params": build_workflow_chat_params(
            spec,
            latest_user_prompt=latest_user_prompt,
            conversation_context=conversation_context,
            context=context,
            uploaded_context=uploaded_context,
        ),
        "endpoint": str(endpoint or "").strip(),
    }
    reasoning = str(reasoning_effort or "").strip()
    if reasoning:
        payload["reasoning_effort"] = reasoning
    conversation_id = str(convo_id or "").strip()
    if conversation_id:
        payload["convo_id"] = conversation_id
    return payload


def format_workflow_conversation_context(history: list[dict[str, Any]] | None) -> str:
    lines: list[str] = []
    for message in history or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role == "system" or not role or not isinstance(content, str):
            continue
        text = content.strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def workflow_chat_response_text(snapshot: dict[str, Any] | None) -> str:
    run = snapshot.get("run") if isinstance(snapshot, dict) else {}
    run = run if isinstance(run, dict) else {}
    status = str(run.get("status") or "unknown").strip() or "unknown"
    run_id = str(run.get("run_id") or "").strip()
    workflow_id = str(run.get("workflow_id") or "").strip()

    if status == "completed":
        text = _last_completed_step_text(snapshot)
        if text:
            return text
        text = _last_artifact_text(snapshot)
        if text:
            return text
        return _workflow_status_message(
            workflow_id=workflow_id,
            run_id=run_id,
            detail="completed, but no text output was produced.",
        )

    if status in {"failed", "cancelled"}:
        detail = _first_failed_step_error(snapshot)
        suffix = f" {detail}" if detail else ""
        return _workflow_status_message(
            workflow_id=workflow_id,
            run_id=run_id,
            detail=f"ended with status `{status}`.{suffix}",
        )

    return _workflow_status_message(
        workflow_id=workflow_id,
        run_id=run_id,
        detail=f"is still in progress with status `{status}`.",
    )


def workflow_chat_run_info(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    run = snapshot.get("run") if isinstance(snapshot, dict) else {}
    run = run if isinstance(run, dict) else {}
    return {
        "workflow": {
            "run_id": str(run.get("run_id") or ""),
            "workflow_id": str(run.get("workflow_id") or ""),
            "status": str(run.get("status") or ""),
        }
    }


def format_workflow_intermediate_content(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    parsed = _parse_json_content(text)
    if parsed is None:
        return text
    if isinstance(parsed, dict):
        return "\n".join(
            f"- {key}: {_compact_json_value(value)}"
            for key, value in parsed.items()
            if value not in (None, "", [], {})
        )
    if isinstance(parsed, list):
        return "\n".join(f"- {_compact_json_value(item)}" for item in parsed)
    return str(parsed)


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


def _workflow_param_keys(spec: dict[str, Any] | None) -> list[str]:
    if not isinstance(spec, dict):
        return []
    schema = spec.get("params_schema")
    if not isinstance(schema, dict):
        return []
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
    return keys


def _parse_json_content(text: str) -> Any:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].strip().startswith("```"):
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _compact_json_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _last_completed_step_text(snapshot: dict[str, Any] | None) -> str:
    steps = snapshot.get("steps") if isinstance(snapshot, dict) else []
    if not isinstance(steps, list):
        return ""
    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("status") != "completed":
            continue
        output = step.get("output_json")
        if not isinstance(output, dict):
            continue
        text = str(output.get("text") or "").strip()
        if text:
            return text
    return ""


def _last_artifact_text(snapshot: dict[str, Any] | None) -> str:
    artifacts = snapshot.get("artifacts") if isinstance(snapshot, dict) else []
    if not isinstance(artifacts, list):
        return ""
    for artifact in reversed(artifacts):
        if not isinstance(artifact, dict):
            continue
        text = str(artifact.get("content_text") or "").strip()
        if text:
            return text
    return ""


def _first_failed_step_error(snapshot: dict[str, Any] | None) -> str:
    steps = snapshot.get("steps") if isinstance(snapshot, dict) else []
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, dict) or step.get("status") != "failed":
            continue
        step_id = str(step.get("step_id") or "unknown").strip()
        error = str(step.get("error") or "").strip()
        if error:
            return f"Failed step `{step_id}`: {error}"
        return f"Failed step `{step_id}`."
    return ""


def _workflow_status_message(
    *,
    workflow_id: str,
    run_id: str,
    detail: str,
) -> str:
    label = "Workflow"
    if workflow_id:
        label += f" `{workflow_id}`"
    if run_id:
        label += f" run `{run_id}`"
    return f"{label} {detail}"
