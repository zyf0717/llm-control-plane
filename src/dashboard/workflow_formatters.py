import json
from typing import Any

from shiny import ui


def format_workflow_choices(workflows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(workflow.get("id")): (
            f"{workflow.get('name') or workflow.get('id')} "
            f"({workflow.get('version') or 'unknown'})"
        )
        for workflow in workflows
        if str(workflow.get("id") or "").strip()
    }


def format_workflow_run_choices(runs: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(run.get("run_id")): (
            f"{run.get('updated_at') or 'unknown'} | "
            f"{run.get('status') or 'unknown'} | {run.get('workflow_id') or ''}"
        )
        for run in runs
        if str(run.get("run_id") or "").strip()
    }


def format_workflow_spec(spec: dict[str, Any] | None) -> ui.Tag:
    if not isinstance(spec, dict) or not spec:
        return ui.card(ui.markdown("**No workflow selected**"))

    steps = spec.get("steps") if isinstance(spec.get("steps"), list) else []
    lines = [
        f"**{spec.get('name') or spec.get('id')}**",
        str(spec.get("description") or ""),
        f"`{len(steps)} steps`",
    ]
    return ui.card(ui.markdown("\n\n".join(line for line in lines if line)))


def format_workflow_run_table(runs: list[dict[str, Any]]) -> ui.Tag:
    if not runs:
        return ui.card(ui.markdown("**No workflow runs found**"))
    rows = [
        "| Run | Workflow | Status | Step | Updated | Convo |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs[:50]:
        rows.append(
            "| "
            + " | ".join(
                [
                    _cell(run.get("run_id")),
                    _cell(run.get("workflow_id")),
                    _cell(run.get("status")),
                    _cell(run.get("current_step_id") or ""),
                    _cell(run.get("updated_at")),
                    _cell(run.get("convo_id")),
                ]
            )
            + " |"
        )
    return ui.card(ui.markdown("\n".join(rows)))


def workflow_progress_summary(snapshot: dict[str, Any] | None) -> dict[str, int]:
    steps = _snapshot_steps(snapshot)
    total = len(steps)
    completed = sum(1 for step in steps if step.get("status") == "completed")
    failed = sum(1 for step in steps if step.get("status") == "failed")
    running = sum(1 for step in steps if step.get("status") == "running")
    pending = sum(1 for step in steps if step.get("status") == "pending")
    percent = round((completed / total) * 100) if total else 0
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "running": running,
        "pending": pending,
        "percent": percent,
    }


def group_workflow_artifacts_by_step(
    snapshot: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    artifacts = []
    if isinstance(snapshot, dict) and isinstance(snapshot.get("artifacts"), list):
        artifacts = snapshot["artifacts"]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        step_id = str(artifact.get("step_id") or "unassigned")
        grouped.setdefault(step_id, []).append(artifact)
    return grouped


def format_step_timeline(snapshot: dict[str, Any] | None) -> ui.Tag:
    if not isinstance(snapshot, dict):
        return ui.card(ui.markdown("**No run selected**"))
    run = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
    steps = _snapshot_steps(snapshot)
    summary = workflow_progress_summary(snapshot)
    artifacts_by_step = group_workflow_artifacts_by_step(snapshot)
    panels = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        status = str(step.get("status") or "unknown")
        step_id = str(step.get("step_id") or "step")
        artifact_count = len(artifacts_by_step.get(step_id, []))
        title = (
            f"{step_id} | {status} | "
            f"{artifact_count} artifact{'s' if artifact_count != 1 else ''}"
        )
        panels.append(
            ui.accordion_panel(
                title,
                format_step_output(step),
            )
        )
    if not panels:
        panels.append(ui.accordion_panel("No steps", ui.markdown("No step rows found.")))
    return ui.card(
        _format_run_header(run),
        ui.tags.div(
            f"{summary['completed']} / {summary['total']} completed",
            style="font-size: 0.875rem; color: #6c757d; margin-bottom: 0.5rem;",
        ),
        ui.div(
            *[
                _format_step_status_box(step, len(artifacts_by_step.get(str(step.get("step_id") or ""), [])))
                for step in steps
                if isinstance(step, dict)
            ],
            style=(
                "display: flex; flex-wrap: wrap; gap: 0.5rem; "
                "align-items: stretch; margin-bottom: 0.75rem;"
            ),
        ),
        ui.accordion(*panels, multiple=True, open=False),
    )


def format_artifacts(snapshot: dict[str, Any] | None) -> ui.Tag:
    artifacts_by_step = group_workflow_artifacts_by_step(snapshot)
    if not artifacts_by_step:
        return ui.card(ui.markdown("**Artifacts**\n\nNo artifacts yet."))
    panels = []
    for step_id, artifacts in artifacts_by_step.items():
        blocks = []
        for artifact in artifacts:
            name = str(artifact.get("name") or artifact.get("artifact_id") or "artifact")
            blocks.extend(
                [
                    ui.markdown(f"**{name}**"),
                    ui.tags.pre(
                        artifact.get("content_text")
                        or json.dumps(artifact.get("content_json"), indent=2),
                        class_="dashboard-trace-json",
                    ),
                ]
            )
        panels.append(
            ui.accordion_panel(
                f"{step_id} ({len(artifacts)})",
                ui.div(*blocks),
            )
        )
    return ui.card(ui.markdown("**Artifacts**"), ui.accordion(*panels, multiple=True))


def format_step_output(step: dict[str, Any]) -> ui.Tag:
    blocks = []
    if step.get("error"):
        blocks.append(ui.markdown(f"**Error:** {step['error']}"))
    for label, key in [("Input", "input_json"), ("Output", "output_json")]:
        value = step.get(key)
        if value is None:
            continue
        blocks.append(ui.markdown(f"**{label}**"))
        blocks.append(
            ui.tags.pre(
                json.dumps(value, indent=2, ensure_ascii=False),
                class_="dashboard-trace-json",
            )
        )
    return ui.div(*blocks) if blocks else ui.markdown("No output yet.")


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


def _format_run_header(run: dict[str, Any]) -> ui.Tag:
    return ui.tags.div(
        _metadata_row(
            ("Run", run.get("run_id")),
            ("Workflow", run.get("workflow_id")),
            ("Status", run.get("status")),
        ),
        _metadata_row(
            ("Current step", run.get("current_step_id")),
            ("Conversation", run.get("convo_id")),
        ),
        _metadata_row(
            ("Updated", run.get("updated_at")),
            ("Completed", run.get("completed_at")),
        ),
        style="display: grid; gap: 0.35rem; margin-bottom: 0.75rem;",
    )


def _metadata_row(*items: tuple[str, Any]) -> ui.Tag:
    return ui.tags.div(
        *[_metadata_item(label, value) for label, value in items],
        style="display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem;",
    )


def _metadata_item(label: str, value: Any) -> ui.Tag:
    text = _compact_value(value)
    return ui.tags.div(
        ui.tags.span(
            f"{label}: ",
            style="font-weight: 600;",
        ),
        ui.tags.code(text),
        style="min-width: 0;",
    )


def _snapshot_steps(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("steps"), list):
        return []
    return [step for step in snapshot["steps"] if isinstance(step, dict)]


def _format_step_status_box(step: dict[str, Any], artifact_count: int) -> ui.Tag:
    status = str(step.get("status") or "unknown")
    step_id = str(step.get("step_id") or "step")
    started = _compact_value(step.get("started_at"))
    completed = _compact_value(step.get("completed_at"))
    output_flag = "output" if isinstance(step.get("output_json"), dict) else "no output"
    artifact_label = f"{artifact_count} artifact{'s' if artifact_count != 1 else ''}"
    error = str(step.get("error") or "").strip()

    blocks: list[ui.Tag] = [
        ui.tags.div(
            ui.tags.span(
                status.upper(),
                style=(
                    f"background: {_status_color(status)}; "
                    f"color: {_status_text_color(status)}; "
                    "border-radius: 0.25rem; font-size: 0.75rem; "
                    "font-weight: 700; min-width: 5.5rem; padding: 0.25rem 0.45rem; "
                    "text-align: center;"
                ),
            ),
            ui.tags.strong(
                step_id,
                style="color: #212529; font-weight: 700;",
            ),
            style=(
                "display: flex; gap: 0.5rem; align-items: center; "
                "min-width: 0;"
            ),
        ),
        ui.tags.div(
            f"Started {started}",
            style="font-size: 0.75rem; color: #6c757d; margin-top: 0.35rem;",
        ),
        ui.tags.div(
            f"Done {completed}",
            style="font-size: 0.75rem; color: #6c757d;",
        ),
        ui.tags.div(
            f"{output_flag} | {artifact_label}",
            style="font-size: 0.75rem; color: #6c757d;",
        ),
    ]
    if error:
        blocks.append(
            ui.tags.div(
                f"Error: {error}",
                style="font-size: 0.875rem; color: #dc3545; margin-top: 0.25rem;",
            )
        )

    return ui.tags.div(
        *blocks,
        style=(
            f"border-left: 0.25rem solid {_status_color(status)}; "
            "padding: 0.5rem 0.75rem; "
            "background: #ffffff; border-top: 1px solid #dee2e6; "
            "border-right: 1px solid #dee2e6; border-bottom: 1px solid #dee2e6; "
            "flex: 1 1 13rem; max-width: 22rem; min-width: 13rem;"
        ),
    )


def _compact_value(value: Any) -> str:
    text = str(value or "").strip()
    return text or "-"


def _status_color(status: str) -> str:
    return {
        "completed": "#198754",
        "running": "#0d6efd",
        "failed": "#dc3545",
        "pending": "#adb5bd",
        "skipped": "#ffc107",
    }.get(status, "#adb5bd")


def _status_text_color(status: str) -> str:
    return "#212529" if status == "skipped" else "#ffffff"
