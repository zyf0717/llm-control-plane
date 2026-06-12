import json
import re
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
    return ui.card(
        _format_run_header(run),
        ui.tags.div(
            f"{summary['completed']} / {summary['total']} completed",
            style=(
                "font-size: 0.875rem; color: var(--bs-secondary-color); "
                "margin-bottom: 0.5rem;"
            ),
        ),
        _format_step_picker(run, steps, artifacts_by_step),
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
    return ui.card(
        ui.markdown("**Artifacts**"),
        ui.accordion(*panels, multiple=True, open=False),
    )


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


def _format_step_picker(
    run: dict[str, Any],
    steps: list[dict[str, Any]],
    artifacts_by_step: dict[str, list[dict[str, Any]]],
) -> ui.Tag:
    if not steps:
        return ui.markdown("No step rows found.")

    prefix = _safe_dom_id(str(run.get("run_id") or "workflow"))
    buttons = []
    panels = []
    for index, step in enumerate(steps):
        step_id = str(step.get("step_id") or f"step-{index + 1}")
        panel_id = f"{prefix}-step-panel-{index}"
        artifact_count = len(artifacts_by_step.get(step_id, []))
        buttons.append(
            ui.tags.button(
                _format_step_status_box(step, artifact_count),
                type="button",
                class_="dashboard-step-button",
                style=f"--dashboard-step-color: {_status_color(str(step.get('status') or 'unknown'))};",
                **{
                    "data-step-panel": panel_id,
                    "aria-controls": panel_id,
                    "aria-expanded": "false",
                },
            )
        )
        panels.append(
            ui.tags.div(
                format_step_detail(step),
                id=panel_id,
                class_="dashboard-step-panel",
                hidden=True,
                **{"data-step-panel-id": panel_id},
            )
        )

    return ui.tags.div(
        ui.tags.div(*buttons, class_=f"dashboard-step-row {_step_grid_class(len(steps))}"),
        ui.tags.div(*panels, class_="dashboard-step-panels"),
        class_="dashboard-step-picker",
    )


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
                style="color: var(--bs-body-color); font-weight: 700;",
            ),
            style=(
                "display: flex; gap: 0.5rem; align-items: center; "
                "min-width: 0;"
            ),
        ),
        ui.tags.div(
            f"Started {started}",
            style=(
                "font-size: 0.75rem; color: var(--bs-secondary-color); "
                "margin-top: 0.35rem;"
            ),
        ),
        ui.tags.div(
            f"Done {completed}",
            style="font-size: 0.75rem; color: var(--bs-secondary-color);",
        ),
        ui.tags.div(
            f"{output_flag} | {artifact_label}",
            style="font-size: 0.75rem; color: var(--bs-secondary-color);",
        ),
    ]
    if error:
        blocks.append(
            ui.tags.div(
                f"Error: {error}",
                style=(
                    "font-size: 0.875rem; color: var(--bs-danger); "
                    "margin-top: 0.25rem;"
                ),
            )
        )

    return ui.tags.div(
        *blocks,
        class_="dashboard-step-box",
        style=(
            f"border-left: 0.25rem solid {_status_color(status)}; "
            "padding: 0.5rem 0.75rem; "
            "background: var(--bs-body-bg); "
            "width: 100%;"
        ),
    )


def format_step_detail(step: dict[str, Any]) -> ui.Tag:
    error = str(step.get("error") or "").strip()
    boxes = [
        _format_step_detail_box("Input", step.get("input_json")),
        _format_step_detail_box("Output", step.get("output_json")),
    ]
    if error:
        boxes.append(
            ui.tags.div(
                f"Error: {error}",
                style=(
                    "grid-column: 1 / -1; color: var(--bs-danger); "
                    "font-size: 0.875rem;"
                ),
            )
        )
    return ui.tags.div(*boxes, class_="dashboard-step-detail-grid")


def _format_step_detail_box(label: str, value: Any) -> ui.Tag:
    return ui.tags.div(
        ui.tags.div(label, class_="dashboard-step-detail-title"),
        ui.tags.div(
            (
                ui.tags.pre(
                    json.dumps(value, indent=2, ensure_ascii=False),
                    class_="dashboard-trace-json",
                )
                if value is not None
                else ui.markdown("No data.")
            ),
            class_="dashboard-step-detail-body",
        ),
        class_="dashboard-step-detail-box",
    )


def _step_grid_class(step_count: int) -> str:
    if step_count <= 1:
        return "cols-1"
    if step_count == 2:
        return "cols-2"
    if step_count in {3, 5, 6}:
        return "cols-3"
    return "cols-4"


def _safe_dom_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return text or "workflow"


def _compact_value(value: Any) -> str:
    text = str(value or "").strip()
    return text or "-"


def _status_color(status: str) -> str:
    return {
        "completed": "var(--bs-success)",
        "running": "var(--bs-primary)",
        "failed": "var(--bs-danger)",
        "pending": "var(--bs-secondary)",
        "skipped": "var(--bs-warning)",
    }.get(status, "var(--bs-secondary)")


def _status_text_color(status: str) -> str:
    return "var(--bs-dark)" if status == "skipped" else "var(--bs-white)"
