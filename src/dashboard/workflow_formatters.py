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


def format_step_timeline(snapshot: dict[str, Any] | None) -> ui.Tag:
    if not isinstance(snapshot, dict):
        return ui.card(ui.markdown("**No run selected**"))
    run = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
    steps = snapshot.get("steps") if isinstance(snapshot.get("steps"), list) else []
    panels = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        status = str(step.get("status") or "unknown")
        title = f"{step.get('step_id') or 'step'} | {status}"
        panels.append(
            ui.accordion_panel(
                title,
                format_step_output(step),
            )
        )
    if not panels:
        panels.append(ui.accordion_panel("No steps", ui.markdown("No step rows found.")))
    return ui.card(
        ui.markdown(
            "\n".join(
                [
                    f"**Run:** `{run.get('run_id') or ''}`",
                    f"**Status:** `{run.get('status') or ''}`",
                    f"**Conversation:** `{run.get('convo_id') or ''}`",
                ]
            )
        ),
        ui.accordion(*panels, multiple=True, open=False),
    )


def format_artifacts(snapshot: dict[str, Any] | None) -> ui.Tag:
    artifacts = []
    if isinstance(snapshot, dict) and isinstance(snapshot.get("artifacts"), list):
        artifacts = snapshot["artifacts"]
    if not artifacts:
        return ui.card(ui.markdown("**Artifacts**\n\nNo artifacts yet."))
    panels = [
        ui.accordion_panel(
            str(artifact.get("name") or artifact.get("artifact_id") or "artifact"),
            ui.tags.pre(
                artifact.get("content_text")
                or json.dumps(artifact.get("content_json"), indent=2),
                class_="dashboard-trace-json",
            ),
        )
        for artifact in artifacts
        if isinstance(artifact, dict)
    ]
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
