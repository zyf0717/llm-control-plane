from __future__ import annotations

import json
from typing import Any

from shiny import ui


def format_graph_choices(graphs: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(graph.get("id")): str(graph.get("name") or graph.get("id"))
        for graph in graphs
        if str(graph.get("id") or "").strip()
    }


def format_graph_run_choices(runs: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(run.get("run_id")): (
            f"{run.get('graph_id') or ''} | {run.get('status') or 'unknown'}"
        )
        for run in runs
        if str(run.get("run_id") or "").strip()
    }


def format_graph_spec(spec: dict[str, Any] | None) -> ui.Tag:
    if not spec:
        return ui.card(ui.markdown("**No graph selected**"))
    return ui.card(
        ui.markdown(
            "\n".join(
                [
                    f"**{_cell(spec.get('name') or spec.get('id'))}**",
                    "",
                    f"- ID: `{_cell(spec.get('id'))}`",
                    f"- Ref: `{_cell(spec.get('graph_ref'))}`",
                    f"- Description: {_cell(spec.get('description'))}",
                ]
            )
        )
    )


def format_graph_run_details(snapshot: dict[str, Any] | None) -> ui.Tag:
    if not snapshot:
        return ui.card(ui.markdown("**No graph run selected**"))
    run = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
    events = snapshot.get("events") if isinstance(snapshot.get("events"), list) else []
    return ui.div(
        ui.card(
            ui.markdown(
                "\n".join(
                    [
                        f"**Run `{_cell(run.get('run_id'))}`**",
                        "",
                        f"- Graph: `{_cell(run.get('graph_id'))}`",
                        f"- Thread: `{_cell(run.get('thread_id'))}`",
                        f"- Status: `{_cell(run.get('status'))}`",
                        f"- Updated: `{_cell(run.get('updated_at'))}`",
                        f"- Events: `{len(events)}`",
                    ]
                )
            )
        ),
        ui.card(
            ui.tags.pre(
                json.dumps(snapshot, indent=2, default=str),
                style="white-space: pre-wrap;",
            )
        ),
    )


def graph_input_template(spec: dict[str, Any] | None) -> str:
    schema = spec.get("input_schema") if isinstance(spec, dict) else {}
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    if not isinstance(properties, dict) or not properties:
        return "{}"
    return json.dumps({str(key): "" for key in properties}, indent=2)


def _cell(value: Any) -> str:
    text = str(value or "").strip()
    return text or "-"

