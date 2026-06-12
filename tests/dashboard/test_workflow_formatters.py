from src.dashboard.workflow_formatters import (
    format_workflow_choices,
    format_artifacts,
    format_step_timeline,
    format_workflow_run_choices,
    group_workflow_artifacts_by_step,
    workflow_progress_summary,
)


def test_format_workflow_choices_uses_id_keys():
    choices = format_workflow_choices(
        [{"id": "implementation_plan", "name": "Implementation Plan", "version": "0.1.0"}]
    )

    assert choices == {"implementation_plan": "Implementation Plan (0.1.0)"}


def test_format_workflow_run_choices_uses_run_id_keys():
    choices = format_workflow_run_choices(
        [
            {
                "run_id": "wf_123",
                "workflow_id": "implementation_plan",
                "status": "completed",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    )

    assert choices["wf_123"].endswith("| completed | implementation_plan")


def test_workflow_progress_summary_counts_statuses_and_percent():
    summary = workflow_progress_summary(
        {
            "steps": [
                {"status": "completed"},
                {"status": "completed"},
                {"status": "running"},
                {"status": "failed"},
            ]
        }
    )

    assert summary == {
        "total": 4,
        "completed": 2,
        "failed": 1,
        "running": 1,
        "pending": 0,
        "percent": 50,
    }


def test_group_workflow_artifacts_by_step_groups_unassigned_artifacts():
    grouped = group_workflow_artifacts_by_step(
        {
            "artifacts": [
                {"step_id": "first", "artifact_id": "a1"},
                {"step_id": "first", "artifact_id": "a2"},
                {"artifact_id": "a3"},
            ]
        }
    )

    assert [artifact["artifact_id"] for artifact in grouped["first"]] == ["a1", "a2"]
    assert grouped["unassigned"][0]["artifact_id"] == "a3"


def test_format_step_timeline_renders_progress_status_error_and_artifact_counts():
    rendered = str(
        format_step_timeline(
            {
                "run": {
                    "run_id": "wf_123",
                    "workflow_id": "implementation_plan",
                    "status": "failed",
                    "current_step_id": "failed_step",
                    "convo_id": "wf_123_convo",
                    "updated_at": "2026-06-12T00:00:00+00:00",
                },
                "steps": [
                    {
                        "step_id": "done",
                        "status": "completed",
                        "input_json": {"current_step": {"name": "Done step"}},
                        "output_json": {"text": "ok"},
                        "started_at": "2026-06-12T00:00:00+00:00",
                        "completed_at": "2026-06-12T00:01:00+00:00",
                    },
                    {
                        "step_id": "failed_step",
                        "status": "failed",
                        "error": "boom",
                    },
                ],
                "artifacts": [
                    {"step_id": "done", "artifact_id": "a1", "content_text": "artifact"}
                ],
            }
        )
    )

    assert "wf_123" in rendered
    assert "1 / 2 completed" in rendered
    assert "COMPLETED" in rendered
    assert "FAILED" in rendered
    assert "Error: boom" in rendered
    assert "1 artifact" in rendered
    assert "- Done step" not in rendered


def test_format_artifacts_groups_panels_by_step():
    rendered = str(
        format_artifacts(
            {
                "artifacts": [
                    {"step_id": "first", "name": "one", "content_text": "first text"},
                    {"step_id": "first", "name": "two", "content_json": {"value": 2}},
                    {"step_id": "second", "name": "three", "content_text": "second text"},
                ]
            }
        )
    )

    assert "first (2)" in rendered
    assert "second (1)" in rendered
    assert "first text" in rendered
    assert '"value": 2' in rendered


def test_format_step_timeline_handles_missing_snapshot():
    assert "No run selected" in str(format_step_timeline(None))


def test_format_step_timeline_renders_empty_current_step_as_placeholder():
    rendered = str(
        format_step_timeline(
            {
                "run": {
                    "run_id": "wf_80bce9409eec",
                    "workflow_id": "contextual_search",
                    "status": "pending",
                    "current_step_id": "",
                    "convo_id": "wf_80bce9409eec_convo",
                    "updated_at": "2026-06-12T03:34:02.297810+00:00",
                    "completed_at": None,
                },
                "steps": [],
                "artifacts": [],
            }
        )
    )

    assert "Current step:" in rendered
    assert "<code>-</code>" in rendered
    assert "Conversation:" in rendered
    assert "wf_80bce9409eec_convo" in rendered
