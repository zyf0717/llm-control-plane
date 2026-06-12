from src.dashboard.workflow_formatters import (
    format_workflow_choices,
    format_workflow_run_choices,
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
