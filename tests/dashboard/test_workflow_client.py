import httpx
import pytest

from src.dashboard.workflow_client import _decode_sse_event, _raise_for_status


def test_raise_for_status_surfaces_fastapi_detail():
    request = httpx.Request("POST", "http://proxy.local/workflows/example/runs")
    response = httpx.Response(
        400,
        json={"detail": "missing required workflow params: latest_user_prompt"},
        request=request,
    )

    with pytest.raises(RuntimeError, match="latest_user_prompt"):
        _raise_for_status(response)


def test_decode_sse_event_uses_event_type_when_missing_from_payload():
    event = _decode_sse_event("step_delta", ['{"content": "hello"}'])

    assert event == {"type": "step_delta", "content": "hello"}
