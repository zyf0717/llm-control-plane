import httpx
import pytest

from src.dashboard.workflow_client import _raise_for_status


def test_raise_for_status_surfaces_fastapi_detail():
    request = httpx.Request("POST", "http://proxy.local/workflows/example/runs")
    response = httpx.Response(
        400,
        json={"detail": "missing required workflow params: latest_user_prompt"},
        request=request,
    )

    with pytest.raises(RuntimeError, match="latest_user_prompt"):
        _raise_for_status(response)
