import json
from unittest.mock import AsyncMock

import pytest
from starlette.responses import JSONResponse, StreamingResponse

from src.orchestrator import upstream_proxy
from src.orchestrator.runtime.clients import ProxyRuntimeLLMClient
from src.llm_metrics import capture_llm_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_workflow_model_metadata_survives_without_entering_prompt(monkeypatch, stream):
    reported = {"model": "served-model", "usage": {"completion_tokens": 2},
                "model_info": {"quantization": "Q4_K_M"},
                "timings": {"prompt_per_second": 1200}}

    async def chunks():
        for part in [
            {"model": "served-model", "choices": [{"delta": {"content": "hello "}}]},
            {"model": "served-model", "choices": [{"delta": {"content": "world"}}]},
            {"choices": [], **reported},
        ]:
            yield "data: " + json.dumps(part) + "\n\n"
        yield "data: [DONE]\n\n"

    response = StreamingResponse(chunks()) if stream else JSONResponse({
        **reported, "choices": [{"message": {"content": "hello world"}}],
    })
    proxy = AsyncMock(return_value=response)
    monkeypatch.setattr(upstream_proxy, "proxy_request", proxy)
    client = ProxyRuntimeLLMClient()
    request = dict(endpoint="node", prompt="hello", conversation_id="workflow",
                   skip_conversation=True)
    with capture_llm_calls() as calls:
        if stream:
            events = [event async for event in client.stream_complete(**request)]
        else:
            result = await client.complete(**request)
    assert len(calls) == 1
    assert calls[0]["metadata"]["model"] == "served-model"
    assert calls[0]["elapsed"] >= 0
    assert (calls[0]["ttft"] is not None) == stream
    if stream:
        metadata = {}
        for event in events:
            metadata.update(event.get("metadata", {}))
        assert sum("model" in event.get("metadata", {}) for event in events) == 1
        assert "".join(event.get("content", "") for event in events) == "hello world"
    else:
        metadata = result["metadata"]
        assert result["text"] == "hello world"
    for key, value in reported.items():
        assert metadata[key] == value
    sent = await proxy.call_args.args[1].json()
    assert sent["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_upstream_http_error_is_a_failed_call(monkeypatch, stream):
    monkeypatch.setattr(upstream_proxy, "proxy_request", AsyncMock(
        return_value=JSONResponse({"error": "unavailable"}, status_code=503)))
    client = ProxyRuntimeLLMClient()
    with capture_llm_calls() as calls:
        with pytest.raises(RuntimeError, match="HTTP 503"):
            request = dict(endpoint="node", prompt="hi", conversation_id="test")
            if stream:
                _ = [event async for event in client.stream_complete(**request)]
            else:
                await client.complete(**request)
    assert len(calls) == 1
    assert calls[0]["status"] == "Failed"
    assert calls[0]["ttft"] is None
