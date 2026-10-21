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
    monkeypatch.setattr(upstream_proxy, "complete_request", proxy)
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
    sent = proxy.call_args.args[0]
    assert sent["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_upstream_http_error_is_a_failed_call(monkeypatch, stream):
    monkeypatch.setattr(upstream_proxy, "complete_request", AsyncMock(
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_context_is_a_reusable_prefix_and_task_stays_last(monkeypatch, stream):
    async def chunks():
        yield 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'

    def response():
        return (
            StreamingResponse(chunks())
            if stream
            else JSONResponse({"choices": [{"message": {"content": "answer"}}]})
        )

    proxy = AsyncMock(side_effect=lambda *a, **kw: response())
    monkeypatch.setattr(upstream_proxy, "complete_request", proxy)
    context = {
        "manual_source_text": "stable instructions",
        "uploaded_source_text": "stable document",
        "thread_state_text": "summary",
        "recent_conversation_messages": "earlier turns",
        "thread_briefing": "summary\n\nearlier turns",
    }
    client = ProxyRuntimeLLMClient()
    for prompt in ("first question + evidence", "second question + new evidence"):
        request = dict(
            endpoint="node",
            prompt=prompt,
            context=context,
            conversation_id="thread",
            skip_conversation=True,
        )
        if stream:
            _ = [event async for event in client.stream_complete(**request)]
        else:
            await client.complete(**request)
    first, second = [call.args[0]["messages"] for call in proxy.await_args_list]
    assert first[:-1] == second[:-1]
    assert first[-1]["content"] == "first question + evidence"
    assert second[-1]["content"] == "second question + new evidence"
    assert [m["role"] for m in first] == ["user"] * 4
    combined = "\n".join(m["content"] for m in first)
    for text in ("stable instructions", "stable document", "summary", "earlier turns"):
        assert combined.count(text) == 1


@pytest.mark.asyncio
async def test_distinct_thread_context_fields_are_not_lost(monkeypatch):
    proxy = AsyncMock(
        return_value=JSONResponse({"choices": [{"message": {"content": "answer"}}]})
    )
    monkeypatch.setattr(upstream_proxy, "complete_request", proxy)
    await ProxyRuntimeLLMClient().complete(
        endpoint="node",
        prompt="task",
        conversation_id="thread",
        context={
            "thread_state_text": "derived summary",
            "recent_conversation_messages": "recent turns",
            "thread_briefing": "additional briefing",
        },
    )
    content = proxy.call_args.args[0]["messages"][0]["content"]
    assert all(
        text in content
        for text in ("derived summary", "recent turns", "additional briefing")
    )


@pytest.mark.asyncio
async def test_graph_json_repair_preserves_context(monkeypatch):
    from src.graphs import common

    client = AsyncMock()
    client.complete.side_effect = [
        {"text": "invalid", "metadata": {}},
        {"text": '{"ok": true}', "metadata": {}},
    ]
    monkeypatch.setattr(common, "LLM_CLIENT", client)
    context = {"thread_briefing": "prior turns"}
    output = await common.llm_json(
        "current task",
        {},
        schema={"type": "object", "required": ["ok"]},
        context=context,
    )
    assert output["json"] == {"ok": True}
    assert len(client.complete.await_args_list) == 2
    assert all(
        call.kwargs["context"] == context for call in client.complete.await_args_list
    )


@pytest.mark.asyncio
async def test_growing_history_keeps_existing_message_boundaries(monkeypatch):
    from src.dashboard.workflow_server_helpers import build_workflow_chat_run_payload

    proxy = AsyncMock(
        return_value=JSONResponse({"choices": [{"message": {"content": "answer"}}]})
    )
    monkeypatch.setattr(upstream_proxy, "complete_request", proxy)
    history = [
        {"role": "system", "content": "hidden"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    for prompt in ("first task", "follow-up task"):
        payload = build_workflow_chat_run_payload(
            {"params_schema": {"properties": {"question": {"type": "string"}}}},
            latest_user_prompt=prompt,
            conversation_messages=list(history),
            endpoint="node",
        )
        await ProxyRuntimeLLMClient().complete(
            endpoint="node",
            prompt=prompt,
            conversation_id="thread",
            context=payload["params"],
        )
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "answer"},
            ]
        )
    first, second = [call.args[0]["messages"] for call in proxy.await_args_list]
    assert all(message["role"] != "system" for message in second)
    assert second[: len(first)] == first
    assert second[-1] == {"role": "user", "content": "follow-up task"}
    assert second[-2] == {"role": "assistant", "content": "answer"}
