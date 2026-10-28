"""Exercise the shipped workflow templates through the shared LLM wire format."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.responses import JSONResponse, StreamingResponse

from src.orchestrator import upstream_proxy
from src.orchestrator.runtime.clients import ProxyRuntimeLLMClient
from src.orchestrator.workflow.registry import WorkflowRegistry
from src.orchestrator.workflow.step_executor import WorkflowStepExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow",
    [
        "repo_context",
        "threaded_rag",
        "threaded_search",
        "url_retrieval",
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_all_workflow_model_steps_put_context_before_task(
    monkeypatch, workflow, stream
):
    registry = WorkflowRegistry()
    registry.load()
    spec = registry.get(workflow)
    params = {
        "latest_user_prompt": "CURRENT_REQUEST",
        "question": "CURRENT_REQUEST",
        "goal": "CURRENT_REQUEST",
        "repo_name": "repo",
        "thread_briefing": "OLD_HISTORY",
        "manual_source_text": "FIXED_INSTRUCTIONS",
        "uploaded_source_text": "FIXED_DOCUMENT",
        "conversation_messages": [
            {"role": "user", "content": "OLD_HISTORY"},
            {"role": "assistant", "content": "OLD_REPLY"},
        ],
    }
    run = SimpleNamespace(
        run_id="test",
        workflow_id=workflow,
        params=params,
        endpoint="node",
        conversation_id="thread",
        reasoning_effort="none",
        retrieval_endpoint=None,
    )
    responses = {
        "plan_search_queries": {
            "rerank_query": "search subject",
            "queries": ["search subject"],
            "reason": "why",
            "source_preferences": [],
        },
        "plan_rag_prompt": {
            "retrieval_queries": ["subject"],
            "retrieval_prompt": "answer subject",
            "reason": "why",
        },
        "plan_repo_context_query": {"query": "Find code"},
    }
    executor = WorkflowStepExecutor(ProxyRuntimeLLMClient())
    for step in spec.steps:
        if step.kind != "llm":
            continue
        text = json.dumps(responses[step.id]) if step.id in responses else "answer"

        async def chunks():
            yield (
                "data: "
                + json.dumps({"choices": [{"delta": {"content": text}}]})
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        async def complete(body, *args, **kwargs):
            return (
                StreamingResponse(chunks())
                if body.get("stream")
                else JSONResponse({"choices": [{"message": {"content": text}}]})
            )

        proxy = AsyncMock(side_effect=complete)
        monkeypatch.setattr(upstream_proxy, "complete_request", proxy)
        inputs = {"params": params, "outputs": {}}
        if stream:
            _ = [
                event
                async for event in executor.stream(
                    run, spec, step, inputs, stream_llm=True
                )
            ]
        else:
            await executor.execute(run, spec, step, inputs)
        messages = proxy.call_args.args[0]["messages"]
        assert [m["content"] for m in messages[:-1]] == [
            "Supplied context:\nFIXED_INSTRUCTIONS",
            "Uploaded source:\nFIXED_DOCUMENT",
            "OLD_HISTORY",
            "OLD_REPLY",
        ]
        assert messages[2:-1] == params["conversation_messages"]
        for value in ("OLD_HISTORY", "FIXED_INSTRUCTIONS", "FIXED_DOCUMENT"):
            assert value not in messages[-1]["content"]


@pytest.mark.asyncio
async def test_workflow_structured_repair_retains_context(monkeypatch):
    registry = WorkflowRegistry()
    registry.load()
    spec = registry.get("repo_context")
    step = spec.steps[0]
    params = {
        "latest_user_prompt": "find validation",
        "repo_name": "repo",
        "thread_briefing": "OLD_HISTORY",
    }
    run = SimpleNamespace(
        params=params,
        endpoint="node",
        conversation_id="thread",
        reasoning_effort="none",
        retrieval_endpoint=None,
    )
    proxy = AsyncMock(
        side_effect=[
            JSONResponse({"choices": [{"message": {"content": text}}]})
            for text in ("invalid", '{"query": "Find validation"}')
        ]
    )
    monkeypatch.setattr(upstream_proxy, "complete_request", proxy)
    await WorkflowStepExecutor(ProxyRuntimeLLMClient()).execute(
        run, spec, step, {"params": params}
    )
    assert len(proxy.await_args_list) == 2
    assert all(
        call.args[0]["messages"][0]["content"]
        == "Prior conversation context:\nOLD_HISTORY"
        for call in proxy.await_args_list
    )
