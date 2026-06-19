from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .common import llm_text, output_text, rerank, search_one, text, with_output


async def search_evidence(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    prompt = f"""{text(state.get("goal"))}

Prior thread context:
{text(state.get("thread_state_text"))}

Recent conversation context:
{text(state.get("recent_conversation_messages"))}

Additional context:
{text(state.get("manual_source_text"))}
{text(state.get("uploaded_source_text"))}
"""
    output = await search_one(prompt, config, count=5, use_query_refiner=True)
    return with_output(state, "search_evidence", output)


async def rerank_search_evidence(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    source_text = f"""Goal:
{text(state.get("goal"))}

Prior thread context:
{text(state.get("thread_state_text"))}

Recent conversation context:
{text(state.get("recent_conversation_messages"))}

Manual context:
{text(state.get("manual_source_text"))}

Uploaded context:
{text(state.get("uploaded_source_text"))}
"""
    output = await rerank(
        text(state.get("goal")),
        outputs.get("search_evidence") or {},
        config,
        source_text=source_text,
    )
    return with_output(state, "search_evidence", output)


async def target_assessment(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    prompt = f"""Goal:
{text(state.get("goal"))}

Prior thread context:
{text(state.get("thread_state_text"))}

Recent conversation context:
{text(state.get("recent_conversation_messages"))}

Manual context:
{text(state.get("manual_source_text"))}

Uploaded context:
{text(state.get("uploaded_source_text"))}

Search context:
{output_text(outputs.get("search_evidence"))}

Identify the target state, constraints, non-goals, risks, and missing information.
"""
    output = await llm_text(prompt, config)
    return with_output(state, "target_assessment", output)


async def architecture_plan(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    prompt = f"""Goal:
{text(state.get("goal"))}

Target assessment:
{output_text(outputs.get("target_assessment"))}

Produce the implementation architecture, module boundaries, data flow, and failure handling.
"""
    output = await llm_text(prompt, config)
    return with_output(state, "architecture_plan", output)


async def commit_plan(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    prompt = f"""Architecture plan:
{output_text(outputs.get("architecture_plan"))}

Produce a sequenced commit plan with tests, acceptance criteria, and rollout notes.
"""
    output = await llm_text(prompt, config)
    return with_output(state, "commit_plan", output, final=output["text"])


builder = StateGraph(dict)
builder.add_node("search_evidence", search_evidence)
builder.add_node("rerank_search_evidence", rerank_search_evidence)
builder.add_node("target_assessment", target_assessment)
builder.add_node("architecture_plan", architecture_plan)
builder.add_node("commit_plan", commit_plan)
builder.add_edge(START, "search_evidence")
builder.add_edge("search_evidence", "rerank_search_evidence")
builder.add_edge("rerank_search_evidence", "target_assessment")
builder.add_edge("target_assessment", "architecture_plan")
builder.add_edge("architecture_plan", "commit_plan")
builder.add_edge("commit_plan", END)
graph = builder.compile()
