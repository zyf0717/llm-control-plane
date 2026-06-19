from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .common import llm_text, output_text, rerank, search_one, text, with_output


async def search_evidence(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    prompt = f"""{text(state.get("question"))}

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
    source_text = f"""Question:
{text(state.get("question"))}

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
        text(state.get("question")),
        outputs.get("search_evidence") or {},
        config,
        source_text=source_text,
    )
    return with_output(state, "search_evidence", output)


async def synthesize_brief(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    prompt = f"""Question:
{text(state.get("question"))}

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

Produce a concise research brief with key findings, uncertainty, and recommended next steps.
"""
    output = await llm_text(prompt, config)
    return with_output(
        state,
        "research_brief",
        output,
        final=output["text"],
    )


builder = StateGraph(dict)
builder.add_node("search_evidence", search_evidence)
builder.add_node("rerank_search_evidence", rerank_search_evidence)
builder.add_node("synthesize_brief", synthesize_brief)
builder.add_edge(START, "search_evidence")
builder.add_edge("search_evidence", "rerank_search_evidence")
builder.add_edge("rerank_search_evidence", "synthesize_brief")
builder.add_edge("synthesize_brief", END)
graph = builder.compile()
