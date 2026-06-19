from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .common import (
    llm_json,
    llm_text,
    output_json,
    output_text,
    rerank,
    search_many,
    text,
    with_output,
)


SEARCH_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rerank_query", "queries", "reason", "source_preferences"],
    "properties": {
        "rerank_query": {"type": "string", "minLength": 3, "maxLength": 240},
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 3, "maxLength": 100},
        },
        "reason": {"type": "string", "minLength": 1},
        "source_preferences": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}


async def plan_search_queries(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    prompt = f"""Latest user prompt:
{text(state.get("latest_user_prompt"))}

Prior thread context:
{text(state.get("thread_state_text"))}

Recent conversation context:
{text(state.get("recent_conversation_messages"))}

Additional context:
{text(state.get("thread_briefing"))}
{text(state.get("manual_source_text"))}
{text(state.get("uploaded_source_text"))}

Infer the user's actual information need from the latest turn in the
full thread context. Decide exactly what should be searched before
any web lookup happens.

Return strict JSON only:
{{
  "rerank_query": "one concise thread-resolved information need for ranking candidate documents",
  "queries": [
    "concise web search query for the primary information need",
    "concise web search query for a materially different evidence angle"
  ],
  "reason": "why these searches answer the latest prompt with the thread context",
  "source_preferences": ["preferred primary source domains or source types"]
}}

Query rules:
- `rerank_query` must resolve pronouns, ellipses, and references from thread context.
- `rerank_query` should describe the evidence needed to answer the user; do not use search operators.
- Return 1 to 5 queries in "queries".
- Use multiple queries only when they target materially different source angles.
- Keep each query under 100 characters.
- Prefer primary sources and exact named entities from the prompt/context.
- Do not include markdown, bullets, or explanatory prose outside the JSON object.
"""
    output = await llm_json(
        prompt,
        config,
        schema=SEARCH_PLAN_SCHEMA,
        default_endpoint="gmktec-evo-x2-utility",
        max_attempts=3,
    )
    return with_output(state, "search_plan", output)


async def search_evidence(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    plan = output_json(outputs.get("search_plan"))
    if not isinstance(plan, dict):
        raise ValueError("threaded_search graph search plan is missing JSON")
    output = await search_many(
        [str(item) for item in plan.get("queries") or []],
        config,
        count=5,
        use_query_refiner=False,
    )
    return with_output(state, "search_evidence", output)


async def rerank_results(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    plan = output_json(outputs.get("search_plan"))
    if not isinstance(plan, dict):
        raise ValueError("threaded_search graph rerank plan is missing JSON")
    source_text = f"""Latest user prompt:
{text(state.get("latest_user_prompt"))}

Prior thread context:
{text(state.get("thread_state_text"))}

Recent conversation context:
{text(state.get("recent_conversation_messages"))}

Search plan reason:
{text(plan.get("reason"))}

Source preferences:
{plan.get("source_preferences") or []}
"""
    output = await rerank(
        text(plan.get("rerank_query")),
        outputs.get("search_evidence") or {},
        config,
        source_text=source_text,
        top_k=5,
    )
    return with_output(state, "search_evidence", output)


async def consolidate_reply(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    prompt = f"""Latest user prompt:
{text(state.get("latest_user_prompt"))}

Prior thread context:
{text(state.get("thread_state_text"))}

Recent conversation context:
{text(state.get("recent_conversation_messages"))}

Additional context:
{text(state.get("thread_briefing"))}
{text(state.get("manual_source_text"))}
{text(state.get("uploaded_source_text"))}

Search plan:
{output_text(outputs.get("search_plan"))}

Search context:
{output_text(outputs.get("search_evidence"))}

Answer the latest user prompt directly. Use the thread context to
preserve intent and constraints, and use the search context for factual
grounding. Surface material uncertainty or missing evidence clearly.
"""
    output = await llm_text(prompt, config)
    return with_output(state, "reply", output, final=output["text"])


builder = StateGraph(dict)
builder.add_node("plan_search_queries", plan_search_queries)
builder.add_node("search_evidence", search_evidence)
builder.add_node("rerank_results", rerank_results)
builder.add_node("consolidate_reply", consolidate_reply)
builder.add_edge(START, "plan_search_queries")
builder.add_edge("plan_search_queries", "search_evidence")
builder.add_edge("search_evidence", "rerank_results")
builder.add_edge("rerank_results", "consolidate_reply")
builder.add_edge("consolidate_reply", END)
graph = builder.compile()
