from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .common import (
    llm_json,
    llm_text,
    output_json,
    output_text,
    retrieve_many,
    text,
    with_output,
)


RAG_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["retrieval_queries", "retrieval_prompt", "reason"],
    "properties": {
        "retrieval_queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {"type": "string", "minLength": 3, "maxLength": 240},
        },
        "retrieval_prompt": {"type": "string", "minLength": 3, "maxLength": 1200},
        "reason": {"type": "string", "minLength": 1},
    },
}


async def plan_rag_prompt(
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
full thread context. Decide exactly what should be sent to Retrieval
before any RAG lookup happens.

Return strict JSON only:
{{
  "retrieval_queries": [
    "focused keyword/phrase probe for the primary entity and evidence need",
    "focused keyword/phrase probe for a distinct entity, project, or evidence angle"
  ],
  "retrieval_prompt": "one self-contained user prompt to answer after Retrieval fanout",
  "reason": "why this prompt preserves the latest user intent with thread context"
}}

RAG prompt rules:
- Resolve pronouns, ellipses, omitted subjects, and references from thread context.
- Preserve the latest user prompt's constraints and requested output shape.
- Build `retrieval_queries` for vector/BM25 chunk retrieval, not web search.
- Prefer compact keyword/phrase probes over full natural-language questions.
- Use exact named entities, project/repo names, product names, symbols, file names, API names, dates, identifiers, and domain terms from the prompt/context.
- Include only the disambiguating action/concept words needed to retrieve relevant chunks.
- For metric/field/methodology questions, prefer retrieval terms such as measured parameters, outcome measures, variables, trial, validation, table, appendix, and any abbreviations present in context.
- Bad retrieval query: "list of physiological metrics measured in Project WOREC's human trials and thermoregulatory model"
- Good retrieval query: "WOREC physiological metrics human trials"
- Good retrieval query: "WOREC measured parameters outcome measures validation"
- Avoid generic verbs and filler such as explain, compare, analyze, overview, latest, best, how, why, what, tell me, information about, docs, guide, tutorial, site, official, or source.
- Never start retrieval queries with generic intent phrases such as list of, specific, details regarding, find out more, determine, provide, or based on.
- Return 1 to 6 queries in `retrieval_queries`.
- Use multiple queries when the latest prompt mentions, compares, or implies multiple entities, projects, repos, products, services, dates, or evidence angles.
- Keep each retrieval query focused on one entity/project/angle where possible.
- Keep each retrieval query under 12 words unless exact identifiers require more.
- Include only context needed to make the latest request self-contained.
- Do not include a transcript, citations invented from memory, markdown fences, or multiple candidate prompts.
- Do not use web search operators, URLs, quoted search strings, boolean syntax, or source-domain preferences.
- `retrieval_prompt` should ask for a direct answer using retrieved context and require uncertainty when Retrieval is insufficient.
- Keep `retrieval_prompt` under 1200 characters.
- Do not include markdown, bullets, or explanatory prose outside the JSON object.
"""
    output = await llm_json(
        prompt,
        config,
        schema=RAG_PLAN_SCHEMA,
        default_endpoint="gmktec-evo-x2-utility",
        max_attempts=3,
        use_retrieval=False,
    )
    return with_output(state, "rag_prompt", output)


async def retrieve_evidence(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    plan = output_json(outputs.get("rag_prompt"))
    if not isinstance(plan, dict):
        raise ValueError("threaded_rag graph plan is missing JSON")
    output = await retrieve_many(
        [str(item) for item in plan.get("retrieval_queries") or []],
        config,
        limit=10,
    )
    return with_output(state, "retrieval_evidence", output)


async def answer_with_rag(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    plan = output_json(outputs.get("rag_prompt"))
    prompt = f"""User prompt:
{text(plan.get("retrieval_prompt") if isinstance(plan, dict) else "")}

Retrieval plan:
{output_text(outputs.get("rag_prompt"))}

Retrieval context:
{output_text(outputs.get("retrieval_evidence"))}

Answer the user prompt directly using the retrieved context. Compare or
combine evidence across all retrieved entities/projects when relevant.
Surface material uncertainty, missing evidence, or retrieval failures.
"""
    output = await llm_text(prompt, config, use_retrieval=False)
    return with_output(state, "reply", output, final=output["text"])


builder = StateGraph(dict)
builder.add_node("plan_rag_prompt", plan_rag_prompt)
builder.add_node("retrieve_evidence", retrieve_evidence)
builder.add_node("answer_with_rag", answer_with_rag)
builder.add_edge(START, "plan_rag_prompt")
builder.add_edge("plan_rag_prompt", "retrieve_evidence")
builder.add_edge("retrieve_evidence", "answer_with_rag")
builder.add_edge("answer_with_rag", END)
graph = builder.compile()
