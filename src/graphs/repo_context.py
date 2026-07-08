from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .common import (
    llm_json,
    llm_text,
    output_json,
    output_text,
    repo_context_explore,
    text,
    with_output,
)


REPO_CONTEXT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query"],
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 320},
    },
}


async def plan_repo_context_query(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    prompt = f"""Latest user prompt:
{text(state.get("latest_user_prompt"))}

Prior thread context:
{text(state.get("thread_state_text"))}

Recent conversation context:
{text(state.get("recent_conversation_messages"))}

Repository:
{text(state.get("repo_name"))}

Produce one repo-context query for citation retrieval.

The query is passed directly to `repo-context explore --query`. It must be
a concrete retrieval instruction, not an analytical question.

Rules:
- Start with an imperative such as "Find", "In <path>, find", or "Return".
- Preserve the latest user prompt's scope. Use prior thread context only
  to resolve pronouns or omitted subject names.
- Preserve exact file paths, symbols, classes, functions, config keys, and
  string literals from the request; wrap identifiers in backticks when useful.
- Do not add unstated constraints: no inferred root directory, directory
  depth, file extension filter, file name pattern, package, provider, or
  technology-specific convention unless the user or thread context says it.
- For location/listing requests such as "where are ..." or "find files",
  keep the query broad and minimal. Do not convert it into a glob, extension
  list, or root-only search unless the user explicitly asked for that.
- For broad architecture requests, translate the ask into code targets to retrieve:
  entrypoints, handlers, routing, auth/security, config, data transforms,
  call sites, tests, and error handling as relevant.
- Avoid vague analysis words in the query such as how, why, explain,
  architecture, behavior, compare, synthesize, tradeoff, and flow.
- Do not include the repository name, markdown, bullets, or multiple
  candidate queries.

Examples:
- User asks where validation happens -> "Find request validation logic."
- User asks where Terraform files are -> "Find Terraform files."
- User names `FooService` -> "Find `FooService` definition and primary call sites."
- User asks about a component's architecture -> "Find the component entrypoints, routing, auth/security, config, and request-processing code."

Return strict JSON only:
{{
  "query": "one concrete repo-context retrieval instruction"
}}
"""
    output = await llm_json(
        prompt,
        config,
        schema=REPO_CONTEXT_PLAN_SCHEMA,
        default_endpoint="gmktec-evo-x2-utility",
        max_attempts=3,
        use_retrieval=False,
    )
    return with_output(state, "repo_context_plan", output)


async def explore_repository(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    plan = output_json(outputs.get("repo_context_plan"))
    if not isinstance(plan, dict):
        raise ValueError("repo_context graph plan is missing JSON")
    output = await repo_context_explore(
        query=text(plan.get("query")),
        repo_name=text(state.get("repo_name")),
        max_turns=6,
    )
    return with_output(state, "repo_context", output)


async def consolidate_reply(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    plan = output_json(outputs.get("repo_context_plan"))
    repo_output = output_json(outputs.get("repo_context"))
    raw_locations = (
        repo_output.get("raw_locations")
        if isinstance(repo_output, dict)
        else []
    )
    prompt = f"""Original request:
{text(state.get("latest_user_prompt"))}

Repository:
{text(state.get("repo_name"))}

Repo-context exploration query:
{text(plan.get("query") if isinstance(plan, dict) else "")}

Repo-context evidence:
{output_text(outputs.get("repo_context"))}

Repo-context raw locations:
{raw_locations}

Answer the original request directly using the repo-context evidence.
Preserve relevant file paths and line ranges. Surface uncertainty or
missing evidence clearly; do not invent repository details.
"""
    output = await llm_text(prompt, config, use_retrieval=False)
    return with_output(state, "reply", output, final=output["text"])


builder = StateGraph(dict)
builder.add_node("plan_repo_context_query", plan_repo_context_query)
builder.add_node("explore_repository", explore_repository)
builder.add_node("consolidate_reply", consolidate_reply)
builder.add_edge(START, "plan_repo_context_query")
builder.add_edge("plan_repo_context_query", "explore_repository")
builder.add_edge("explore_repository", "consolidate_reply")
builder.add_edge("consolidate_reply", END)
graph = builder.compile()
