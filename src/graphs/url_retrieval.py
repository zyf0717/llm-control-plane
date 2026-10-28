"""Standalone deterministic URL retrieval graph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.url_retrieval import extract_url_output, fetch_url_output


def extract_urls(state: dict[str, Any]) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    outputs["url_candidates"] = extract_url_output(state["latest_user_prompt"])
    return {**state, "outputs": outputs}


async def fetch_urls(state: dict[str, Any]) -> dict[str, Any]:
    outputs = dict(state["outputs"])
    output = await fetch_url_output(outputs["url_candidates"]["json"]["urls"])
    outputs["url_documents"] = output
    return {**state, "outputs": outputs, "final": output["text"]}


builder = StateGraph(dict)
builder.add_node("extract_urls", extract_urls)
builder.add_node("fetch_urls", fetch_urls)
builder.add_edge(START, "extract_urls")
builder.add_edge("extract_urls", "fetch_urls")
builder.add_edge("fetch_urls", END)
graph = builder.compile()
