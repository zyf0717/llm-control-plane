from __future__ import annotations

import json
from typing import Any

from .models import WorkflowSpec, WorkflowStepSpec
from .template import parse_json_text


def extract_search_queries(prompt: str) -> list[str]:
    stripped = str(prompt or "").strip()
    parsed = parse_json_text(stripped)
    raw_queries: list[Any] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("queries"), list):
            raw_queries.extend(parsed["queries"])
        elif parsed.get("query") is not None:
            raw_queries.append(parsed.get("query"))
    elif isinstance(parsed, list):
        raw_queries.extend(parsed)
    elif isinstance(parsed, str):
        raw_queries.append(parsed)
    else:
        raw_queries.append(stripped)

    queries: list[str] = []
    seen: set[str] = set()
    for raw_query in raw_queries:
        query = str(raw_query or "").strip()
        key = " ".join(query.lower().split())
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries or [stripped]


def merge_search_results(
    queries: list[str], results: list[dict[str, Any]], *, use_query_refiner: bool
) -> dict[str, Any]:
    if len(results) == 1:
        merged = dict(results[0])
        merged.setdefault("query", queries[0] if queries else "")
        merged["workflow_search"] = {
            "planned_by_workflow": not use_query_refiner,
            "queries": list(queries),
        }
        return merged

    seen_urls: set[str] = set()
    merged_results: list[Any] = []
    warnings: list[str] = []
    per_query: list[dict[str, Any]] = []
    degraded = False
    providers: set[str] = set()

    for query, result in zip(queries, results):
        response = dict(result)
        response["query"] = str(response.get("query") or query)
        per_query.append(response)
        degraded = degraded or bool(response.get("degraded"))
        provider = str(response.get("provider") or "").strip()
        if provider and provider != "none":
            providers.add(provider)
        if isinstance(response.get("warnings"), list):
            warnings.extend(str(item) for item in response["warnings"])
        for item in response.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            key = url or json.dumps(item, sort_keys=True)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            merged_results.append(item)

    return {
        "query": queries[0] if queries else "",
        "queries": list(queries),
        "provider": next(iter(providers)) if len(providers) == 1 else "fanout",
        "results": merged_results,
        "warnings": warnings,
        "degraded": degraded
        or any(not (result.get("results") or []) for result in results),
        "per_query": per_query,
        "workflow_search": {
            "planned_by_workflow": True,
            "queries": list(queries),
        },
    }


def select_rerank_source(
    spec: WorkflowSpec, step: WorkflowStepSpec, step_input: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    outputs = step_input.get("previous_outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    step_specs = {candidate.id: candidate for candidate in spec.steps}
    source_keys = [
        step_specs[dependency].output_key or dependency
        for dependency in step.depends_on or []
        if dependency in step_specs
    ]
    if not source_keys:
        source_keys = list(outputs.keys())

    candidates: list[tuple[str, dict[str, Any]]] = []
    seen_keys: set[str] = set()
    for source_key in source_keys:
        if source_key in seen_keys:
            continue
        seen_keys.add(source_key)
        source = search_json_from_step_output(outputs.get(source_key))
        if source is not None:
            candidates.append((source_key, source))

    if not candidates:
        raise ValueError(
            "workflow rerank step found no dependency output with search results"
        )
    if len(candidates) > 1:
        raise ValueError(
            "workflow rerank step has multiple dependency outputs with search results"
        )
    return candidates[0]


def search_json_from_step_output(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, dict):
        return None
    search_json = output.get("json")
    if not isinstance(search_json, dict):
        return None
    if not isinstance(search_json.get("results"), list):
        return None
    return dict(search_json)


def search_output_query(output: dict[str, Any]) -> str:
    query = str(output.get("query") or "").strip()
    if query:
        return query
    queries = output.get("queries")
    if isinstance(queries, list):
        for item in queries:
            query = str(item or "").strip()
            if query:
                return query
    return ""


def merge_reranked_search_result(
    source: dict[str, Any], reranked: dict[str, Any]
) -> dict[str, Any]:
    warnings = [
        *[str(item) for item in source.get("warnings") or []],
        *[str(item) for item in reranked.get("warnings") or []],
    ]
    merged = dict(reranked) | {
        key: value
        for key, value in source.items()
        if key not in {"results", "reranking", "warnings", "wrapped_results"}
    }
    merged["warnings"] = warnings
    return merged


def reranking_metadata(
    result: dict[str, Any], *, used: bool, degraded: bool, path: str
) -> dict[str, Any]:
    reranking = result.get("reranking")
    metadata = dict(reranking) if isinstance(reranking, dict) else {}
    metadata.setdefault("used", used)
    metadata.setdefault("degraded", degraded)
    metadata["path"] = path
    return metadata


def reranking_path(result: dict[str, Any]) -> str:
    reranking = result.get("reranking")
    if not isinstance(reranking, dict):
        return "none"
    path = str(reranking.get("path") or "").strip().lower()
    if path:
        return path
    if reranking.get("used") is True:
        backend = str(reranking.get("backend") or "").strip().lower()
        if backend in {"dedicated", "llm"}:
            return backend
    return "none"
