from __future__ import annotations

import asyncio
import json
from typing import Any

import jsonschema

from src.orchestrator.runtime import ProxyRuntimeLLMClient, ProxyRuntimeSearchClient


LLM_CLIENT = ProxyRuntimeLLMClient()
SEARCH_CLIENT = ProxyRuntimeSearchClient()


def configurable(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config if isinstance(config, dict) else {}
    value = config.get("configurable")
    return value if isinstance(value, dict) else {}


def thread_id(config: dict[str, Any] | None) -> str:
    value = str(configurable(config).get("thread_id") or "").strip()
    return value or "graph"


def endpoint(config: dict[str, Any] | None, *, default: str = "smart") -> str:
    value = str(configurable(config).get("endpoint") or "").strip()
    return value or default


def reasoning_effort(config: dict[str, Any] | None, *, default: str = "high") -> str | None:
    value = str(configurable(config).get("reasoning_effort") or default).strip()
    return value or None


def retrieval_endpoint(config: dict[str, Any] | None) -> str | None:
    value = str(configurable(config).get("retrieval_endpoint") or "").strip()
    return value or None


def search_provider(config: dict[str, Any] | None) -> str | None:
    value = str(configurable(config).get("search_provider") or "").strip()
    return value or None


def max_tokens(config: dict[str, Any] | None) -> int | None:
    value = configurable(config).get("max_tokens")
    if value in {None, ""}:
        return None
    return int(value)


def text(value: Any) -> str:
    return str(value or "").strip()


def output_text(output: dict[str, Any] | None) -> str:
    return str((output or {}).get("text") or "").strip()


def output_json(output: dict[str, Any] | None) -> Any:
    return (output or {}).get("json")


def with_output(
    state: dict[str, Any],
    key: str,
    output: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    outputs = dict(state.get("outputs") or {})
    outputs[key] = output
    return {"outputs": outputs, **extra}


def output_from_payload(
    payload: dict[str, Any],
    *,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "text": json.dumps(payload, indent=2, default=str),
        "json": payload,
        "metadata": {"kind": kind, **dict(metadata or {})},
    }


async def llm_text(
    prompt: str,
    config: dict[str, Any] | None,
    *,
    default_endpoint: str = "smart",
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    response = await LLM_CLIENT.complete(
        endpoint=endpoint(config, default=default_endpoint),
        prompt=prompt,
        conversation_id=thread_id(config),
        reasoning_effort=reasoning_effort(config),
        retrieval_endpoint=retrieval_endpoint(config),
        max_tokens=max_output_tokens if max_output_tokens is not None else max_tokens(config),
        skip_conversation=True,
    )
    return {
        "text": str(response.get("text") or ""),
        "json": None,
        "metadata": dict(response.get("metadata") or {}),
    }


async def llm_json(
    prompt: str,
    config: dict[str, Any] | None,
    *,
    schema: dict[str, Any],
    default_endpoint: str = "smart",
    max_attempts: int = 3,
) -> dict[str, Any]:
    current_prompt = prompt
    last_text = ""
    last_error = ""
    for attempt in range(max(1, max_attempts)):
        output = await llm_text(
            current_prompt,
            config,
            default_endpoint=default_endpoint,
        )
        last_text = output["text"]
        try:
            parsed = parse_json_object(last_text)
            jsonschema.validate(parsed, schema)
            output["json"] = parsed
            return output
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 >= max_attempts:
                break
            current_prompt = repair_json_prompt(
                original_prompt=prompt,
                invalid_text=last_text,
                error=last_error,
                schema=schema,
            )
    raise ValueError(f"graph LLM JSON output invalid: {last_error}; text={last_text[:500]}")


async def search_one(
    query: str,
    config: dict[str, Any] | None,
    *,
    count: int = 5,
    use_query_refiner: bool = True,
) -> dict[str, Any]:
    payload = await SEARCH_CLIENT.search(
        query=query,
        provider=search_provider(config),
        count=count,
        use_query_refiner=use_query_refiner,
    )
    return output_from_payload(payload, kind="search")


async def search_many(
    queries: list[str],
    config: dict[str, Any] | None,
    *,
    count: int = 5,
    use_query_refiner: bool = False,
) -> dict[str, Any]:
    cleaned = dedupe_texts(queries)
    if not cleaned:
        raise ValueError("graph search produced no queries")
    if len(cleaned) == 1:
        return await search_one(
            cleaned[0],
            config,
            count=count,
            use_query_refiner=use_query_refiner,
        )
    payloads = await asyncio.gather(
        *[
            SEARCH_CLIENT.search(
                query=query,
                provider=search_provider(config),
                count=count,
                use_query_refiner=use_query_refiner,
            )
            for query in cleaned
        ]
    )
    merged = merge_search_payloads(cleaned, payloads)
    return output_from_payload(merged, kind="search")


async def rerank(
    query: str,
    source_output: dict[str, Any],
    config: dict[str, Any] | None,
    *,
    source_text: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    source = output_json(source_output)
    if not isinstance(source, dict):
        raise ValueError("graph rerank source output has no JSON payload")
    results = [item for item in source.get("results") or [] if isinstance(item, dict)]
    if not results:
        payload = dict(source)
        payload["workflow_rerank"] = {
            "query": query,
            "path": "none",
            "skipped": "no-results",
        }
        return output_from_payload(payload, kind="rerank")
    payload = await SEARCH_CLIENT.rerank_results(
        query=query,
        results=results,
        source_text=source_text,
        top_k=top_k,
    )
    payload["graph_rerank"] = {
        "query": query,
        "source_text_provided": bool(source_text),
    }
    return output_from_payload(payload, kind="rerank")


def parse_json_object(raw: str) -> dict[str, Any]:
    text_value = raw.strip()
    if text_value.startswith("```"):
        lines = text_value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text_value = "\n".join(lines).strip()
    try:
        parsed = json.loads(text_value)
    except json.JSONDecodeError:
        start = text_value.find("{")
        end = text_value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text_value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def repair_json_prompt(
    *,
    original_prompt: str,
    invalid_text: str,
    error: str,
    schema: dict[str, Any],
) -> str:
    return f"""The previous response did not satisfy the required JSON contract.

Validation error:
{error}

Required JSON Schema:
{json.dumps(schema, indent=2)}

Original task:
{original_prompt}

Invalid response:
{invalid_text}

Return only corrected strict JSON. Do not include markdown or commentary.
"""


def dedupe_texts(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        item = str(value or "").strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def merge_search_payloads(
    queries: list[str],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_results: list[dict[str, Any]] = []
    seen: set[str] = set()
    warnings: list[str] = []
    per_query: list[dict[str, Any]] = []
    for query, payload in zip(queries, payloads):
        payload = payload if isinstance(payload, dict) else {}
        per_query.append(
            {
                "query": query,
                "provider": payload.get("provider"),
                "result_count": len(payload.get("results") or []),
            }
        )
        warnings.extend(str(item) for item in payload.get("warnings") or [])
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("url") or item.get("title") or "").strip().lower()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged_results.append(dict(item))
    return {
        "query": " | ".join(queries),
        "queries": queries,
        "per_query": per_query,
        "results": merged_results,
        "warnings": warnings,
        "graph_search": {
            "fanout": len(queries),
            "merged_results": len(merged_results),
        },
    }

