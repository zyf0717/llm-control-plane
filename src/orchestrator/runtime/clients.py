from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from src.llm_metrics import CallMetrics, llm_response_metadata

from src.search import SearchArgs, wrap_search_results
from src.search.types import SearchResponse, SearchResult


def _context_messages(prompt: str, context: dict | None) -> list[dict]:
    """Keep reusable background in separate messages before the changing task."""
    context = context or {}
    messages = context.get("conversation_messages")
    history = str(context.get("thread_state_text") or "").strip()
    if messages is None:
        briefing = str(context.get("thread_briefing") or "").strip()
        history = "\n\n".join(
            [
                *(
                    str(context[key]).strip()
                    for key in ("thread_state_text", "recent_conversation_messages")
                    if context.get(key) and str(context[key]).strip() not in briefing
                ),
                briefing,
            ]
        ).strip()
    sections = (
        ("Supplied context", context.get("manual_source_text")),
        ("Uploaded source", context.get("uploaded_source_text")),
        ("Prior conversation context", history),
    )
    return (
        [
            {"role": "user", "content": f"{label}:\n{str(value).strip()}"}
            for label, value in sections
            if value and str(value).strip()
        ]
        + [
            {"role": message["role"], "content": message["content"]}
            for message in messages or []
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and message.get("content")
        ]
        + [{"role": "user", "content": prompt}]
    )


class ProxyRuntimeLLMClient:
    async def _request(
        self,
        stream: bool,
        *,
        prompt: str,
        conversation_id: str,
        context: dict | None = None,
        endpoint: str | None = None,
        workload_type: str | None = None,
        reasoning_effort: str | None = None,
        retrieval_endpoint: str | None = None,
        max_tokens: int | None = None,
        skip_conversation: bool = False,
    ):
        from ..upstream_proxy import complete_request

        payload = {"stream": stream, "messages": _context_messages(prompt, context)}
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        headers = {"X-Conversation-ID": conversation_id}
        if reasoning_effort:
            headers["X-Reasoning-Effort"] = reasoning_effort
        if retrieval_endpoint:
            headers["X-Retrieval-Endpoint"] = retrieval_endpoint
        if skip_conversation:
            headers["X-LLMCP-Skip-Conversation"] = "true"
        if endpoint and endpoint != "smart":
            headers["X-Allow-Route-Switch"] = "true"
        return await complete_request(
            payload, headers, endpoint=endpoint, workload_type=workload_type
        )

    async def complete(self, **request) -> Dict[str, Any]:
        from ..utils import extract_assistant_text

        endpoint = request.get("endpoint") or request.get("workload_type") or "smart"
        with CallMetrics(metadata={"endpoint": endpoint}) as call:
            response = await self._request(False, **request)
            if response.status_code >= 400:
                raise RuntimeError(f"LLM request failed (HTTP {response.status_code})")
            data = json.loads(response.body)
            metadata = _metadata_from_headers(endpoint, dict(response.headers))
            metadata.update(
                status_code=response.status_code, **llm_response_metadata(data)
            )
            call.metadata.update(metadata)
            return {"text": extract_assistant_text(data) or "", "metadata": metadata}

    async def stream_complete(self, **request) -> AsyncIterator[Dict[str, Any]]:
        endpoint = request.get("endpoint") or request.get("workload_type") or "smart"
        with CallMetrics(metadata={"endpoint": endpoint}) as call:
            response = await self._request(True, **request)
            if response.status_code >= 400:
                raise RuntimeError(f"LLM request failed (HTTP {response.status_code})")
            metadata = _metadata_from_headers(endpoint, dict(response.headers))
            call.metadata.update(metadata)
            if metadata:
                yield {"channel": "metadata", "metadata": dict(metadata)}
            async for data in _iter_sse_data(response):
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if str(obj.get("type") or "").startswith("proxy."):
                    raise RuntimeError(
                        str(obj.get("detail") or obj.get("error") or obj["type"])
                    )
                new_metadata = {
                    key: value
                    for key, value in llm_response_metadata(obj).items()
                    if call.metadata.get(key) != value
                }
                if new_metadata:
                    call.metadata.update(new_metadata)
                    yield {"channel": "metadata", "metadata": new_metadata}
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                for channel, content in (
                    (
                        "reasoning",
                        delta.get("reasoning") or delta.get("reasoning_content"),
                    ),
                    ("content", delta.get("content")),
                ):
                    if content:
                        call.on_first_token()
                        yield {"channel": channel, "content": str(content)}


class ProxyRuntimeSearchClient:
    async def search(
        self,
        *,
        query: str,
        provider: Optional[str] = None,
        count: int = 5,
        use_query_refiner: bool = True,
    ) -> Dict[str, Any]:
        from .. import proxy_services as services

        response = await services.search_service.search(
            SearchArgs(
                query=query,
                provider=provider or "auto",
                count=count,
                use_query_refiner=use_query_refiner,
                use_reranker=False,
            )
        )
        payload = response.to_dict()
        payload["search_evidence"] = wrap_search_results(response)
        return payload

    async def rerank_results(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        source_text: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        from .. import proxy_services as services

        reranker = getattr(services.search_service, "reranker", None)
        search_results = [_search_result_from_dict(item) for item in results]
        if reranker is None or not search_results:
            if top_k is not None:
                search_results = search_results[: max(1, int(top_k))]
            response = SearchResponse(
                query=query,
                provider="fanout",
                results=search_results,
            )
            payload = response.to_dict()
            payload["search_evidence"] = wrap_search_results(response)
            return payload

        reranking = await reranker.rerank(
            query=query,
            results=search_results,
            source_text=source_text,
            top_k=top_k,
        )
        response = SearchResponse(
            query=query,
            provider="fanout",
            results=reranking.results,
            reranking=reranking.to_public_dict(),
        )
        if reranking.warning:
            response.warnings.append(f"reranker: {reranking.warning}")
        payload = response.to_dict()
        payload["search_evidence"] = wrap_search_results(response)
        return payload


class ProxyRuntimeRetrievalClient:
    async def retrieve(
        self,
        *,
        query: str,
        retrieval_endpoint: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        from .. import proxy_services as services
        from ..request_processor import RequestProcessor
        from ..utils import HeaderManager

        normalized_endpoint = RequestProcessor._normalize_retrieval_endpoint(
            retrieval_endpoint
        )
        if not normalized_endpoint:
            raise ValueError("retrieval endpoint is required")

        request_headers = HeaderManager.create_auth_headers()
        request_headers["Content-Type"] = "application/json"
        request_payload = {
            "query": str(query or "").strip(),
            "limit": int(limit or services.RETRIEVAL_TOP_K),
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    normalized_endpoint,
                    json=request_payload,
                    headers=request_headers,
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return {
                "query": request_payload["query"],
                "retrieval_endpoint": normalized_endpoint,
                "context_blocks": [],
                "evidence_blocks": [],
                "grounded_user_message": "",
                "warnings": [f"retrieval request failed: {str(exc)}"],
                "degraded": True,
            }

        data = dict(payload) if isinstance(payload, dict) else {}
        context_blocks = data.get("context_blocks")
        if not isinstance(context_blocks, list):
            context_blocks = []
        evidence_blocks = data.get("evidence_blocks")
        if not isinstance(evidence_blocks, list):
            evidence_blocks = context_blocks
        warnings = []
        if isinstance(data.get("warnings"), list):
            warnings = [
                str(item) for item in data.get("warnings", []) if str(item).strip()
            ]

        grounded_user_message = str(data.get("grounded_user_message") or "").strip()
        if not grounded_user_message and not context_blocks and not evidence_blocks:
            warnings.append("retrieval returned no context")

        return {
            **data,
            "query": request_payload["query"],
            "retrieval_endpoint": normalized_endpoint,
            "context_blocks": context_blocks,
            "evidence_blocks": evidence_blocks,
            "grounded_user_message": grounded_user_message,
            "warnings": warnings,
            "degraded": bool(data.get("degraded"))
            or (
                not grounded_user_message and not context_blocks and not evidence_blocks
            ),
        }


async def _iter_sse_data(response: Any) -> AsyncIterator[str]:
    buffer = ""
    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        return
    async for chunk in body_iterator:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", errors="replace")
        else:
            buffer += str(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line.startswith("data: "):
                yield line[6:].strip()
    for line in buffer.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            yield line[6:].strip()


def _metadata_from_headers(
    endpoint_key: str, headers: Dict[str, Any]
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "endpoint": endpoint_key,
    }
    for header_name, header_value in headers.items():
        lower_name = str(header_name).lower()
        if lower_name == "x-trace-id":
            metadata["trace_id"] = str(header_value)
        elif lower_name.startswith("x-route-"):
            metadata.setdefault("routing", {})[
                lower_name.replace("x-route-", "").replace("-", "_")
            ] = str(header_value)
        elif lower_name.startswith("x-retrieval-"):
            metadata.setdefault("retrieval", {})[
                lower_name.replace("x-retrieval-", "").replace("-", "_")
            ] = str(header_value)
    metadata["endpoint"] = metadata.get("routing", {}).get("decision", endpoint_key)
    return metadata


def _search_result_from_dict(item: dict[str, Any]) -> SearchResult:
    return SearchResult(
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        snippet=(str(item.get("snippet")) if item.get("snippet") is not None else None),
        rank=int(item.get("rank") or 0),
        provider=str(item.get("provider") or ""),
        engine=str(item.get("engine") or ""),
        fetched_at=str(item.get("fetched_at") or ""),
        score=(float(item["score"]) if item.get("score") is not None else None),
        ranking=dict(item.get("ranking") or {}),
    )
