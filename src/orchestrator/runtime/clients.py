from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from src.search import SearchArgs, wrap_search_results
from src.search.types import SearchResponse, SearchResult


class _RuntimeRequest:
    def __init__(self, *, path: str, body: Dict[str, Any], headers: Dict[str, str]):
        self.method = "POST"
        self.headers = headers
        self.query_params: Dict[str, str] = {}
        self.url = type("URL", (), {"path": path})()
        self._body = json.dumps(body).encode("utf-8")
        self._json = body

    async def body(self) -> bytes:
        return self._body

    async def json(self) -> Dict[str, Any]:
        return self._json


class ProxyRuntimeLLMClient:
    async def complete(
        self,
        *,
        endpoint: str,
        prompt: str,
        conversation_id: str,
        reasoning_effort: Optional[str] = None,
        retrieval_endpoint: Optional[str] = None,
        max_tokens: Optional[int] = None,
        skip_conversation: bool = False,
    ) -> Dict[str, Any]:
        from ..upstream_proxy import proxy_request
        from ..utils import extract_assistant_text

        endpoint_key = str(endpoint or "smart").strip() or "smart"
        payload: Dict[str, Any] = {
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        headers = {
            "Content-Type": "application/json",
            "X-Conversation-ID": conversation_id,
        }
        if reasoning_effort:
            headers["X-Reasoning-Effort"] = reasoning_effort
        if retrieval_endpoint:
            headers["X-Retrieval-Endpoint"] = retrieval_endpoint
        if skip_conversation:
            headers["X-LLMCP-Skip-Conversation"] = "true"
        if endpoint_key != "smart":
            headers["X-Allow-Route-Switch"] = "true"

        request = _RuntimeRequest(
            path=f"/{endpoint_key}",
            body=payload,
            headers=headers,
        )
        if endpoint_key == "smart":
            from ..proxy import smart_route

            response = await smart_route(request)
        else:
            response = await proxy_request(endpoint_key, request)
        response_body = getattr(response, "body", b"") or b""
        try:
            response_json = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            response_json = {}

        text = extract_assistant_text(response_json) or ""
        response_headers = dict(getattr(response, "headers", {}) or {})
        metadata = {
            "endpoint": endpoint_key,
            "status_code": getattr(response, "status_code", None),
        }
        for header_name, header_value in response_headers.items():
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

        for key in ("usage", "stats", "model_info", "runtime", "timings"):
            if isinstance(response_json.get(key), dict):
                metadata[key] = response_json[key]

        return {"text": text, "metadata": metadata}

    async def stream_complete(
        self,
        *,
        endpoint: str,
        prompt: str,
        conversation_id: str,
        reasoning_effort: Optional[str] = None,
        retrieval_endpoint: Optional[str] = None,
        max_tokens: Optional[int] = None,
        skip_conversation: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        from ..upstream_proxy import proxy_request

        endpoint_key = str(endpoint or "smart").strip() or "smart"
        payload: Dict[str, Any] = {
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        headers = {
            "Content-Type": "application/json",
            "X-Conversation-ID": conversation_id,
        }
        if reasoning_effort:
            headers["X-Reasoning-Effort"] = reasoning_effort
        if retrieval_endpoint:
            headers["X-Retrieval-Endpoint"] = retrieval_endpoint
        if skip_conversation:
            headers["X-LLMCP-Skip-Conversation"] = "true"
        if endpoint_key != "smart":
            headers["X-Allow-Route-Switch"] = "true"

        request = _RuntimeRequest(
            path=f"/{endpoint_key}",
            body=payload,
            headers=headers,
        )
        if endpoint_key == "smart":
            from ..proxy import smart_route

            response = await smart_route(request)
        else:
            response = await proxy_request(endpoint_key, request)

        response_headers = dict(getattr(response, "headers", {}) or {})
        metadata = _metadata_from_headers(endpoint_key, response_headers)
        if metadata:
            yield {"channel": "metadata", "metadata": metadata}

        async for data in _iter_sse_data(response):
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if str(obj.get("type") or "").startswith("proxy."):
                detail = obj.get("detail") or obj.get("error") or obj.get("type")
                raise RuntimeError(str(detail))
            for key in ("usage", "stats", "model_info", "runtime", "timings"):
                if isinstance(obj.get(key), dict):
                    yield {"channel": "metadata", "metadata": {key: obj[key]}}
            choices = obj.get("choices") if isinstance(obj, dict) else []
            if not choices:
                continue
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else {}
            if not isinstance(delta, dict):
                continue
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                yield {"channel": "reasoning", "content": str(reasoning)}
            content = delta.get("content")
            if content:
                yield {"channel": "content", "content": str(content)}


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
                str(item)
                for item in data.get("warnings", [])
                if str(item).strip()
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
            or (not grounded_user_message and not context_blocks and not evidence_blocks),
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


def _metadata_from_headers(endpoint_key: str, headers: Dict[str, Any]) -> Dict[str, Any]:
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
    return metadata


def _search_result_from_dict(item: dict[str, Any]) -> SearchResult:
    return SearchResult(
        title=str(item.get("title") or ""),
        url=str(item.get("url") or ""),
        snippet=(
            str(item.get("snippet"))
            if item.get("snippet") is not None
            else None
        ),
        rank=int(item.get("rank") or 0),
        provider=str(item.get("provider") or ""),
        engine=str(item.get("engine") or ""),
        fetched_at=str(item.get("fetched_at") or ""),
        score=(
            float(item["score"])
            if item.get("score") is not None
            else None
        ),
        ranking=dict(item.get("ranking") or {}),
    )
