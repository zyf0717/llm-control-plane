from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

from src.search import SearchArgs, wrap_search_results
from src.search.types import SearchResponse, SearchResult

from . import proxy_services as services
from .upstream_proxy import proxy_request
from .utils import extract_assistant_text


class _WorkflowRequest:
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


class ProxyWorkflowLLMClient:
    async def complete(
        self,
        *,
        endpoint: str,
        prompt: str,
        convo_id: str,
        reasoning_effort: Optional[str] = None,
        rag_endpoint: Optional[str] = None,
        max_tokens: Optional[int] = None,
        skip_history: bool = False,
    ) -> Dict[str, Any]:
        endpoint_key = str(endpoint or "smart").strip() or "smart"
        payload: Dict[str, Any] = {
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        headers = {
            "Content-Type": "application/json",
            "X-Convo-ID": convo_id,
        }
        if reasoning_effort:
            headers["X-Reasoning-Effort"] = reasoning_effort
        if rag_endpoint:
            headers["X-RAG-Endpoint"] = rag_endpoint
        if skip_history:
            headers["X-LLMCP-Skip-History"] = "true"
        if endpoint_key != "smart":
            headers["X-Allow-Route-Switch"] = "true"

        request = _WorkflowRequest(
            path=f"/{endpoint_key}",
            body=payload,
            headers=headers,
        )
        if endpoint_key == "smart":
            from .proxy import smart_route

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
            elif lower_name.startswith("x-rag-"):
                metadata.setdefault("rag", {})[
                    lower_name.replace("x-rag-", "").replace("-", "_")
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
        convo_id: str,
        reasoning_effort: Optional[str] = None,
        rag_endpoint: Optional[str] = None,
        max_tokens: Optional[int] = None,
        skip_history: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
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
            "X-Convo-ID": convo_id,
        }
        if reasoning_effort:
            headers["X-Reasoning-Effort"] = reasoning_effort
        if rag_endpoint:
            headers["X-RAG-Endpoint"] = rag_endpoint
        if skip_history:
            headers["X-LLMCP-Skip-History"] = "true"
        if endpoint_key != "smart":
            headers["X-Allow-Route-Switch"] = "true"

        request = _WorkflowRequest(
            path=f"/{endpoint_key}",
            body=payload,
            headers=headers,
        )
        if endpoint_key == "smart":
            from .proxy import smart_route

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


class ProxyWorkflowSearchClient:
    async def search(
        self,
        *,
        query: str,
        provider: Optional[str] = None,
        count: int = 5,
        use_query_refiner: bool = True,
    ) -> Dict[str, Any]:
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
        payload["wrapped_results"] = wrap_search_results(response)
        return payload

    async def rerank_results(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        context: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
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
            payload["wrapped_results"] = wrap_search_results(response)
            return payload

        reranking = await reranker.rerank(
            query=query,
            results=search_results,
            context=context,
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
        payload["wrapped_results"] = wrap_search_results(response)
        return payload


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
        elif lower_name.startswith("x-rag-"):
            metadata.setdefault("rag", {})[
                lower_name.replace("x-rag-", "").replace("-", "_")
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
