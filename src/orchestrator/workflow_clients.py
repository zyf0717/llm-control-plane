from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.search import SearchArgs, wrap_search_results

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
            )
        )
        payload = response.to_dict()
        payload["wrapped_results"] = wrap_search_results(response)
        return payload
