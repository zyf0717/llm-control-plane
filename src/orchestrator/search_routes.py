from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from src.search import SearchArgs, wrap_search_results

from . import proxy_services as services

router = APIRouter()


def search_request_bool(
    body: Dict[str, Any],
    *,
    snake_key: str,
    camel_key: str,
    default: bool,
) -> bool:
    for key in (snake_key, camel_key):
        if key not in body:
            continue
        raw = body[key]
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"false", "0", "no", "off"}:
                return False
            if normalized in {"true", "1", "yes", "on"}:
                return True
        return bool(raw)
    return default


@router.post("/search/web")
async def search_web(request: Request):
    """Return normalized candidate links from configured search providers."""
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON object")

    try:
        args = SearchArgs(
            query=str(body.get("query", "")).strip(),
            count=body.get("count"),
            provider=str(body.get("provider", "auto") or "auto"),
            context=body.get("context"),
            language=body.get("language"),
            region=body.get("region"),
            safe_search=body.get("safeSearch") or body.get("safe_search"),
            freshness=body.get("freshness"),
            use_query_refiner=search_request_bool(
                body,
                snake_key="use_query_refiner",
                camel_key="useQueryRefiner",
                default=True,
            ),
            use_reranker=search_request_bool(
                body,
                snake_key="use_reranker",
                camel_key="useReranker",
                default=True,
            ),
            rerank_context=body.get("rerank_context") or body.get("rerankContext"),
        )
        response = await services.search_service.search(args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status_code = 503 if "disabled" in str(exc).lower() else 500
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    payload = response.to_dict()
    payload["wrapped_results"] = wrap_search_results(response)
    return payload
