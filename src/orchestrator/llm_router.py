"""Ordered workload routing and shared OpenAI-compatible model resolution."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from time import monotonic

import httpx
from fastapi import HTTPException

from src.llm_metrics import CallMetrics, llm_response_metadata
from .utils import HeaderManager, extract_assistant_text, filter_unsafe_headers

logger = logging.getLogger(__name__)
WORKLOADS = {"utility", "complex", "responsive", "vision"}


class LLMRouter:
    def __init__(self, config: dict):
        self.endpoints = {
            name: {**endpoint, "name": name, "url": endpoint["url"].rstrip("/")}
            for name, endpoint in config.get("endpoints", {}).items()
        }
        self.workloads = config.get("workloads", {})
        for workload, names in self.workloads.items():
            if workload not in WORKLOADS or not isinstance(names, list):
                raise ValueError(f"Invalid workload preference: {workload}")
            if unknown := set(names) - self.endpoints.keys():
                raise ValueError(f"Unknown endpoints in {workload}: {sorted(unknown)}")
        self._catalogues = {}
        self._locks = {}

    async def models(self, url: str, *, refresh: bool = False) -> list[dict]:
        """Cache successes and failures; endpoints sharing a URL share one probe."""
        async with self._locks.setdefault(url, asyncio.Lock()):
            cached = self._catalogues.get(url)
            if refresh or cached is None or monotonic() - cached[0] >= 30:
                try:
                    async with (
                        asyncio.timeout(2),
                        httpx.AsyncClient(timeout=2) as client,
                    ):
                        response = await client.get(
                            f"{url}/v1/models",
                            headers=HeaderManager.create_auth_headers(),
                        )
                        response.raise_for_status()
                        data = response.json()
                        models = data if isinstance(data, list) else data["data"]
                        if not isinstance(models, list):
                            raise ValueError("Model catalogue must be a list")
                        result = list(
                            {
                                m["id"]: m
                                for m in models
                                if isinstance(m, dict)
                                and isinstance(m.get("id"), str)
                                and m["id"]
                            }.values()
                        )
                except (
                    httpx.HTTPError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as exc:
                    result = exc
                cached = self._catalogues[url] = (monotonic(), result)
            if isinstance(cached[1], Exception):
                raise cached[1]
            return cached[1]

    async def resolve(
        self, name: str, *, model: str | None = None, refresh=False
    ) -> dict:
        if name not in self.endpoints:
            raise HTTPException(404, f"Endpoint not found: {name}")
        endpoint = self.endpoints[name]
        preferred = model or endpoint.get("model")
        try:
            models = await self.models(endpoint["url"], refresh=refresh)
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            if preferred:
                logger.warning(
                    "Discovery failed for %s; trying configured model: %s", name, exc
                )
                return {**endpoint, "model": preferred}
            raise HTTPException(
                503, f"Cannot discover model for {name}: {exc}"
            ) from exc
        ids = [m["id"] for m in models]
        if preferred in ids:
            return {**endpoint, "model": preferred}
        if not model and len(ids) == 1:
            if preferred:
                logger.warning(
                    "Model %s missing on %s; using %s", preferred, name, ids[0]
                )
            return {**endpoint, "model": ids[0]}
        raise HTTPException(
            409 if ids else 503,
            f"Configure an explicit advertised model for {name}; available models: {ids}",
        )

    @asynccontextmanager
    async def request(
        self,
        body,
        *,
        endpoint=None,
        workload_type=None,
        headers=None,
        timeout=120,
        before_request=None,
    ):
        """Fail over only before yielding a successful response, including for SSE."""
        if endpoint and workload_type:
            raise ValueError("Specify endpoint or workload_type, not both")
        names = [endpoint] if endpoint else self.workloads.get(workload_type, [])
        failures = {}
        upstream_headers = filter_unsafe_headers(dict(headers or {}))
        upstream_headers.update(HeaderManager.create_auth_headers())
        upstream_headers["Content-Type"] = "application/json"
        if body.get("stream"):
            upstream_headers["Accept"] = "text/event-stream"
        async with httpx.AsyncClient(timeout=timeout) as client:
            for name in names:
                for attempt in range(2):
                    async with AsyncExitStack() as stack:
                        try:
                            target = await self.resolve(
                                name, model=body.get("model"), refresh=bool(attempt)
                            )
                            payload = {**body, "model": target["model"]}
                            if before_request:
                                await before_request(target, payload)
                            url = target["url"] + "/v1/chat/completions"
                            if body.get("stream"):
                                response = await stack.enter_async_context(
                                    client.stream(
                                        "POST",
                                        url,
                                        json=payload,
                                        headers=upstream_headers,
                                    )
                                )
                            else:
                                response = await client.post(
                                    url, json=payload, headers=upstream_headers
                                )
                            if response.status_code >= 400:
                                await response.aread()
                                if (
                                    _model_rejected(response)
                                    and not body.get("model")
                                    and not attempt
                                ):
                                    continue
                            response.raise_for_status()
                        except HTTPException as exc:
                            if endpoint:
                                raise
                            failures[name] = exc.detail
                            break
                        except (httpx.HTTPError, TimeoutError) as exc:
                            if isinstance(exc, httpx.HTTPStatusError):
                                status = exc.response.status_code
                                if (
                                    status != 429
                                    and status < 500
                                    and not _model_rejected(exc.response)
                                ):
                                    raise
                            if endpoint:
                                raise
                            failures[name] = str(exc)
                            break
                        # Deliberately outside the retry handler: stream failures never replay.
                        yield target, response
                        return
        raise HTTPException(
            503,
            {
                "message": f"No usable endpoint for {workload_type}",
                "endpoints": failures,
            },
        )

    async def complete(self, payload: dict, *, timeout: float = 7) -> dict:
        """Injected utility completion callback; callers own parsing and telemetry."""
        async with asyncio.timeout(timeout):
            async with self.request(
                payload,
                workload_type="utility",
                timeout=httpx.Timeout(timeout, connect=min(timeout, 2)),
            ) as (target, response):
                return {**response.json(), "endpoint": target["name"]}

    async def classify(self, messages: list[dict]) -> str:
        if has_images(messages):
            return "vision"
        content = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        try:
            with CallMetrics(label="Routing classification") as call:
                data = await self.complete(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "Classify the request. Reply complex for reasoning, analysis, or programming; otherwise responsive. Reply with one word only.",
                            },
                            {"role": "user", "content": content},
                        ],
                        "max_tokens": 20,
                        "temperature": 0,
                        "reasoning_effort": "none",
                        "stream": False,
                    },
                    timeout=10,
                )
                call.metadata.update(llm_response_metadata(data))
                call.metadata["endpoint"] = data["endpoint"]
                result = (extract_assistant_text(data) or "").strip().lower()
                if result in {"complex", "responsive"}:
                    return result
                call.on_error("Invalid classification")
        except (
            httpx.HTTPError,
            HTTPException,
            TimeoutError,
            ValueError,
            KeyError,
            TypeError,
        ):
            logger.warning("Classification failed; using responsive", exc_info=True)
        return "responsive"

    async def list_models(self) -> dict:
        errors = {}

        async def collect(name):
            try:
                target = await self.resolve(name)
                try:
                    advertised = await self.models(target["url"])
                except (
                    httpx.HTTPError,
                    TimeoutError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as exc:
                    advertised = []
                    errors[name] = str(exc)
                metadata = next(
                    (m for m in advertised if m["id"] == target["model"]), {}
                )
                return {
                    **metadata,
                    **{
                        k: v
                        for k, v in target.items()
                        if k in {"gpu", "vram", "soc", "cpu", "ram"}
                    },
                    "id": target["model"],
                    "object": "model",
                    "endpoint": name,
                    "endpoint_url": target["url"],
                }
            except HTTPException as exc:
                errors[name] = exc.detail

        models = await asyncio.gather(*(collect(name) for name in self.endpoints))
        return {"object": "list", "data": [m for m in models if m], "errors": errors}


def has_images(messages: list[dict]) -> bool:
    return any(
        isinstance(m.get("content"), list)
        and any(
            isinstance(part, dict)
            and part.get("type") in {"image", "image_url", "input_image"}
            for part in m["content"]
        )
        for m in messages
    )


def _model_rejected(response: httpx.Response) -> bool:
    if response.status_code not in {400, 404, 422}:
        return False
    text = response.text.lower()
    return "model" in text and any(
        term in text
        for term in (
            "not found",
            "not_found",
            "unknown",
            "does not exist",
            "not loaded",
            "invalid model",
        )
    )
