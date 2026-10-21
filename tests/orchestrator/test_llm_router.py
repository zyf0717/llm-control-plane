import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException

from src.orchestrator.llm_router import LLMRouter
from src.llm_metrics import capture_llm_calls


@pytest.fixture
def router():
    return LLMRouter(
        {
            "endpoints": {
                "fast": {"url": "https://fast.local", "model": "configured"},
                "strong": {"url": "https://strong.local"},
            },
            "workloads": {
                "utility": ["fast", "strong"],
                "complex": ["strong", "fast"],
                "responsive": ["fast", "strong"],
                "vision": ["strong"],
            },
        }
    )


def catalogue(*ids):
    return httpx.Response(200, json={"data": [{"id": name} for name in ids]})


def completion(text="OK", model="served"):
    return httpx.Response(
        200, json={"model": model, "choices": [{"message": {"content": text}}]}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "advertised", "selected"),
    [
        ("chosen", ["chosen", "other"], "chosen"),
        (None, ["only"], "only"),
        ("stale", ["only"], "only"),
    ],
)
async def test_model_resolution(router, http_backend, configured, advertised, selected):
    router.endpoints["fast"]["model"] = configured
    requests = http_backend(
        lambda r: catalogue(*advertised) if r.method == "GET" else completion()
    )
    await router.complete({"messages": []})
    assert json.loads(requests[-1].content)["model"] == selected


@pytest.mark.asyncio
@pytest.mark.parametrize("advertised", [[], ["a", "b"]])
async def test_unresolved_default_is_not_guessed(router, http_backend, advertised):
    http_backend(lambda r: catalogue(*advertised))
    with pytest.raises(HTTPException) as error:
        await router.resolve("fast")
    assert error.value.status_code == (409 if advertised else 503)
    assert "available models" in error.value.detail


@pytest.mark.asyncio
async def test_explicit_request_model_is_never_replaced(router, http_backend):
    requests = http_backend(lambda r: catalogue("served"))
    with pytest.raises(HTTPException, match="advertised"):
        async with router.request({"model": "requested"}, endpoint="fast"):
            pytest.fail("Invalid explicit model must not generate")
    assert all(r.method == "GET" for r in requests)


@pytest.mark.asyncio
async def test_shared_url_discovers_once_and_keeps_separate_models(
    router, http_backend
):
    router.endpoints["strong"].update(url="https://fast.local", model="second")
    requests = http_backend(lambda r: catalogue("configured", "second"))
    listed = await router.list_models()
    assert {m["endpoint"]: m["id"] for m in listed["data"]} == {
        "fast": "configured",
        "strong": "second",
    }
    assert len(requests) == 1
    assert not listed["errors"]


@pytest.mark.asyncio
async def test_discovery_failure_tries_configured_model(router, http_backend):
    def backend(request):
        return httpx.Response(404) if request.method == "GET" else completion()

    requests = http_backend(backend)
    data = await router.complete({"messages": []})
    assert data["endpoint"] == "fast"
    assert json.loads(requests[-1].content)["model"] == "configured"


@pytest.mark.asyncio
async def test_rejected_model_refreshes_and_retries_once(router, http_backend):
    probes = 0

    def backend(request):
        nonlocal probes
        if request.method == "GET":
            probes += 1
            return catalogue("configured" if probes == 1 else "new")
        model = json.loads(request.content)["model"]
        return (
            httpx.Response(404, json={"error": "model not found"})
            if model == "configured"
            else completion()
        )

    requests = http_backend(backend)
    assert (await router.complete({"messages": []}))["endpoint"] == "fast"
    assert [json.loads(r.content)["model"] for r in requests if r.method == "POST"] == [
        "configured",
        "new",
    ]
    assert probes == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connect", 429, 503, "ambiguous"])
async def test_ordered_fallback(router, http_backend, failure):
    def backend(request):
        if request.method == "GET":
            return (
                catalogue("a", "b")
                if failure == "ambiguous" and request.url.host == "fast.local"
                else catalogue("served")
            )
        if request.url.host == "fast.local":
            if failure == "connect":
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(failure)
        return completion()

    http_backend(backend)
    assert (await router.complete({"messages": []}))["endpoint"] == "strong"


@pytest.mark.asyncio
async def test_request_errors_do_not_fail_over(router, http_backend):
    requests = http_backend(
        lambda r: (
            catalogue("served")
            if r.method == "GET"
            else httpx.Response(400, json={"error": "invalid messages"})
        )
    )
    with pytest.raises(httpx.HTTPStatusError):
        await router.complete({"messages": []})
    assert all(r.url.host == "fast.local" for r in requests)


@pytest.mark.asyncio
async def test_no_fallback_outside_preference_list(router, http_backend):
    router.workloads["utility"] = []
    requests = http_backend(lambda r: completion())
    with pytest.raises(HTTPException) as exc:
        await router.complete({"messages": []})
    assert exc.value.status_code == 503
    assert requests == []


@pytest.mark.asyncio
async def test_probe_timeout_is_bounded_and_other_endpoints_survive(
    router, http_backend
):
    async def backend(request):
        if request.url.host == "fast.local":
            await asyncio.sleep(30)
        return catalogue("served")

    http_backend(backend)
    router.endpoints["fast"].pop("model")
    async with asyncio.timeout(3):
        listed = await router.list_models()
    assert [m["endpoint"] for m in listed["data"]] == ["strong"]
    assert "fast" in listed["errors"]


@pytest.mark.asyncio
async def test_discovery_cache_expires(router, http_backend, monkeypatch):
    import src.orchestrator.llm_router as routing

    now = 1
    monkeypatch.setattr(routing, "monotonic", lambda: now)
    requests = http_backend(lambda r: catalogue("served"))
    await router.resolve("fast")
    await router.resolve("fast")
    assert len(requests) == 1
    now += 31
    await router.resolve("fast")
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_shared_transport_sets_auth_and_actual_model(router, http_backend):
    requests = http_backend(
        lambda r: catalogue("actual") if r.method == "GET" else completion()
    )
    await router.complete({"messages": [], "reasoning_effort": "none"})
    for request in requests:
        assert request.headers["cf-access-client-id"] == "test-api-key-id"
        assert request.headers["cf-access-client-secret"] == "test-api-secret"
    assert json.loads(requests[-1].content)["reasoning_effort"] == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("complex", "complex"),
        ("responsive", "responsive"),
        ("unexpected", "responsive"),
    ],
)
async def test_classifier_uses_utility_and_disables_reasoning(
    router, http_backend, answer, expected
):
    requests = http_backend(
        lambda r: catalogue("actual") if r.method == "GET" else completion(answer)
    )
    with capture_llm_calls() as calls:
        assert (
            await router.classify([{"role": "user", "content": "write code"}])
            == expected
        )
    payload = json.loads(requests[-1].content)
    assert requests[-1].url.host == "fast.local"
    assert payload["reasoning_effort"] == "none"
    assert payload["model"] == "actual"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_classifier_outage_falls_back_to_responsive(router, http_backend):
    http_backend(lambda r: httpx.Response(503))
    assert await router.classify([{"role": "user", "content": "hi"}]) == "responsive"


def test_bad_config_references_fail_at_startup():
    with pytest.raises(ValueError, match="Unknown endpoints"):
        LLMRouter({"workloads": {"utility": ["missing"]}})


@pytest.mark.asyncio
async def test_stream_read_failure_after_success_never_tries_another_endpoint(
    router, http_backend
):
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("stream lost")

    requests = http_backend(
        lambda r: (
            catalogue("served")
            if r.method == "GET"
            else httpx.Response(200, stream=BrokenStream())
        )
    )
    with pytest.raises(httpx.ReadError, match="stream lost"):
        async with router.request({"stream": True}, workload_type="utility") as (
            target,
            response,
        ):
            assert target["name"] == "fast"
            _ = [line async for line in response.aiter_lines()]
    assert [r.url.host for r in requests if r.method == "POST"] == ["fast.local"]


@pytest.mark.asyncio
async def test_ambiguous_model_after_rejection_does_not_retry_a_guessed_model(
    router, http_backend
):
    probes = 0

    def backend(request):
        nonlocal probes
        if request.method == "GET":
            probes += 1
            return catalogue("configured") if probes == 1 else catalogue("a", "b")
        return httpx.Response(404, json={"error": "model not found"})

    requests = http_backend(backend)
    with pytest.raises(HTTPException) as exc:
        async with router.request({}, endpoint="fast"):
            pytest.fail("Ambiguous replacement must not generate")
    assert exc.value.status_code == 409
    assert len([r for r in requests if r.method == "POST"]) == 1
