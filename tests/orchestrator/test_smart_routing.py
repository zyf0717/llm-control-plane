import json

import httpx
import pytest
from fastapi.testclient import TestClient

from src.orchestrator import proxy_services as services
from src.orchestrator.conversation_store import MemoryConversationStore
from src.orchestrator.llm_router import LLMRouter
from src.orchestrator.proxy import app


@pytest.fixture
def routing(monkeypatch):
    router = LLMRouter(
        {
            "endpoints": {
                "fast": {"url": "https://fast.local", "model": "small"},
                "strong": {
                    "url": "https://strong.local",
                    "model": "big",
                    "soc": "test device",
                },
            },
            "workloads": {
                "utility": ["fast", "strong"],
                "complex": ["strong", "fast"],
                "responsive": ["fast", "strong"],
                "vision": ["strong"],
            },
        }
    )
    monkeypatch.setattr(services, "llm_router", router)
    monkeypatch.setattr(services, "conversation_store", MemoryConversationStore())
    return router


def upstream(classification="complex", failing=None):
    def respond(request):
        model = "small" if request.url.host == "fast.local" else "big"
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": model}]})
        payload = json.loads(request.content)
        classifier = payload.get("max_tokens") == 20
        if failing == request.url.host and not classifier:
            return httpx.Response(503)
        if payload.get("stream"):
            return httpx.Response(
                200,
                text='data: {"model":"'
                + model
                + '","choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [
                    {"message": {"content": classification if classifier else "answer"}}
                ],
            },
        )

    return respond


@pytest.mark.parametrize("path", ["/", "/smart", "/smart/v1/chat/completions"])
def test_automatic_classification_and_actual_routing_metadata(
    routing, http_backend, path
):
    requests = http_backend(upstream())
    response = TestClient(app).post(
        path, json={"messages": [{"role": "user", "content": "write code"}]}
    )
    assert response.status_code == 200
    assert response.json()["model"] == "big"
    assert response.headers["x-route-decision"] == "strong"
    assert response.headers["x-route-strategy"] == "complex"
    assert response.headers["x-route-soc"] == "test device"
    assert "x-route-confidence" not in response.headers
    assert [r.url.host for r in requests if r.method == "POST"] == [
        "fast.local",
        "strong.local",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"messages": []},
        {"messages": [{"role": "system", "content": "hi"}]},
        {"messages": ["invalid"]},
        [],
    ],
)
def test_invalid_messages_are_rejected_before_upstream(routing, http_backend, payload):
    requests = http_backend(upstream())
    response = TestClient(app).post("/smart", json=payload)
    assert response.status_code == 400
    assert requests == []


def test_invalid_json_is_rejected(routing):
    assert TestClient(app).post("/smart", content="invalid").status_code == 400


@pytest.mark.parametrize("stream", [False, True])
def test_fallback_uses_actual_endpoint_and_pins_only_success(
    routing, http_backend, stream
):
    requests = http_backend(upstream(failing="strong.local"))
    response = TestClient(app).post(
        "/smart",
        headers={"X-Conversation-ID": "fallback"},
        json={
            "stream": stream,
            "messages": [{"role": "user", "content": "reason"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-route-decision"] == "fast"
    assert (
        services.conversation_store.conversation_control_states["fallback"][
            "route_endpoint"
        ]
        == "fast"
    )
    assert services.conversation_store.conversations["fallback"] == [
        {"role": "user", "content": "reason"},
        {"role": "assistant", "content": "answer"},
    ]
    assert [r.url.host for r in requests if r.method == "POST"] == [
        "fast.local",
        "strong.local",
        "fast.local",
    ]


def test_existing_pin_skips_classifier_and_does_not_silently_fail_over(
    routing, http_backend
):
    requests = http_backend(upstream(failing="strong.local"))
    services.conversation_store.conversation_control_states["pinned"] = {
        "route_endpoint": "strong",
        "reasoning_effort": "none",
        "slots": {},
    }
    response = TestClient(app).post(
        "/smart",
        headers={"X-Conversation-ID": "pinned"},
        json={
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 503
    assert [r.url.host for r in requests] == ["strong.local", "strong.local"]


def test_stale_pin_is_cleared(routing, http_backend):
    http_backend(upstream("responsive"))
    services.conversation_store.conversation_control_states["stale"] = {
        "route_endpoint": "removed",
        "slots": {},
    }
    response = TestClient(app).post(
        "/smart",
        headers={"X-Conversation-ID": "stale"},
        json={
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-route-pin-stale"] == "true"
    assert response.headers["x-route-decision"] == "fast"


VISION = [
    {"type": "text", "text": "describe"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
]


def test_vision_bypasses_classifier_and_forwards_content_parts(routing, http_backend):
    requests = http_backend(upstream())
    response = TestClient(app).post(
        "/smart", json={"messages": [{"role": "user", "content": VISION}]}
    )
    assert response.status_code == 200
    assert response.headers["x-route-strategy"] == "vision"
    posts = [r for r in requests if r.method == "POST"]
    assert len(posts) == 1
    assert posts[0].url.host == "strong.local"
    assert json.loads(posts[0].content)["messages"][0]["content"] == VISION


def test_vision_does_not_fall_back_to_text_only_endpoint(routing, http_backend):
    requests = http_backend(upstream(failing="strong.local"))
    response = TestClient(app).post(
        "/smart", json={"messages": [{"role": "user", "content": VISION}]}
    )
    assert response.status_code == 503
    assert all(r.url.host == "strong.local" for r in requests)


@pytest.mark.parametrize("allow_switch", [False, True])
def test_vision_respects_route_switch_control(routing, http_backend, allow_switch):
    http_backend(upstream())
    services.conversation_store.conversation_control_states["vision-pin"] = {
        "route_endpoint": "fast",
        "slots": {},
    }
    response = TestClient(app).post(
        "/smart",
        headers={
            "X-Conversation-ID": "vision-pin",
            "X-Allow-Route-Switch": str(allow_switch),
        },
        json={"messages": [{"role": "user", "content": VISION}]},
    )
    assert response.status_code == (200 if allow_switch else 409)
    if allow_switch:
        assert response.headers["x-route-decision"] == "strong"
        assert response.headers["x-route-switched"] == "true"
    else:
        assert "vision-pin" not in services.conversation_store.conversations


def test_models_uses_same_resolver_and_reports_ambiguity(routing, http_backend):
    def backend(request):
        ids = ["a", "b"] if request.url.host == "fast.local" else ["big"]
        return httpx.Response(200, json={"data": [{"id": name} for name in ids]})

    http_backend(backend)
    response = TestClient(app).get("/models")
    assert response.status_code == 200
    assert [m["endpoint"] for m in response.json()["data"]] == ["strong"]
    assert "fast" in response.json()["errors"]
