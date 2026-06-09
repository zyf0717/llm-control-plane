import json

from src.logging_config import LOG_DIR_ENV
from src.search.safety import EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER
from src.orchestrator.trace import RequestTrace


def test_request_trace_emits_sanitized_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv(LOG_DIR_ENV, str(tmp_path))
    trace = RequestTrace(convo_id="convo-1", endpoint="primary")
    trace.capture_headers(
        {
            "X-Route-Decision": "primary",
            "X-RAG-Injected": "true",
            "X-Upstream-Slot-ID": "2",
            "Authorization": "secret-token",
        }
    )
    trace.capture_search_from_body(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{EPHEMERAL_WEB_SEARCH_CONTEXT_MARKER}\n"
                        '{"provider":"duckduckgo_html","query":"secret prompt",'
                        '"results":[{"title":"secret result"}],"degraded":true,'
                        '"warnings":["secret warning"]}'
                    ),
                }
            ]
        }
    )

    trace.emit(status_code=200)
    trace.emit(status_code=500)

    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    serialized = json.dumps(payload)
    assert payload["request_id"] == trace.request_id
    assert payload["convo_id"] == "convo-1"
    assert payload["endpoint"] == "primary"
    assert payload["route"] == {"decision": "primary"}
    assert payload["rag"] == {"injected": "true"}
    assert payload["slot"] == {"id": "2"}
    assert payload["search"] == {
        "injected": "true",
        "provider": "duckduckgo_html",
        "result_count": 1,
        "degraded": True,
        "warning_count": 1,
    }
    assert payload["status_code"] == 200
    assert "secret-token" not in serialized
    assert "secret prompt" not in serialized
    assert "secret result" not in serialized
    assert "secret warning" not in serialized
