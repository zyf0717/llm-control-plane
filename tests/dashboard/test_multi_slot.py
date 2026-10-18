import asyncio

import pytest

from src.dashboard import multi_slot
from src.dashboard.multi_slot import SlotConversation, parse_generation_settings


def test_generation_settings_keep_defaults_and_explicit_zero():
    assert parse_generation_settings("", "  ") == {}
    assert parse_generation_settings("0", "32") == {"temperature": 0.0, "max_tokens": 32}
    assert parse_generation_settings(" 0.7 ", " 1024 ") == {
        "temperature": 0.7, "max_tokens": 1024,
    }


@pytest.mark.parametrize("temperature", ["-1", "nan", "inf", "-inf", "hello"])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        parse_generation_settings(temperature, "")


@pytest.mark.parametrize("max_tokens", ["0", "-1", "2.5", "nan", "hello"])
def test_invalid_token_cap(max_tokens):
    with pytest.raises(ValueError, match="positive integer"):
        parse_generation_settings("", max_tokens)


def test_prepare_reserves_pane_and_snapshots_settings():
    pane = SlotConversation(history=[{"role": "user", "content": "prior turn"}])
    settings = {"temperature": 0.0, "endpoints_dict": {"node": {"models": []}}}
    request = pane.prepare(" hello ", settings)
    settings["temperature"] = 1.0
    settings["endpoints_dict"].clear()
    pane.history.clear()

    assert request["text"] == "hello"
    assert request["temperature"] == 0.0
    assert "node" in request["endpoints_dict"]
    assert request["extra_turn_messages"] == [{"role": "user", "content": "prior turn"}]
    with pytest.raises(ValueError, match="in progress"):
        pane.prepare("duplicate", {})


def test_empty_prompt_does_not_reserve_pane():
    pane = SlotConversation()
    with pytest.raises(ValueError, match="Enter a message"):
        pane.prepare("  ", {})
    assert not pane.busy


@pytest.mark.asyncio
async def test_independent_requests_overlap_and_keep_histories_separate(monkeypatch):
    started = []
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def fake_stream(**request):
        started.append(request)
        if len(started) == 3:
            all_started.set()
        await release.wait()
        yield f"Answer to {request['text']}"

    monkeypatch.setattr(multi_slot, "stream_chat_response", fake_stream)
    panes = [SlotConversation() for _ in range(3)]
    settings = {"endpoint_key": "node", "endpoints_dict": {"node": {"models": []}}}

    async def collect(pane, request):
        try:
            return "".join([chunk async for chunk in pane.response(request)])
        finally:
            pane.finish()

    tasks = [
        asyncio.create_task(collect(pane, pane.prepare(f"prompt-{i}", settings)))
        for i, pane in enumerate(panes)
    ]
    try:
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert all(not task.done() for task in tasks)
        assert all(pane.busy for pane in panes)
    finally:
        release.set()
        responses = await asyncio.gather(*tasks)

    assert responses == [f"Answer to prompt-{i}" for i in range(3)]
    for i, pane in enumerate(panes):
        assert not pane.busy
        assert pane.status == "Completed"
        assert pane.elapsed >= 0
        request = pane.prepare("follow-up", settings)
        assert request["extra_turn_messages"] == [
            {"role": "user", "content": f"prompt-{i}"},
            {"role": "assistant", "content": f"Answer to prompt-{i}"},
        ]
        assert "conversation_id" not in request
        assert "id_slot" not in request
    assert {request["endpoint_key"] for request in started} == {"node"}


@pytest.mark.asyncio
async def test_failure_does_not_enter_history_or_break_other_panes(monkeypatch):
    async def fake_stream(**request):
        if request["text"] == "fail":
            yield "partial"
            request["on_error"]("Upstream failed")
            yield "Error: Upstream failed"
        else:
            yield "healthy response"

    monkeypatch.setattr(multi_slot, "stream_chat_response", fake_stream)
    failed, healthy = SlotConversation(), SlotConversation()

    async def collect(pane, text):
        result = [chunk async for chunk in pane.response(pane.prepare(text, {}))]
        pane.finish()
        return result

    failure, success = await asyncio.gather(collect(failed, "fail"), collect(healthy, "ok"))
    assert failure == ["partial", "Error: Upstream failed"]
    assert success == ["healthy response"]
    assert failed.history == []
    assert failed.status == "Failed"
    assert healthy.status == "Completed"
    assert healthy.history[-1]["content"] == "healthy response"


@pytest.mark.asyncio
async def test_cancellation_closes_stream_and_keeps_partial_reply_out_of_history(monkeypatch):
    started = asyncio.Event()
    closed = asyncio.Event()

    async def fake_stream(**_request):
        try:
            yield "partial"
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(multi_slot, "stream_chat_response", fake_stream)
    pane = SlotConversation()
    request = pane.prepare("hello", {})

    async def collect():
        try:
            return [chunk async for chunk in pane.response(request)]
        finally:
            pane.finish(cancelled=True)

    task = asyncio.create_task(collect())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert pane.status == "Stopped"
    assert not pane.busy
    assert pane.elapsed >= 0
    assert pane.history == []
