import asyncio

import pytest

from src.llm_metrics import CallMetrics, capture_llm_calls, capture_llm_stream, without_call_metrics


@pytest.mark.asyncio
async def test_nested_parallel_calls_are_collected_once_and_requests_are_isolated():
    async def request(name):
        with capture_llm_calls() as outer:
            async def branch(index):
                with capture_llm_calls() as inner:
                    with CallMetrics(label=f"{name}-{index}"):
                        await asyncio.sleep(0)
                assert len(inner) == 1
                return inner[0]
            nested = await asyncio.gather(branch(1), branch(2))
        assert {call["id"] for call in outer} == {call["id"] for call in nested}
        return outer

    first, second = await asyncio.gather(request("a"), request("b"))
    assert {call["label"] for call in first} == {"a-1", "a-2"}
    assert {call["label"] for call in second} == {"b-1", "b-2"}


@pytest.mark.parametrize("error,status", [(RuntimeError("lost"), "Failed"),
                                          (asyncio.CancelledError(), "Stopped")])
def test_failed_and_cancelled_calls_keep_partial_metadata(error, status):
    with capture_llm_calls() as calls:
        with pytest.raises(type(error)):
            with CallMetrics(metadata={"usage": {"completion_tokens": 2}}) as call:
                call.on_first_token()
                raise error
    assert len(calls) == 1
    assert calls[0]["status"] == status
    assert 0 <= calls[0]["ttft"] <= calls[0]["elapsed"]
    assert calls[0]["metadata"]["usage"]["completion_tokens"] == 2
    call.metadata["usage"]["completion_tokens"] = 99
    assert calls[0]["metadata"]["usage"]["completion_tokens"] == 2


def test_model_inputs_exclude_nested_telemetry_without_mutating_outputs():
    call = CallMetrics().record()
    payload = {"text": "answer", "metadata": {"llm_calls": [call]},
               "artifacts": [{"content_json": {"llm_calls": [call]}}],
               "user_data": {"llm_calls": ["ordinary data"]}}
    cleaned = without_call_metrics(payload)
    assert cleaned == {"text": "answer", "metadata": {},
                       "artifacts": [{"content_json": {}}],
                       "user_data": {"llm_calls": ["ordinary data"]}}
    assert payload["metadata"]["llm_calls"] == [call]


@pytest.mark.asyncio
async def test_stream_scope_does_not_capture_consumer_work_and_closes_in_other_task():
    async def source():
        with CallMetrics(label="streamed"):
            yield "first"
            yield "second"

    calls = []
    stream = capture_llm_stream(source(), calls)
    assert await anext(stream) == "first"
    with CallMetrics(label="unrelated"):
        pass
    # Async generators may be closed by a finalizer task rather than their consumer.
    await asyncio.create_task(stream.aclose())
    assert [call["label"] for call in calls] == ["streamed"]
    assert calls[0]["status"] == "Stopped"
