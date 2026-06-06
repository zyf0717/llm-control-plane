import asyncio
import json
import os
import time
from contextlib import suppress
from typing import Any, AsyncGenerator, Callable, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL")


MetadataCallback = Callable[[Optional[Dict[str, Any]]], None]
StateCallback = Callable[[str], None]
RuntimeCallback = Callable[[float], None]


def _extract_metadata(
    obj: Dict[str, Any],
    *,
    endpoint_key: str,
    response_headers: Optional[Dict[str, Any]] = None,
    preserve_routing_from: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract metadata from response payloads and headers."""
    metadata_keys = ("stats", "usage", "model_info", "runtime", "timings")
    combined: Dict[str, Any] = {}

    if preserve_routing_from and "routing" in preserve_routing_from:
        combined["routing"] = preserve_routing_from["routing"]

    for key in metadata_keys:
        if key in obj and isinstance(obj[key], dict):
            combined[key] = obj[key]

    is_auto_routing = (
        endpoint_key == "smart"
        or endpoint_key == "Auto"
        or (
            response_headers
            and any(h.lower().startswith("x-route-") for h in response_headers.keys())
        )
    )

    if response_headers and is_auto_routing:
        routing_info = {}
        for header_name, header_value in response_headers.items():
            if header_name.lower().startswith("x-route-"):
                key = header_name.lower().replace("x-route-", "")
                routing_info[key] = header_value

        if routing_info:
            combined["routing"] = routing_info

    if response_headers:
        rag_info = {}
        for header_name, header_value in response_headers.items():
            if header_name.lower().startswith("x-rag-"):
                key = header_name.lower().replace("x-rag-", "")
                rag_info[key] = header_value
        if rag_info:
            combined["rag"] = rag_info

    return combined


async def _close_stream_response(response: httpx.Response) -> None:
    with suppress(Exception):
        await response.aclose()


async def _iter_stream_lines(
    response: httpx.Response,
    stop_event: Optional[asyncio.Event],
) -> AsyncGenerator[str, None]:
    line_iter = response.aiter_lines()
    while True:
        if stop_event is not None and stop_event.is_set():
            await _close_stream_response(response)
            return

        line_task = asyncio.create_task(anext(line_iter))
        stop_task = None
        try:
            if stop_event is not None:
                stop_task = asyncio.create_task(stop_event.wait())
                done, _pending = await asyncio.wait(
                    {line_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done:
                    line_task.cancel()
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        await line_task
                    await _close_stream_response(response)
                    return
                line = line_task.result()
            else:
                line = await line_task
        except StopAsyncIteration:
            return
        finally:
            if stop_task is not None and not stop_task.done():
                stop_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stop_task

        yield line


async def stream_chat_response(
    *,
    endpoint_key: str,
    text: str,
    endpoints_dict: Dict[str, Any],
    stream: bool = True,
    output_json: bool = False,
    reasoning_effort: str = "medium",
    output_reasoning: bool = False,
    convo_id: Optional[str] = None,
    current_routing_info: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    rag_endpoint: Optional[str] = None,
    stop_event: Optional[asyncio.Event] = None,
    on_metadata: Optional[MetadataCallback] = None,
    on_send_button_state: Optional[StateCallback] = None,
    on_runtime: Optional[RuntimeCallback] = None,
) -> AsyncGenerator[str, None]:
    """Proxy a dashboard chat request and stream the response back to the UI."""
    text = (text or "Hello! What model are you?").strip()
    if not text:
        yield ""
        return

    started_at = time.time()
    if on_send_button_state:
        on_send_button_state("busy")

    try:
        if endpoint_key == "Auto":
            url = f"{PROXY_BASE_URL}/smart"
        else:
            if not endpoints_dict or endpoint_key not in endpoints_dict:
                yield f"Error: Endpoint '{endpoint_key}' not available"
                return
            url = f"{PROXY_BASE_URL}/{endpoint_key}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        payload = {"messages": messages}
        headers = {
            "CF-Access-Client-Id": API_KEY_ID,
            "CF-Access-Client-Secret": API_KEY_SECRET,
            "Content-Type": "application/json",
            "X-Reasoning-Effort": reasoning_effort,
        }
        if convo_id:
            headers["X-Convo-ID"] = convo_id
        if rag_endpoint:
            headers["X-RAG-Endpoint"] = rag_endpoint
        timeout = httpx.Timeout(connect=5, read=None, write=5, pool=10)
        routing_info_holder = {"routing": current_routing_info or {}}

        def publish_metadata(
            obj: Dict[str, Any], response_headers: Optional[Dict[str, Any]] = None
        ) -> None:
            combined = _extract_metadata(
                obj,
                endpoint_key=endpoint_key,
                response_headers=response_headers,
                preserve_routing_from=routing_info_holder,
            )
            if "routing" in combined:
                routing_info_holder["routing"] = combined["routing"]
            if on_metadata:
                on_metadata(combined if combined else None)

        async with httpx.AsyncClient(timeout=timeout) as client:
            if stream:
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    publish_metadata({}, dict(response.headers))

                    reasoning_chunk_found = False
                    md_code_wrap = True

                    async for line in _iter_stream_lines(response, stop_event):
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break

                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if output_json:
                            if md_code_wrap:
                                yield "```json\n"
                                md_code_wrap = False
                            yield f"{json.dumps(obj, indent=2)},\n"
                            continue

                        choices = obj.get("choices", [])
                        if choices:
                            if output_reasoning:
                                delta = choices[0].get("delta", {})
                                reasoning_chunk = delta.get("reasoning") or delta.get(
                                    "reasoning_content", ""
                                )
                                if reasoning_chunk:
                                    if reasoning_chunk_found is False:
                                        reasoning_chunk_found = True
                                        yield "<em>"
                                    yield reasoning_chunk

                            content_chunk = choices[0].get("delta", {}).get(
                                "content", ""
                            )
                            if content_chunk:
                                if reasoning_chunk_found:
                                    yield "</em>\n\n---\n\n"
                                    reasoning_chunk_found = False
                                yield content_chunk

                        publish_metadata(obj)

                    if reasoning_chunk_found:
                        yield "</em>"
                    if not md_code_wrap:
                        yield "```\n"
            else:
                response = await client.post(
                    url, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                data = response.json()
                publish_metadata(data, dict(response.headers))

                if output_json:
                    yield f"```json\n{json.dumps(data, indent=2)}\n```"
                else:
                    if output_reasoning:
                        message = data.get("choices", [{}])[0].get("message", {})
                        reasoning = message.get("reasoning") or message.get(
                            "reasoning_content", ""
                        )
                        if reasoning:
                            yield f"<em>{str(reasoning)}</em>\n\n---\n\n"
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    content = content.replace("<think>", "<em>")
                    content = content.replace("</think>", "</em>\n\n---\n\n")
                    content = content.replace(
                        "<|channel|>analysis<|message|>", "<em>"
                    )
                    content = content.replace(
                        "<|end|><|start|>assistant<|channel|>final<|message|>",
                        "</em>\n\n---\n\n",
                    )
                    yield str(content)
    except asyncio.CancelledError:
        return
    except httpx.HTTPStatusError as exc:
        reason = getattr(exc.response, "reason_phrase", "Unknown Error")
        yield f"HTTP Error {exc.response.status_code}: {reason}"
    except httpx.RequestError as exc:
        if stop_event is not None and stop_event.is_set():
            return
        yield f"Request Error: {str(exc)}"
    except Exception as exc:
        if stop_event is not None and stop_event.is_set():
            return
        print(
            f"Unexpected error in stream_chat_response: {type(exc).__name__}: {str(exc)}"
        )
        import traceback

        traceback.print_exc()
        yield f"Error: {str(exc)}"
    finally:
        if on_runtime:
            on_runtime(time.time() - started_at)
        if on_send_button_state:
            on_send_button_state("ready")
