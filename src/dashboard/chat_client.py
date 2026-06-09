import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Callable, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL")


from .cache_telemetry import normalize_cache_telemetry

logger = logging.getLogger(__name__)
MetadataCallback = Callable[[Optional[Dict[str, Any]]], None]
StateCallback = Callable[[str], None]
RuntimeCallback = Callable[[float], None]


def build_chat_messages(
    *,
    text: str,
    system_prompt: Optional[str] = None,
    extra_turn_messages: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, str]]:
    """Build outbound chat messages in stable order for one dashboard turn."""
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if extra_turn_messages:
        for message in extra_turn_messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip()
            content = message.get("content")
            if not role or not isinstance(content, str) or not content.strip():
                continue
            messages.append({"role": role, "content": content.strip()})

    messages.append({"role": "user", "content": text})
    return messages


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
        trace_id = ""
        for header_name, header_value in response_headers.items():
            if header_name.lower() == "x-trace-id":
                trace_id = str(header_value or "").strip()
                break
        if trace_id:
            combined["trace"] = {"id": trace_id}

        rag_info = {}
        for header_name, header_value in response_headers.items():
            if header_name.lower().startswith("x-rag-"):
                key = header_name.lower().replace("x-rag-", "")
                rag_info[key] = header_value
        if rag_info:
            combined["rag"] = rag_info

    return combined


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
    extra_turn_messages: Optional[list[dict[str, Any]]] = None,
    rag_endpoint: Optional[str] = None,
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

        payload = {
            "messages": build_chat_messages(
                text=text,
                system_prompt=system_prompt,
                extra_turn_messages=extra_turn_messages,
            )
        }
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
        metadata_holder: Dict[str, Any] = {}

        def publish_metadata(
            obj: Dict[str, Any], response_headers: Optional[Dict[str, Any]] = None
        ) -> None:
            combined = _extract_metadata(
                obj,
                endpoint_key=endpoint_key,
                response_headers=response_headers,
                preserve_routing_from=routing_info_holder,
            )
            if not combined:
                return

            if "routing" in combined:
                routing_info_holder["routing"] = combined["routing"]

            cache = normalize_cache_telemetry(combined)
            if cache["status"] != "unknown":
                combined["cache"] = cache

            updated_metadata = dict(metadata_holder)
            updated_metadata.update(combined)
            if updated_metadata == metadata_holder:
                return

            metadata_holder.clear()
            metadata_holder.update(updated_metadata)

            if on_metadata:
                on_metadata(dict(metadata_holder))

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

                    async for line in response.aiter_lines():
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

                            content_chunk = (
                                choices[0].get("delta", {}).get("content", "")
                            )
                            if content_chunk:
                                if reasoning_chunk_found:
                                    yield "</em>\n\n---\n\n"
                                    reasoning_chunk_found = False
                                yield content_chunk

                        publish_metadata(obj)

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
                    content = content.replace("<|channel|>analysis<|message|>", "<em>")
                    content = content.replace(
                        "<|end|><|start|>assistant<|channel|>final<|message|>",
                        "</em>\n\n---\n\n",
                    )
                    yield str(content)
    except httpx.HTTPStatusError as exc:
        reason = getattr(exc.response, "reason_phrase", "Unknown Error")
        yield f"HTTP Error {exc.response.status_code}: {reason}"
    except httpx.RequestError as exc:
        yield f"Request Error: {str(exc)}"
    except Exception as exc:
        logger.exception(
            "Unexpected error in stream_chat_response: %s: %s",
            type(exc).__name__,
            exc,
        )
        yield f"Error: {str(exc)}"
    finally:
        if on_runtime:
            on_runtime(time.time() - started_at)
        if on_send_button_state:
            on_send_button_state("ready")
