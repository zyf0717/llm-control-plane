import json
import logging
import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from utils import SSEAccumulator

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
GPT_OSS_20B_API_URL = os.getenv("GPT_OSS_20B_API_URL")
QWEN_3_4B_API_URL = os.getenv("QWEN_3_4B_API_URL")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/")
async def root_chat(request: Request):
    # Route root POST requests directly to chat/completions
    return await proxy_with_context("", request)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def custom_endpoints(path: str, request: Request):
    # Skip health endpoint
    if path == "health":
        return await health_check()
    else:
        return await proxy_with_context(path, request)


# simple global store (not persistent across restarts)
convo_history = {}


async def proxy_with_context(path: str, request: Request):
    # Prepare headers
    headers = dict(request.headers)
    headers["CF-Access-Client-Id"] = API_KEY_ID
    headers["CF-Access-Client-Secret"] = API_KEY_SECRET
    headers.pop("host", None)
    headers.pop("content-length", None)

    endpoint_map = {
        "gpt-oss-20b": f"{GPT_OSS_20B_API_URL}/v1/chat/completions",
        "gpt-oss-20b-api": f"{GPT_OSS_20B_API_URL}/api/v0/chat/completions",
        "qwen3-4b": f"{QWEN_3_4B_API_URL}/v1/chat/completions",
        "qwen3-4b-api": f"{QWEN_3_4B_API_URL}/api/v0/chat/completions",
    }
    target_endpoint = endpoint_map.get(
        path, f"{GPT_OSS_20B_API_URL}/v1/chat/completions"
    )

    # Handle body
    body = await request.body()
    is_streaming = False
    if body:
        try:
            body_json = json.loads(body)
            if isinstance(body_json, dict):
                # Check if streaming is requested
                is_streaming = body_json.get("stream", False)

                # Ensure default overflow policy
                body_json.setdefault("contextOverflowPolicy", "rollingWindow")

                # Only inject history if there's a messages list
                if "messages" in body_json and isinstance(body_json["messages"], list):
                    # Get convo_id
                    convo_id = request.headers.get("X-Convo-ID")
                    headers["X-Convo-ID"] = convo_id

                    # Init history if not present
                    if convo_id not in convo_history:
                        convo_history[convo_id] = []

                    # Append new messages to history
                    convo_history[convo_id].extend(body_json["messages"])

                    # Replace payload messages with full history
                    body_json["messages"] = convo_history[convo_id]

                # Re-encode body
                body = json.dumps(body_json).encode()

        except Exception:
            logger.warning("Failed to parse request body as JSON")

    logger.info(
        "Proxying %s %s to %s (streaming: %s)",
        request.method,
        request.url.path,
        target_endpoint,
        is_streaming,
    )

    # Send request and receive response
    async with httpx.AsyncClient() as client:
        if is_streaming or request.query_params.get("stream") in {"true", "1"}:
            timeout = httpx.Timeout(connect=20, read=None, write=20, pool=20)
            upstream_headers = {
                k: v
                for k, v in headers.items()
                if k.lower() not in {"content-length", "host"}
            }
            upstream_headers["Accept"] = "text/event-stream"

            acc = SSEAccumulator()

            async def stream_response():
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client2:
                        async with client2.stream(
                            method=request.method,
                            url=target_endpoint,
                            headers=upstream_headers,
                            content=body if body else None,
                            params=dict(request.query_params),
                        ) as resp:
                            resp.raise_for_status()
                            async for chunk in resp.aiter_bytes():
                                # tap to accumulate assistant text
                                acc.feed(chunk)
                                # forward raw SSE bytes (`data: ...\n\n`)
                                yield chunk
                except httpx.HTTPStatusError as e:
                    msg = f'data: {{"type":"proxy.error","status":{e.response.status_code},"detail":{json.dumps(e.response.text)}}}\n\n'
                    yield msg.encode("utf-8")
                except Exception as e:
                    msg = f'data: {{"type":"proxy.error","detail":{json.dumps(repr(e))}}}\n\n'
                    yield msg.encode("utf-8")
                finally:
                    # append assembled assistant message to history
                    try:
                        assembled = acc.text()
                        if (
                            assembled
                            and convo_id
                            and convo_history.get(convo_id) is not None
                        ):
                            convo_history[convo_id].append(
                                {"role": "assistant", "content": assembled}
                            )
                    except Exception:
                        # don't let history write failures affect client stream
                        pass

            resp_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            if convo_id:
                resp_headers["X-Convo-ID"] = convo_id

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers=resp_headers,
            )

        else:  # Non-streaming response
            resp = await client.request(
                method=request.method,
                url=target_endpoint,
                headers=headers,
                content=body if body else None,
                params=dict(request.query_params),
                timeout=120,
            )

            assistant_text = None
            finish_reason = None

            # Log / extract without dumping content
            try:
                resp_json = resp.json()

                # standard logs
                if "created" in resp_json:
                    ts = datetime.fromtimestamp(resp_json["created"])
                    logger.info("Response created: %s", ts)
                if "model" in resp_json:
                    logger.info("Response model: %s", resp_json["model"])
                if "usage" in resp_json:
                    logger.info("Usage: %s", resp_json["usage"])

                # assistant text (OpenAI-style)
                ch0 = (resp_json.get("choices") or [None])[0] or {}
                # new-style
                if isinstance(ch0.get("message"), dict):
                    assistant_text = ch0["message"].get("content")
                # v0/legacy fallback
                if not assistant_text:
                    assistant_text = ch0.get("text")

                finish_reason = ch0.get("finish_reason")

            except Exception:
                logger.info("Response: %s", resp.text[-500:])

            # Append assistant message to convo history
            if assistant_text and convo_id and convo_history.get(convo_id) is not None:
                convo_history[convo_id].append(
                    {"role": "assistant", "content": assistant_text}
                )
                if finish_reason:
                    logger.info("Finish reason: %s", finish_reason)

            # Safe response headers + echo convo id for traceability
            safe_headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower()
                not in {"content-encoding", "transfer-encoding", "connection"}
            }
            if convo_id:
                safe_headers["X-Convo-ID"] = convo_id

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=safe_headers,
            )
