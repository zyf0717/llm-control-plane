import json
import logging
import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response

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
    return await passthrough("", request)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def custom_endpoints(path: str, request: Request):
    # Skip health endpoint
    if path == "health":
        return await health_check()
    else:
        return await passthrough(path, request)


async def passthrough(path: str, request: Request):
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
    if body:
        try:
            body_json = json.loads(body)
            if isinstance(body_json, dict) and "contextOverflowPolicy" not in body_json:
                body_json["contextOverflowPolicy"] = "rollingWindow"
                body = json.dumps(body_json).encode()
        except Exception:
            logger.warning("Failed to parse request body as JSON")

    logger.info(
        "Proxying %s %s to %s", request.method, request.url.path, target_endpoint
    )

    # Forward request
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=target_endpoint,
            headers=headers,
            content=body if body else None,
            params=dict(request.query_params),
            timeout=120,
        )

    # Log response (excluding message content)
    try:
        resp_json = resp.json()
        if "created" in resp_json:
            ts = datetime.fromtimestamp(resp_json["created"])
            logger.info("Response created: %s", ts)
        if "model" in resp_json:
            logger.info("Response model: %s", resp_json["model"])
        if "usage" in resp_json:
            logger.info("Usage: %s", resp_json["usage"])
    except Exception:
        logger.info("Response: %s", resp.text[-500:])

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in {"content-encoding", "transfer-encoding", "connection"}
        },
    )
