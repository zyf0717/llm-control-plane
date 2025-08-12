import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")

urls = [
    "https://llm.paperclips.dev/qwen3-4b",  # v1/chat/completions (proxy)
    "https://llm.paperclips.dev/gpt-oss-20b",  # v1/chat/completions (proxy)
    "https://llm.paperclips.dev/",  # root → default model (proxy)
]


def handle_line(decoded_line: str):
    if not decoded_line.startswith("data: "):
        return False
    data = decoded_line[6:].strip()
    if data == "[DONE]":
        print("\n[stream complete]")
        return True
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return False

    # Branch: Responses API (api/v0 typed events)
    if "type" in obj:
        t = obj["type"]
        if t == "response.output_text.delta":
            print(obj.get("delta", ""), end="", flush=True)
        elif t in ("response.error", "response.completed"):
            # you may inspect/print obj further here
            pass
        return False

    # Branch: Chat Completions (OpenAI-compatible)
    if "choices" in obj and obj["choices"]:
        delta = obj["choices"][0].get("delta", {})
        content = delta.get("content")
        if content:
            print(content, end="", flush=True)
    return False


for url in urls:
    print(f"\nTesting {url}:")
    payload = {
        "messages": [{"role": "user", "content": "Output 100 words of text."}],
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "CF-Access-Client-Id": API_KEY_ID,
        "CF-Access-Client-Secret": API_KEY_SECRET,
    }

    try:
        with requests.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=(10, 300),
        ) as resp:
            print(f"Status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Error: {resp.text[:500]}")
                continue

            for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
                if not line:
                    continue
                if handle_line(line):
                    break
            print()  # newline after stream

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
