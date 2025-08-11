import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")

# Test the proxy server with streaming
urls = [
    "https://llm.paperclips.dev/qwen3-4b",
    "https://llm.paperclips.dev/gpt-oss-20b",
    "https://llm.paperclips.dev/",
]

for url in urls:
    print(f"\nTesting {url}:")
    payload = {
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "CF-Access-Client-Id": API_KEY_ID,
        "CF-Access-Client-Secret": API_KEY_SECRET,
    }

    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=60, stream=True
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print("Streaming response:")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data = decoded_line[6:]  # Remove 'data: ' prefix
                        if data.strip() == "[DONE]":
                            print("Stream complete")
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
            print()
        else:
            print(f"Error: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        print(f"Request failed: {e}")
        print(f"Request failed: {e}")
