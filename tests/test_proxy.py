import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")

# Test the proxy server
urls = [
    "https://llm.paperclips.dev/health",
    "https://llm.paperclips.dev/qwen3-4b",
    "https://llm.paperclips.dev/gpt-oss-20b",
    "https://llm.paperclips.dev/",
]

for url in urls:
    payload = {
        "messages": [{"role": "user", "content": "Hello, what model are you?"}],
    }
    headers = {
        "CF-Access-Client-Id": API_KEY_ID,
        "CF-Access-Client-Secret": API_KEY_SECRET,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print("URL:", url)
        print("Status:", response.status_code)
        print(
            "Response:",
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "<no content>")
            .strip(),
        )
        print()
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
