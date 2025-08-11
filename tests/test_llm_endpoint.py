import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
LLM_API_URL = os.getenv("QWEN_3_4B_API_URL")

# Test endpoint directly
url = f"{LLM_API_URL}/v1/chat/completions"
payload = {
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
}
headers = {
    "CF-Access-Client-Id": API_KEY_ID,
    "CF-Access-Client-Secret": API_KEY_SECRET,
    "Content-Type": "application/json",
}

try:
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    print("Status:", r.status_code)
    print("Response:", r.text)
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
