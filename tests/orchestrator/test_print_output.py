import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
LLM_API_URL = os.getenv("GPT_OSS_20B_API_URL")

# Test GET /models endpoint with .env headers
url = "https://llm.paperclips.dev/models"

# Load all .env variables as headers (underscores -> hyphens)
env_headers = {
    "CF-Access-Client-Id": API_KEY_ID,
    "CF-Access-Client-Secret": API_KEY_SECRET,
    "Content-Type": "application/json",
}

try:
    r = requests.get(url, headers=env_headers, timeout=30)
    print("Status:", r.status_code)
    print("Response:", json.dumps(r.json(), indent=2))
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
