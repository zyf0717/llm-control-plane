import json
import os

import httpx
from dotenv import load_dotenv


def main() -> None:
    """Manually inspect /models without making requests during test collection."""
    load_dotenv()
    headers = {
        "CF-Access-Client-Id": os.getenv("API_KEY_ID", ""),
        "CF-Access-Client-Secret": os.getenv("API_KEY_SECRET", ""),
        "Content-Type": "application/json",
    }
    url = os.getenv("PROXY_BASE_URL", "https://llm.paperclips.dev").rstrip("/") + "/models"
    try:
        response = httpx.get(url, headers=headers, timeout=30)
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
    except httpx.RequestError as exc:
        print("Request failed:", exc)


if __name__ == "__main__":
    main()
