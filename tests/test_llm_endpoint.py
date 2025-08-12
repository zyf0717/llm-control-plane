import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY_ID = os.getenv("API_KEY_ID")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
LLM_API_URL = os.getenv("GPT_OSS_20B_API_URL")

# Test endpoint directly
url = f"{LLM_API_URL}/api/v0/chat/completions"
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Hello, who are you?",
        }
    ],
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


# Status: 200
# Response: {
#     "id": "chatcmpl-6yg3xn1uz14pvk27mp8mj",
#     "object": "chat.completion",
#     "created": 1754982783,
#     "model": "openai/gpt-oss-20b",
#     "choices": [
#         {
#             "index": 0,
#             "logprobs": null,
#             "finish_reason": "stop",
#             "message": {
#                 "role": "assistant",
#                 "content": "Hi there! I’m ChatGPT—an AI language model created by OpenAI. I’m here to help answer questions, offer explanations, brainstorm ideas, or just chat about whatever’s on your mind. How can I assist you today?",
#                 "reasoning_content": "Need to respond politely.",
#             },
#         }
#     ],
#     "usage": {"prompt_tokens": 73, "completion_tokens": 52, "total_tokens": 125},
#     "stats": {
#         "tokens_per_second": 99.38193898895442,
#         "time_to_first_token": 0.227,
#         "generation_time": 0.634,
#         "stop_reason": "stop",
#     },
#     "model_info": {
#         "arch": "gpt-oss",
#         "quant": "MXFP4",
#         "format": "gguf",
#         "context_length": 15000,
#     },
#     "runtime": {
#         "name": "llama.cpp-win-x86_64-nvidia-cuda12-avx2",
#         "version": "1.44.0",
#         "supported_formats": ["gguf"],
#     },
# }
