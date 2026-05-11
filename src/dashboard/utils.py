import logging
import os
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
load_dotenv()
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
DEFAULT_RAG_ENDPOINT = "localhost:8100"


async def fetch_models_data():
    """Fetch raw models data from the proxy /models endpoint for telemetry."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{PROXY_BASE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Failed to fetch models data: {e}")
        return {}


def load_rag_endpoint_config():
    """Load configured RAG endpoints and default selection from config.yaml."""
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logging.warning("config.yaml not found; using default RAG endpoint")
        config = {}
    except yaml.YAMLError as e:
        logging.warning("Failed to parse config.yaml for RAG endpoints: %s", e)
        config = {}

    rag_config = config.get("rag", {}) if isinstance(config, dict) else {}
    configured_endpoints = rag_config.get("endpoints", [])

    endpoints = []
    seen_retrieve_urls = set()
    for endpoint in configured_endpoints:
        if isinstance(endpoint, str):
            retrieve_url = endpoint.strip()
            health_url = ""
            name = retrieve_url
        elif isinstance(endpoint, dict):
            retrieve_url = str(
                endpoint.get("retrieve_url") or endpoint.get("url") or ""
            ).strip()
            health_url = str(endpoint.get("health_url") or "").strip()
            name = str(endpoint.get("name") or retrieve_url).strip()
        else:
            continue

        if not retrieve_url or retrieve_url in seen_retrieve_urls:
            continue

        endpoints.append(
            {
                "name": name or retrieve_url,
                "retrieve_url": retrieve_url,
                "health_url": health_url,
            }
        )
        seen_retrieve_urls.add(retrieve_url)

    if not endpoints:
        endpoints = [
            {
                "name": DEFAULT_RAG_ENDPOINT,
                "retrieve_url": f"http://{DEFAULT_RAG_ENDPOINT}/api/retrieve",
                "health_url": f"http://{DEFAULT_RAG_ENDPOINT}/api/health",
            }
        ]

    default_endpoint = str(rag_config.get("default_endpoint") or "").strip()
    if not default_endpoint:
        default_endpoint = endpoints[0]["retrieve_url"]

    return endpoints, default_endpoint


async def fetch_available_rag_endpoints():
    """Return healthy RAG endpoint choices and the selected default."""
    endpoint_configs, default_endpoint = load_rag_endpoint_config()
    healthy_choices = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        for endpoint in endpoint_configs:
            health_url = endpoint["health_url"]
            if not health_url:
                continue

            try:
                response = await client.get(health_url)
                response.raise_for_status()
            except Exception as e:
                logging.warning("RAG health check failed for %s: %s", health_url, e)
                continue

            retrieve_url = endpoint["retrieve_url"]
            label = endpoint["name"]
            if label != retrieve_url:
                label = f"{label} ({retrieve_url})"
            healthy_choices[label] = retrieve_url

    selected = None
    if default_endpoint in healthy_choices.values():
        selected = default_endpoint
    elif healthy_choices:
        selected = next(iter(healthy_choices.values()))

    return healthy_choices, selected


async def fetch_available_endpoints():
    """Fetch available endpoints grouped by endpoint name."""
    try:
        data = await fetch_models_data()

        # data: Raw response from /models endpoint with structure:
        #   {
        #     "data": [
        #       {
        #         "id": "model-name",           # Model identifier (e.g., "llama-3.1-8b")
        #         "endpoint": "endpoint-name",  # Endpoint name (e.g., "Mac Mini")
        #         "endpoint_url": "http://...", # Full URL to the endpoint
        #         ...                          # Additional model metadata
        #       },
        #       ...
        #     ]
        #   }

        # Group models by endpoint
        endpoints = {}
        for model in data.get("data", []):
            endpoint_name = model.get("endpoint")
            endpoint_url = model.get("endpoint_url")
            model_id = model.get("id")
            if endpoint_name and endpoint_url and model_id:
                if endpoint_name not in endpoints:
                    endpoints[endpoint_name] = {
                        "endpoint_url": endpoint_url,
                        "models": [],
                    }
                endpoints[endpoint_name]["models"].append(model)

        # Always add Auto/Smart routing option
        endpoints["Auto"] = {
            "endpoint_url": f"{PROXY_BASE_URL}/smart",
            "models": [
                {
                    "id": "auto-router",
                    "object": "model",
                    "endpoint": "Auto",
                    "endpoint_url": f"{PROXY_BASE_URL}/smart",
                    "description": "Intelligent routing to best available endpoint",
                }
            ],
        }

        logging.info(f"Fetched endpoints: {endpoints}")
        logging.info(f"Raw data: {data}")
        return endpoints, data  # Return both endpoints and raw data
    except Exception as e:
        logging.error(f"Failed to fetch endpoints: {e}")
        return {}, {}


def create_endpoint_display_choices(endpoints_data):
    """Create display choices and mapping for endpoint dropdown."""
    choices = {}
    mapping = {}

    # Always put Auto first if it exists
    if "Auto" in endpoints_data:
        choices["Auto (auto-router)"] = "Auto"
        mapping["Auto (auto-router)"] = "Auto"

    for endpoint_name, endpoint_data in endpoints_data.items():
        if endpoint_name == "Auto":
            continue  # Already handled above

        models = endpoint_data.get("models", [])
        if models:
            # Use the first model's name for display
            model_name = models[0].get("id", "")
            if model_name:
                display_name = f"{endpoint_name} ({model_name})"
                choices[display_name] = endpoint_name
                mapping[display_name] = endpoint_name
            else:
                choices[endpoint_name] = endpoint_name
                mapping[endpoint_name] = endpoint_name
        else:
            choices[endpoint_name] = endpoint_name
            mapping[endpoint_name] = endpoint_name

    return choices, mapping


def find_model_by_endpoint(endpoint_data, endpoint_key):
    """Find the first model for a given endpoint."""
    if not endpoint_data or not endpoint_key:
        return None

    for model in endpoint_data.get("data", []):
        if model.get("endpoint") == endpoint_key:
            return model
    return None


async def fetch_convo_history(convo_id):
    """Fetch conversation history for a given convo_id."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{PROXY_BASE_URL}/conversations/retrieve", json={"convo_id": convo_id}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Failed to fetch conversation history: {e}")
        return {}
