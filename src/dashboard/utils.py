import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
load_dotenv()
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
DEFAULT_RAG_ENDPOINT = "localhost:8100"
NONE_RAG_OPTION_LABEL = "None"
NONE_RAG_OPTION_VALUE = ""
NONE_SEARCH_PROVIDER_LABEL = "None"
NONE_SEARCH_PROVIDER_VALUE = ""
DEFAULT_SEARCH_PLANNER_MAX_CONTEXT_CHARS = 12000
HISTORY_DISPLAY_TIMEZONE = timezone(timedelta(hours=8))
SEARCH_PROVIDER_DISPLAY_NAMES = {
    "duckduckgo_html": "DuckDuckGo",
    "marginalia_html": "Marginalia",
    "mojeek_html": "Mojeek",
    "searxng_html": "SearXNG",
    "wikipedia_opensearch": "Wikipedia",
}


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
                "retrieve_url": f"http://{DEFAULT_RAG_ENDPOINT}/api/retrieve/context",
                "health_url": f"http://{DEFAULT_RAG_ENDPOINT}/api/health",
            }
        ]

    default_endpoint = str(rag_config.get("default_endpoint") or "").strip()
    if not default_endpoint:
        default_endpoint = endpoints[0]["retrieve_url"]

    return endpoints, default_endpoint


async def fetch_available_rag_endpoints():
    """Return configured RAG endpoint choices with a persistent None option."""
    endpoint_configs, default_endpoint = load_rag_endpoint_config()
    rag_choices = {NONE_RAG_OPTION_VALUE: NONE_RAG_OPTION_LABEL}

    async with httpx.AsyncClient(timeout=5.0) as client:
        for endpoint in endpoint_configs:
            health_url = endpoint["health_url"]
            if health_url:
                try:
                    response = await client.get(health_url)
                    response.raise_for_status()
                except Exception as e:
                    logging.warning("RAG health check failed for %s: %s", health_url, e)

            retrieve_url = endpoint["retrieve_url"]
            label = endpoint["name"]
            if label != retrieve_url:
                label = f"{label} ({retrieve_url})"
            rag_choices[retrieve_url] = label

    return rag_choices, NONE_RAG_OPTION_VALUE


def format_search_provider_label(provider_id: str) -> str:
    """Return a stable display label for a configured search provider id."""
    normalized = str(provider_id or "").strip()
    if not normalized:
        return "Unknown"
    return SEARCH_PROVIDER_DISPLAY_NAMES.get(
        normalized,
        normalized.replace("_", " ").title(),
    )


def load_search_provider_config() -> list[dict[str, Any]]:
    """Load enabled search providers from config.yaml, ordered by priority."""
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logging.warning("config.yaml not found; search providers disabled")
        return []
    except yaml.YAMLError as e:
        logging.warning("Failed to parse config.yaml for search providers: %s", e)
        return []

    if not isinstance(config, dict):
        return []

    search_config = config.get("search", {})
    if not isinstance(search_config, dict) or not bool(search_config.get("enabled")):
        return []

    provider_configs = search_config.get("providers", {})
    if not isinstance(provider_configs, dict):
        return []

    providers = []
    for provider_id, provider_config in provider_configs.items():
        if not isinstance(provider_config, dict) or not provider_config.get("enabled"):
            continue
        providers.append(
            {
                "id": str(provider_id).strip(),
                "name": format_search_provider_label(provider_id),
                "priority": int(provider_config.get("priority", 100)),
            }
        )

    providers.sort(key=lambda item: (item["priority"], item["name"], item["id"]))
    return providers


def fetch_available_search_providers() -> tuple[dict[str, str], str]:
    """Return enabled search provider choices with a persistent None option."""
    choices = {NONE_SEARCH_PROVIDER_VALUE: NONE_SEARCH_PROVIDER_LABEL}

    for provider in load_search_provider_config():
        provider_id = provider["id"]
        provider_name = provider["name"]
        choices[provider_id] = f"{provider_name} ({provider_id})"

    return choices, NONE_SEARCH_PROVIDER_VALUE


def load_search_planner_max_context_chars() -> int:
    """Return the configured client-side planner context bound."""
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return DEFAULT_SEARCH_PLANNER_MAX_CONTEXT_CHARS

    search_config = config.get("search", {}) if isinstance(config, dict) else {}
    try:
        value = int(
            search_config.get(
                "planner_max_context_chars",
                DEFAULT_SEARCH_PLANNER_MAX_CONTEXT_CHARS,
            )
        )
    except (TypeError, ValueError):
        return DEFAULT_SEARCH_PLANNER_MAX_CONTEXT_CHARS
    return max(0, value)


def trim_search_context(context: Any, max_chars: int) -> str:
    """Trim planner context from the front, preserving the newest request tail."""
    text = str(context or "").strip()
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    marker = "[trimmed earlier context]\n"
    if max_chars <= len(marker):
        return text[-max_chars:]
    return marker + text[-(max_chars - len(marker)) :]


async def fetch_search_results(
    *, query: str, provider: str, count: int = 5, context: Any = None
) -> dict[str, Any]:
    """Fetch normalized search candidates from the proxy search endpoint."""
    payload = {
        "query": str(query or "").strip(),
        "provider": str(provider or "").strip(),
        "count": int(count),
    }
    trimmed_context = trim_search_context(
        context,
        load_search_planner_max_context_chars(),
    )
    if trimmed_context:
        payload["context"] = trimmed_context

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{PROXY_BASE_URL}/search/web", json=payload)
        response.raise_for_status()
        return response.json()


async def fetch_available_endpoints():
    """Fetch available endpoints grouped by endpoint name."""
    try:
        data = await fetch_models_data()

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
        return endpoints, data
    except Exception as e:
        logging.error(f"Failed to fetch endpoints: {e}")
        return {}, {}


def create_endpoint_display_choices(endpoints_data):
    """Create display choices and mapping for endpoint dropdown."""
    choices = {}
    mapping = {}

    for endpoint_name, endpoint_data in endpoints_data.items():
        if endpoint_name == "Auto":
            continue

        models = endpoint_data.get("models", [])
        if models:
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


def format_history_choice_label(conversation):
    """Build a stable history dropdown label from conversation metadata."""
    convo_id = str(conversation.get("convo_id") or "").strip()
    raw_last_updated = conversation.get("last_updated")
    timestamp_label = str(raw_last_updated or "Unknown time")

    if isinstance(raw_last_updated, str) and raw_last_updated.strip():
        normalized = raw_last_updated.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp_label = parsed.astimezone(HISTORY_DISPLAY_TIMEZONE).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    return f"{timestamp_label} | {convo_id}"


def create_history_select_choices(conversations):
    """Build History tab select choices keyed by convo id."""
    return {
        str(conversation.get("convo_id") or "").strip(): format_history_choice_label(
            conversation
        )
        for conversation in conversations
        if str(conversation.get("convo_id") or "").strip()
    }


async def fetch_conversation_summaries():
    """Fetch conversation metadata for the History tab selector."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{PROXY_BASE_URL}/conversations")
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        print(f"Failed to fetch conversation summaries: {e}")
        return []

    if not isinstance(data, list):
        return []

    summaries = []
    for conversation in data:
        if not isinstance(conversation, dict):
            continue
        convo_id = str(conversation.get("convo_id") or "").strip()
        if not convo_id:
            continue
        summaries.append(conversation)

    return sorted(
        summaries,
        key=lambda conversation: str(conversation.get("last_updated") or ""),
        reverse=True,
    )
