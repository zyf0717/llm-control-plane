"""
LLM Router for intelligent endpoint selection.

Determines the most appropriate endpoint based on workload type and endpoint preferences.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml

from .utils import HeaderManager

logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Available workload types."""

    REASONING = "reasoning"
    TTFT_CONTENT = "ttft_content"
    TOKENS_PER_SECOND = "tokens_per_second"
    PROGRAMMING = "programming"


@dataclass
class EndpointConfig:
    """Endpoint configuration."""

    name: str
    url: str
    viable_models: List[str]
    gpu: Optional[str] = None
    vram: Optional[str] = None
    cpu: Optional[str] = None
    ram: Optional[str] = None
    soc: Optional[str] = None


@dataclass
class RouteDecision:
    """Routing decision result."""

    endpoint: str
    confidence: float
    reason: str
    workload_type: WorkloadType


class WorkloadClassifier:
    """Classifies text to determine appropriate workload type."""

    async def classify_with_llm(
        self, text: str, classification_endpoint_url: str
    ) -> Optional[WorkloadType]:
        """Use LLM to classify workload type."""
        try:
            system_prompt = """You are a classifier only.
Classify the request into exactly one of these categories:
- reasoning
- programming
- ttft_content
- tokens_per_second

Rules in descending order of priority:
1. If multi-step analysis/trade-offs → reasoning
2. If code-related tasks → programming
3. If emphasis on speed/throughput/TTFT/TPS → tokens_per_second
4. Else → ttft_content

Output exactly one label, lowercase, no punctuation."""

            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "model": "classification",
                "max_tokens": 20,
                "temperature": 0.1,
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = HeaderManager.create_auth_headers()
                headers["Content-Type"] = "application/json"

                response = await client.post(
                    classification_endpoint_url, json=payload, headers=headers
                )
                response.raise_for_status()

                result = response.json()
                if "choices" in result and result["choices"]:
                    content = (
                        result["choices"][0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                        .lower()
                    )

                    for workload_type in WorkloadType:
                        if workload_type.value in content:
                            return workload_type

        except (httpx.RequestError, httpx.HTTPStatusError, KeyError) as e:
            logger.warning("LLM classification failed: %s", e)

        return None


class LLMRouter:
    """Main LLM router for endpoint selection."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize router with configuration."""
        self.config_path = Path(config_path)
        self.endpoints: Dict[str, EndpointConfig] = {}
        self.workload_preferences: Dict[WorkloadType, List[str]] = {}
        self.classifier = WorkloadClassifier()
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        with self.config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Load endpoints
        for ep_config in config.get("endpoints", []):
            endpoint = EndpointConfig(
                name=ep_config["name"],
                url=ep_config["url"],
                viable_models=ep_config.get("viable_models", []),
                gpu=ep_config.get("gpu"),
                vram=ep_config.get("vram"),
                cpu=ep_config.get("cpu"),
                ram=ep_config.get("ram"),
                soc=ep_config.get("soc"),
            )
            self.endpoints[endpoint.name] = endpoint

        # Load workload preferences
        for workload_config in config.get("workloads", []):
            workload_type = WorkloadType(workload_config["type"])
            self.workload_preferences[workload_type] = workload_config[
                "endpoint_preference"
            ]

    async def route_request(
        self, text: str, workload_type: Optional[WorkloadType] = None
    ) -> RouteDecision:
        """Route request to best endpoint."""
        # Determine workload type if not provided
        if workload_type is None:
            workload_type = await self._classify_workload(text)

        # Get preferred endpoints for this workload type
        preferred_endpoints = self.workload_preferences.get(workload_type, [])

        # Find first available endpoint from preferences
        selected_endpoint_name = preferred_endpoints[
            0
        ]  # Assume at least 1 endpoint available
        selected_endpoint = self.endpoints[selected_endpoint_name]

        return RouteDecision(
            endpoint=selected_endpoint.name,
            confidence=0.9,
            reason=f"Selected for {workload_type.value} workload",
            workload_type=workload_type,
        )

    async def _classify_workload(self, text: str) -> WorkloadType:
        """Classify text to determine workload type."""
        # Try LLM classification first with fastest endpoint
        fastest_endpoint = self._get_fastest_endpoint()
        if fastest_endpoint:
            classification_url = f"{fastest_endpoint.url}/api/v0/chat/completions"
            llm_result = await self.classifier.classify_with_llm(
                text, classification_url
            )
            if llm_result:
                return llm_result

    def _get_fastest_endpoint(self) -> Optional[EndpointConfig]:
        """Get fastest endpoint for classification (prefer SOC-based)."""
        # Prefer Apple Silicon endpoints for speed
        for endpoint in self.endpoints.values():
            if endpoint.soc and "apple" in endpoint.soc.lower():
                return endpoint

        # Fallback to first available endpoint
        return next(iter(self.endpoints.values()), None)

    def get_endpoint_by_name(self, name: str) -> Optional[EndpointConfig]:
        """Get endpoint by name."""
        return self.endpoints.get(name)

    def list_endpoints(self) -> List[str]:
        """List all endpoint names."""
        return list(self.endpoints.keys())


# Global router instance
_router_instance: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    """Get global router instance."""
    # Using global is necessary for singleton pattern
    global _router_instance  # noqa: PLW0603
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


def reset_router():
    """Reset global router instance."""
    # Using global is necessary for singleton pattern
    global _router_instance  # noqa: PLW0603
    _router_instance = None


async def route_text(
    text: str, workload_type: Optional[WorkloadType] = None
) -> RouteDecision:
    """Convenience function to route text."""
    router = get_router()
    return await router.route_request(text, workload_type)
