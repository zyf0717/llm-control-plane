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
    CLASSIFICATION = "classification"


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
            system_prompt = """
You are a classifier only, and your job is to classify the request into exactly ONE of the following categories:
- reasoning
- programming
- tokens_per_second
- ttft_content

Rules:
- If the request needs multi-step thinking, analysis, or comparing trade-offs → reasoning
- If the request asks to write, debug, or explain code → programming
- If the request expects a long or detailed output with minimal reasoning/programming → tokens_per_second
- Otherwise (simple Q&A, general chat, basic requests) → ttft_content

Examples:
- "Compare <service A> vs <service B> for cost efficiency." → reasoning
- "Analyze the pros and cons of <option 1> and <option 2>." → reasoning

- "Write <programming language> to parse <log format>." → programming
- "Implement <programming language> function to <task>." → programming

- "Explain <topic> in detail (without code or comparisons)." → tokens_per_second
- "Describe <topic> comprehensively (without trade-offs)." → tokens_per_second

- "What is the capital of <country>?" → ttft_content
- "Define <term>." → ttft_content

Pick only ONE of the four categories, the MOST LIKELY based on the rules above.

Respond with ONLY the category name and nothing else.
"""

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
        self,
        text: str,
        reachable_endpoints: List[str],
        workload_type: Optional[WorkloadType] = None,
    ) -> RouteDecision:
        """Route request to best endpoint."""
        # Determine workload type if not provided
        if workload_type is None:
            workload_type = await self._classify_workload(text, reachable_endpoints)

        # Get preferred endpoints for this workload type
        preferred_endpoints = self.workload_preferences.get(workload_type, [])

        # Keep only reachable endpoints
        available_preferred = [
            ep for ep in preferred_endpoints if ep in reachable_endpoints
        ]

        # Select endpoint with fallback logic
        if available_preferred:
            selected_endpoint_name = available_preferred[0]
            reason = f"Selected preferred endpoint for {workload_type.value} workload"
        elif reachable_endpoints:
            # Fallback to any reachable endpoint
            selected_endpoint_name = reachable_endpoints[0]
            reason = f"No preferred endpoints available, using fallback for {workload_type.value} workload"
        else:
            # No endpoints available at all
            raise RuntimeError("No reachable endpoints available")

        selected_endpoint = self.endpoints[selected_endpoint_name]

        return RouteDecision(
            endpoint=selected_endpoint.name,
            confidence=0.9 if available_preferred else 0.5,
            reason=reason,
            workload_type=workload_type,
        )

    async def _classify_workload(
        self, text: str, reachable_endpoints: List[str]
    ) -> WorkloadType:
        """Classify text to determine workload type."""
        # Try LLM classification first with fastest endpoint
        fastest_endpoint = self._get_fastest_endpoint(reachable_endpoints)
        logger.info("Fastest endpoint for classification: %s", fastest_endpoint)
        if fastest_endpoint:
            classification_url = f"{fastest_endpoint.url}/v1/chat/completions"
            llm_result = await self.classifier.classify_with_llm(
                text, classification_url
            )
            if llm_result:
                return llm_result

        # Fallback to tokens_per_second if classification fails
        return WorkloadType.TOKENS_PER_SECOND

    def _get_fastest_endpoint(
        self, reachable_endpoints: List[str]
    ) -> Optional[EndpointConfig]:
        """Get fastest endpoint for classification using reachable endpoints and classification preferences."""
        # Get classification endpoint preferences from config
        classification_preferences = self.workload_preferences.get(
            WorkloadType.CLASSIFICATION, []
        )

        # Find first endpoint that is both preferred for classification and reachable
        for preferred_endpoint in classification_preferences:
            if (
                preferred_endpoint in reachable_endpoints
                and preferred_endpoint in self.endpoints
            ):
                return self.endpoints[preferred_endpoint]

        # Fallback: use first reachable endpoint
        for endpoint_name in reachable_endpoints:
            if endpoint_name in self.endpoints:
                return self.endpoints[endpoint_name]

        # Last resort: any configured endpoint
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
