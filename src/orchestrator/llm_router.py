"""
LLM Router for intelligent endpoint selection.

Determines the most appropriate endpoint based on request characteristics.
Currently supports reasoning detection with extensible framework for future routing logic.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

logger = logging.getLogger(__name__)


class RouteStrategy(Enum):
    """Available routing strategies."""

    REASONING = "reasoning"
    PERFORMANCE = "performance"
    COST = "cost"
    # Future strategies can be added here


@dataclass
class EndpointConfig:
    """Endpoint configuration."""

    name: str
    url: str
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
    strategy: RouteStrategy


class BaseRouter(ABC):
    """Base class for routing strategies."""

    @abstractmethod
    async def should_route(self, text: str, **kwargs) -> bool:
        """Determine if this router should handle the request."""
        pass

    @abstractmethod
    async def select_endpoint(
        self, text: str, endpoints: List[EndpointConfig], **kwargs
    ) -> RouteDecision:
        """Select the best endpoint for the request."""
        pass


class ReasoningRouter(BaseRouter):
    """Router for reasoning-intensive tasks."""

    # Patterns that indicate reasoning is needed
    REASONING_PATTERNS = [
        r"\b(think|reason|analyze|solve|calculate|deduce|infer)\b",
        r"\b(step by step|step-by-step|explain why|how does|what if)\b",
        r"\b(problem|puzzle|logic|proof|derive|conclude)\b",
        r"\b(compare|contrast|evaluate|assess|judge)\b",
        r"\b(plan|strategy|approach|method|solution)\b",
        r"\b(why|how)\b.*\?",  # Why/How questions are typically reasoning
        r"\bwhat if\b.*\?",  # What-if scenarios require reasoning
        r"\b(because|therefore|thus|hence|consequently)\b",
    ]

    # Keywords that strongly suggest reasoning
    STRONG_REASONING_KEYWORDS = [
        "analyze",
        "reasoning",
        "logic",
        "proof",
        "derive",
        "calculate",
        "step by step",
        "think through",
        "problem solving",
        "critical thinking",
    ]

    def __init__(
        self, lightweight_endpoint: str = None, reasoning_endpoint: str = None
    ):
        """
        Initialize reasoning router.

        Args:
            lightweight_endpoint: Name of lightweight model for classification
            reasoning_endpoint: Name of reasoning-capable model for complex tasks
        """
        self.lightweight_endpoint = (
            lightweight_endpoint or "Mac Mini"
        )  # Default to fastest
        self.reasoning_endpoint = (
            reasoning_endpoint or "HRPC-CISR HPC"
        )  # Default to most powerful

    async def should_route(self, text: str, **kwargs) -> bool:
        """Check if text indicates reasoning is needed."""
        text_lower = text.lower()

        # Quick pattern-based check first
        for pattern in self.REASONING_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        # Check for strong reasoning keywords
        for keyword in self.STRONG_REASONING_KEYWORDS:
            if keyword in text_lower:
                return True

        # For ambiguous cases, could use lightweight LLM (not implemented yet)
        return False

    async def select_endpoint(
        self, text: str, endpoints: List[EndpointConfig], **kwargs
    ) -> RouteDecision:
        """Select endpoint based on reasoning requirements."""
        needs_reasoning = await self.should_route(text)

        if needs_reasoning:
            # Find the most powerful endpoint for reasoning
            target_endpoint = self._find_endpoint_by_name(
                endpoints, self.reasoning_endpoint
            )
            if not target_endpoint:
                target_endpoint = self._select_most_powerful(endpoints)

            return RouteDecision(
                endpoint=target_endpoint.name,
                confidence=0.8,
                reason="Complex reasoning task detected",
                strategy=RouteStrategy.REASONING,
            )
        else:
            # Use lightweight endpoint for simple tasks
            target_endpoint = self._find_endpoint_by_name(
                endpoints, self.lightweight_endpoint
            )
            if not target_endpoint:
                target_endpoint = self._select_fastest(endpoints)

            return RouteDecision(
                endpoint=target_endpoint.name,
                confidence=0.7,
                reason="Simple task, using efficient endpoint",
                strategy=RouteStrategy.REASONING,
            )

    def _find_endpoint_by_name(
        self, endpoints: List[EndpointConfig], name: str
    ) -> Optional[EndpointConfig]:
        """Find endpoint by name."""
        return next((ep for ep in endpoints if ep.name == name), None)

    def _select_most_powerful(self, endpoints: List[EndpointConfig]) -> EndpointConfig:
        """Select the most powerful endpoint based on GPU/hardware."""
        # Prefer NVIDIA GPUs with high VRAM
        nvidia_endpoints = [
            ep for ep in endpoints if ep.gpu and "nvidia" in ep.gpu.lower()
        ]
        if nvidia_endpoints:
            return max(nvidia_endpoints, key=lambda ep: self._extract_vram_gb(ep.vram))

        # Fallback to first endpoint
        return endpoints[0] if endpoints else None

    def _select_fastest(self, endpoints: List[EndpointConfig]) -> EndpointConfig:
        """Select the fastest endpoint (Apple Silicon preferred for speed)."""
        # Prefer Apple Silicon for speed
        apple_endpoints = [
            ep for ep in endpoints if ep.soc and "apple" in ep.soc.lower()
        ]
        if apple_endpoints:
            return apple_endpoints[0]

        # Fallback to first endpoint
        return endpoints[0] if endpoints else None

    def _extract_vram_gb(self, vram_str: Optional[str]) -> int:
        """Extract VRAM in GB from string."""
        if not vram_str:
            return 0
        match = re.search(r"(\d+)", vram_str)
        return int(match.group(1)) if match else 0


class LLMRouter:
    """Main LLM router that orchestrates different routing strategies."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the router with configuration."""
        self.config_path = Path(config_path)
        self.endpoints = self._load_endpoints()
        self.routers = {RouteStrategy.REASONING: ReasoningRouter()}
        self.default_strategy = RouteStrategy.REASONING

    def _load_endpoints(self) -> List[EndpointConfig]:
        """Load endpoint configurations from YAML."""
        try:
            with self.config_path.open("r") as f:
                config = yaml.safe_load(f)

            endpoints = []
            for ep_config in config.get("endpoints", []):
                endpoints.append(
                    EndpointConfig(
                        name=ep_config["name"],
                        url=ep_config["url"],
                        gpu=ep_config.get("gpu"),
                        vram=ep_config.get("vram"),
                        cpu=ep_config.get("cpu"),
                        ram=ep_config.get("ram"),
                        soc=ep_config.get("soc"),
                    )
                )

            return endpoints
        except Exception as e:
            logger.error(f"Failed to load endpoints config: {e}")
            return []

    async def route_request(
        self, text: str, strategy: Optional[RouteStrategy] = None, **kwargs
    ) -> RouteDecision:
        """
        Route a request to the most appropriate endpoint.

        Args:
            text: The input text/prompt to analyze
            strategy: Optional specific strategy to use
            **kwargs: Additional parameters for routing

        Returns:
            RouteDecision with selected endpoint and reasoning
        """
        if not self.endpoints:
            raise ValueError("No endpoints configured")

        # Use specified strategy or default
        strategy = strategy or self.default_strategy
        router = self.routers.get(strategy)

        if not router:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        try:
            decision = await router.select_endpoint(text, self.endpoints, **kwargs)
            logger.info(f"Routed to {decision.endpoint}: {decision.reason}")
            return decision
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Fallback to first endpoint
            return RouteDecision(
                endpoint=self.endpoints[0].name,
                confidence=0.1,
                reason=f"Fallback due to routing error: {e}",
                strategy=strategy,
            )

    def add_router(self, strategy: RouteStrategy, router: BaseRouter):
        """Add a new routing strategy."""
        self.routers[strategy] = router

    def get_endpoint_by_name(self, name: str) -> Optional[EndpointConfig]:
        """Get endpoint configuration by name."""
        return next((ep for ep in self.endpoints if ep.name == name), None)

    def list_endpoints(self) -> List[str]:
        """List all available endpoint names."""
        return [ep.name for ep in self.endpoints]


# Global router instance
_router_instance = None


def get_router() -> LLMRouter:
    """Get the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


async def route_text(
    text: str, strategy: Optional[RouteStrategy] = None
) -> RouteDecision:
    """Convenience function to route text."""
    router = get_router()
    return await router.route_request(text, strategy)
