"""
LLM Router for intelligent endpoint selection.

Determines the most appropriate endpoint based on request characteristics.
Currently supports reasoning detection with extensible framework for future routing logic.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import httpx
import yaml

from .utils import HeaderManager

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
        self,
        lightweight_endpoint: str = None,
        reasoning_endpoint: str = None,
        use_llm_classification: bool = True,
    ):
        """
        Initialize reasoning router.

        Args:
            lightweight_endpoint: Name of lightweight model for classification
            reasoning_endpoint: Name of reasoning-capable model for complex tasks
            use_llm_classification: Whether to use LLM for classification (vs pattern matching)
        """
        self.lightweight_endpoint = (
            lightweight_endpoint or "Mac Mini"
        )  # Default to fastest
        self.reasoning_endpoint = (
            reasoning_endpoint or "HRPC-CISR HPC"
        )  # Default to most powerful
        self.use_llm_classification = use_llm_classification

    async def _classify_with_llm(
        self, text: str, endpoints: List[EndpointConfig]
    ) -> Optional[bool]:
        """Use lightweight LLM to classify if reasoning is needed."""
        try:
            # Find the lightweight endpoint
            lightweight_ep = self._find_endpoint_by_name(
                endpoints, self.lightweight_endpoint
            )
            if not lightweight_ep:
                logger.warning(
                    "Lightweight endpoint not found, falling back to pattern matching"
                )
                return None

            # Prepare the classification prompt
            classification_prompt = f"""Analyze the following user request and determine if it requires complex reasoning, step-by-step thinking, problem-solving, or chain-of-thought processing.

User request: "{text}"

Consider these factors:
- Does it ask for analysis, reasoning, or problem-solving?
- Does it require step-by-step thinking?
- Does it involve calculations, logic, or complex explanations?
- Is it asking "how" or "why" questions that need detailed reasoning?

Respond with only "YES" if complex reasoning is required, or "NO" if it's a simple task.

Response:"""

            # Make request to lightweight endpoint
            endpoint_url = f"{lightweight_ep.url}/api/v0/chat/completions"

            payload = {
                "messages": [{"role": "user", "content": classification_prompt}],
                "model": "classification",
                "max_tokens": 10,
                "temperature": 0.1,  # Low temperature for consistent classification
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                # Create authentication headers
                headers = HeaderManager.create_auth_headers()
                headers["Content-Type"] = "application/json"

                # Debug logging
                logger.debug(f"Making LLM classification request to: {endpoint_url}")
                logger.debug(f"Headers: {headers}")

                response = await client.post(
                    endpoint_url, json=payload, headers=headers
                )
                response.raise_for_status()

                result = response.json()
                if "choices" in result and result["choices"]:
                    content = (
                        result["choices"][0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                        .upper()
                    )

                    if "YES" in content:
                        return True
                    elif "NO" in content:
                        return False
                    else:
                        logger.warning(
                            f"Unexpected LLM classification response: {content}"
                        )
                        return None

        except Exception as e:
            logger.warning(
                f"LLM classification failed: {e}, falling back to pattern matching"
            )
            return None

        return None

    async def should_route(
        self, text: str, endpoints: List[EndpointConfig] = None, **kwargs
    ) -> bool:
        """Check if text indicates reasoning is needed."""
        # Try LLM classification first if enabled and endpoints available
        if self.use_llm_classification and endpoints:
            llm_result = await self._classify_with_llm(text, endpoints)
            if llm_result is not None:
                logger.info(
                    f"LLM classification: {'reasoning' if llm_result else 'simple'} task"
                )
                return llm_result

        # Fallback to pattern-based classification
        text_lower = text.lower()

        # Quick pattern-based check
        for pattern in self.REASONING_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        # Check for strong reasoning keywords
        for keyword in self.STRONG_REASONING_KEYWORDS:
            if keyword in text_lower:
                return True

        return False

    async def select_endpoint(
        self, text: str, endpoints: List[EndpointConfig], **kwargs
    ) -> RouteDecision:
        """Select endpoint based on reasoning requirements."""
        needs_reasoning = await self.should_route(text, endpoints=endpoints)

        if needs_reasoning:
            # Find the most powerful endpoint for reasoning
            target_endpoint = self._find_endpoint_by_name(
                endpoints, self.reasoning_endpoint
            )
            if not target_endpoint:
                target_endpoint = self._select_most_powerful(endpoints)

            # Higher confidence when using LLM classification
            confidence = 0.9 if self.use_llm_classification else 0.8
            reason = (
                "Complex reasoning task detected (LLM classified)"
                if self.use_llm_classification
                else "Complex reasoning task detected"
            )

            return RouteDecision(
                endpoint=target_endpoint.name,
                confidence=confidence,
                reason=reason,
                strategy=RouteStrategy.REASONING,
            )
        else:
            # Use lightweight endpoint for simple tasks
            target_endpoint = self._find_endpoint_by_name(
                endpoints, self.lightweight_endpoint
            )
            if not target_endpoint:
                target_endpoint = self._select_fastest(endpoints)

            # Higher confidence when using LLM classification
            confidence = 0.85 if self.use_llm_classification else 0.7
            reason = (
                "Simple task, using efficient endpoint (LLM classified)"
                if self.use_llm_classification
                else "Simple task, using efficient endpoint"
            )

            return RouteDecision(
                endpoint=target_endpoint.name,
                confidence=confidence,
                reason=reason,
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

    def __init__(
        self, config_path: str = "config.yaml", use_llm_classification: bool = True
    ):
        """Initialize the router with configuration."""
        self.config_path = Path(config_path)
        self.endpoints = self._load_endpoints()
        self.use_llm_classification = use_llm_classification
        self.routers = {
            RouteStrategy.REASONING: ReasoningRouter(
                use_llm_classification=use_llm_classification
            )
        }
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


def get_router(use_llm_classification: bool = True) -> LLMRouter:
    """Get the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter(use_llm_classification=use_llm_classification)
    return _router_instance


def reset_router():
    """Reset the global router instance (useful for testing)."""
    global _router_instance
    _router_instance = None


async def route_text(
    text: str, strategy: Optional[RouteStrategy] = None
) -> RouteDecision:
    """Convenience function to route text."""
    router = get_router()
    return await router.route_request(text, strategy)
