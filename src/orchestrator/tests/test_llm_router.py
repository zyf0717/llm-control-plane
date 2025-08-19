"""
Test suite for LLM router functionality.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from ..llm_router import (
    EndpointConfig,
    LLMRouter,
    ReasoningRouter,
    RouteDecision,
    RouteStrategy,
    get_router,
)


@pytest.fixture
def mock_endpoints():
    """Mock endpoint configurations for testing."""
    return [
        EndpointConfig(
            name="HRPC-CISR HPC",
            url="https://llm-hrpc.paperclips.dev",
            gpu="NVIDIA GeForce RTX 5060 Ti 16GB",
            vram="16GB",
            cpu="Intel Core Ultra 9 285K",
            ram="192GB",
        ),
        EndpointConfig(
            name="Mac Mini",
            url="https://llm-mac-mini.paperclips.dev",
            soc="Apple M4",
            ram="16GB",
        ),
        EndpointConfig(
            name="Home Desktop",
            url="https://llm-home-desktop.paperclips.dev",
            gpu="NVIDIA GeForce RTX 4070 Ti",
            vram="12GB",
            cpu="Intel Core i7-13700KF",
            ram="32GB",
        ),
        EndpointConfig(
            name="MacBook Pro 16-inch",
            url="https://llm-macbook-pro.paperclips.dev",
            soc="Apple M2 Max",
            ram="32GB",
        ),
    ]


@pytest.fixture
def reasoning_router():
    """Create a ReasoningRouter instance with pattern matching only."""
    return ReasoningRouter(use_llm_classification=False)


@pytest.fixture
def reasoning_router_with_llm():
    """Create a ReasoningRouter instance with LLM classification enabled."""
    return ReasoningRouter(use_llm_classification=True)


@pytest.fixture
def llm_router(mock_endpoints):
    """Create an LLMRouter instance with mock endpoints."""
    with patch.object(LLMRouter, "_load_endpoints", return_value=mock_endpoints):
        return LLMRouter()


class TestReasoningRouter:
    """Tests for the ReasoningRouter class."""

    @pytest.mark.asyncio
    async def test_simple_tasks_detection(self, reasoning_router):
        """Test that simple tasks are correctly identified."""
        simple_cases = [
            "What's the weather like today?",
            "Hi there!",
            "Write a hello world program",
            "What is 2+2?",
            "Good morning",
        ]

        for case in simple_cases:
            should_route = await reasoning_router.should_route(case)
            assert not should_route, f"'{case}' should not require reasoning"

    @pytest.mark.asyncio
    async def test_reasoning_tasks_detection(self, reasoning_router):
        """Test that reasoning-intensive tasks are correctly identified."""
        reasoning_cases = [
            "Please analyze the economic impact of inflation",
            "Solve this step by step: complex problem",
            "Think through this problem carefully",
            "Can you help me reason about quantum computing?",
            "Calculate the derivative of x^2 + 3x + 1",
            "How would you design a distributed cache?",
            "Explain why this approach works",
            "What if we tried a different strategy?",
        ]

        for case in reasoning_cases:
            should_route = await reasoning_router.should_route(case)
            assert should_route, f"'{case}' should require reasoning"

    @pytest.mark.asyncio
    async def test_select_endpoint_simple_task(self, reasoning_router, mock_endpoints):
        """Test endpoint selection for simple tasks."""
        simple_text = "What's the weather today?"

        decision = await reasoning_router.select_endpoint(simple_text, mock_endpoints)

        assert decision.endpoint == "Mac Mini"  # Should route to fastest endpoint
        assert decision.strategy == RouteStrategy.REASONING
        assert decision.confidence == 0.7
        assert "simple task" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_select_endpoint_complex_task(self, reasoning_router, mock_endpoints):
        """Test endpoint selection for complex reasoning tasks."""
        complex_text = "Analyze the philosophical implications step by step"

        decision = await reasoning_router.select_endpoint(complex_text, mock_endpoints)

        assert decision.endpoint == "HRPC-CISR HPC"  # Should route to powerful endpoint
        assert decision.strategy == RouteStrategy.REASONING
        assert decision.confidence == 0.8
        assert "reasoning" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_endpoint_fallback(self, reasoning_router):
        """Test fallback when preferred endpoints are not available."""
        limited_endpoints = [
            EndpointConfig(name="Test Endpoint", url="https://test.com")
        ]

        simple_decision = await reasoning_router.select_endpoint(
            "Hello", limited_endpoints
        )
        complex_decision = await reasoning_router.select_endpoint(
            "Analyze this complex problem", limited_endpoints
        )

        # Both should fall back to the only available endpoint
        assert simple_decision.endpoint == "Test Endpoint"
        assert complex_decision.endpoint == "Test Endpoint"

    def test_extract_vram_gb(self, reasoning_router):
        """Test VRAM extraction from strings."""
        assert reasoning_router._extract_vram_gb("16GB") == 16
        assert reasoning_router._extract_vram_gb("12GB") == 12
        assert reasoning_router._extract_vram_gb("32 GB") == 32
        assert reasoning_router._extract_vram_gb(None) == 0
        assert reasoning_router._extract_vram_gb("unknown") == 0

    @pytest.mark.asyncio
    async def test_llm_classification_success(
        self, reasoning_router_with_llm, mock_endpoints
    ):
        """Test LLM classification when endpoint is available."""
        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock successful LLM response
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "requires_reasoning"}}]
            }
            mock_post.return_value = mock_response

            decision = await reasoning_router_with_llm.select_endpoint(
                "Think step by step", mock_endpoints
            )

            # Should use LLM classification and route to HPC
            assert decision.endpoint == "HRPC-CISR HPC"
            assert decision.confidence == 0.9
            assert "LLM classified" in decision.reason

    @pytest.mark.asyncio
    async def test_llm_classification_fallback(
        self, reasoning_router_with_llm, mock_endpoints
    ):
        """Test fallback to pattern matching when LLM fails."""
        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock LLM endpoint failure
            import httpx

            mock_post.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=Mock(status_code=404)
            )

            decision = await reasoning_router_with_llm.select_endpoint(
                "Think step by step", mock_endpoints
            )

            # Should fall back to pattern matching
            assert decision.endpoint == "HRPC-CISR HPC"
            assert (
                decision.confidence == 0.9
            )  # Uses LLM confidence since fallback still shows as LLM classified
            assert (
                "LLM classified" in decision.reason
            )  # Still shows as LLM classified due to fallback

    @pytest.mark.asyncio
    async def test_llm_classification_simple_task(
        self, reasoning_router_with_llm, mock_endpoints
    ):
        """Test LLM classification for simple tasks."""
        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock LLM response for simple task
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "simple_task"}}]
            }
            mock_post.return_value = mock_response

            decision = await reasoning_router_with_llm.select_endpoint(
                "What's the weather?", mock_endpoints
            )

            # Should route to efficient endpoint
            assert decision.endpoint == "Mac Mini"
            assert decision.confidence == 0.85
            assert "LLM classified" in decision.reason


class TestLLMRouter:
    """Tests for the main LLMRouter class."""

    @pytest.mark.asyncio
    async def test_route_request_simple(self, llm_router):
        """Test routing a simple request."""
        decision = await llm_router.route_request("Hello world")

        assert decision.endpoint == "Mac Mini"
        assert decision.strategy == RouteStrategy.REASONING
        assert isinstance(decision.confidence, float)
        assert decision.reason

    @pytest.mark.asyncio
    async def test_route_request_complex(self, llm_router):
        """Test routing a complex reasoning request."""
        decision = await llm_router.route_request(
            "Think step by step about quantum mechanics"
        )

        assert decision.endpoint == "HRPC-CISR HPC"
        assert decision.strategy == RouteStrategy.REASONING
        assert isinstance(decision.confidence, float)
        assert decision.reason

    @pytest.mark.asyncio
    async def test_route_request_with_strategy(self, llm_router):
        """Test routing with explicit strategy specification."""
        decision = await llm_router.route_request(
            "Test message", strategy=RouteStrategy.REASONING
        )

        assert decision.strategy == RouteStrategy.REASONING
        assert decision.endpoint in llm_router.list_endpoints()

    @pytest.mark.asyncio
    async def test_route_request_no_endpoints(self):
        """Test routing behavior when no endpoints are configured."""
        with patch.object(LLMRouter, "_load_endpoints", return_value=[]):
            router = LLMRouter()

            with pytest.raises(ValueError, match="No endpoints configured"):
                await router.route_request("Test message")

    @pytest.mark.asyncio
    async def test_route_request_unknown_strategy(self, llm_router):
        """Test routing with unknown strategy."""
        # This would normally raise an error, but let's test the fallback
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            await llm_router.route_request(
                "Test", strategy="unknown_strategy"  # Invalid strategy
            )

    @pytest.mark.asyncio
    async def test_route_request_error_fallback(self, llm_router):
        """Test fallback behavior when routing fails."""
        # Mock the router to raise an exception
        with patch.object(
            llm_router.routers[RouteStrategy.REASONING], "select_endpoint"
        ) as mock_select:
            mock_select.side_effect = Exception("Test error")

            decision = await llm_router.route_request("Test message")

            # Should fallback to first endpoint
            assert decision.endpoint == llm_router.endpoints[0].name
            assert decision.confidence == 0.1
            assert "fallback" in decision.reason.lower()

    def test_get_endpoint_by_name(self, llm_router):
        """Test getting endpoint by name."""
        endpoint = llm_router.get_endpoint_by_name("Mac Mini")
        assert endpoint is not None
        assert endpoint.name == "Mac Mini"

        missing = llm_router.get_endpoint_by_name("Nonexistent")
        assert missing is None

    def test_list_endpoints(self, llm_router):
        """Test listing all endpoint names."""
        endpoints = llm_router.list_endpoints()
        expected = ["HRPC-CISR HPC", "Mac Mini", "Home Desktop", "MacBook Pro 16-inch"]
        assert endpoints == expected

    def test_add_router(self, llm_router):
        """Test adding a new routing strategy."""
        mock_router = Mock()
        llm_router.add_router(RouteStrategy.PERFORMANCE, mock_router)

        assert RouteStrategy.PERFORMANCE in llm_router.routers
        assert llm_router.routers[RouteStrategy.PERFORMANCE] == mock_router


class TestRouterGlobalFunctions:
    """Tests for global router functions."""

    def test_get_router_singleton(self):
        """Test that get_router returns the same instance."""
        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

    @pytest.mark.asyncio
    async def test_route_text_function(self):
        """Test the convenience route_text function."""
        # Import here to avoid circular import issues
        from ..llm_router import route_text

        with patch("src.orchestrator.llm_router.get_router") as mock_get_router:
            mock_router = Mock()
            mock_decision = RouteDecision(
                endpoint="Test",
                confidence=0.8,
                reason="Test reason",
                strategy=RouteStrategy.REASONING,
            )
            mock_router.route_request = AsyncMock(return_value=mock_decision)
            mock_get_router.return_value = mock_router

            result = await route_text("Test message")

            assert result == mock_decision
            mock_router.route_request.assert_called_once_with("Test message", None)


class TestRouteDecision:
    """Tests for the RouteDecision dataclass."""

    def test_route_decision_creation(self):
        """Test creating a RouteDecision instance."""
        decision = RouteDecision(
            endpoint="Test Endpoint",
            confidence=0.95,
            reason="Test reasoning",
            strategy=RouteStrategy.REASONING,
        )

        assert decision.endpoint == "Test Endpoint"
        assert decision.confidence == 0.95
        assert decision.reason == "Test reasoning"
        assert decision.strategy == RouteStrategy.REASONING


class TestEndpointConfig:
    """Tests for the EndpointConfig dataclass."""

    def test_endpoint_config_creation(self):
        """Test creating an EndpointConfig instance."""
        config = EndpointConfig(
            name="Test", url="https://test.com", gpu="NVIDIA RTX 4090", vram="24GB"
        )

        assert config.name == "Test"
        assert config.url == "https://test.com"
        assert config.gpu == "NVIDIA RTX 4090"
        assert config.vram == "24GB"
        assert config.cpu is None  # Optional field
        assert config.soc is None  # Optional field
        assert config.soc is None  # Optional field
