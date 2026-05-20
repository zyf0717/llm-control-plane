"""
Test suite for LLM router functionality.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.orchestrator.llm_router import (
    EndpointConfig,
    LLMRouter,
    RouteDecision,
    WorkloadClassifier,
    WorkloadType,
    get_router,
    reset_router,
)


@pytest.fixture
def mock_config():
    """Mock configuration data."""
    return {
        "endpoints": [
            {
                "name": "hrpc-cisr-hpc",
                "url": "https://llm-hrpc-cisr-hpc.paperclips.dev",
                "gpu": "NVIDIA GeForce RTX 5060 Ti 16GB",
                "vram": "16GB",
                "cpu": "Intel Core Ultra 9 285K",
                "ram": "192GB",
                "viable_models": ["gpt-oss-20b", "deepseek-r1-0528-qwen3-8b"],
            },
            {
                "name": "mac-mini",
                "url": "https://llm-mac-mini.paperclips.dev",
                "soc": "Apple M4",
                "ram": "16GB",
                "viable_models": ["qwen3-4b-2507"],
            },
            {
                "name": "gmktec-evo-x2",
                "url": "https://llm-evo-x2.paperclips.dev",
                "soc": "AMD Ryzen AI Max+ 395",
                "ram": "128GB",
                "viable_models": ["gpt-oss-120b", "Llama-3.3-70B-Instruct"],
            },
        ],
        "workloads": [
            {
                "type": "reasoning",
                "endpoint_preference": ["gmktec-evo-x2", "hrpc-cisr-hpc", "mac-mini"],
            },
            {
                "type": "programming",
                "endpoint_preference": ["gmktec-evo-x2", "hrpc-cisr-hpc", "mac-mini"],
            },
            {
                "type": "ttft_content",
                "endpoint_preference": ["mac-mini", "hrpc-cisr-hpc", "gmktec-evo-x2"],
            },
            {
                "type": "tokens_per_second",
                "endpoint_preference": ["hrpc-cisr-hpc", "mac-mini", "gmktec-evo-x2"],
            },
        ],
    }


@pytest.fixture
def mock_endpoints():
    """Mock endpoint configurations for testing."""
    return {
        "hrpc-cisr-hpc": EndpointConfig(
            name="hrpc-cisr-hpc",
            url="https://llm-hrpc-cisr-hpc.paperclips.dev",
            viable_models=["gpt-oss-20b", "deepseek-r1-0528-qwen3-8b"],
            gpu="NVIDIA GeForce RTX 5060 Ti 16GB",
            vram="16GB",
            cpu="Intel Core Ultra 9 285K",
            ram="192GB",
        ),
        "mac-mini": EndpointConfig(
            name="mac-mini",
            url="https://llm-mac-mini.paperclips.dev",
            viable_models=["qwen3-4b-2507"],
            soc="Apple M4",
            ram="16GB",
        ),
        "gmktec-evo-x2": EndpointConfig(
            name="gmktec-evo-x2",
            url="https://llm-evo-x2.paperclips.dev",
            viable_models=["gpt-oss-120b", "Llama-3.3-70B-Instruct"],
            soc="AMD Ryzen AI Max+ 395",
            ram="128GB",
        ),
    }


@pytest.fixture
def workload_preferences():
    """Mock workload preferences."""
    return {
        WorkloadType.REASONING: ["gmktec-evo-x2", "hrpc-cisr-hpc", "mac-mini"],
        WorkloadType.PROGRAMMING: ["gmktec-evo-x2", "hrpc-cisr-hpc", "mac-mini"],
        WorkloadType.TTFT_CONTENT: ["mac-mini", "hrpc-cisr-hpc", "gmktec-evo-x2"],
        WorkloadType.TOKENS_PER_SECOND: ["hrpc-cisr-hpc", "mac-mini", "gmktec-evo-x2"],
    }


@pytest.fixture
def llm_router(mock_config):
    """Create an LLMRouter instance with mocked configuration."""
    with patch("yaml.safe_load", return_value=mock_config):
        with patch("pathlib.Path.open"):
            return LLMRouter()


@pytest.fixture
def workload_classifier():
    """Create a WorkloadClassifier instance."""
    return WorkloadClassifier()


class TestWorkloadClassifier:
    """Tests for the WorkloadClassifier class."""

    @pytest.mark.asyncio
    async def test_classify_programming_patterns(self, workload_classifier):
        """Test classification of programming-related text using LLM."""
        programming_case = "Write a Python function to sort a list"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "programming"}}]
            }
            mock_response.raise_for_status = Mock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await workload_classifier.classify_with_llm(
                programming_case, "https://test.example.com/v1/chat/completions"
            )
            assert result == WorkloadType.PROGRAMMING

    @pytest.mark.asyncio
    async def test_classify_reasoning_patterns(self, workload_classifier):
        """Test classification of reasoning-intensive text using LLM."""
        reasoning_case = "Analyze the economic impact of inflation step by step"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "reasoning"}}]
            }
            mock_response.raise_for_status = Mock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await workload_classifier.classify_with_llm(
                reasoning_case, "https://test.example.com/v1/chat/completions"
            )
            assert result == WorkloadType.REASONING

    @pytest.mark.asyncio
    async def test_classify_default_content(self, workload_classifier):
        """Test that simple content can be classified as TTFT_CONTENT."""
        simple_case = "Hello, good morning!"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "ttft_content"}}]
            }
            mock_response.raise_for_status = Mock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await workload_classifier.classify_with_llm(
                simple_case, "https://test.example.com/v1/chat/completions"
            )
            assert result == WorkloadType.TTFT_CONTENT

    @pytest.mark.asyncio
    async def test_llm_classification_success(self, workload_classifier):
        """Test successful LLM classification."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "programming task detected"}}]
            }
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await workload_classifier.classify_with_llm(
                "Write a function", "https://test.com/api"
            )

            assert result == WorkloadType.PROGRAMMING

    @pytest.mark.asyncio
    async def test_llm_classification_failure(self, workload_classifier):
        """Test LLM classification failure handling."""
        with patch("httpx.AsyncClient") as mock_client_class:
            import httpx

            mock_client = Mock()
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=Mock(status_code=404)
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await workload_classifier.classify_with_llm(
                "test text", "https://test.com/api"
            )

            assert result is None


class TestLLMRouter:
    """Tests for the main LLMRouter class."""

    def test_load_config(self, llm_router):
        """Test that configuration is loaded correctly."""
        assert len(llm_router.endpoints) == 3
        assert "hrpc-cisr-hpc" in llm_router.endpoints
        assert "mac-mini" in llm_router.endpoints
        assert "gmktec-evo-x2" in llm_router.endpoints

        assert len(llm_router.workload_preferences) == 4
        assert WorkloadType.REASONING in llm_router.workload_preferences
        assert WorkloadType.PROGRAMMING in llm_router.workload_preferences

    @pytest.mark.asyncio
    async def test_route_reasoning_request(self, llm_router):
        """Test routing a reasoning request."""
        reachable = ["hrpc-cisr-hpc", "mac-mini", "gmktec-evo-x2"]
        decision = await llm_router.route_request(
            "Analyze this complex problem step by step",
            reachable,
            WorkloadType.REASONING,
        )

        assert decision.endpoint == "gmktec-evo-x2"  # First in reasoning preferences
        assert decision.workload_type == WorkloadType.REASONING
        assert decision.confidence == 0.9
        assert "reasoning workload" in decision.reason

    @pytest.mark.asyncio
    async def test_route_programming_request(self, llm_router):
        """Test routing a programming request."""
        reachable = ["hrpc-cisr-hpc", "mac-mini", "gmktec-evo-x2"]
        decision = await llm_router.route_request(
            "Write a Python function to parse JSON", reachable, WorkloadType.PROGRAMMING
        )

        assert decision.endpoint == "gmktec-evo-x2"  # First in programming preferences
        assert decision.workload_type == WorkloadType.PROGRAMMING
        assert decision.confidence == 0.9
        assert "programming workload" in decision.reason

    @pytest.mark.asyncio
    async def test_route_content_request(self, llm_router):
        """Test routing a simple content request."""
        reachable = ["hrpc-cisr-hpc", "mac-mini", "gmktec-evo-x2"]
        decision = await llm_router.route_request(
            "Tell me a joke", reachable, WorkloadType.TTFT_CONTENT
        )

        assert decision.endpoint == "mac-mini"  # First in content preferences
        assert decision.workload_type == WorkloadType.TTFT_CONTENT
        assert decision.confidence == 0.9
        assert "ttft_content workload" in decision.reason

    @pytest.mark.asyncio
    async def test_route_with_explicit_workload(self, llm_router):
        """Test routing with explicitly specified workload type."""
        reachable = ["hrpc-cisr-hpc", "mac-mini", "gmktec-evo-x2"]
        decision = await llm_router.route_request(
            "Hello world", reachable, workload_type=WorkloadType.TOKENS_PER_SECOND
        )

        assert (
            decision.endpoint == "hrpc-cisr-hpc"
        )  # First in tokens_per_second preferences
        assert decision.workload_type == WorkloadType.TOKENS_PER_SECOND
        assert decision.confidence == 0.9

    def test_get_fastest_endpoint(self, llm_router):
        """Test getting the fastest endpoint (prefers classification preferences)."""
        reachable = ["hrpc-cisr-hpc", "mac-mini", "gmktec-evo-x2"]
        fastest = llm_router._get_fastest_endpoint(reachable)
        assert fastest is not None
        assert fastest.name in reachable

    def test_get_endpoint_by_name(self, llm_router):
        """Test getting endpoint by name."""
        endpoint = llm_router.get_endpoint_by_name("mac-mini")
        assert endpoint is not None
        assert endpoint.name == "mac-mini"

        missing = llm_router.get_endpoint_by_name("nonexistent")
        assert missing is None

    def test_list_endpoints(self, llm_router):
        """Test listing all endpoint names."""
        endpoints = llm_router.list_endpoints()
        expected = ["hrpc-cisr-hpc", "mac-mini", "gmktec-evo-x2"]
        assert set(endpoints) == set(expected)


class TestGlobalFunctions:
    """Tests for global router functions."""

    def test_get_router_singleton(self):
        """Test that get_router returns the same instance."""
        reset_router()  # Ensure clean state

        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

    def test_reset_router(self):
        """Test router reset functionality."""
        router1 = get_router()
        reset_router()
        router2 = get_router()
        assert router1 is not router2


class TestDataClasses:
    """Tests for data classes."""

    def test_endpoint_config_creation(self):
        """Test creating an EndpointConfig instance."""
        config = EndpointConfig(
            name="test-endpoint",
            url="https://test.com",
            viable_models=["model1", "model2"],
            gpu="NVIDIA RTX 4090",
            vram="24GB",
        )

        assert config.name == "test-endpoint"
        assert config.url == "https://test.com"
        assert config.viable_models == ["model1", "model2"]
        assert config.gpu == "NVIDIA RTX 4090"
        assert config.vram == "24GB"
        assert config.soc is None  # Optional field

    def test_route_decision_creation(self):
        """Test creating a RouteDecision instance."""
        decision = RouteDecision(
            endpoint="test-endpoint",
            confidence=0.95,
            reason="Test reasoning",
            workload_type=WorkloadType.REASONING,
        )

        assert decision.endpoint == "test-endpoint"
        assert decision.confidence == 0.95
        assert decision.reason == "Test reasoning"
        assert decision.workload_type == WorkloadType.REASONING

    def test_workload_type_enum(self):
        """Test WorkloadType enum values."""
        assert WorkloadType.REASONING.value == "reasoning"
        assert WorkloadType.PROGRAMMING.value == "programming"
        assert WorkloadType.TTFT_CONTENT.value == "ttft_content"
        assert WorkloadType.TOKENS_PER_SECOND.value == "tokens_per_second"
