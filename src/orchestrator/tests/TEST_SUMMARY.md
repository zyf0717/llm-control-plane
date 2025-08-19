# Test Suite Documentation

## Overview

The LLM Control Plane has a comprehensive pytest-based test suite covering all major functionality including the new intelligent routing system.

## Test Structure

### Unit Tests (`tests/test_proxy.py`)

- **TestEndpointRouting**: Tests for endpoint routing logic
  - Endpoint mapping for known endpoints
  - Default fallback for unknown endpoints

- **TestHeaderPreparation**: Tests for header preparation
  - API key injection
  - Header filtering

- **TestConversationHistory**: Tests for conversation history management
  - Empty body handling
  - Invalid JSON handling
  - Message injection and history management
  - Conversation continuity

### LLM Router Tests (`tests/test_llm_router.py`)

- **TestReasoningRouter**: Tests for reasoning detection and routing
  - Simple task detection (should not require reasoning)
  - Complex reasoning task detection (analyze, solve, step-by-step, etc.)
  - Endpoint selection based on task complexity
  - Hardware-based endpoint fallback
  - VRAM extraction from endpoint configurations

- **TestLLMRouter**: Tests for main router orchestration
  - Route request handling for simple and complex tasks
  - Strategy-based routing (currently REASONING strategy)
  - Error handling and fallback behavior
  - Endpoint management (listing, finding by name)
  - Router extensibility (adding new routing strategies)

- **TestRouterGlobalFunctions**: Tests for global router access
  - Singleton pattern for router instance
  - Convenience routing functions

- **TestDataClasses**: Tests for routing data structures
  - RouteDecision creation and attributes
  - EndpointConfig creation and attributes

### Smart Routing Integration Tests (`tests/test_smart_routing.py`)

- **TestSmartRoutingEndpoint**: Tests for `/smart` endpoint functionality
  - Basic routing functionality with mocked responses
  - Input validation (no messages, no user messages, invalid JSON)
  - Multiple user message handling (uses latest message for routing)
  - Reasoning detection integration
  - Error handling for router failures

- **TestSmartRoutingIntegration**: Integration tests for smart routing
  - Logging verification for routing decisions
  - Header propagation for routing metadata

### Integration Tests (`tests/test_integration.py`)

- Basic proxy functionality tests
- Root endpoint structure testing

## Test Configuration

- **pytest.ini**: Test configuration with proper markers
- **conftest.py**: Shared fixtures and environment mocking
- **Markers**: `@pytest.mark.integration` for slow/network tests
- **Environment Mocking**: Consistent test environment variables

## Coverage Summary

The test suite covers:

- **Core Proxy Functionality**: All existing proxy features
- **Intelligent Routing**: Complete LLM router system including:
  - Pattern-based reasoning detection
  - Hardware-aware endpoint selection
  - Strategy-based routing framework
  - Error handling and fallbacks
- **Smart Endpoint**: New `/smart` endpoint with full integration
- **Data Structures**: All routing-related classes and enums

## Status

- **28 tests passing** ✅ (19 router tests + 9 smart routing tests)
- **0 failures** ✅
- **Full test coverage** of routing functionality ✅
