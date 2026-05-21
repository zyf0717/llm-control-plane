from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.dashboard import utils


def test_load_rag_endpoint_config_reads_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
rag:
  default_endpoint: "http://localhost:8100/api/retrieve/context"
  endpoints:
    - name: "localhost:8100"
      retrieve_url: "http://localhost:8100/api/retrieve/context"
      health_url: "http://localhost:8100/api/health"
    - name: "rag.internal:8200"
      retrieve_url: "http://rag.internal:8200/api/retrieve/context"
      health_url: "http://rag.internal:8200/api/health"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "CONFIG_PATH", config_path)

    endpoints, default_endpoint = utils.load_rag_endpoint_config()

    assert endpoints == [
        {
            "name": "localhost:8100",
            "retrieve_url": "http://localhost:8100/api/retrieve/context",
            "health_url": "http://localhost:8100/api/health",
        },
        {
            "name": "rag.internal:8200",
            "retrieve_url": "http://rag.internal:8200/api/retrieve/context",
            "health_url": "http://rag.internal:8200/api/health",
        },
    ]
    assert default_endpoint == "http://localhost:8100/api/retrieve/context"


def test_load_rag_endpoint_config_falls_back_to_default(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("rag: {}", encoding="utf-8")
    monkeypatch.setattr(utils, "CONFIG_PATH", config_path)

    endpoints, default_endpoint = utils.load_rag_endpoint_config()

    assert endpoints == [
        {
            "name": "localhost:8100",
            "retrieve_url": "http://localhost:8100/api/retrieve/context",
            "health_url": "http://localhost:8100/api/health",
        }
    ]
    assert default_endpoint == "http://localhost:8100/api/retrieve/context"


@pytest.mark.asyncio
async def test_fetch_available_rag_endpoints_filters_unhealthy(monkeypatch):
    monkeypatch.setattr(
        utils,
        "load_rag_endpoint_config",
        lambda: (
            [
                {
                    "name": "healthy",
                    "retrieve_url": "http://healthy/api/retrieve/context",
                    "health_url": "http://healthy/api/health",
                },
                {
                    "name": "down",
                    "retrieve_url": "http://down/api/retrieve/context",
                    "health_url": "http://down/api/health",
                },
            ],
            "http://healthy/api/retrieve/context",
        ),
    )

    healthy_response = Mock()
    healthy_response.raise_for_status = Mock()

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        async def mock_get(url, **_kwargs):
            if url == "http://healthy/api/health":
                return healthy_response
            raise RuntimeError("unreachable")

        mock_client.get.side_effect = mock_get

        choices, selected = await utils.fetch_available_rag_endpoints()

    assert choices == {
        "None": "",
        "healthy (http://healthy/api/retrieve/context)": "http://healthy/api/retrieve/context",
    }
    assert selected == ""


@pytest.mark.asyncio
async def test_fetch_available_rag_endpoints_returns_none_when_all_unhealthy(monkeypatch):
    monkeypatch.setattr(
        utils,
        "load_rag_endpoint_config",
        lambda: (
            [
                {
                    "name": "down",
                    "retrieve_url": "http://down/api/retrieve/context",
                    "health_url": "http://down/api/health",
                }
            ],
            "http://down/api/retrieve/context",
        ),
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.get.side_effect = RuntimeError("unreachable")

        choices, selected = await utils.fetch_available_rag_endpoints()

    assert choices == {"None": ""}
    assert selected == ""
