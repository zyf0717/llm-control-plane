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
async def test_fetch_available_rag_endpoints_keeps_unhealthy_configured_options(
    monkeypatch,
):
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
        utils.NONE_RAG_OPTION_VALUE: "None",
        "http://healthy/api/retrieve/context": "healthy (http://healthy/api/retrieve/context)",
        "http://down/api/retrieve/context": "down (http://down/api/retrieve/context)",
    }
    assert selected == utils.NONE_RAG_OPTION_VALUE


@pytest.mark.asyncio
async def test_fetch_available_rag_endpoints_does_not_auto_select_configured_default(
    monkeypatch,
):
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

    assert choices == {
        utils.NONE_RAG_OPTION_VALUE: "None",
        "http://down/api/retrieve/context": "down (http://down/api/retrieve/context)",
    }
    assert selected == utils.NONE_RAG_OPTION_VALUE


def test_create_history_select_choices_uses_convo_id_as_value():
    conversations = [
        {
            "convo_id": "c3630b242ac7",
            "last_updated": "2026-05-21T05:48:40+00:00",
        }
    ]

    assert utils.create_history_select_choices(conversations) == {
        "c3630b242ac7": "2026-05-21 13:48:40 | c3630b242ac7"
    }
