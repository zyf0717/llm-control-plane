from pathlib import Path

import pytest

from src.search.providers.duckduckgo_html import DuckDuckGoHtmlProvider
from src.search.providers.marginalia_html import MarginaliaHtmlProvider
from src.search.providers.mojeek_html import MojeekHtmlProvider
from src.search.types import SearchArgs

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_duckduckgo_provider_parses_and_dedupes_results():
    provider = DuckDuckGoHtmlProvider()

    results = provider.parse(_load("duckduckgo.simple.html"), SearchArgs(query="alpha"))

    assert [result.title for result in results] == ["Alpha Result", "Beta Result"]
    assert results[0].url == "https://example.com/alpha"
    assert results[0].snippet == "Alpha snippet"
    assert results[1].rank == 2


def test_duckduckgo_provider_detects_challenge_pages():
    provider = DuckDuckGoHtmlProvider()

    with pytest.raises(ValueError, match="challenge"):
        provider.parse(_load("duckduckgo.challenge.html"), SearchArgs(query="alpha"))


def test_marginalia_provider_handles_missing_snippet():
    provider = MarginaliaHtmlProvider()

    results = provider.parse(_load("marginalia.simple.html"), SearchArgs(query="alpha"))

    assert len(results) == 2
    assert results[0].snippet == "Independent web result"
    assert results[1].snippet is None


def test_mojeek_provider_parses_simple_results():
    provider = MojeekHtmlProvider()

    results = provider.parse(_load("mojeek.simple.html"), SearchArgs(query="docs"))

    assert len(results) == 1
    assert results[0].title == "Docs Page"
    assert results[0].url == "https://docs.example.org/page"
