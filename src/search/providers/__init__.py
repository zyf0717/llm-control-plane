"""Search provider registry."""

from .duckduckgo_html import DuckDuckGoHtmlProvider
from .marginalia_html import MarginaliaHtmlProvider
from .mojeek_html import MojeekHtmlProvider
from .searxng_html import SearxngHtmlProvider
from .wikipedia_opensearch import WikipediaOpenSearchProvider

__all__ = [
    "DuckDuckGoHtmlProvider",
    "MarginaliaHtmlProvider",
    "MojeekHtmlProvider",
    "SearxngHtmlProvider",
    "WikipediaOpenSearchProvider",
]
