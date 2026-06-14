from __future__ import annotations

from typing import Any, Dict, Optional

from src.search.safety import EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER

from .prompt_state import normalize_system_prompt
from .utils import format_search_provider_label


def build_search_success_state(search_response: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a successful proxy search response for UI and request shaping."""
    provider_id = str(search_response.get("provider") or "").strip()
    warnings = [
        str(warning).strip()
        for warning in search_response.get("warnings", [])
        if str(warning).strip()
    ]
    results = search_response.get("results", [])
    if not isinstance(results, list):
        results = []

    query_refinement = search_response.get("query_refinement")
    if not isinstance(query_refinement, dict):
        query_refinement = {}

    return {
        "provider": provider_id,
        "provider_label": format_search_provider_label(provider_id),
        "query": str(search_response.get("query") or "").strip(),
        "query_refinement": query_refinement,
        "degraded": bool(search_response.get("degraded", False)),
        "warnings": warnings,
        "results": results,
        "result_count": len(results),
        "search_evidence": (
            search_response.get("search_evidence")
            if isinstance(search_response.get("search_evidence"), str)
            else None
        ),
        "show_preface": True,
    }


def build_query_refiner_source_text(
    *,
    system_prompt: Optional[str],
    history: Any,
    user_input: str,
) -> str:
    """Build compact source text for search query refinement."""
    sections = []
    prompt = normalize_system_prompt(system_prompt)
    if prompt:
        sections.append(f"System prompt:\n{prompt}")

    history_lines = []
    if isinstance(history, list):
        for message in history:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if role and content:
                history_lines.append(f"{role}: {content}")
    if history_lines:
        sections.append("Conversation history:\n" + "\n".join(history_lines))

    request = str(user_input or "").strip()
    if request:
        sections.append(f"Current user request:\n{request}")

    return "\n\n".join(sections)


def build_search_failure_state(provider_id: str, error: Any) -> Dict[str, Any]:
    """Build a non-blocking search failure state for metadata only."""
    return {
        "provider": str(provider_id or "").strip(),
        "provider_label": format_search_provider_label(provider_id),
        "degraded": True,
        "warnings": [f"search request failed: {str(error)}"],
        "results": [],
        "result_count": 0,
        "search_evidence": None,
        "show_preface": False,
    }


def _format_search_evidence(search_state: Dict[str, Any]) -> Optional[str]:
    """Render turn-local search evidence for model consumption, not conversation."""
    results = search_state.get("results")
    if not isinstance(results, list) or not results:
        return None

    search_evidence = search_state.get("search_evidence")
    if not isinstance(search_evidence, str) or not search_evidence.strip():
        return None

    return "\n".join([EPHEMERAL_WEB_SEARCH_EVIDENCE_MARKER, search_evidence.strip()])


def build_search_turn_messages(
    search_state: Optional[Dict[str, Any]],
) -> list[Dict[str, str]]:
    """Convert successful search state into turn-local injected messages."""
    if not isinstance(search_state, dict):
        return []

    content = _format_search_evidence(search_state)
    if not content:
        return []

    return [{"role": "user", "content": content}]


def _query_refinement_queries(
    query_refinement: Dict[str, Any], fallback_query: str
) -> list[str]:
    raw_queries = query_refinement.get("queries")
    queries = raw_queries if isinstance(raw_queries, list) else []
    cleaned = []
    seen = set()
    for item in [*queries, fallback_query]:
        query = " ".join(str(item or "").split())
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
    return cleaned


def build_search_preface(search_state: Optional[Dict[str, Any]]) -> Optional[str]:
    """Render a markdown transcript preface for successful search calls."""
    if not isinstance(search_state, dict) or not search_state.get("show_preface"):
        return None

    provider_label = str(search_state.get("provider_label") or "Search").strip()
    query_refinement = (
        search_state.get("query_refinement")
        if isinstance(search_state.get("query_refinement"), dict)
        else {}
    )
    query = " ".join(str(search_state.get("query") or "").split())
    refined_queries = _query_refinement_queries(query_refinement, query)
    results = search_state.get("results")
    warnings = search_state.get("warnings") or []
    degraded = bool(search_state.get("degraded", False))
    heading = f"**Search candidates** via {provider_label}"
    if query_refinement.get("used") is True and refined_queries:
        quoted_queries = "; ".join(f'"{item}"' for item in refined_queries)
        heading = f"{heading} for {quoted_queries}"
    lines = [heading]

    if degraded:
        lines.append("_Search returned degraded results._")

    if isinstance(results, list) and results:
        for index, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            title = str(result.get("title") or result.get("url") or f"Result {index}")
            url = str(result.get("url") or "").strip()
            snippet = str(result.get("snippet") or "").strip()
            line = f"{index}. [{title}]({url})" if url else f"{index}. {title}"
            if snippet:
                line = f"{line} - {snippet}"
            lines.append(line)
    else:
        lines.append("No candidates found.")

    if warnings:
        lines.append(f"Warnings: {'; '.join(str(warning) for warning in warnings)}")

    return "\n".join(lines)


def merge_run_info(
    metadata: Optional[Dict[str, Any]], search_state: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge search metadata into the runtime panel payload."""
    merged = dict(metadata or {})

    if isinstance(search_state, dict):
        merged["search"] = {
            "provider": search_state.get("provider_label")
            or search_state.get("provider")
            or "Unknown",
            "provider_id": search_state.get("provider") or "",
            "result_count": search_state.get("result_count", 0),
            "degraded": bool(search_state.get("degraded", False)),
            "warnings": list(search_state.get("warnings") or []),
        }
        query_refinement = search_state.get("query_refinement")
        query = str(search_state.get("query") or "").strip()
        if (
            isinstance(query_refinement, dict)
            and query_refinement.get("used") is True
            and query
        ):
            merged["search"]["query"] = query
            queries = _query_refinement_queries(query_refinement, query)
            if len(queries) > 1:
                merged["search"]["queries"] = queries

    return merged or None
