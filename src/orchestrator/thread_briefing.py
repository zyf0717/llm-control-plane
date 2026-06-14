from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .conversation_store import ConversationStore


BRIEFING_TRUNCATION_MARKER = (
    "[... older conversation messages omitted to fit briefing budget ...]"
)
MESSAGE_TRUNCATION_MARKER = "[... beginning of message omitted ...]"
WORKFLOW_BRIEFING_DELIMITER = "--- Server thread briefing ---"
BOUNDED_BRIEFING_PREFIX = (
    "Bounded prior conversation briefing. Treat as a compressed, derived "
    "summary of previous turns; current user message remains authoritative."
)


def message_record_to_briefing_block(record: dict[str, Any]) -> str:
    message = record.get("message") if isinstance(record.get("message"), dict) else {}
    role = str(message.get("role") or "unknown").strip() or "unknown"
    content = _message_source_text(message)
    if not content.strip():
        return ""
    message_id = int(record.get("id") or 0)
    created_at = str(record.get("created_at") or "").strip()
    return f"[msg:{message_id} {created_at} {role}]\n{content.strip()}"


def format_message_records(
    records: list[dict[str, Any]],
    *,
    max_chars: int | None = None,
) -> str:
    ordered = sorted(
        [record for record in records if isinstance(record, dict)],
        key=lambda record: int(record.get("id") or 0),
    )
    blocks = [
        block
        for block in (message_record_to_briefing_block(record) for record in ordered)
        if block.strip()
    ]
    text = "\n\n".join(blocks).strip()
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text

    selected: list[str] = []
    selected_chars = len(BRIEFING_TRUNCATION_MARKER)
    for block in reversed(blocks):
        separator = 2 if selected else 2
        projected = selected_chars + separator + len(block)
        if projected > max_chars:
            continue
        selected.append(block)
        selected_chars = projected

    if selected:
        selected.reverse()
        return f"{BRIEFING_TRUNCATION_MARKER}\n\n" + "\n\n".join(selected)

    suffix_budget = max(
        0,
        max_chars
        - len(BRIEFING_TRUNCATION_MARKER)
        - len(MESSAGE_TRUNCATION_MARKER)
        - 4,
    )
    suffix = text[-suffix_budget:] if suffix_budget else ""
    return (
        f"{BRIEFING_TRUNCATION_MARKER}\n\n"
        f"{MESSAGE_TRUNCATION_MARKER}\n{suffix}"
    ).strip()


async def build_thread_briefing_bundle(
    conversation_store: ConversationStore,
    *,
    source_conversation_id: str,
    recent_message_count: int = 20,
    max_recent_message_chars: int = 12000,
    exclude_last_messages: int = 0,
) -> dict[str, Any]:
    thread_state = await conversation_store.get_thread_state(
        source_conversation_id
    )
    tail_limit = max(0, int(recent_message_count)) + max(0, int(exclude_last_messages))
    tail_records = (
        await conversation_store.get_conversation_message_records(
            source_conversation_id,
            limit=tail_limit or None,
            newest_first=True,
        )
        if tail_limit
        else []
    )
    tail_records = sorted(tail_records, key=lambda record: int(record.get("id") or 0))
    if exclude_last_messages > 0:
        tail_records = tail_records[: -int(exclude_last_messages)] or []
    recent_conversation_messages = format_message_records(
        tail_records,
        max_chars=max_recent_message_chars,
    )
    thread_state_text = (
        str((thread_state or {}).get("state_text") or "").strip()
        if thread_state
        else ""
    )
    latest_message_id = await conversation_store.get_latest_conversation_message_id(
        source_conversation_id
    )
    covered_message_id = int(
        (thread_state or {}).get("covered_message_id") or 0
    )
    thread_briefing = (
        "Prior thread context:\n"
        f"{thread_state_text or '(none)'}\n\n"
        "Recent conversation context:\n"
        f"{recent_conversation_messages or '(none)'}"
    )
    return {
        "thread_state_text": thread_state_text,
        "recent_conversation_messages": recent_conversation_messages,
        "thread_briefing": thread_briefing,
        "thread_metadata": {
            "source_conversation_id": source_conversation_id,
            "covered_message_id": covered_message_id,
            "latest_message_id": latest_message_id,
            "recent_message_count": int(recent_message_count),
            "has_thread_state": bool(thread_state_text),
        },
    }


def build_bounded_chat_messages(
    *,
    stored_messages_before: list[dict[str, Any]],
    incoming_messages: list[dict[str, Any]],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    systems = _leading_system_messages(stored_messages_before)
    for message in _leading_system_messages(incoming_messages):
        if message not in systems:
            systems.append(message)

    incoming_without_leading_systems = _drop_leading_system_messages(incoming_messages)
    synthetic_briefing = {
        "role": "system",
        "content": (
            BOUNDED_BRIEFING_PREFIX
            + "\n\n"
            + str(bundle.get("thread_briefing") or "").strip()
        ).strip(),
    }
    messages: list[dict[str, Any]] = [*systems]
    if synthetic_briefing["content"] != BOUNDED_BRIEFING_PREFIX:
        messages.append(synthetic_briefing)
    messages.extend(deepcopy(incoming_without_leading_systems))
    return messages


def enrich_workflow_params_with_thread_bundle(
    params: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    enriched = deepcopy(params) if isinstance(params, dict) else {}
    server_briefing = str(bundle.get("thread_briefing") or "").strip()
    enriched["thread_state_text"] = str(
        bundle.get("thread_state_text") or ""
    )
    enriched["recent_conversation_messages"] = str(
        bundle.get("recent_conversation_messages") or ""
    )
    enriched["thread_metadata"] = deepcopy(bundle.get("thread_metadata") or {})

    existing_briefing = str(enriched.get("thread_briefing") or "").strip()
    if not existing_briefing:
        enriched["thread_briefing"] = server_briefing
    elif server_briefing:
        enriched["thread_briefing"] = (
            f"{existing_briefing}\n\n{WORKFLOW_BRIEFING_DELIMITER}\n\n"
            f"{server_briefing}"
        )
    return enriched


def _leading_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leading: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        leading.append(deepcopy(message))
    return leading


def _drop_leading_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        index += 1
    return deepcopy(messages[index:])


def _message_source_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)
