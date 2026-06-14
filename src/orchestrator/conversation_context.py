from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .history_store import HistoryStore


CONTEXT_TRUNCATION_MARKER = (
    "[... older conversation messages omitted to fit context budget ...]"
)
MESSAGE_TRUNCATION_MARKER = "[... beginning of message omitted ...]"
WORKFLOW_CONTEXT_DELIMITER = "--- Server bounded conversation context ---"
BOUNDED_CONTEXT_PREFIX = (
    "Bounded prior conversation context. Treat as a compressed, derived "
    "summary of previous turns; current user message remains authoritative."
)


def message_record_to_context_block(record: dict[str, Any]) -> str:
    message = record.get("message") if isinstance(record.get("message"), dict) else {}
    role = str(message.get("role") or "unknown").strip() or "unknown"
    content = _message_content_text(message)
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
        for block in (message_record_to_context_block(record) for record in ordered)
        if block.strip()
    ]
    text = "\n\n".join(blocks).strip()
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text

    selected: list[str] = []
    selected_chars = len(CONTEXT_TRUNCATION_MARKER)
    for block in reversed(blocks):
        separator = 2 if selected else 2
        projected = selected_chars + separator + len(block)
        if projected > max_chars:
            continue
        selected.append(block)
        selected_chars = projected

    if selected:
        selected.reverse()
        return f"{CONTEXT_TRUNCATION_MARKER}\n\n" + "\n\n".join(selected)

    suffix_budget = max(
        0,
        max_chars
        - len(CONTEXT_TRUNCATION_MARKER)
        - len(MESSAGE_TRUNCATION_MARKER)
        - 4,
    )
    suffix = text[-suffix_budget:] if suffix_budget else ""
    return (
        f"{CONTEXT_TRUNCATION_MARKER}\n\n"
        f"{MESSAGE_TRUNCATION_MARKER}\n{suffix}"
    ).strip()


async def build_conversation_context_bundle(
    history_store: HistoryStore,
    *,
    source_convo_id: str,
    recent_tail_messages: int = 20,
    max_recent_tail_chars: int = 12000,
    exclude_last_messages: int = 0,
) -> dict[str, Any]:
    compacted_state = await history_store.get_compacted_conversation_state(
        source_convo_id
    )
    tail_limit = max(0, int(recent_tail_messages)) + max(0, int(exclude_last_messages))
    tail_records = (
        await history_store.get_conversation_message_records(
            source_convo_id,
            limit=tail_limit or None,
            newest_first=True,
        )
        if tail_limit
        else []
    )
    tail_records = sorted(tail_records, key=lambda record: int(record.get("id") or 0))
    if exclude_last_messages > 0:
        tail_records = tail_records[: -int(exclude_last_messages)] or []
    recent_tail = format_message_records(
        tail_records,
        max_chars=max_recent_tail_chars,
    )
    compacted_text = (
        str((compacted_state or {}).get("state_text") or "").strip()
        if compacted_state
        else ""
    )
    latest_message_id = await history_store.get_latest_conversation_message_id(
        source_convo_id
    )
    covered_message_id = int(
        (compacted_state or {}).get("covered_message_id") or 0
    )
    server_context = (
        "Compacted prior conversation state:\n"
        f"{compacted_text or '(none)'}\n\n"
        "Recent raw conversation tail:\n"
        f"{recent_tail or '(none)'}"
    )
    return {
        "compacted_thread_state": compacted_text,
        "recent_conversation_tail": recent_tail,
        "server_conversation_context": server_context,
        "context_state": {
            "source_convo_id": source_convo_id,
            "covered_message_id": covered_message_id,
            "latest_message_id": latest_message_id,
            "recent_tail_messages": int(recent_tail_messages),
            "has_compacted_state": bool(compacted_text),
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
    synthetic_context = {
        "role": "system",
        "content": (
            BOUNDED_CONTEXT_PREFIX
            + "\n\n"
            + str(bundle.get("server_conversation_context") or "").strip()
        ).strip(),
    }
    messages: list[dict[str, Any]] = [*systems]
    if synthetic_context["content"] != BOUNDED_CONTEXT_PREFIX:
        messages.append(synthetic_context)
    messages.extend(deepcopy(incoming_without_leading_systems))
    return messages


def enrich_workflow_params_with_context_bundle(
    params: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    enriched = deepcopy(params) if isinstance(params, dict) else {}
    server_context = str(bundle.get("server_conversation_context") or "").strip()
    enriched["compacted_thread_state"] = str(
        bundle.get("compacted_thread_state") or ""
    )
    enriched["recent_conversation_tail"] = str(
        bundle.get("recent_conversation_tail") or ""
    )
    enriched["server_conversation_context"] = server_context
    enriched["context_state"] = deepcopy(bundle.get("context_state") or {})

    existing_context = str(enriched.get("conversation_context") or "").strip()
    if not existing_context:
        enriched["conversation_context"] = server_context
    elif server_context:
        enriched["conversation_context"] = (
            f"{existing_context}\n\n{WORKFLOW_CONTEXT_DELIMITER}\n\n{server_context}"
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


def _message_content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)
