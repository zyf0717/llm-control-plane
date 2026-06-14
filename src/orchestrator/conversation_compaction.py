from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .conversation_context import format_message_records
from .history_store import HistoryStore
from .workflow.models import WorkflowDefaults, WorkflowRun, WorkflowSpec, WorkflowStepSpec
from .workflow.step_executor import WorkflowLLMClient, WorkflowStepExecutor


CONVERSATION_COMPACTION_GOAL = (
    "Preserve durable conversation intent, decisions, constraints, open tasks, "
    "facts, source references, contradictions, uncertainty, and user preferences. "
    "Do not treat the latest raw tail as omitted truth; it remains available "
    "separately."
)


@dataclass(slots=True)
class ConversationCompactionSettings:
    enabled: bool = True
    recent_tail_messages: int = 20
    trigger_delta_chars: int = 16000
    trigger_delta_messages: int = 20
    compaction_trigger_chars: int = 12000
    compaction_chunk_chars: int = 10000
    compaction_target_chars: int = 6000
    compaction_max_output_chars: int = 8000
    compaction_max_rounds: int = 2


async def maybe_refresh_compacted_conversation_state(
    *,
    history_store: HistoryStore,
    llm_client: WorkflowLLMClient,
    source_convo_id: str,
    endpoint: str,
    reasoning_effort: str | None = None,
    settings: ConversationCompactionSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    settings = settings or ConversationCompactionSettings()
    if not settings.enabled and not force:
        return {"updated": False, "reason": "disabled"}

    compacted_state = await history_store.get_compacted_conversation_state(
        source_convo_id
    )
    covered_message_id = int((compacted_state or {}).get("covered_message_id") or 0)
    all_records = await history_store.get_conversation_message_records(source_convo_id)
    if not all_records:
        return {"updated": False, "reason": "no-messages"}

    records_after = [
        record for record in all_records if int(record.get("id") or 0) > covered_message_id
    ]
    if not records_after:
        return {
            "updated": False,
            "reason": "no-new-messages",
            "covered_message_id": covered_message_id,
            "latest_message_id": int(all_records[-1].get("id") or 0),
        }

    compactable_records = _compactable_records(
        all_records,
        records_after,
        recent_tail_messages=settings.recent_tail_messages,
        force=force,
    )
    if not compactable_records:
        return {
            "updated": False,
            "reason": "recent-tail-only",
            "covered_message_id": covered_message_id,
            "latest_message_id": int(all_records[-1].get("id") or 0),
        }

    new_messages_text = format_message_records(compactable_records)
    delta_chars = len(new_messages_text)
    if (
        not force
        and len(compactable_records) < settings.trigger_delta_messages
        and delta_chars < settings.trigger_delta_chars
    ):
        return {
            "updated": False,
            "reason": "below-threshold",
            "delta_messages": len(compactable_records),
            "delta_chars": delta_chars,
            "covered_message_id": covered_message_id,
            "latest_message_id": int(all_records[-1].get("id") or 0),
        }

    previous_state_text = str((compacted_state or {}).get("state_text") or "").strip()
    source_text = (
        "Previous compacted state:\n"
        f"{previous_state_text or '(none)'}\n\n"
        "New raw conversation messages:\n"
        f"{new_messages_text}"
    ).strip()
    compacted_output = await _run_conversation_compaction(
        llm_client=llm_client,
        endpoint=endpoint,
        convo_id=f"{source_convo_id}:conversation_compaction",
        source_text=source_text,
        reasoning_effort=reasoning_effort,
        settings=settings,
    )
    output_json = (
        compacted_output.get("json") if isinstance(compacted_output.get("json"), dict) else {}
    )
    covered_to = max(int(record.get("id") or 0) for record in compactable_records)
    state = await history_store.upsert_compacted_conversation_state(
        source_convo_id,
        covered_message_id=covered_to,
        state_text=str(compacted_output.get("text") or "").strip(),
        state_json=(
            output_json.get("compact_output")
            if isinstance(output_json.get("compact_output"), dict)
            else None
        ),
        metadata_json={
            "source_convo_id": source_convo_id,
            "delta_messages": len(compactable_records),
            "delta_chars": delta_chars,
            "latest_message_id": int(all_records[-1].get("id") or 0),
            "compaction": output_json,
        },
    )
    return {
        "updated": True,
        "reason": "force" if force else "threshold",
        "delta_messages": len(compactable_records),
        "delta_chars": delta_chars,
        "covered_message_id": covered_to,
        "latest_message_id": int(all_records[-1].get("id") or 0),
        "state": state,
    }


def _compactable_records(
    all_records: list[dict[str, Any]],
    records_after: list[dict[str, Any]],
    *,
    recent_tail_messages: int,
    force: bool,
) -> list[dict[str, Any]]:
    if force:
        return records_after
    tail_count = max(0, int(recent_tail_messages))
    if not tail_count:
        return records_after
    if len(all_records) <= tail_count:
        return []
    max_compactable_id = int(all_records[-tail_count - 1].get("id") or 0)
    return [
        record
        for record in records_after
        if int(record.get("id") or 0) <= max_compactable_id
    ]


async def _run_conversation_compaction(
    *,
    llm_client: WorkflowLLMClient,
    endpoint: str,
    convo_id: str,
    source_text: str,
    reasoning_effort: str | None,
    settings: ConversationCompactionSettings,
) -> dict[str, Any]:
    trigger_chars = max(1, min(settings.compaction_trigger_chars, len(source_text)))
    now = "1970-01-01T00:00:00+00:00"
    run = WorkflowRun(
        run_id="conversation_compaction",
        workflow_id="conversation_compaction",
        workflow_version="0.1.0",
        status="running",
        convo_id=convo_id,
        params={},
        endpoint=endpoint,
        reasoning_effort=reasoning_effort,
        rag_endpoint=None,
        search_provider=None,
        current_step_id="compact",
        created_at=now,
        updated_at=now,
        completed_at=None,
    )
    step = WorkflowStepSpec(
        id="compact",
        name="Compact conversation",
        kind="compact_context",
        prompt=None,
        output_key="compacted_thread_state",
        compaction_trigger_chars=trigger_chars,
        compaction_chunk_chars=settings.compaction_chunk_chars,
        compaction_target_chars=settings.compaction_target_chars,
        compaction_max_output_chars=settings.compaction_max_output_chars,
        compaction_max_rounds=settings.compaction_max_rounds,
        compaction_goal=CONVERSATION_COMPACTION_GOAL,
    )
    spec = WorkflowSpec(
        id="conversation_compaction",
        name="Conversation Compaction",
        description=None,
        version="0.1.0",
        params_schema={"type": "object"},
        defaults=WorkflowDefaults(reasoning_effort=reasoning_effort),
        steps=[step],
    )
    execution = await WorkflowStepExecutor(llm_client)._execute_compact_context_step(
        run=run,
        spec=spec,
        step=step,
        source=source_text,
        step_input={},
    )
    return execution.output
