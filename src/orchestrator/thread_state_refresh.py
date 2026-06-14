from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .thread_briefing import format_message_records
from .conversation_store import ConversationStore
from .workflow.models import WorkflowDefaults, WorkflowRun, WorkflowSpec, WorkflowStepSpec
from .workflow.step_executor import WorkflowLLMClient, WorkflowStepExecutor


THREAD_STATE_REFRESH_GOAL = (
    "Preserve durable conversation intent, decisions, constraints, open tasks, "
    "facts, source references, contradictions, uncertainty, and user preferences. "
    "Do not treat the latest raw tail as omitted truth; it remains available "
    "separately."
)


@dataclass(slots=True)
class ThreadStateRefreshSettings:
    enabled: bool = True
    recent_message_count: int = 20
    trigger_delta_chars: int = 16000
    trigger_delta_messages: int = 20
    compression_trigger_chars: int = 12000
    compression_chunk_chars: int = 10000
    compression_target_chars: int = 6000
    compression_max_output_chars: int = 8000
    compression_max_rounds: int = 2


async def maybe_refresh_thread_state(
    *,
    conversation_store: ConversationStore,
    llm_client: WorkflowLLMClient,
    source_conversation_id: str,
    endpoint: str,
    reasoning_effort: str | None = None,
    settings: ThreadStateRefreshSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    settings = settings or ThreadStateRefreshSettings()
    if not settings.enabled and not force:
        return {"updated": False, "reason": "disabled"}

    thread_state = await conversation_store.get_thread_state(
        source_conversation_id
    )
    covered_message_id = int((thread_state or {}).get("covered_message_id") or 0)
    all_records = await conversation_store.get_conversation_message_records(source_conversation_id)
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

    compressible_records = _compressible_records(
        all_records,
        records_after,
        recent_message_count=settings.recent_message_count,
        force=force,
    )
    if not compressible_records:
        return {
            "updated": False,
            "reason": "recent-tail-only",
            "covered_message_id": covered_message_id,
            "latest_message_id": int(all_records[-1].get("id") or 0),
        }

    new_messages_text = format_message_records(compressible_records)
    delta_chars = len(new_messages_text)
    if (
        not force
        and len(compressible_records) < settings.trigger_delta_messages
        and delta_chars < settings.trigger_delta_chars
    ):
        return {
            "updated": False,
            "reason": "below-threshold",
            "delta_messages": len(compressible_records),
            "delta_chars": delta_chars,
            "covered_message_id": covered_message_id,
            "latest_message_id": int(all_records[-1].get("id") or 0),
        }

    previous_state_text = str((thread_state or {}).get("state_text") or "").strip()
    source_text = (
        "Previous thread state:\n"
        f"{previous_state_text or '(none)'}\n\n"
        "New raw conversation messages:\n"
        f"{new_messages_text}"
    ).strip()
    compressed_output = await _run_thread_state_refresh(
        llm_client=llm_client,
        endpoint=endpoint,
        conversation_id=f"{source_conversation_id}:thread_state_refresh",
        source_text=source_text,
        reasoning_effort=reasoning_effort,
        settings=settings,
    )
    output_json = (
        compressed_output.get("json") if isinstance(compressed_output.get("json"), dict) else {}
    )
    covered_to = max(int(record.get("id") or 0) for record in compressible_records)
    state = await conversation_store.upsert_thread_state(
        source_conversation_id,
        covered_message_id=covered_to,
        state_text=str(compressed_output.get("text") or "").strip(),
        state_json=(
            output_json.get("compressed_output")
            if isinstance(output_json.get("compressed_output"), dict)
            else None
        ),
        metadata_json={
            "source_conversation_id": source_conversation_id,
            "delta_messages": len(compressible_records),
            "delta_chars": delta_chars,
            "latest_message_id": int(all_records[-1].get("id") or 0),
            "compression": output_json,
        },
    )
    return {
        "updated": True,
        "reason": "force" if force else "threshold",
        "delta_messages": len(compressible_records),
        "delta_chars": delta_chars,
        "covered_message_id": covered_to,
        "latest_message_id": int(all_records[-1].get("id") or 0),
        "state": state,
    }


def _compressible_records(
    all_records: list[dict[str, Any]],
    records_after: list[dict[str, Any]],
    *,
    recent_message_count: int,
    force: bool,
) -> list[dict[str, Any]]:
    if force:
        return records_after
    tail_count = max(0, int(recent_message_count))
    if not tail_count:
        return records_after
    if len(all_records) <= tail_count:
        return []
    max_compressible_id = int(all_records[-tail_count - 1].get("id") or 0)
    return [
        record
        for record in records_after
        if int(record.get("id") or 0) <= max_compressible_id
    ]


async def _run_thread_state_refresh(
    *,
    llm_client: WorkflowLLMClient,
    endpoint: str,
    conversation_id: str,
    source_text: str,
    reasoning_effort: str | None,
    settings: ThreadStateRefreshSettings,
) -> dict[str, Any]:
    trigger_chars = max(1, min(settings.compression_trigger_chars, len(source_text)))
    now = "1970-01-01T00:00:00+00:00"
    run = WorkflowRun(
        run_id="thread_state_refresh",
        workflow_id="thread_state_refresh",
        workflow_version="0.1.0",
        status="running",
        conversation_id=conversation_id,
        params={},
        endpoint=endpoint,
        reasoning_effort=reasoning_effort,
        retrieval_endpoint=None,
        search_provider=None,
        current_step_id="compress",
        created_at=now,
        updated_at=now,
        completed_at=None,
    )
    step = WorkflowStepSpec(
        id="compress",
        name="Refresh thread state",
        kind="compress_source",
        prompt=None,
        output_key="thread_state_text",
        compression_trigger_chars=trigger_chars,
        compression_chunk_chars=settings.compression_chunk_chars,
        compression_target_chars=settings.compression_target_chars,
        compression_max_output_chars=settings.compression_max_output_chars,
        compression_max_rounds=settings.compression_max_rounds,
        compression_goal=THREAD_STATE_REFRESH_GOAL,
    )
    spec = WorkflowSpec(
        id="thread_state_refresh",
        name="Thread State Refresh",
        description=None,
        version="0.1.0",
        params_schema={"type": "object"},
        defaults=WorkflowDefaults(reasoning_effort=reasoning_effort),
        steps=[step],
    )
    execution = await WorkflowStepExecutor(llm_client)._execute_compress_source_step(
        run=run,
        spec=spec,
        step=step,
        source=source_text,
        step_input={},
    )
    return execution.output
