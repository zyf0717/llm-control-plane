from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from .models import WorkflowRun, WorkflowSpec, WorkflowStepSpec
from .registry import WorkflowRegistry
from .search_results import reranking_path
from .step_executor import (
    WorkflowLLMClient,
    WorkflowSearchClient,
    WorkflowStepExecution,
    WorkflowStepExecutor,
)
from .store import SQLiteWorkflowStore
from .template import render_template

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    def __init__(
        self,
        registry: WorkflowRegistry,
        store: SQLiteWorkflowStore,
        llm_client: WorkflowLLMClient,
        search_client: WorkflowSearchClient | None = None,
    ):
        self.registry = registry
        self.store = store
        self.llm_client = llm_client
        self.search_client = search_client
        self.step_executor = WorkflowStepExecutor(llm_client, search_client)

    async def create_run(
        self,
        workflow_id: str,
        *,
        params: dict[str, Any] | None = None,
        convo_id: str | None = None,
        endpoint: str | None = None,
        reasoning_effort: str | None = None,
        rag_endpoint: str | None = None,
        search_provider: str | None = None,
    ) -> dict[str, Any]:
        spec = self.registry.get(workflow_id)
        cleaned_params = params if isinstance(params, dict) else {}
        self._validate_required_params(spec, cleaned_params)
        run_endpoint = str(endpoint or "").strip()
        if not run_endpoint:
            raise ValueError("workflow run endpoint is required")
        if run_endpoint.lower() == "smart":
            raise ValueError("workflow run endpoint must be a concrete endpoint")

        run_id = f"wf_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        run = WorkflowRun(
            run_id=run_id,
            workflow_id=spec.id,
            workflow_version=spec.version,
            status="pending",
            convo_id=str(convo_id or "").strip() or f"{run_id}_convo",
            params=cleaned_params,
            endpoint=run_endpoint,
            reasoning_effort=reasoning_effort or spec.defaults.reasoning_effort,
            rag_endpoint=_optional_str(rag_endpoint),
            search_provider=_optional_str(search_provider),
            current_step_id=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )
        await self.store.create_run(run, [step.id for step in spec.steps])
        logger.info(
            "workflow run created: run_id=%s workflow_id=%s version=%s steps=%d endpoint=%s",
            run.run_id,
            run.workflow_id,
            run.workflow_version,
            len(spec.steps),
            run.endpoint,
        )
        snapshot = await self.store.snapshot(run_id)
        if snapshot is None:
            raise RuntimeError(f"created workflow run not found: {run_id}")
        return snapshot

    async def advance(self, run_id: str) -> dict[str, Any]:
        run = await self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown workflow run: {run_id}")
        if run.status in {"completed", "failed", "cancelled"}:
            snapshot = await self.store.snapshot(run_id)
            if snapshot is None:
                raise KeyError(f"unknown workflow run: {run_id}")
            return snapshot

        spec = self.registry.get(run.workflow_id)
        snapshot = await self.store.snapshot(run_id)
        if snapshot is None:
            raise KeyError(f"unknown workflow run: {run_id}")

        next_step = self._next_runnable_step(spec, snapshot)
        if next_step is None:
            if self._all_steps_completed(snapshot):
                await self.store.mark_run_status(
                    run_id, "completed", current_step_id=None, completed=True
                )
                logger.info(
                    "workflow run completed: run_id=%s workflow_id=%s",
                    run_id,
                    run.workflow_id,
                )
                completed = await self.store.snapshot(run_id)
                if completed is None:
                    raise KeyError(f"unknown workflow run: {run_id}")
                return completed
            return snapshot

        step_input = self._build_step_input(spec, snapshot, next_step)
        await self.store.mark_step_running(run_id, next_step.id, step_input)
        self._log_step_started(run, next_step)
        try:
            execution = await self.step_executor.execute(
                run, spec, next_step, step_input
            )
        except Exception as exc:
            await self.store.mark_step_failed(run_id, next_step.id, str(exc))
            logger.exception(
                "workflow step failed: run_id=%s workflow_id=%s step_id=%s kind=%s",
                run_id,
                run.workflow_id,
                next_step.id,
                next_step.kind,
            )
            failed = await self.store.snapshot(run_id)
            if failed is None:
                raise KeyError(f"unknown workflow run: {run_id}")
            return failed

        await self.store.mark_step_completed(run_id, next_step.id, execution.output)
        self._log_step_completed(run, next_step, execution)
        if execution.artifact_text:
            await self.store.create_artifact(
                artifact_id=f"wfa_{uuid.uuid4().hex[:12]}",
                run_id=run_id,
                step_id=next_step.id,
                artifact_type="text",
                name=next_step.output_key or next_step.id,
                content_json=execution.output,
                content_text=execution.artifact_text,
            )
        advanced = await self.store.snapshot(run_id)
        if advanced is None:
            raise KeyError(f"unknown workflow run: {run_id}")
        if self._next_runnable_step(spec, advanced) is None:
            if self._all_steps_completed(advanced):
                await self.store.mark_run_status(
                    run_id, "completed", current_step_id=None, completed=True
                )
                logger.info(
                    "workflow run completed: run_id=%s workflow_id=%s",
                    run_id,
                    run.workflow_id,
                )
                advanced = await self.store.snapshot(run_id)
                if advanced is None:
                    raise KeyError(f"unknown workflow run: {run_id}")
        return advanced

    async def run_to_completion(
        self, run_id: str, *, max_steps: int | None = None
    ) -> dict[str, Any]:
        budget = max(1, int(max_steps or 100))
        snapshot: dict[str, Any] | None = None
        for _ in range(budget):
            snapshot = await self.advance(run_id)
            status = str(snapshot.get("run", {}).get("status") or "")
            if status in {"completed", "failed", "cancelled"}:
                return snapshot
        if snapshot is None:
            snapshot = await self.store.snapshot(run_id)
        if snapshot is None:
            raise KeyError(f"unknown workflow run: {run_id}")
        return snapshot

    async def run_to_completion_stream(
        self,
        run_id: str,
        *,
        max_steps: int | None = None,
        stream_llm: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        budget = max(1, int(max_steps or 100))
        initial = await self.store.snapshot(run_id)
        if initial is None:
            yield {
                "type": "error",
                "run_id": run_id,
                "error": f"unknown workflow run: {run_id}",
            }
            return

        run = initial.get("run") if isinstance(initial.get("run"), dict) else {}
        yield {
            "type": "run_started",
            "run_id": run_id,
            "workflow_id": str(run.get("workflow_id") or ""),
        }
        yield {"type": "snapshot", "run_id": run_id, "snapshot": initial}

        snapshot = initial
        for _ in range(budget):
            async for event in self._advance_stream(run_id, stream_llm=stream_llm):
                if event.get("type") == "snapshot" and isinstance(
                    event.get("snapshot"), dict
                ):
                    snapshot = event["snapshot"]
                yield event
            status = str(snapshot.get("run", {}).get("status") or "")
            if status in {"completed", "failed", "cancelled"}:
                yield {
                    "type": "run_completed",
                    "run_id": run_id,
                    "workflow_id": str(
                        snapshot.get("run", {}).get("workflow_id") or ""
                    ),
                    "status": status,
                    "snapshot": snapshot,
                }
                return

        yield {
            "type": "run_completed",
            "run_id": run_id,
            "workflow_id": str(snapshot.get("run", {}).get("workflow_id") or ""),
            "status": str(snapshot.get("run", {}).get("status") or ""),
            "snapshot": snapshot,
        }

    async def _advance_stream(
        self, run_id: str, *, stream_llm: bool
    ) -> AsyncIterator[dict[str, Any]]:
        run = await self.store.get_run(run_id)
        if run is None:
            yield {
                "type": "error",
                "run_id": run_id,
                "error": f"unknown workflow run: {run_id}",
            }
            return
        if run.status in {"completed", "failed", "cancelled"}:
            snapshot = await self.store.snapshot(run_id)
            if snapshot is not None:
                yield {"type": "snapshot", "run_id": run_id, "snapshot": snapshot}
            return

        spec = self.registry.get(run.workflow_id)
        snapshot = await self.store.snapshot(run_id)
        if snapshot is None:
            yield {
                "type": "error",
                "run_id": run_id,
                "error": f"unknown workflow run: {run_id}",
            }
            return

        next_step = self._next_runnable_step(spec, snapshot)
        if next_step is None:
            if self._all_steps_completed(snapshot):
                await self.store.mark_run_status(
                    run_id, "completed", current_step_id=None, completed=True
                )
                logger.info(
                    "workflow run completed: run_id=%s workflow_id=%s",
                    run_id,
                    run.workflow_id,
                )
                completed = await self.store.snapshot(run_id)
                if completed is not None:
                    yield {"type": "snapshot", "run_id": run_id, "snapshot": completed}
            else:
                yield {"type": "snapshot", "run_id": run_id, "snapshot": snapshot}
            return

        step_input = self._build_step_input(spec, snapshot, next_step)
        await self.store.mark_step_running(run_id, next_step.id, step_input)
        self._log_step_started(run, next_step)
        yield self._step_event(run, next_step, "step_started")
        running_snapshot = await self.store.snapshot(run_id)
        if running_snapshot is not None:
            yield {"type": "snapshot", "run_id": run_id, "snapshot": running_snapshot}

        execution: WorkflowStepExecution | None = None
        try:
            async for event in self.step_executor.stream(
                run, spec, next_step, step_input, stream_llm=stream_llm
            ):
                if event.get("type") == "_execution_result":
                    execution = event["execution"]
                else:
                    yield event
            if execution is None:
                raise RuntimeError("workflow step produced no execution result")
        except Exception as exc:
            await self.store.mark_step_failed(run_id, next_step.id, str(exc))
            logger.exception(
                "workflow step failed: run_id=%s workflow_id=%s step_id=%s kind=%s",
                run_id,
                run.workflow_id,
                next_step.id,
                next_step.kind,
            )
            failed = await self.store.snapshot(run_id)
            yield self._step_event(
                run,
                next_step,
                "error",
                error=str(exc),
                snapshot=failed,
            )
            if failed is not None:
                yield {"type": "snapshot", "run_id": run_id, "snapshot": failed}
            return

        await self.store.mark_step_completed(run_id, next_step.id, execution.output)
        self._log_step_completed(run, next_step, execution)
        if execution.artifact_text:
            await self.store.create_artifact(
                artifact_id=f"wfa_{uuid.uuid4().hex[:12]}",
                run_id=run_id,
                step_id=next_step.id,
                artifact_type="text",
                name=next_step.output_key or next_step.id,
                content_json=execution.output,
                content_text=execution.artifact_text,
            )
        advanced = await self.store.snapshot(run_id)
        if advanced is not None and self._next_runnable_step(spec, advanced) is None:
            if self._all_steps_completed(advanced):
                await self.store.mark_run_status(
                    run_id, "completed", current_step_id=None, completed=True
                )
                logger.info(
                    "workflow run completed: run_id=%s workflow_id=%s",
                    run_id,
                    run.workflow_id,
                )
                advanced = await self.store.snapshot(run_id)

        yield self._step_event(
            run,
            next_step,
            "step_completed",
            content=self._visible_step_content(next_step, execution),
            streamed=execution.streamed,
            snapshot=advanced,
        )
        if advanced is not None:
            yield {"type": "snapshot", "run_id": run_id, "snapshot": advanced}

    async def retry_step(self, run_id: str, step_id: str) -> dict[str, Any]:
        run = await self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown workflow run: {run_id}")
        spec = self.registry.get(run.workflow_id)
        await self.store.retry_step(run_id, step_id, self._retry_step_ids(spec, step_id))
        snapshot = await self.store.snapshot(run_id)
        if snapshot is None:
            raise KeyError(f"unknown workflow run: {run_id}")
        return snapshot

    @staticmethod
    def _validate_required_params(spec: WorkflowSpec, params: dict[str, Any]) -> None:
        required = spec.params_schema.get("required", [])
        if not isinstance(required, list):
            return
        missing = [
            str(field)
            for field in required
            if not str(field) in params or params.get(str(field)) in {None, ""}
        ]
        if missing:
            raise ValueError(f"missing required workflow params: {', '.join(missing)}")

    @staticmethod
    def _next_runnable_step(
        spec: WorkflowSpec, snapshot: dict[str, Any]
    ) -> WorkflowStepSpec | None:
        steps_by_id = {
            str(step.get("step_id")): step
            for step in snapshot.get("steps", [])
            if isinstance(step, dict)
        }
        for step in spec.steps:
            state = steps_by_id.get(step.id)
            if not state or state.get("status") != "pending":
                continue
            dependencies = step.depends_on or []
            if all(
                steps_by_id.get(dependency, {}).get("status") == "completed"
                for dependency in dependencies
            ):
                return step
        return None

    @staticmethod
    def _all_steps_completed(snapshot: dict[str, Any]) -> bool:
        steps = [step for step in snapshot.get("steps", []) if isinstance(step, dict)]
        return bool(steps) and all(step.get("status") == "completed" for step in steps)

    @staticmethod
    def _retry_step_ids(spec: WorkflowSpec, step_id: str) -> list[str]:
        step_ids = [step.id for step in spec.steps]
        if step_id not in step_ids:
            return [step_id]

        affected = {step_id}
        changed = True
        while changed:
            changed = False
            for step in spec.steps:
                if step.id in affected:
                    continue
                if any(dependency in affected for dependency in step.depends_on or []):
                    affected.add(step.id)
                    changed = True
        return [candidate for candidate in step_ids if candidate in affected]

    @staticmethod
    def _build_step_input(
        spec: WorkflowSpec, snapshot: dict[str, Any], step: WorkflowStepSpec
    ) -> dict[str, Any]:
        run = snapshot.get("run") if isinstance(snapshot.get("run"), dict) else {}
        outputs = WorkflowExecutor._previous_outputs(spec, snapshot)
        return {
            "params": dict(run.get("params") or {}),
            "previous_outputs": outputs,
            "outputs": outputs,
            "artifacts": list(snapshot.get("artifacts") or []),
            "current_step": step.to_dict(),
        }

    @staticmethod
    def _previous_outputs(
        spec: WorkflowSpec, snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        step_specs = {step.id: step for step in spec.steps}
        outputs: dict[str, Any] = {}
        for step_run in snapshot.get("steps", []):
            if not isinstance(step_run, dict) or step_run.get("status") != "completed":
                continue
            step_id = str(step_run.get("step_id") or "")
            output = step_run.get("output_json")
            if not isinstance(output, dict):
                continue
            key = step_specs.get(step_id).output_key if step_specs.get(step_id) else None
            outputs[key or step_id] = output
        return outputs


    @staticmethod
    def _log_step_started(run: WorkflowRun, step: WorkflowStepSpec) -> None:
        logger.info(
            "workflow step started: run_id=%s workflow_id=%s step_id=%s kind=%s "
            "depends_on=%d",
            run.run_id,
            run.workflow_id,
            step.id,
            step.kind,
            len(step.depends_on or []),
        )

    @staticmethod
    def _log_step_completed(
        run: WorkflowRun,
        step: WorkflowStepSpec,
        execution: WorkflowStepExecution,
    ) -> None:
        output = execution.output if isinstance(execution.output, dict) else {}
        output_json = output.get("json")
        result_count = None
        reranker_path = None
        endpoint = None

        if isinstance(output_json, dict):
            raw_results = output_json.get("results")
            if isinstance(raw_results, list):
                result_count = len(raw_results)
            reranker_path = reranking_path(output_json)

        metadata = output.get("metadata")
        if isinstance(metadata, dict):
            endpoint = metadata.get("endpoint")

        logger.info(
            "workflow step completed: run_id=%s workflow_id=%s step_id=%s kind=%s "
            "streamed=%s artifact=%s output_json=%s results=%s reranker_path=%s "
            "endpoint=%s",
            run.run_id,
            run.workflow_id,
            step.id,
            step.kind,
            execution.streamed,
            bool(execution.artifact_text),
            output_json is not None,
            result_count,
            reranker_path,
            endpoint,
        )

    @staticmethod
    def _visible_step_content(
        step: WorkflowStepSpec, execution: WorkflowStepExecution
    ) -> str:
        if step.chat_visibility == "hidden":
            return ""
        return str(execution.output.get("text") or "").strip()

    @staticmethod
    def _step_event(
        run: WorkflowRun,
        step: WorkflowStepSpec,
        event_type: str,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "type": event_type,
            "run_id": run.run_id,
            "workflow_id": run.workflow_id,
            "step_id": step.id,
            "step_name": step.name,
            "chat_visibility": step.chat_visibility,
            **extra,
        }



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
