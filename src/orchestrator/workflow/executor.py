from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from .models import WorkflowRun, WorkflowSpec, WorkflowStepSpec
from .registry import WorkflowRegistry
from .store import SQLiteWorkflowStore


class WorkflowLLMClient(Protocol):
    async def complete(
        self,
        *,
        endpoint: str,
        prompt: str,
        convo_id: str,
        reasoning_effort: str | None = None,
        rag_endpoint: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        ...


class WorkflowSearchClient(Protocol):
    async def search(
        self,
        *,
        query: str,
        provider: str | None = None,
        count: int = 5,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class WorkflowStepExecution:
    output: dict[str, Any]
    artifact_text: str | None = None


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

    async def create_run(
        self,
        workflow_id: str,
        *,
        params: dict[str, Any] | None = None,
        convo_id: str | None = None,
        endpoint: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        spec = self.registry.get(workflow_id)
        cleaned_params = params if isinstance(params, dict) else {}
        self._validate_required_params(spec, cleaned_params)

        run_id = f"wf_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        run = WorkflowRun(
            run_id=run_id,
            workflow_id=spec.id,
            workflow_version=spec.version,
            status="pending",
            convo_id=str(convo_id or "").strip() or f"{run_id}_convo",
            params=cleaned_params,
            endpoint=endpoint or spec.defaults.endpoint,
            reasoning_effort=reasoning_effort or spec.defaults.reasoning_effort,
            current_step_id=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )
        await self.store.create_run(run, [step.id for step in spec.steps])
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
            await self.store.mark_run_status(
                run_id, "completed", current_step_id=None, completed=True
            )
            completed = await self.store.snapshot(run_id)
            if completed is None:
                raise KeyError(f"unknown workflow run: {run_id}")
            return completed

        step_input = self._build_step_input(spec, snapshot, next_step)
        await self.store.mark_step_running(run_id, next_step.id, step_input)
        try:
            execution = await self._execute_step(run, spec, next_step, step_input)
        except Exception as exc:
            await self.store.mark_step_failed(run_id, next_step.id, str(exc))
            failed = await self.store.snapshot(run_id)
            if failed is None:
                raise KeyError(f"unknown workflow run: {run_id}")
            return failed

        await self.store.mark_step_completed(run_id, next_step.id, execution.output)
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
            await self.store.mark_run_status(
                run_id, "completed", current_step_id=None, completed=True
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

    async def retry_step(self, run_id: str, step_id: str) -> dict[str, Any]:
        await self.store.retry_step(run_id, step_id)
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

    async def _execute_step(
        self,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        step_input: dict[str, Any],
    ) -> WorkflowStepExecution:
        if step.kind == "manual":
            return WorkflowStepExecution(
                output={"text": "manual step acknowledged", "json": None, "metadata": {}}
            )

        prompt = render_template(step.prompt or "", step_input)
        if step.kind == "search":
            if self.search_client is None:
                return WorkflowStepExecution(
                    output={
                        "text": prompt,
                        "json": None,
                        "metadata": {"kind": "search", "skipped": "no-search-client"},
                    }
                )
            result = await self.search_client.search(
                query=prompt,
                provider=step.search_provider or spec.defaults.search_provider,
            )
            return WorkflowStepExecution(
                output={"text": json.dumps(result, indent=2), "json": result, "metadata": {"kind": "search"}},
                artifact_text=json.dumps(result, indent=2),
            )

        endpoint = step.endpoint or run.endpoint or spec.defaults.endpoint or "smart"
        reasoning_effort = (
            step.reasoning_effort or run.reasoning_effort or spec.defaults.reasoning_effort
        )
        result = await self.llm_client.complete(
            endpoint=endpoint,
            prompt=prompt,
            convo_id=run.convo_id,
            reasoning_effort=reasoning_effort,
            rag_endpoint=step.rag_endpoint or spec.defaults.rag_endpoint,
            max_tokens=step.max_tokens or spec.defaults.max_tokens,
        )
        text = str(result.get("text") or "")
        parsed = _parse_json_text(text)
        output = {
            "text": text,
            "json": parsed,
            "metadata": dict(result.get("metadata") or {}),
        }
        return WorkflowStepExecution(output=output, artifact_text=text or None)


_TEMPLATE_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*}}")


def render_template(template: str, context: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value: Any = context
        for part in match.group(1).split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        return "" if value is None else str(value)

    return _TEMPLATE_PATTERN.sub(replace, template)


def _parse_json_text(text: str) -> Any:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
