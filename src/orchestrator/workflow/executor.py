from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Protocol

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

    async def stream_complete(
        self,
        *,
        endpoint: str,
        prompt: str,
        convo_id: str,
        reasoning_effort: str | None = None,
        rag_endpoint: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        ...


class WorkflowSearchClient(Protocol):
    async def search(
        self,
        *,
        query: str,
        provider: str | None = None,
        count: int = 5,
        use_query_refiner: bool = True,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class WorkflowStepExecution:
    output: dict[str, Any]
    artifact_text: str | None = None
    streamed: bool = False


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
                completed = await self.store.snapshot(run_id)
                if completed is None:
                    raise KeyError(f"unknown workflow run: {run_id}")
                return completed
            return snapshot

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
            if self._all_steps_completed(advanced):
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
                completed = await self.store.snapshot(run_id)
                if completed is not None:
                    yield {"type": "snapshot", "run_id": run_id, "snapshot": completed}
            else:
                yield {"type": "snapshot", "run_id": run_id, "snapshot": snapshot}
            return

        step_input = self._build_step_input(spec, snapshot, next_step)
        await self.store.mark_step_running(run_id, next_step.id, step_input)
        yield self._step_event(run, next_step, "step_started")
        running_snapshot = await self.store.snapshot(run_id)
        if running_snapshot is not None:
            yield {"type": "snapshot", "run_id": run_id, "snapshot": running_snapshot}

        execution: WorkflowStepExecution | None = None
        try:
            async for event in self._execute_step_stream(
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
            queries = _extract_search_queries(prompt)
            provider = (
                run.search_provider
                or step.search_provider
                or spec.defaults.search_provider
            )
            use_query_refiner = len(queries) == 1 and queries[0] == prompt.strip()
            if not any(query.strip() for query in queries):
                raise ValueError(
                    "workflow search step produced no query; check upstream JSON output"
                )
            results = await asyncio.gather(
                *(
                    self.search_client.search(
                        query=query,
                        provider=provider,
                        use_query_refiner=use_query_refiner,
                    )
                    for query in queries
                )
            )
            result = _merge_search_results(
                queries,
                results,
                use_query_refiner=use_query_refiner,
            )
            return WorkflowStepExecution(
                output={
                    "text": json.dumps(result, indent=2),
                    "json": result,
                    "metadata": {"kind": "search"},
                },
                artifact_text=json.dumps(result, indent=2),
            )

        endpoint = run.endpoint
        if not endpoint:
            raise ValueError("workflow step endpoint is required")
        if endpoint.lower() == "smart":
            raise ValueError("workflow step endpoint must be a concrete endpoint")
        reasoning_effort = (
            step.reasoning_effort or run.reasoning_effort or spec.defaults.reasoning_effort
        )
        result = await self.llm_client.complete(
            endpoint=endpoint,
            prompt=prompt,
            convo_id=run.convo_id,
            reasoning_effort=reasoning_effort,
            rag_endpoint=run.rag_endpoint or step.rag_endpoint or spec.defaults.rag_endpoint,
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

    async def _execute_step_stream(
        self,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        step_input: dict[str, Any],
        *,
        stream_llm: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        if step.kind != "llm" or not self._should_stream_step(step, stream_llm):
            execution = await self._execute_step(run, spec, step, step_input)
            yield {"type": "_execution_result", "execution": execution}
            return

        prompt = render_template(step.prompt or "", step_input)
        endpoint = run.endpoint
        if not endpoint:
            raise ValueError("workflow step endpoint is required")
        if endpoint.lower() == "smart":
            raise ValueError("workflow step endpoint must be a concrete endpoint")
        reasoning_effort = (
            step.reasoning_effort or run.reasoning_effort or spec.defaults.reasoning_effort
        )

        stream_complete = getattr(self.llm_client, "stream_complete", None)
        if stream_complete is None:
            execution = await self._execute_step(run, spec, step, step_input)
            yield {"type": "_execution_result", "execution": execution}
            return

        text_parts: list[str] = []
        metadata: dict[str, Any] = {}
        async for chunk in stream_complete(
            endpoint=endpoint,
            prompt=prompt,
            convo_id=run.convo_id,
            reasoning_effort=reasoning_effort,
            rag_endpoint=run.rag_endpoint or step.rag_endpoint or spec.defaults.rag_endpoint,
            max_tokens=step.max_tokens or spec.defaults.max_tokens,
        ):
            if not isinstance(chunk, dict):
                continue
            channel = str(chunk.get("channel") or "").strip()
            content = str(chunk.get("content") or "")
            if channel == "content" and content:
                text_parts.append(content)
            if isinstance(chunk.get("metadata"), dict):
                metadata.update(chunk["metadata"])
            if channel in {"content", "reasoning"} and content:
                yield self._step_event(
                    run,
                    step,
                    "step_delta",
                    channel=channel,
                    content=content,
                )

        text = "".join(text_parts)
        parsed = _parse_json_text(text)
        output = {
            "text": text,
            "json": parsed,
            "metadata": metadata,
        }
        yield {
            "type": "_execution_result",
            "execution": WorkflowStepExecution(
                output=output,
                artifact_text=text or None,
                streamed=True,
            ),
        }

    @staticmethod
    def _should_stream_step(step: WorkflowStepSpec, stream_llm: bool) -> bool:
        return (
            bool(stream_llm)
            and step.chat_visibility != "hidden"
            and step.chat_stream is not False
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


_TEMPLATE_PATTERN = re.compile(
    r"{{\s*(?:(json)\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*}}"
)


def render_template(template: str, context: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value: Any = context
        for part in match.group(2).split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return ""
        if match.group(1) == "json":
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        return "" if value is None else str(value)

    return _TEMPLATE_PATTERN.sub(replace, template)


def _parse_json_text(text: str) -> Any:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    fenced = _strip_json_code_fence(stripped)
    if fenced != stripped:
        stripped = fenced
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _strip_json_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _extract_search_queries(prompt: str) -> list[str]:
    stripped = str(prompt or "").strip()
    parsed = _parse_json_text(stripped)
    raw_queries: list[Any] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("queries"), list):
            raw_queries.extend(parsed["queries"])
        elif parsed.get("query") is not None:
            raw_queries.append(parsed.get("query"))
    elif isinstance(parsed, list):
        raw_queries.extend(parsed)
    elif isinstance(parsed, str):
        raw_queries.append(parsed)
    else:
        raw_queries.append(stripped)

    queries: list[str] = []
    seen: set[str] = set()
    for raw_query in raw_queries:
        query = str(raw_query or "").strip()
        key = " ".join(query.lower().split())
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries or [stripped]


def _merge_search_results(
    queries: list[str], results: list[dict[str, Any]], *, use_query_refiner: bool
) -> dict[str, Any]:
    if len(results) == 1:
        merged = dict(results[0])
        merged.setdefault("query", queries[0] if queries else "")
        merged["workflow_search"] = {
            "planned_by_workflow": not use_query_refiner,
            "queries": list(queries),
        }
        return merged

    seen_urls: set[str] = set()
    merged_results: list[Any] = []
    warnings: list[str] = []
    per_query: list[dict[str, Any]] = []
    degraded = False
    providers: set[str] = set()

    for query, result in zip(queries, results):
        response = dict(result)
        response["query"] = str(response.get("query") or query)
        per_query.append(response)
        degraded = degraded or bool(response.get("degraded"))
        provider = str(response.get("provider") or "").strip()
        if provider and provider != "none":
            providers.add(provider)
        if isinstance(response.get("warnings"), list):
            warnings.extend(str(item) for item in response["warnings"])
        for item in response.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            key = url or json.dumps(item, sort_keys=True)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            merged_results.append(item)

    return {
        "query": queries[0] if queries else "",
        "queries": list(queries),
        "provider": next(iter(providers)) if len(providers) == 1 else "fanout",
        "results": merged_results,
        "warnings": warnings,
        "degraded": degraded or any(not (result.get("results") or []) for result in results),
        "per_query": per_query,
        "workflow_search": {
            "planned_by_workflow": True,
            "queries": list(queries),
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
