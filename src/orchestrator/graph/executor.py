from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, AsyncIterator

import jsonschema

from .registry import GraphRegistry
from .store import SQLiteGraphRunStore


class GraphExecutor:
    def __init__(self, registry: GraphRegistry, store: SQLiteGraphRunStore):
        self.registry = registry
        self.store = store

    async def create_run(
        self,
        graph_id: str,
        *,
        input: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = self.registry.get(graph_id)
        run_input = input if isinstance(input, dict) else {}
        if spec.input_schema:
            jsonschema.validate(run_input, spec.input_schema)
        run_config = _merge_config(spec.defaults, config)
        run_id = f"gr_{uuid.uuid4().hex[:12]}"
        thread_id = _thread_id(run_config) or run_id
        _set_thread_id(run_config, thread_id)
        await self.store.create_run(
            run_id=run_id,
            graph_id=spec.id,
            thread_id=thread_id,
            input_json=run_input,
            config_json=run_config,
        )
        snapshot = await self.store.snapshot(run_id)
        if snapshot is None:
            raise RuntimeError(f"created graph run not found: {run_id}")
        return {
            "run_id": run_id,
            "graph_id": spec.id,
            "status": snapshot["run"]["status"],
            "thread_id": thread_id,
            "snapshot": snapshot,
        }

    async def run(self, run_id: str) -> dict[str, Any]:
        run = await self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown graph run: {run_id}")
        if run["status"] in {"completed", "failed", "cancelled"}:
            snapshot = await self.store.snapshot(run_id)
            if snapshot is None:
                raise KeyError(f"unknown graph run: {run_id}")
            return snapshot

        spec = self.registry.get(run["graph_id"])
        await self.store.mark_status(run_id, "running")
        try:
            output = await _invoke_graph(
                spec.graph,
                run["input"],
                config=run["config"],
            )
        except Exception as exc:
            await self.store.mark_status(
                run_id,
                "failed",
                error=str(exc),
                completed=True,
            )
            failed = await self.store.snapshot(run_id)
            if failed is None:
                raise KeyError(f"unknown graph run: {run_id}")
            return failed

        await self.store.mark_status(
            run_id,
            "completed",
            output_json=_coerce_output(output),
            error=None,
            completed=True,
        )
        snapshot = await self.store.snapshot(run_id)
        if snapshot is None:
            raise KeyError(f"unknown graph run: {run_id}")
        return snapshot

    async def stream(
        self,
        run_id: str,
        *,
        stream_mode: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        run = await self.store.get_run(run_id)
        if run is None:
            yield {
                "type": "error",
                "run_id": run_id,
                "error": f"unknown graph run: {run_id}",
            }
            return
        if run["status"] in {"completed", "failed", "cancelled"}:
            snapshot = await self.store.snapshot(run_id)
            if snapshot is not None:
                yield {"type": "snapshot", "run_id": run_id, "snapshot": snapshot}
            return

        spec = self.registry.get(run["graph_id"])
        yield {
            "type": "graph_run_started",
            "run_id": run_id,
            "graph_id": run["graph_id"],
            "thread_id": run["thread_id"],
        }
        await self.store.mark_status(run_id, "running")
        snapshot = await self.store.snapshot(run_id)
        if snapshot is not None:
            yield {"type": "snapshot", "run_id": run_id, "snapshot": snapshot}

        last_event: dict[str, Any] | None = None
        try:
            if hasattr(spec.graph, "astream"):
                async for raw_event in spec.graph.astream(
                    run["input"],
                    config=run["config"],
                    stream_mode=stream_mode or "updates",
                ):
                    event = _graph_stream_event(
                        raw_event,
                        run_id=run_id,
                        graph_id=run["graph_id"],
                    )
                    last_event = event
                    await self.store.append_event(
                        run_id,
                        event_type=str(event.get("type") or "graph_event"),
                        node_name=_node_name(event),
                        event_json=event,
                    )
                    yield event
            elif hasattr(spec.graph, "stream"):
                for raw_event in await asyncio.to_thread(
                    lambda: list(
                        spec.graph.stream(
                            run["input"],
                            config=run["config"],
                            stream_mode=stream_mode or "updates",
                        )
                    )
                ):
                    event = _graph_stream_event(
                        raw_event,
                        run_id=run_id,
                        graph_id=run["graph_id"],
                    )
                    last_event = event
                    await self.store.append_event(
                        run_id,
                        event_type=str(event.get("type") or "graph_event"),
                        node_name=_node_name(event),
                        event_json=event,
                    )
                    yield event
            else:
                snapshot = await self.run(run_id)
                yield {
                    "type": "graph_run_completed",
                    "run_id": run_id,
                    "graph_id": run["graph_id"],
                    "status": snapshot["run"]["status"],
                    "snapshot": snapshot,
                }
                return
        except Exception as exc:
            await self.store.mark_status(run_id, "failed", error=str(exc), completed=True)
            failed = await self.store.snapshot(run_id)
            yield {
                "type": "error",
                "run_id": run_id,
                "graph_id": run["graph_id"],
                "error": str(exc),
                "snapshot": failed,
            }
            return

        await self.store.mark_status(
            run_id,
            "completed",
            output_json={"last_event": last_event} if last_event is not None else {},
            error=None,
            completed=True,
        )
        completed = await self.store.snapshot(run_id)
        yield {
            "type": "graph_run_completed",
            "run_id": run_id,
            "graph_id": run["graph_id"],
            "status": "completed",
            "snapshot": completed,
        }

    async def resume(
        self,
        run_id: str,
        *,
        resume: Any = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = await self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown graph run: {run_id}")
        spec = self.registry.get(run["graph_id"])
        run_config = _merge_config(run["config"], config)
        try:
            from langgraph.types import Command

            command: Any = Command(resume=resume)
        except Exception:
            command = {"resume": resume}
        await self.store.mark_status(run_id, "running")
        try:
            output = await _invoke_graph(spec.graph, command, config=run_config)
        except Exception as exc:
            await self.store.mark_status(run_id, "failed", error=str(exc), completed=True)
            failed = await self.store.snapshot(run_id)
            if failed is None:
                raise KeyError(f"unknown graph run: {run_id}")
            return failed
        await self.store.mark_status(
            run_id,
            "completed",
            output_json=_coerce_output(output),
            error=None,
            completed=True,
        )
        snapshot = await self.store.snapshot(run_id)
        if snapshot is None:
            raise KeyError(f"unknown graph run: {run_id}")
        return snapshot


async def _invoke_graph(graph: Any, input_value: Any, *, config: dict[str, Any]) -> Any:
    if hasattr(graph, "ainvoke"):
        return await graph.ainvoke(input_value, config=config)
    if hasattr(graph, "invoke"):
        result = graph.invoke(input_value, config=config)
        if inspect.isawaitable(result):
            return await result
        return result
    raise ValueError("graph object does not expose invoke/ainvoke")


def _merge_config(
    defaults: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(defaults or {})
    override = override if isinstance(override, dict) else {}
    for key, value in override.items():
        if (
            key == "configurable"
            and isinstance(value, dict)
            and isinstance(merged.get("configurable"), dict)
        ):
            configurable = dict(merged["configurable"])
            configurable.update(value)
            merged["configurable"] = configurable
        else:
            merged[key] = value
    return merged


def _thread_id(config: dict[str, Any]) -> str | None:
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    thread_id = str(configurable.get("thread_id") or "").strip()
    return thread_id or None


def _set_thread_id(config: dict[str, Any], thread_id: str) -> None:
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        configurable = {}
    configurable["thread_id"] = thread_id
    config["configurable"] = configurable


def _coerce_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    return {"value": output}


def _graph_stream_event(
    raw_event: Any,
    *,
    run_id: str,
    graph_id: str,
) -> dict[str, Any]:
    if isinstance(raw_event, tuple) and len(raw_event) == 2:
        mode, payload = raw_event
        return {
            "type": "graph_event",
            "run_id": run_id,
            "graph_id": graph_id,
            "mode": str(mode),
            "payload": payload,
        }
    return {
        "type": "graph_event",
        "run_id": run_id,
        "graph_id": graph_id,
        "payload": raw_event,
    }


def _node_name(event: dict[str, Any]) -> str | None:
    payload = event.get("payload")
    if isinstance(payload, dict) and len(payload) == 1:
        return str(next(iter(payload)))
    return None

