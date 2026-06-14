from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

from .compaction import (
    build_compaction_prompt,
    build_compaction_repair_prompt,
    check_compaction_input_payload,
    check_compaction_output_payload,
    chunk_compaction_units,
    chunk_text_by_budget,
    compaction_needs_reduce,
    compaction_output_payload,
    compaction_source_excerpt,
    compaction_units,
    estimate_tokens,
    extract_compaction_anchors,
    parse_compaction_input,
    top_compaction_evidence,
    validate_compaction_output_text,
)
from .models import WorkflowRun, WorkflowSpec, WorkflowStepSpec
from .search_results import (
    extract_search_queries,
    merge_reranked_search_result,
    merge_search_results,
    reranking_metadata,
    reranking_path,
    search_output_query,
    select_rerank_source,
)
from .structured_output import (
    build_repair_prompt,
    build_retry_prompt,
    build_structured_output_instructions,
    contract_requires_structure,
    parse_structured_output,
    validate_structured_output,
)
from .template import parse_json_text, render_template


logger = logging.getLogger(__name__)


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

    async def rerank_results(
        self,
        *,
        query: str,
        results: list[dict[str, Any]],
        context: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class WorkflowStepExecution:
    output: dict[str, Any]
    artifact_text: str | None = None
    streamed: bool = False


class WorkflowStepExecutor:
    def __init__(
        self,
        llm_client: WorkflowLLMClient,
        search_client: WorkflowSearchClient | None = None,
    ):
        self.llm_client = llm_client
        self.search_client = search_client

    async def execute(
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
        if step.kind == "compact_context":
            return await self._execute_compact_context_step(
                run=run,
                spec=spec,
                step=step,
                source=prompt,
                step_input=step_input,
            )

        if step.kind == "search":
            return await self._execute_search_step(run, spec, step, prompt)

        if step.kind == "rerank":
            return await self._execute_rerank_step(run, spec, step, step_input, prompt)

        endpoint = run.endpoint
        if not endpoint:
            raise ValueError("workflow step endpoint is required")
        if endpoint.lower() == "smart":
            raise ValueError("workflow step endpoint must be a concrete endpoint")
        reasoning_effort = (
            step.reasoning_effort or run.reasoning_effort or spec.defaults.reasoning_effort
        )
        return await self._execute_llm_step(
            run=run,
            spec=spec,
            step=step,
            prompt=prompt,
            reasoning_effort=reasoning_effort,
        )

    async def stream(
        self,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        step_input: dict[str, Any],
        *,
        stream_llm: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        if step.kind != "llm" or not _should_stream_step(step, stream_llm):
            execution = await self.execute(run, spec, step, step_input)
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
            execution = await self.execute(run, spec, step, step_input)
            yield {"type": "_execution_result", "execution": execution}
            return

        text_parts: list[str] = []
        metadata: dict[str, Any] = {}
        async for chunk in stream_complete(
            endpoint=endpoint,
            prompt=_prompt_with_contract(prompt, step),
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
                yield _step_event(
                    run,
                    step,
                    "step_delta",
                    channel=channel,
                    content=content,
                )

        text = "".join(text_parts)
        output = await self._coerce_llm_output(
            run=run,
            spec=spec,
            step=step,
            prompt=prompt,
            reasoning_effort=reasoning_effort,
            text=text,
            metadata=metadata,
        )
        yield {
            "type": "_execution_result",
            "execution": WorkflowStepExecution(
                output=output,
                artifact_text=text or None,
                streamed=True,
            ),
        }

    async def _execute_search_step(
        self,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        prompt: str,
    ) -> WorkflowStepExecution:
        if self.search_client is None:
            return WorkflowStepExecution(
                output={
                    "text": prompt,
                    "json": None,
                    "metadata": {"kind": "search", "skipped": "no-search-client"},
                }
            )
        queries = extract_search_queries(prompt)
        provider = run.search_provider or step.search_provider or spec.defaults.search_provider
        use_query_refiner = (
            bool(step.use_query_refiner)
            if step.use_query_refiner is not None
            else len(queries) == 1 and queries[0] == prompt.strip()
        )
        if not any(query.strip() for query in queries):
            raise ValueError(
                "workflow search step produced no query; check upstream JSON output"
            )
        results = await asyncio.gather(
            *(
                self.search_client.search(
                    query=query,
                    provider=provider,
                    count=step.search_count or 5,
                    use_query_refiner=use_query_refiner,
                )
                for query in queries
            )
        )
        result = merge_search_results(
            queries,
            results,
            use_query_refiner=use_query_refiner,
        )
        text = json.dumps(result, indent=2)
        return WorkflowStepExecution(
            output={"text": text, "json": result, "metadata": {"kind": "search"}},
            artifact_text=text,
        )

    async def _execute_rerank_step(
        self,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        step_input: dict[str, Any],
        prompt: str,
    ) -> WorkflowStepExecution:
        source_key, source = select_rerank_source(spec, step, step_input)
        source_results = [
            dict(item)
            for item in source.get("results") or []
            if isinstance(item, dict)
        ]
        query = prompt.strip() or search_output_query(source)
        if source_results and not query:
            raise ValueError("workflow rerank step produced no query")

        rerank_context = (
            render_template(step.rerank_context, step_input)
            if step.rerank_context
            else None
        )
        if not source_results:
            logger.info(
                "workflow rerank skipped: run_id=%s workflow_id=%s step_id=%s "
                "source_output=%s reason=no-results",
                run.run_id,
                run.workflow_id,
                step.id,
                source_key,
            )
            result = dict(source)
            result["warnings"] = [str(item) for item in result.get("warnings") or []]
            result["reranking"] = reranking_metadata(
                result, used=False, degraded=False, path="none"
            )
        else:
            result = await self._run_rerank(
                run=run,
                step=step,
                source_key=source_key,
                source=source,
                source_results=source_results,
                query=query,
                rerank_context=rerank_context,
            )

        result["workflow_rerank"] = {
            "source_output": source_key,
            "query": query,
            "context_provided": bool(rerank_context),
            "path": reranking_path(result),
        }
        if not source_results:
            result["workflow_rerank"]["skipped"] = "no-results"
        text = json.dumps(result, indent=2)
        return WorkflowStepExecution(
            output={"text": text, "json": result, "metadata": {"kind": "rerank"}},
            artifact_text=text,
        )

    async def _run_rerank(
        self,
        *,
        run: WorkflowRun,
        step: WorkflowStepSpec,
        source_key: str,
        source: dict[str, Any],
        source_results: list[dict[str, Any]],
        query: str,
        rerank_context: str | None,
    ) -> dict[str, Any]:
        rerank_results = (
            getattr(self.search_client, "rerank_results", None)
            if self.search_client is not None
            else None
        )
        if rerank_results is None:
            logger.warning(
                "workflow rerank fallback: run_id=%s workflow_id=%s step_id=%s "
                "source_output=%s reason=no-rerank-client",
                run.run_id,
                run.workflow_id,
                step.id,
                source_key,
            )
            result = dict(source)
            result["warnings"] = [
                *[str(item) for item in result.get("warnings") or []],
                "reranker: no-rerank-client",
            ]
            result["reranking"] = reranking_metadata(
                result, used=False, degraded=True, path="none"
            )
            return result

        logger.info(
            "workflow rerank requested: run_id=%s workflow_id=%s step_id=%s "
            "source_output=%s results=%d context=%s",
            run.run_id,
            run.workflow_id,
            step.id,
            source_key,
            len(source_results),
            bool(rerank_context),
        )
        reranked = await rerank_results(
            query=query,
            results=source_results,
            context=rerank_context,
            top_k=step.rerank_top_k,
        )
        result = merge_reranked_search_result(source, reranked)
        logger.info(
            "workflow rerank completed: run_id=%s workflow_id=%s step_id=%s "
            "source_output=%s path=%s degraded=%s",
            run.run_id,
            run.workflow_id,
            step.id,
            source_key,
            reranking_path(result),
            (result.get("reranking") or {}).get("degraded")
            if isinstance(result.get("reranking"), dict)
            else None,
        )
        return result

    async def _execute_compact_context_step(
        self,
        *,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        source: str,
        step_input: dict[str, Any],
    ) -> WorkflowStepExecution:
        source_text = str(source or "")
        source_format, parsed_source = parse_compaction_input(source_text, step)
        normalized_units = compaction_units(source_text, source_format, parsed_source)
        normalized_source = "\n\n".join(normalized_units).strip() or source_text
        goal = (
            render_template(step.compaction_goal, step_input)
            if step.compaction_goal
            else ""
        ).strip() or (
            "Preserve user intent, constraints, claims, concept relationships, "
            "high-value evidence snippets, source references, contradictions, "
            "and uncertainty."
        )
        anchors = extract_compaction_anchors(normalized_source)
        evidence = top_compaction_evidence(normalized_source, anchors, goal)
        warnings: list[str] = []
        source_chars = len(normalized_source)
        source_tokens = estimate_tokens(normalized_source)

        if source_chars < step.compaction_trigger_chars:
            checked = await self._check_compaction_output(
                run=run,
                spec=spec,
                step=step,
                text=normalized_source,
                source_text=normalized_source,
                goal=goal,
                anchors=anchors,
                allow_repair=False,
                phase="pass_through",
            )
            warnings.extend(checked["warnings"])
            output = compaction_output_payload(
                text=checked["text"],
                parsed_output=checked.get("parsed_output"),
                step=step,
                compacted=False,
                source_format=source_format,
                source_chars=source_chars,
                source_tokens=source_tokens,
                max_request_bytes=0,
                chunks=1 if normalized_source else 0,
                rounds=0,
                evidence=evidence,
                anchors=anchors,
                warnings=warnings,
                method="pass_through",
            )
            check_compaction_output_payload(output, step)
            return WorkflowStepExecution(
                output=output,
                artifact_text=checked["text"] or None,
            )

        chunks = chunk_compaction_units(normalized_units, step.compaction_chunk_chars)
        if not chunks and normalized_source.strip():
            raise ValueError("compaction_input_over_budget: no compactable input chunks")

        reasoning_effort = (
            step.reasoning_effort or run.reasoning_effort or spec.defaults.reasoning_effort
        )
        summaries: list[str] = []
        request_bytes: list[int] = []
        for index, chunk in enumerate(chunks, start=1):
            checked, request_byte_count = await self._compact_chunk(
                run=run,
                spec=spec,
                step=step,
                chunk=chunk,
                goal=goal,
                anchors=anchors,
                evidence=evidence,
                reasoning_effort=reasoning_effort,
                phase="chunk",
                chunk_index=index,
                chunk_count=len(chunks),
            )
            request_bytes.append(request_byte_count)
            warnings.extend(checked["warnings"])
            summaries.append(checked["text"])

        rounds = 0
        while compaction_needs_reduce(summaries, step):
            if rounds >= step.compaction_max_rounds:
                break
            rounds += 1
            summaries = await self._reduce_compaction_summaries(
                run=run,
                spec=spec,
                step=step,
                summaries=summaries,
                goal=goal,
                anchors=anchors,
                evidence=evidence,
                reasoning_effort=reasoning_effort,
                round_index=rounds,
                request_bytes=request_bytes,
                warnings=warnings,
            )

        if step.compaction_output_format in {"json", "yaml"} and len(summaries) != 1:
            raise ValueError(
                "compaction_output_shape_invalid: structured compaction did not "
                "reduce to one output"
            )

        final_text = "\n\n".join(summary for summary in summaries if summary.strip())
        checked = await self._check_compaction_output(
            run=run,
            spec=spec,
            step=step,
            text=final_text,
            source_text=normalized_source,
            goal=goal,
            anchors=anchors,
            allow_repair=bool(final_text),
            phase="final",
        )
        warnings.extend(checked["warnings"])
        final_text = checked["text"]
        if len(final_text) > step.compaction_target_chars:
            warnings.append("target_exceeded")

        output = compaction_output_payload(
            text=final_text,
            parsed_output=checked.get("parsed_output"),
            step=step,
            compacted=True,
            source_format=source_format,
            source_chars=source_chars,
            source_tokens=source_tokens,
            max_request_bytes=max(request_bytes) if request_bytes else 0,
            chunks=len(chunks),
            rounds=rounds,
            evidence=evidence,
            anchors=anchors,
            warnings=warnings,
            method="llm_compaction",
        )
        check_compaction_output_payload(output, step)
        return WorkflowStepExecution(output=output, artifact_text=final_text or None)

    async def _compact_chunk(
        self,
        *,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        chunk: str,
        goal: str,
        anchors: list[str],
        evidence: list[dict[str, Any]],
        reasoning_effort: str | None,
        phase: str,
        chunk_index: int,
        chunk_count: int,
        round_index: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        chunk_anchors = extract_compaction_anchors(chunk) or anchors
        prompt = build_compaction_prompt(
            source=chunk,
            goal=goal,
            anchors=chunk_anchors,
            evidence=evidence,
            step=step,
            phase=phase,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            round_index=round_index,
        )
        request_bytes = check_compaction_input_payload(prompt, chunk, step)
        result = await self.llm_client.complete(
            endpoint=str(run.endpoint or ""),
            prompt=prompt,
            convo_id=run.convo_id,
            reasoning_effort=reasoning_effort,
            rag_endpoint=run.rag_endpoint or step.rag_endpoint or spec.defaults.rag_endpoint,
            max_tokens=step.max_tokens or spec.defaults.max_tokens,
        )
        checked = await self._check_compaction_output(
            run=run,
            spec=spec,
            step=step,
            text=str(result.get("text") or ""),
            source_text=chunk,
            goal=goal,
            anchors=chunk_anchors,
            allow_repair=True,
            phase=phase,
        )
        return checked, request_bytes

    async def _reduce_compaction_summaries(
        self,
        *,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        summaries: list[str],
        goal: str,
        anchors: list[str],
        evidence: list[dict[str, Any]],
        reasoning_effort: str | None,
        round_index: int,
        request_bytes: list[int],
        warnings: list[str],
    ) -> list[str]:
        combined = "\n\n".join(summary for summary in summaries if summary.strip())
        reduce_chunks = chunk_text_by_budget(combined, step.compaction_chunk_chars)
        next_summaries: list[str] = []
        for index, chunk in enumerate(reduce_chunks, start=1):
            checked, request_byte_count = await self._compact_chunk(
                run=run,
                spec=spec,
                step=step,
                chunk=chunk,
                goal=goal,
                anchors=anchors,
                evidence=evidence,
                reasoning_effort=reasoning_effort,
                phase="reduce",
                chunk_index=index,
                chunk_count=len(reduce_chunks),
                round_index=round_index,
            )
            request_bytes.append(request_byte_count)
            warnings.extend(checked["warnings"])
            next_summaries.append(checked["text"])
        return next_summaries

    async def _check_compaction_output(
        self,
        *,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        text: str,
        source_text: str,
        goal: str,
        anchors: list[str],
        allow_repair: bool,
        phase: str,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        checked = validate_compaction_output_text(text, step, anchors)
        if checked["valid"]:
            return {**checked, "warnings": warnings}
        if not allow_repair:
            raise ValueError(checked["error"])

        prompt = build_compaction_repair_prompt(
            text=text,
            error=str(checked["error"]),
            source_text=source_text,
            goal=goal,
            anchors=anchors,
            step=step,
            phase=phase,
        )
        check_compaction_input_payload(
            prompt, compaction_source_excerpt(source_text, step), step
        )
        reasoning_effort = (
            step.reasoning_effort or run.reasoning_effort or spec.defaults.reasoning_effort
        )
        result = await self.llm_client.complete(
            endpoint=str(run.endpoint or ""),
            prompt=prompt,
            convo_id=run.convo_id,
            reasoning_effort=reasoning_effort,
            rag_endpoint=run.rag_endpoint or step.rag_endpoint or spec.defaults.rag_endpoint,
            max_tokens=step.max_tokens or spec.defaults.max_tokens,
        )
        repaired = validate_compaction_output_text(
            str(result.get("text") or ""), step, anchors
        )
        if not repaired["valid"]:
            raise ValueError(repaired["error"])
        warnings.append("output_repaired")
        return {**repaired, "warnings": warnings}

    async def _execute_llm_step(
        self,
        *,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        prompt: str,
        reasoning_effort: str | None,
    ) -> WorkflowStepExecution:
        result = await self.llm_client.complete(
            endpoint=str(run.endpoint or ""),
            prompt=_prompt_with_contract(prompt, step),
            convo_id=run.convo_id,
            reasoning_effort=reasoning_effort,
            rag_endpoint=run.rag_endpoint or step.rag_endpoint or spec.defaults.rag_endpoint,
            max_tokens=step.max_tokens or spec.defaults.max_tokens,
        )
        text = str(result.get("text") or "")
        metadata = dict(result.get("metadata") or {})
        output = await self._coerce_llm_output(
            run=run,
            spec=spec,
            step=step,
            prompt=prompt,
            reasoning_effort=reasoning_effort,
            text=text,
            metadata=metadata,
        )
        return WorkflowStepExecution(output=output, artifact_text=output.get("text") or None)

    async def _coerce_llm_output(
        self,
        *,
        run: WorkflowRun,
        spec: WorkflowSpec,
        step: WorkflowStepSpec,
        prompt: str,
        reasoning_effort: str | None,
        text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        contract = step.output_contract
        if not contract_requires_structure(contract):
            return {
                "text": text,
                "json": parse_json_text(text),
                "metadata": metadata,
            }
        assert contract is not None

        attempts = 1
        repair_used = False
        parsed = parse_structured_output(text, contract)
        validation = validate_structured_output(parsed, contract)
        if validation.valid:
            return _structured_output_payload(
                text=text,
                value=validation.value,
                metadata=metadata,
                contract_format=contract.format,
                attempts=attempts,
                repair_used=repair_used,
                errors=[],
            )

        errors = validation.errors
        for mode in _correction_modes(contract):
            repair_used = repair_used or mode == "repair"
            retry_prompt = (
                build_repair_prompt(text, errors, contract)
                if mode == "repair"
                else build_retry_prompt(prompt, errors, contract)
            )
            result = await self.llm_client.complete(
                endpoint=str(run.endpoint or ""),
                prompt=retry_prompt,
                convo_id=run.convo_id,
                reasoning_effort=reasoning_effort,
                rag_endpoint=(
                    run.rag_endpoint or step.rag_endpoint or spec.defaults.rag_endpoint
                ),
                max_tokens=step.max_tokens or spec.defaults.max_tokens,
            )
            attempts += 1
            text = str(result.get("text") or "")
            metadata.update(dict(result.get("metadata") or {}))
            parsed = parse_structured_output(text, contract)
            validation = validate_structured_output(parsed, contract)
            if validation.valid:
                return _structured_output_payload(
                    text=text,
                    value=validation.value,
                    metadata=metadata,
                    contract_format=contract.format,
                    attempts=attempts,
                    repair_used=repair_used,
                    errors=[],
                )
            errors = validation.errors

        raise ValueError(
            "workflow step failed: "
            f"step_id={step.id} reason=structured_output_invalid "
            f"errors={'; '.join(errors)}"
        )


def _prompt_with_contract(prompt: str, step: WorkflowStepSpec) -> str:
    instructions = build_structured_output_instructions(step.output_contract)
    if not instructions:
        return prompt
    return f"{prompt.rstrip()}\n\n{instructions}".strip()


def _correction_modes(step_contract: Any) -> list[str]:
    policy = step_contract.on_invalid if isinstance(step_contract.on_invalid, dict) else {}
    action = str(policy.get("action") or "retry").strip().lower()
    if action == "fail":
        return []
    max_attempts = _safe_nonnegative_int(policy.get("max_attempts"), default=2)
    if max_attempts <= 0:
        return []
    repair = (
        bool(policy["repair"])
        if "repair" in policy
        else action in {"repair", "retry"}
    )
    if action == "repair":
        return ["repair"] * max_attempts
    if action == "retry":
        modes = ["retry"] * max_attempts
        if repair:
            modes[0] = "repair"
        return modes
    return []


def _structured_output_payload(
    *,
    text: str,
    value: Any,
    metadata: dict[str, Any],
    contract_format: str,
    attempts: int,
    repair_used: bool,
    errors: list[str],
) -> dict[str, Any]:
    output_metadata = dict(metadata)
    output_metadata["structured_output"] = {
        "format": contract_format,
        "valid": True,
        "schema_enforced": True,
        "attempts": attempts,
        "repair_used": repair_used,
        "errors": list(errors),
    }
    return {"text": text, "json": value, "metadata": output_metadata}


def _safe_nonnegative_int(value: Any, *, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _should_stream_step(step: WorkflowStepSpec, stream_llm: bool) -> bool:
    return (
        bool(stream_llm)
        and step.chat_visibility != "hidden"
        and step.chat_stream is not False
    )


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
