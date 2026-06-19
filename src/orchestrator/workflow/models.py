from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


WorkflowStepKind = Literal[
    "llm",
    "search",
    "rerank",
    "manual",
    "compress_source",
    "repo_context",
]
WorkflowChatVisibility = Literal["hidden", "intermediate", "final"]
WorkflowOutputFormat = Literal["json", "yaml", "text"]
WorkflowCompressionInputFormat = Literal["auto", "text", "json", "yaml"]
WorkflowCompressionOutputFormat = Literal["text", "json", "yaml"]
WorkflowRunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
WorkflowStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]


COMPRESSION_DEFAULT_TRIGGER_CHARS = 48_000
COMPRESSION_DEFAULT_CHUNK_CHARS = 32_000
COMPRESSION_DEFAULT_TARGET_CHARS = 16_000
COMPRESSION_DEFAULT_MAX_OUTPUT_CHARS = 20_000
COMPRESSION_DEFAULT_MAX_OUTPUT_JSON_BYTES = 256_000
COMPRESSION_DEFAULT_MAX_ROUNDS = 3
COMPRESSION_DEFAULT_INPUT_FORMAT = "auto"
COMPRESSION_DEFAULT_OUTPUT_FORMAT = "text"
COMPRESSION_DEFAULT_GOAL = (
    "Preserve user intent, constraints, claims, concept relationships, "
    "high-value evidence snippets, source references, contradictions, and uncertainty."
)


def _builtin_compression_contract(
    output_format: WorkflowOutputFormat,
) -> "WorkflowOutputContract":
    return WorkflowOutputContract(
        format=output_format,
        required=True,
        schema={
            "type": "object",
            "additionalProperties": True,
            "required": [
                "summary",
                "preserved_keywords",
                "evidence_snippets",
                "uncertainties",
                "source_refs",
            ],
            "properties": {
                "summary": {"type": "string", "minLength": 1},
                "preserved_keywords": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "evidence_snippets": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "source_refs": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
        on_invalid={"action": "repair", "max_attempts": 1, "repair": True},
    )


@dataclass(slots=True)
class WorkflowDefaults:
    reasoning_effort: str | None = None
    retrieval_endpoint: str | None = None
    search_provider: str | None = None
    stream: bool = False
    max_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WorkflowDefaults":
        data = data if isinstance(data, dict) else {}
        return cls(
            reasoning_effort=_optional_str(data.get("reasoning_effort")),
            retrieval_endpoint=_optional_str(data.get("retrieval_endpoint")),
            search_provider=_optional_str(data.get("search_provider")),
            stream=bool(data.get("stream", False)),
            max_tokens=_optional_int(data.get("max_tokens")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowOutputContract:
    format: WorkflowOutputFormat = "text"
    required: bool = False
    schema: dict[str, Any] | None = None
    on_invalid: dict[str, Any] | None = None
    raw_text: bool = True

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, step_id: str
    ) -> "WorkflowOutputContract":
        if not isinstance(data, dict):
            raise ValueError(f"workflow step {step_id} output_contract must be an object")

        output_format = _optional_str(data.get("format")) or "text"
        if output_format not in {"json", "yaml", "text"}:
            raise ValueError(
                f"workflow step {step_id} output_contract.format is unsupported: "
                f"{output_format}"
            )

        schema = data.get("schema")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError(
                f"workflow step {step_id} output_contract.schema must be an object"
            )

        on_invalid = data.get("on_invalid")
        if on_invalid is not None and not isinstance(on_invalid, dict):
            raise ValueError(
                f"workflow step {step_id} output_contract.on_invalid must be an object"
            )

        return cls(
            format=output_format,  # type: ignore[arg-type]
            required=bool(data.get("required", False)),
            schema=schema,
            on_invalid=on_invalid,
            raw_text=bool(data.get("raw_text", True)),
        )

@dataclass(slots=True)
class WorkflowStepSpec:
    id: str
    name: str
    kind: WorkflowStepKind
    prompt: str | None = None
    depends_on: list[str] | None = None
    output_key: str | None = None
    output_contract: WorkflowOutputContract | None = None
    endpoint: str | None = None
    reasoning_effort: str | None = None
    retrieval_endpoint: str | None = None
    search_provider: str | None = None
    search_count: int | None = None
    use_query_refiner: bool | None = None
    rerank_source_text: str | None = None
    rerank_top_k: int | None = None
    max_tokens: int | None = None
    compression_trigger_chars: int = COMPRESSION_DEFAULT_TRIGGER_CHARS
    compression_chunk_chars: int = COMPRESSION_DEFAULT_CHUNK_CHARS
    compression_target_chars: int = COMPRESSION_DEFAULT_TARGET_CHARS
    compression_max_output_chars: int = COMPRESSION_DEFAULT_MAX_OUTPUT_CHARS
    compression_max_output_json_bytes: int = COMPRESSION_DEFAULT_MAX_OUTPUT_JSON_BYTES
    compression_max_rounds: int = COMPRESSION_DEFAULT_MAX_ROUNDS
    compression_input_format: WorkflowCompressionInputFormat = COMPRESSION_DEFAULT_INPUT_FORMAT
    compression_output_format: WorkflowCompressionOutputFormat = COMPRESSION_DEFAULT_OUTPUT_FORMAT
    compression_goal: str | None = None
    repo_context_repo: str | None = None
    repo_context_max_turns: int | None = None
    chat_visibility: WorkflowChatVisibility = "hidden"
    chat_stream: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStepSpec":
        if not isinstance(data, dict):
            raise ValueError("workflow step must be an object")

        step_id = _required_str(data, "id", "workflow step")
        kind = _required_str(data, "kind", f"workflow step {step_id}")
        if kind not in {
            "llm",
            "search",
            "rerank",
            "manual",
            "compress_source",
            "repo_context",
        }:
            raise ValueError(f"workflow step {step_id} has unsupported kind: {kind}")
        if "use_reranker" in data:
            raise ValueError(
                f"workflow step {step_id} use_reranker is no longer supported; "
                "add an explicit rerank step"
            )
        if "rerank_context" in data:
            raise ValueError(
                f"workflow step {step_id} rerank_context is no longer supported; "
                "use rerank_source_text"
            )
        if "compact_context" in data or any(
            str(key).startswith("compaction_") for key in data
        ):
            raise ValueError(
                f"workflow step {step_id} compaction fields are no longer supported; "
                "use compress_source and compression_*"
            )
        if "rerank_source_text" in data and kind != "rerank":
            raise ValueError(
                f"workflow step {step_id} rerank_source_text is only supported on rerank steps"
            )
        if (
            "repo_context_repo" in data or "repo_context_max_turns" in data
        ) and kind != "repo_context":
            raise ValueError(
                f"workflow step {step_id} repo_context fields are only supported "
                "on repo_context steps"
            )
        endpoint = _optional_str(data.get("endpoint"))
        if endpoint is not None and kind not in {"llm", "compress_source"}:
            raise ValueError(
                f"workflow step {step_id} endpoint is only supported on model-backed steps"
            )
        if endpoint is not None and endpoint.lower() == "smart":
            raise ValueError(
                f"workflow step {step_id} endpoint must be a concrete endpoint"
            )

        depends_on = data.get("depends_on", [])
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list) or not all(
            isinstance(item, str) and item.strip() for item in depends_on
        ):
            raise ValueError(f"workflow step {step_id} depends_on must be a string list")

        if "output_schema" in data:
            raise ValueError(
                f"workflow step {step_id} output_schema is no longer supported; "
                "use output_contract"
            )
        output_contract_data = data.get("output_contract")
        output_contract = None
        if output_contract_data is not None:
            output_contract = WorkflowOutputContract.from_dict(
                output_contract_data,
                step_id=step_id,
            )

        compression_input_format = (
            _optional_str(data.get("compression_input_format"))
            or COMPRESSION_DEFAULT_INPUT_FORMAT
        )
        if compression_input_format not in {"auto", "text", "json", "yaml"}:
            raise ValueError(
                f"workflow step {step_id} compression_input_format is unsupported: "
                f"{compression_input_format}"
            )
        compression_output_format = (
            _optional_str(data.get("compression_output_format"))
            or COMPRESSION_DEFAULT_OUTPUT_FORMAT
        )
        if compression_output_format not in {"text", "json", "yaml"}:
            raise ValueError(
                f"workflow step {step_id} compression_output_format is unsupported: "
                f"{compression_output_format}"
            )
        if kind == "compress_source":
            _validate_compression_budgets(data, step_id=step_id)
            if (
                output_contract is not None
                and output_contract.format != compression_output_format
            ):
                raise ValueError(
                    f"workflow step {step_id} output_contract.format must match "
                    "compression_output_format"
                )
            if compression_output_format in {"json", "yaml"} and output_contract is None:
                output_contract = _builtin_compression_contract(
                    compression_output_format  # type: ignore[arg-type]
                )
        if kind == "repo_context":
            if not _optional_str(data.get("repo_context_repo")):
                raise ValueError(
                    f"workflow step {step_id} repo_context_repo is required"
                )
            max_turns = _optional_int(data.get("repo_context_max_turns"))
            if max_turns is not None and max_turns <= 0:
                raise ValueError(
                    f"workflow step {step_id} repo_context_max_turns must be positive"
                )
        else:
            max_turns = None

        chat_visibility = _optional_str(data.get("chat_visibility")) or "hidden"
        if chat_visibility not in {"hidden", "intermediate", "final"}:
            raise ValueError(
                f"workflow step {step_id} has unsupported chat_visibility: "
                f"{chat_visibility}"
            )

        return cls(
            id=step_id,
            name=_optional_str(data.get("name")) or step_id,
            kind=kind,  # type: ignore[arg-type]
            prompt=_optional_str(data.get("prompt")),
            depends_on=[str(item).strip() for item in depends_on],
            output_key=_optional_str(data.get("output_key")),
            output_contract=output_contract,
            endpoint=endpoint,
            reasoning_effort=_optional_str(data.get("reasoning_effort")),
            retrieval_endpoint=_optional_str(data.get("retrieval_endpoint")),
            search_provider=_optional_str(data.get("search_provider")),
            search_count=_optional_int(data.get("search_count")),
            use_query_refiner=_optional_bool(data.get("use_query_refiner")),
            rerank_source_text=_optional_str(data.get("rerank_source_text")),
            rerank_top_k=_optional_int(data.get("rerank_top_k")),
            max_tokens=_optional_int(data.get("max_tokens")),
            compression_trigger_chars=_compression_budget_value(
                data, "compression_trigger_chars", COMPRESSION_DEFAULT_TRIGGER_CHARS
            ),
            compression_chunk_chars=_compression_budget_value(
                data, "compression_chunk_chars", COMPRESSION_DEFAULT_CHUNK_CHARS
            ),
            compression_target_chars=_compression_budget_value(
                data, "compression_target_chars", COMPRESSION_DEFAULT_TARGET_CHARS
            ),
            compression_max_output_chars=_compression_budget_value(
                data, "compression_max_output_chars", COMPRESSION_DEFAULT_MAX_OUTPUT_CHARS
            ),
            compression_max_output_json_bytes=_compression_budget_value(
                data,
                "compression_max_output_json_bytes",
                COMPRESSION_DEFAULT_MAX_OUTPUT_JSON_BYTES,
            ),
            compression_max_rounds=_compression_budget_value(
                data, "compression_max_rounds", COMPRESSION_DEFAULT_MAX_ROUNDS
            ),
            compression_input_format=compression_input_format,  # type: ignore[arg-type]
            compression_output_format=compression_output_format,  # type: ignore[arg-type]
            compression_goal=_optional_str(data.get("compression_goal")),
            repo_context_repo=_optional_str(data.get("repo_context_repo")),
            repo_context_max_turns=max_turns,
            chat_visibility=chat_visibility,  # type: ignore[arg-type]
            chat_stream=_optional_bool(data.get("chat_stream")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["depends_on"] = list(self.depends_on or [])
        return data


@dataclass(slots=True)
class WorkflowSpec:
    id: str
    name: str
    description: str | None
    version: str
    params_schema: dict[str, Any]
    defaults: WorkflowDefaults
    steps: list[WorkflowStepSpec]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowSpec":
        if not isinstance(data, dict):
            raise ValueError("workflow spec must be an object")

        params_schema = data.get("params_schema", {})
        if not isinstance(params_schema, dict):
            raise ValueError("workflow params_schema must be an object")

        steps_data = data.get("steps")
        if not isinstance(steps_data, list) or not steps_data:
            raise ValueError("workflow steps must be a non-empty list")

        return cls(
            id=_required_str(data, "id", "workflow"),
            name=_optional_str(data.get("name")) or _required_str(data, "id", "workflow"),
            description=_optional_str(data.get("description")),
            version=_optional_str(data.get("version")) or "0.1.0",
            params_schema=params_schema,
            defaults=WorkflowDefaults.from_dict(data.get("defaults")),
            steps=[WorkflowStepSpec.from_dict(step) for step in steps_data],
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "params_schema": self.params_schema,
            "defaults": self.defaults.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(slots=True)
class WorkflowRun:
    run_id: str
    workflow_id: str
    workflow_version: str
    status: WorkflowRunStatus
    conversation_id: str
    params: dict[str, Any]
    endpoint: str | None
    reasoning_effort: str | None
    retrieval_endpoint: str | None
    search_provider: str | None
    current_step_id: str | None
    created_at: str
    updated_at: str
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowStepRun:
    run_id: str
    step_id: str
    status: WorkflowStepStatus
    input_json: dict[str, Any]
    output_json: dict[str, Any] | None
    error: str | None
    started_at: str | None
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected integer, got {value!r}") from exc


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"expected boolean, got {value!r}")


def _compression_budget_value(
    data: dict[str, Any], field: str, default: int
) -> int:
    value = _optional_int(data.get(field))
    return default if value is None else value


def _validate_compression_budgets(data: dict[str, Any], *, step_id: str) -> None:
    trigger = _compression_budget_value(
        data, "compression_trigger_chars", COMPRESSION_DEFAULT_TRIGGER_CHARS
    )
    chunk = _compression_budget_value(
        data, "compression_chunk_chars", COMPRESSION_DEFAULT_CHUNK_CHARS
    )
    target = _compression_budget_value(
        data, "compression_target_chars", COMPRESSION_DEFAULT_TARGET_CHARS
    )
    max_output = _compression_budget_value(
        data, "compression_max_output_chars", COMPRESSION_DEFAULT_MAX_OUTPUT_CHARS
    )
    max_json = _compression_budget_value(
        data,
        "compression_max_output_json_bytes",
        COMPRESSION_DEFAULT_MAX_OUTPUT_JSON_BYTES,
    )
    max_rounds = _compression_budget_value(
        data, "compression_max_rounds", COMPRESSION_DEFAULT_MAX_ROUNDS
    )
    values = {
        "compression_trigger_chars": trigger,
        "compression_chunk_chars": chunk,
        "compression_target_chars": target,
        "compression_max_output_chars": max_output,
        "compression_max_output_json_bytes": max_json,
        "compression_max_rounds": max_rounds,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"workflow step {step_id} {name} must be positive")
    if not target <= max_output < trigger:
        raise ValueError(
            f"workflow step {step_id} compression budgets must satisfy "
            "compression_target_chars <= compression_max_output_chars < "
            "compression_trigger_chars"
        )
    if chunk > trigger:
        raise ValueError(
            f"workflow step {step_id} compression_chunk_chars must be <= "
            "compression_trigger_chars"
        )


def _required_str(data: dict[str, Any], field: str, location: str) -> str:
    value = _optional_str(data.get(field))
    if not value:
        raise ValueError(f"{location} missing required field: {field}")
    return value
