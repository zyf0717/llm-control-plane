from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


WorkflowStepKind = Literal["llm", "search", "manual"]
WorkflowChatVisibility = Literal["hidden", "intermediate", "final"]
WorkflowRunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
WorkflowStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]


@dataclass(slots=True)
class WorkflowDefaults:
    reasoning_effort: str | None = None
    rag_endpoint: str | None = None
    search_provider: str | None = None
    stream: bool = False
    max_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WorkflowDefaults":
        data = data if isinstance(data, dict) else {}
        return cls(
            reasoning_effort=_optional_str(data.get("reasoning_effort")),
            rag_endpoint=_optional_str(data.get("rag_endpoint")),
            search_provider=_optional_str(data.get("search_provider")),
            stream=bool(data.get("stream", False)),
            max_tokens=_optional_int(data.get("max_tokens")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowStepSpec:
    id: str
    name: str
    kind: WorkflowStepKind
    prompt: str | None = None
    depends_on: list[str] | None = None
    output_key: str | None = None
    output_schema: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    rag_endpoint: str | None = None
    search_provider: str | None = None
    max_tokens: int | None = None
    chat_visibility: WorkflowChatVisibility = "hidden"
    chat_stream: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStepSpec":
        if not isinstance(data, dict):
            raise ValueError("workflow step must be an object")

        step_id = _required_str(data, "id", "workflow step")
        kind = _required_str(data, "kind", f"workflow step {step_id}")
        if kind not in {"llm", "search", "manual"}:
            raise ValueError(f"workflow step {step_id} has unsupported kind: {kind}")

        depends_on = data.get("depends_on", [])
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list) or not all(
            isinstance(item, str) and item.strip() for item in depends_on
        ):
            raise ValueError(f"workflow step {step_id} depends_on must be a string list")

        output_schema = data.get("output_schema")
        if output_schema is not None and not isinstance(output_schema, dict):
            raise ValueError(f"workflow step {step_id} output_schema must be an object")
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
            output_schema=output_schema,
            reasoning_effort=_optional_str(data.get("reasoning_effort")),
            rag_endpoint=_optional_str(data.get("rag_endpoint")),
            search_provider=_optional_str(data.get("search_provider")),
            max_tokens=_optional_int(data.get("max_tokens")),
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
    convo_id: str
    params: dict[str, Any]
    endpoint: str | None
    reasoning_effort: str | None
    rag_endpoint: str | None
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


def _required_str(data: dict[str, Any], field: str, context: str) -> str:
    value = _optional_str(data.get(field))
    if not value:
        raise ValueError(f"{context} missing required field: {field}")
    return value
