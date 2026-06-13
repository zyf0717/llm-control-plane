from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from yaml.events import AliasEvent, CollectionEndEvent, Event, ScalarEvent

from .models import WorkflowOutputContract


@dataclass(slots=True)
class ParsedOutput:
    raw_text: str
    format: str
    value: Any
    parse_error: str | None = None


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: list[str]
    value: Any


def contract_requires_structure(contract: WorkflowOutputContract | None) -> bool:
    return bool(
        contract
        and contract.format in {"json", "yaml"}
        and (contract.required or contract.schema is not None)
    )


def parse_structured_output(
    text: str, contract: WorkflowOutputContract
) -> ParsedOutput:
    raw_text = str(text or "")
    if contract.required and not raw_text.strip():
        return ParsedOutput(
            raw_text=raw_text,
            format=contract.format,
            value=None,
            parse_error="output is required but empty",
        )
    if not raw_text.strip():
        return ParsedOutput(raw_text=raw_text, format=contract.format, value=None)

    if contract.format == "json":
        return _parse_json(raw_text)
    if contract.format == "yaml":
        return _parse_yaml(raw_text)
    return ParsedOutput(raw_text=raw_text, format=contract.format, value=raw_text)


def validate_structured_output(
    parsed: ParsedOutput, contract: WorkflowOutputContract
) -> ValidationResult:
    errors: list[str] = []
    if parsed.parse_error:
        errors.append(parsed.parse_error)
    elif contract.required and parsed.value is None:
        errors.append("output is required but empty")

    if not errors and contract.schema is not None:
        validator = Draft202012Validator(contract.schema)
        errors.extend(
            _format_schema_error(error)
            for error in sorted(validator.iter_errors(parsed.value), key=str)
        )

    return ValidationResult(valid=not errors, errors=errors, value=parsed.value)


def build_structured_output_instructions(
    contract: WorkflowOutputContract | None,
) -> str:
    if not contract_requires_structure(contract):
        return ""
    assert contract is not None
    schema = (
        json.dumps(contract.schema, indent=2, sort_keys=True)
        if contract.schema is not None
        else "{}"
    )
    if contract.format == "json":
        return (
            "You must return only valid JSON matching this schema.\n"
            "Do not include markdown fences.\n"
            "Do not include explanation.\n"
            "Schema:\n"
            f"{schema}"
        )
    return (
        "You must return only valid YAML matching this schema.\n"
        "Do not include markdown fences.\n"
        "Do not include explanation.\n"
        "Use only plain scalars, lists, and mappings.\n"
        "Schema:\n"
        f"{schema}"
    )


def build_repair_prompt(
    text: str,
    errors: list[str],
    contract: WorkflowOutputContract,
) -> str:
    schema = (
        json.dumps(contract.schema, indent=2, sort_keys=True)
        if contract.schema is not None
        else "{}"
    )
    return (
        "The previous output did not satisfy the required machine-readable contract.\n\n"
        f"Required format: {contract.format.upper()}\n"
        "Schema:\n"
        f"{schema}\n\n"
        "Validation errors:\n"
        f"{json.dumps(errors, indent=2)}\n\n"
        "Previous output:\n"
        f"{text}\n\n"
        f"Return only valid {contract.format.upper()}. No prose. No markdown fence."
    )


def build_retry_prompt(
    prompt: str,
    errors: list[str],
    contract: WorkflowOutputContract,
) -> str:
    instructions = build_structured_output_instructions(contract)
    return (
        f"{prompt.rstrip()}\n\n"
        "The previous response was invalid for this machine-readable contract.\n"
        "Validation errors:\n"
        f"{json.dumps(errors, indent=2)}\n\n"
        f"{instructions}"
    ).strip()


def _parse_json(text: str) -> ParsedOutput:
    body = _strip_code_fence(text, {"json"})
    try:
        return ParsedOutput(
            raw_text=text,
            format="json",
            value=json.loads(body),
        )
    except json.JSONDecodeError as exc:
        return ParsedOutput(
            raw_text=text,
            format="json",
            value=None,
            parse_error=f"invalid JSON: {exc.msg}",
        )


def _parse_yaml(text: str) -> ParsedOutput:
    body = _strip_code_fence(text, {"yaml", "yml"})
    try:
        _reject_nonportable_yaml(body)
        value = yaml.safe_load(body)
        _ensure_json_compatible(value)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        return ParsedOutput(
            raw_text=text,
            format="yaml",
            value=None,
            parse_error=f"invalid YAML: {exc}",
        )
    return ParsedOutput(raw_text=text, format="yaml", value=value)


def _strip_code_fence(text: str, languages: set[str]) -> str:
    stripped = str(text or "").strip()
    match = re.fullmatch(
        r"```([A-Za-z0-9_-]*)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if not match:
        return stripped
    language = str(match.group(1) or "").strip().lower()
    if language and language not in languages:
        return stripped
    return match.group(2).strip()


def _reject_nonportable_yaml(text: str) -> None:
    for event in yaml.parse(text):
        if isinstance(event, AliasEvent):
            raise ValueError("aliases are not supported")
        if _event_anchor(event) is not None:
            raise ValueError("anchors are not supported")
        if isinstance(event, CollectionEndEvent):
            continue
        if isinstance(event, ScalarEvent) and str(event.tag).endswith(":null"):
            continue
        tag = str(getattr(event, "tag", "") or "")
        if tag.startswith("!") or tag.startswith("tag:yaml.org,2002:python/"):
            raise ValueError(f"custom YAML tag is not supported: {tag}")


def _event_anchor(event: Event) -> str | None:
    anchor = getattr(event, "anchor", None)
    return str(anchor) if anchor else None


def _ensure_json_compatible(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not JSON-compatible")
        return
    if isinstance(value, (datetime, date, bytes, bytearray, set, tuple)):
        raise ValueError(f"{type(value).__name__} is not JSON-compatible")
    if isinstance(value, list):
        for item in value:
            _ensure_json_compatible(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            _ensure_json_compatible(item)
        return
    raise ValueError(f"{type(value).__name__} is not JSON-compatible")


def _format_schema_error(error: Any) -> str:
    path = "$"
    for part in error.absolute_path:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return f"{path}: {error.message}"
