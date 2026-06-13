import pytest

from src.orchestrator.workflow.models import WorkflowOutputContract
from src.orchestrator.workflow.structured_output import (
    build_repair_prompt,
    parse_structured_output,
    validate_structured_output,
)


def contract(schema):
    return WorkflowOutputContract(
        format="json",
        required=True,
        schema=schema,
    )


def test_valid_raw_json_object_passes_schema():
    output = parse_structured_output(
        '{"queries": ["alpha"]}',
        contract(
            {
                "type": "object",
                "required": ["queries"],
                "properties": {"queries": {"type": "array"}},
            }
        ),
    )

    result = validate_structured_output(
        output,
        contract(
            {
                "type": "object",
                "required": ["queries"],
                "properties": {"queries": {"type": "array"}},
            }
        ),
    )

    assert result.valid is True
    assert result.value == {"queries": ["alpha"]}


def test_valid_fenced_json_passes_schema():
    output = parse_structured_output(
        '```json\n{"queries": ["alpha"]}\n```',
        contract({"type": "object"}),
    )

    result = validate_structured_output(output, contract({"type": "object"}))

    assert result.valid is True
    assert result.value == {"queries": ["alpha"]}


def test_invalid_json_reports_parse_error():
    output = parse_structured_output("{nope", contract({"type": "object"}))
    result = validate_structured_output(output, contract({"type": "object"}))

    assert result.valid is False
    assert result.errors == ["invalid JSON: Expecting property name enclosed in double quotes"]


def test_valid_raw_yaml_object_passes_schema():
    yaml_contract = WorkflowOutputContract(
        format="yaml",
        required=True,
        schema={
            "type": "object",
            "required": ["reranker"],
            "properties": {"reranker": {"type": "object"}},
        },
    )

    output = parse_structured_output(
        "reranker:\n  enabled: true\n  backend: llm\n",
        yaml_contract,
    )
    result = validate_structured_output(output, yaml_contract)

    assert result.valid is True
    assert result.value == {"reranker": {"enabled": True, "backend": "llm"}}


def test_valid_fenced_yaml_object_passes_schema():
    yaml_contract = WorkflowOutputContract(
        format="yaml",
        required=True,
        schema={"type": "object"},
    )

    output = parse_structured_output("```yaml\nfoo: bar\n```", yaml_contract)
    result = validate_structured_output(output, yaml_contract)

    assert result.valid is True
    assert result.value == {"foo": "bar"}


def test_yaml_scalar_rejected_when_object_expected():
    yaml_contract = WorkflowOutputContract(
        format="yaml",
        required=True,
        schema={"type": "object"},
    )

    output = parse_structured_output("hello", yaml_contract)
    result = validate_structured_output(output, yaml_contract)

    assert result.valid is False
    assert "$:" in result.errors[0]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("!Custom\nfoo: bar", "custom YAML tag"),
        ("first: &base {x: 1}\nsecond: *base", "anchors are not supported"),
        ("when: 2026-06-13", "date is not JSON-compatible"),
    ],
)
def test_yaml_nonportable_features_rejected(text, expected):
    yaml_contract = WorkflowOutputContract(format="yaml", required=True)

    output = parse_structured_output(text, yaml_contract)

    assert output.parse_error is not None
    assert expected in output.parse_error


def test_schema_error_path_formatting_and_additional_properties():
    json_contract = contract(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["queries"],
            "properties": {"queries": {"type": "array"}},
        }
    )

    output = parse_structured_output('{"queries": "alpha", "extra": true}', json_contract)
    result = validate_structured_output(output, json_contract)

    assert result.valid is False
    assert "$.queries:" in result.errors[0] or "$:" in result.errors[0]
    assert any("Additional properties" in error for error in result.errors)


def test_empty_required_output_rejected():
    json_contract = contract({"type": "object"})

    output = parse_structured_output("", json_contract)
    result = validate_structured_output(output, json_contract)

    assert result.valid is False
    assert result.errors == ["output is required but empty"]


def test_repair_prompt_demands_only_contract_format():
    json_contract = contract({"type": "object"})

    prompt = build_repair_prompt("bad", ["invalid JSON"], json_contract)

    assert "Return only valid JSON" in prompt
    assert "invalid JSON" in prompt
