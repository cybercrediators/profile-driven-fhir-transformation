"""Schema derivation and local validation of untrusted model output."""

from typing import List, Optional

import pytest
from pydantic import BaseModel

from llm.errors import LLMResponseValidationError, LLMSchemaError
from llm.models import StructuredOutputMode
from llm.structured_output import (
    json_schema_for,
    parse_structured_response,
    response_format_for,
    schema_digest,
    schema_instructions,
    schema_name,
)

pytestmark = pytest.mark.unit


class Nested(BaseModel):
    label: str
    weight: Optional[float] = None


class Selection(BaseModel):
    candidate_id: str
    nested: List[Nested]
    reason: Optional[str] = None


class NonNullableDefault(BaseModel):
    attempts: int = 3


def test_strict_schema_forbids_unknown_properties_at_every_level():
    schema = json_schema_for(Selection, strict=True)

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Nested"]["additionalProperties"] is False


def test_strict_schema_requires_every_declared_property():
    schema = json_schema_for(Selection, strict=True)

    assert set(schema["required"]) == {"candidate_id", "nested", "reason"}
    assert set(schema["$defs"]["Nested"]["required"]) == {"label", "weight"}


def test_strict_schema_drops_defaults_that_carry_no_validation_meaning():
    schema = json_schema_for(Selection, strict=True)

    assert "default" not in schema["properties"]["reason"]
    assert "default" not in schema["$defs"]["Nested"]["properties"]["weight"]


def test_non_nullable_optional_field_is_rejected_with_an_actionable_message():
    with pytest.raises(LLMSchemaError) as excinfo:
        json_schema_for(NonNullableDefault, strict=True)

    message = str(excinfo.value)
    assert "attempts" in message
    assert "Optional[...] = None" in message
    assert "json-object" in message


def test_non_strict_schema_leaves_the_model_untouched():
    schema = json_schema_for(NonNullableDefault, strict=False)

    assert schema["properties"]["attempts"]["default"] == 3
    assert "additionalProperties" not in schema


def test_response_format_per_mode_is_explicit():
    strict = response_format_for(Selection, StructuredOutputMode.JSON_SCHEMA)
    assert strict["type"] == "json_schema"
    assert strict["json_schema"]["strict"] is True
    assert strict["json_schema"]["name"] == "Selection"

    assert response_format_for(Selection, StructuredOutputMode.JSON_OBJECT) == {
        "type": "json_object"
    }
    # Prompt-constrained mode requests no server-side constraint at all.
    assert response_format_for(Selection, StructuredOutputMode.PROMPT) is None


def test_schema_name_is_response_format_safe():
    """`response_format.json_schema.name` only accepts [A-Za-z0-9_-]."""

    dynamic = type("Odd.Name Model", (BaseModel,), {"__annotations__": {"a": str}})

    assert schema_name(Selection) == "Selection"
    assert schema_name(dynamic) == "Odd_Name_Model"


def test_schema_digest_changes_when_the_model_changes():
    class V1(BaseModel):
        a: str

    class V2(BaseModel):
        a: str
        b: str

    assert schema_digest(V1) == schema_digest(V1)
    assert schema_digest(V1) != schema_digest(V2)


def test_valid_payload_is_returned_as_the_typed_model():
    payload = '{"candidate_id": "c-1", "nested": [{"label": "x", "weight": 0.5}], "reason": null}'

    value = parse_structured_response(payload, Selection, StructuredOutputMode.JSON_SCHEMA)

    assert value.candidate_id == "c-1"
    assert value.nested[0].label == "x"
    assert value.reason is None


def test_schema_violation_is_rejected_even_when_the_json_parses():
    payload = '{"candidate_id": "c-1"}'

    with pytest.raises(LLMResponseValidationError, match="did not satisfy"):
        parse_structured_response(payload, Selection, StructuredOutputMode.JSON_SCHEMA)


def test_malformed_json_is_rejected():
    with pytest.raises(LLMResponseValidationError, match="not valid JSON"):
        parse_structured_response("{oops", Selection, StructuredOutputMode.JSON_SCHEMA)


def test_empty_response_is_rejected():
    with pytest.raises(LLMResponseValidationError, match="empty response"):
        parse_structured_response("   ", Selection, StructuredOutputMode.JSON_SCHEMA)


def test_server_constrained_modes_do_not_scrape_prose_around_the_json():
    """A server-enforced mode that returns prose is a failure, not a parse problem.

    Recovering JSON from surrounding text here would hide a misconfigured
    endpoint that is silently ignoring the requested response format.
    """
    payload = 'Sure! Here you go:\n```json\n{"candidate_id": "c", "nested": [], "reason": null}\n```'

    with pytest.raises(LLMResponseValidationError, match="not valid JSON"):
        parse_structured_response(payload, Selection, StructuredOutputMode.JSON_SCHEMA)


def test_prompt_mode_recovers_json_from_a_fenced_block():
    payload = 'Here:\n```json\n{"candidate_id": "c", "nested": [], "reason": null}\n```\nDone.'

    value = parse_structured_response(payload, Selection, StructuredOutputMode.PROMPT)

    assert value.candidate_id == "c"


def test_prompt_mode_ignores_reasoning_model_think_blocks():
    payload = (
        "<think>The label is probably x, but the schema wants no label here.</think>\n"
        '{"candidate_id": "c", "nested": [], "reason": null}'
    )

    value = parse_structured_response(payload, Selection, StructuredOutputMode.PROMPT)

    assert value.candidate_id == "c"


def test_prompt_mode_still_enforces_the_schema():
    with pytest.raises(LLMResponseValidationError, match="did not satisfy"):
        parse_structured_response(
            '```json\n{"candidate_id": 5, "nested": []}\n```',
            Selection,
            StructuredOutputMode.PROMPT,
        )


def test_prompt_instructions_embed_the_schema():
    text = schema_instructions(Selection)

    assert "candidate_id" in text
    assert "single JSON document" in text


def test_validation_error_retains_the_raw_text_for_the_run_report():
    with pytest.raises(LLMResponseValidationError) as excinfo:
        parse_structured_response("not json", Selection, StructuredOutputMode.JSON_SCHEMA)

    assert excinfo.value.raw_text == "not json"
