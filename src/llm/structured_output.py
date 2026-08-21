"""Schema derivation and validation of structure llm output"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from llm.errors import LLMResponseValidationError, LLMSchemaError
from llm.models import LLMMessage, Role, StructuredOutputMode

TModel = TypeVar("TModel", bound=BaseModel)

_SCHEMA_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def schema_name(model: Type[BaseModel]) -> str:
    """Return a response-format-safe name for *model*."""

    return _SCHEMA_NAME_RE.sub("_", model.__name__)


def _admits_null(node: Mapping[str, Any]) -> bool:
    """True when the schema node permits an explicit null value."""

    node_type = node.get("type")
    if node_type == "null":
        return True
    if isinstance(node_type, list) and "null" in node_type:
        return True
    for keyword in ("anyOf", "oneOf"):
        for branch in node.get(keyword, []) or []:
            if isinstance(branch, Mapping) and _admits_null(branch):
                return True
    return False


def _harden_object(node: Dict[str, Any], model_name: str, pointer: str) -> None:
    """ensure an object node satisfies strict JSON schema structured output"""

    properties = node.get("properties")
    if not isinstance(properties, dict):
        return

    node["additionalProperties"] = False
    declared_required = set(node.get("required") or ())
    for prop_name, prop_schema in properties.items():
        if prop_name in declared_required:
            continue
        if not isinstance(prop_schema, Mapping) or not _admits_null(prop_schema):
            where = f"{pointer}/{prop_name}" if pointer else prop_name
            raise LLMSchemaError(
                model_name,
                (
                    f"field '{where}' is optional but cannot be null, so it cannot "
                    "be expressed in strict json-schema mode. Declare it as "
                    "Optional[...] = None, make it required, or select the "
                    "'json-object' structured-output mode explicitly."
                ),
            )
    node["required"] = list(properties)


def _harden(node: Any, model_name: str, pointer: str = "") -> None:
    """Recursively apply strict-mode requirements in place."""

    if isinstance(node, dict):
        node.pop("default", None)
        if node.get("type") == "object" or "properties" in node:
            _harden_object(node, model_name, pointer)
        for key, child in node.items():
            if key in ("properties", "$defs", "definitions") and isinstance(child, dict):
                for child_name, child_schema in child.items():
                    _harden(child_schema, model_name, f"{pointer}/{child_name}".lstrip("/"))
            elif key == "items":
                _harden(child, model_name, f"{pointer}[]")
            elif key in ("anyOf", "oneOf", "allOf") and isinstance(child, list):
                for branch in child:
                    _harden(branch, model_name, pointer)
    elif isinstance(node, list):
        for item in node:
            _harden(item, model_name, pointer)


def json_schema_for(model: Type[BaseModel], *, strict: bool) -> Dict[str, Any]:
    """return JSON schema of a model"""

    schema = model.model_json_schema()
    if strict:
        _harden(schema, model.__name__)
    return schema


def schema_digest(model: Type[BaseModel]) -> str:
    """retrieve model schema"""

    payload = json.dumps(
        model.model_json_schema(), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def response_format_for(
    model: Type[BaseModel], mode: StructuredOutputMode
) -> Optional[Dict[str, Any]]:
    """create compatible response format for mode for openai compatible endpoint"""

    if mode is StructuredOutputMode.JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name(model),
                "strict": True,
                "schema": json_schema_for(model, strict=True),
            },
        }
    if mode is StructuredOutputMode.JSON_OBJECT:
        return {"type": "json_object"}
    return None


def schema_instructions(model: Type[BaseModel]) -> str:
    """Render schema guidance for prompt-constrained mode."""

    schema = json.dumps(json_schema_for(model, strict=False), indent=2, default=str)
    return (
        "Respond with a single JSON document and nothing else. "
        "It must validate against this JSON Schema:\n"
        f"{schema}"
    )


def messages_with_output_contract(
    messages: Sequence[LLMMessage],
    model: Type[BaseModel],
    mode: StructuredOutputMode,
) -> List[LLMMessage]:
    """return exact messages to send for the corresponding structured output mode"""

    if mode is not StructuredOutputMode.PROMPT:
        return list(messages)

    instruction = schema_instructions(model)
    prepared = list(messages)
    for index in range(len(prepared) - 1, -1, -1):
        if prepared[index].role is Role.USER:
            prepared[index] = LLMMessage(
                Role.USER, f"{prepared[index].content}\n\n{instruction}"
            )
            return prepared

    prepared.append(LLMMessage.user(instruction))
    return prepared


def _extract_json_text(text: str) -> str:
    """extract the JSON of a returned document of a prompt-based output"""

    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    fenced = _FENCED_JSON_RE.search(cleaned)
    if fenced:
        return fenced.group(1).strip()

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            return cleaned[start : end + 1]
    return cleaned


def parse_structured_response(
    text: str, model: Type[TModel], mode: StructuredOutputMode
) -> TModel:
    """parse/validated the given text against the given structured output mode"""

    if text is None or not text.strip():
        raise LLMResponseValidationError("The provider returned an empty response.", text)

    candidate = (
        _extract_json_text(text) if mode is StructuredOutputMode.PROMPT else text.strip()
    )

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMResponseValidationError(
            f"Response was not valid JSON ({mode.value} mode): {exc}", text
        ) from exc

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseValidationError(
            f"Response did not satisfy the '{model.__name__}' schema: {exc}", text
        ) from exc
