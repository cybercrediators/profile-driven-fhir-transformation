"""Backward-compatible rule-IR primitives used by mapping-table values.

The original contract is a JSON object of ``source-path -> target-path``
strings. Collection rules extend a value of that object instead of replacing
the format, so existing projects and automapping output remain valid.
"""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any, Optional, Tuple

from fhir.resources.R4B.structuremap import (
    StructureMapGroup,
    StructureMapGroupInput,
    StructureMapGroupRule,
    StructureMapGroupRuleDependent,
    StructureMapGroupRuleSource,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
    StructureMapStructure,
)


SOURCE_LIST_MODES = frozenset(
    {"first", "not_first", "last", "not_last", "only_one"}
)
TARGET_LIST_MODES = frozenset({"first", "share", "last", "collate"})
_FHIR_ID = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")
RULE_DOCUMENT_KEYS = frozenset(
    {"$imports", "$structures", "$groups", "$rules", "$references"}
)
STRUCTURE_MODES = frozenset({"source", "queried", "target", "produced"})
GROUP_TYPE_MODES = frozenset({"none", "types", "type-and-types"})
MAPPING_TRANSFORMS = frozenset(
    {
        "create",
        "copy",
        "truncate",
        "escape",
        "cast",
        "append",
        "translate",
        "reference",
        "dateOp",
        "uuid",
        "pointer",
        "evaluate",
        "cc",
        "c",
        "qty",
        "id",
        "cp",
    }
)
REFERENCE_STRATEGIES = frozenset(
    {"direct", "bundled", "conditional", "contained", "canonical"}
)
REFERENCE_MATCH_POLICIES = frozenset({"byOrder", "singleton", "identifier", "all"})
REFERENCE_VALUE_MODES = frozenset({"urn", "relative"})
_COLLECTION_PROPERTIES = frozenset(
    {
        "target",
        "sourceListMode",
        "targetListMode",
        "listRuleId",
        "correlation",
    }
)


@dataclass(frozen=True)
class CollectionRuleSpec:
    """Normalized collection rule attached to a source/target parent pair."""

    source: str
    target: str
    source_list_mode: Optional[str] = None
    target_list_modes: Tuple[str, ...] = ()
    list_rule_id: Optional[str] = None
    source_key: Optional[str] = None
    target_key: Optional[str] = None
    invalid: bool = False


@dataclass
class CompiledRuleDocument:
    """Profile-scoped additions compiled from the typed mapping-table envelope."""

    imports: list[str] = field(default_factory=list)
    structures: list[StructureMapStructure] = field(default_factory=list)
    primary_rules: list[StructureMapGroupRule] = field(default_factory=list)
    groups: list[StructureMapGroup] = field(default_factory=list)
    reference_paths: set[str] = field(default_factory=set)
    diagnostics: list[dict] = field(default_factory=list)


class RuleIRValidationError(ValueError):
    """An authored typed-rule declaration cannot be represented safely."""


def mapping_target_path(value: Any) -> Optional[str]:
    """Return the target path from a legacy or typed mapping-table value."""

    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        target = value.get("target")
        return target if isinstance(target, str) and target else None
    return None


def _default_list_rule_id(target: str) -> str:
    """Create a deterministic FHIR ``id`` for a target share group."""

    readable = re.sub(r"[^A-Za-z0-9\-.]", "-", target).strip("-.")
    candidate = f"collection-{readable}" if readable else "collection"
    if len(candidate) <= 64:
        return candidate
    digest = hashlib.sha1(target.encode("utf-8")).hexdigest()[:10]
    return f"{candidate[:53]}-{digest}"


def parse_collection_rule(source: str, value: Any) -> Optional[CollectionRuleSpec]:
    """Parse one typed collection value.

    ``None`` means that ``value`` is a legacy mapping. Invalid typed rules raise
    ``ValueError`` so callers can diagnose them without inventing semantics.
    """

    if not isinstance(value, dict):
        return None

    unknown = sorted(set(value) - _COLLECTION_PROPERTIES)
    if unknown:
        raise ValueError(
            "unsupported collection rule properties: " + ", ".join(unknown)
        )

    target = mapping_target_path(value)
    if not target:
        raise ValueError("collection rule requires a non-empty string 'target'")

    source_mode = value.get("sourceListMode")
    if source_mode is not None and source_mode not in SOURCE_LIST_MODES:
        raise ValueError(
            f"sourceListMode must be one of {sorted(SOURCE_LIST_MODES)}, "
            f"got {source_mode!r}"
        )

    raw_target_modes = value.get("targetListMode")
    if raw_target_modes is None:
        target_modes = ()
    elif isinstance(raw_target_modes, str):
        target_modes = (raw_target_modes,)
    elif isinstance(raw_target_modes, list) and all(
        isinstance(mode, str) for mode in raw_target_modes
    ):
        target_modes = tuple(raw_target_modes)
    else:
        raise ValueError("targetListMode must be a string or a list of strings")

    invalid_modes = sorted(set(target_modes) - TARGET_LIST_MODES)
    if invalid_modes:
        raise ValueError(
            f"targetListMode must contain only {sorted(TARGET_LIST_MODES)}, "
            f"got {invalid_modes}"
        )
    if len(set(target_modes)) != len(target_modes):
        raise ValueError("targetListMode must not contain duplicate modes")

    list_rule_id = value.get("listRuleId")
    if list_rule_id is not None and (
        not isinstance(list_rule_id, str) or not _FHIR_ID.fullmatch(list_rule_id)
    ):
        raise ValueError("listRuleId must be a valid FHIR id (1..64 characters)")
    if list_rule_id and "share" not in target_modes:
        raise ValueError("listRuleId is only meaningful with targetListMode 'share'")
    if "share" in target_modes and not list_rule_id:
        list_rule_id = _default_list_rule_id(target)

    source_key = target_key = None
    correlation = value.get("correlation")
    if isinstance(correlation, str):
        source_key = target_key = correlation
    elif correlation is not None:
        if not isinstance(correlation, dict):
            raise ValueError("correlation must be a string or an object")
        unknown_correlation = sorted(
            set(correlation) - {"sourceKey", "targetKey"}
        )
        if unknown_correlation:
            raise ValueError(
                "unsupported correlation properties: "
                + ", ".join(unknown_correlation)
            )
        source_key = correlation.get("sourceKey")
        target_key = correlation.get("targetKey")
        if not isinstance(source_key, str) or not source_key:
            raise ValueError("correlation.sourceKey must be a non-empty string")
        if target_key is not None and (
            not isinstance(target_key, str) or not target_key
        ):
            raise ValueError(
                "correlation.targetKey must be a non-empty string when present"
            )

    return CollectionRuleSpec(
        source=source,
        target=target,
        source_list_mode=source_mode,
        target_list_modes=target_modes,
        list_rule_id=list_rule_id,
        source_key=source_key,
        target_key=target_key,
    )


def is_rule_document_key(key: Any) -> bool:
    return isinstance(key, str) and key in RULE_DOCUMENT_KEYS


def _require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _FHIR_ID.fullmatch(value):
        raise RuleIRValidationError(
            f"{label} must be a valid FHIR id (1..64 characters)"
        )
    return value


def _as_string_list(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise RuleIRValidationError(f"{label} must be a string or list of strings")
    return value


def _scope_matches(
    declaration: dict,
    profile_id: str,
    profile_url: str,
    resource_type: str,
) -> bool:
    profiles = declaration.get("profiles")
    if profiles is not None:
        profiles = _as_string_list(profiles, "profiles")
        canonical = (profile_url or "").split("|", 1)[0]
        profile_names = {
            profile_id,
            canonical,
            canonical.rstrip("/").rsplit("/", 1)[-1] if canonical else "",
        }
        if not profile_names.intersection(profiles):
            return False
    resource_types = declaration.get("resourceTypes")
    if resource_types is not None:
        resource_types = _as_string_list(resource_types, "resourceTypes")
        if resource_type not in resource_types:
            return False
    return True


def _check_unknown(raw: dict, allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise RuleIRValidationError(
            f"unsupported {label} properties: {', '.join(unknown)}"
        )


_PARAMETER_FIELDS = {
    "valueId",
    "valueString",
    "valueBoolean",
    "valueInteger",
    "valueDecimal",
}


def _compile_parameter(raw: Any) -> StructureMapGroupRuleTargetParameter:
    if isinstance(raw, str):
        raw = {"valueString": raw}
    if not isinstance(raw, dict):
        raise RuleIRValidationError(
            "target parameters must be strings or single-value objects"
        )
    unknown = sorted(set(raw) - _PARAMETER_FIELDS)
    populated = [key for key in _PARAMETER_FIELDS if key in raw]
    if unknown or len(populated) != 1:
        raise RuleIRValidationError(
            "a target parameter must contain exactly one of "
            + ", ".join(sorted(_PARAMETER_FIELDS))
        )
    key = populated[0]
    value = raw[key]
    if key in ("valueId", "valueString") and not isinstance(value, str):
        raise RuleIRValidationError(f"{key} must be a string")
    if key == "valueBoolean" and not isinstance(value, bool):
        raise RuleIRValidationError("valueBoolean must be boolean")
    if key == "valueInteger" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        raise RuleIRValidationError("valueInteger must be an integer")
    if key == "valueDecimal":
        if not isinstance(value, (int, float, str, Decimal)) or isinstance(
            value, bool
        ):
            raise RuleIRValidationError("valueDecimal must be numeric")
        try:
            value = Decimal(str(value))
        except InvalidOperation as exc:
            raise RuleIRValidationError("valueDecimal must be numeric") from exc
    return StructureMapGroupRuleTargetParameter.model_construct(**{key: value})


_SOURCE_PROPERTIES = {
    "context",
    "element",
    "variable",
    "condition",
    "check",
    "logMessage",
    "listMode",
    "min",
    "max",
    "type",
}


def _compile_source(raw: Any, index: int) -> StructureMapGroupRuleSource:
    if isinstance(raw, str):
        raw = {"element": raw}
    if not isinstance(raw, dict):
        raise RuleIRValidationError("rule sources must be strings or objects")
    allowed = set(_SOURCE_PROPERTIES)
    allowed.update(
        key
        for key in StructureMapGroupRuleSource.model_fields
        if key.startswith("defaultValue")
    )
    _check_unknown(raw, allowed, "source")
    context = raw.get("context", "source")
    _require_id(context, "source.context")
    variable = raw.get("variable")
    if variable is not None:
        _require_id(variable, "source.variable")
    list_mode = raw.get("listMode")
    if list_mode is not None and list_mode not in SOURCE_LIST_MODES:
        raise RuleIRValidationError(
            f"source.listMode must be one of {sorted(SOURCE_LIST_MODES)}"
        )
    values = dict(raw)
    values["context"] = context
    if not variable and raw.get("element"):
        values["variable"] = f"src-{index}"
    return StructureMapGroupRuleSource.model_construct(**values)


_TARGET_PROPERTIES = {
    "context",
    "contextType",
    "element",
    "variable",
    "transform",
    "parameters",
    "parameter",
    "targetListMode",
    "listMode",
    "listRuleId",
}


def _compile_target(raw: Any) -> StructureMapGroupRuleTarget:
    if isinstance(raw, str):
        raw = {"element": raw}
    if not isinstance(raw, dict):
        raise RuleIRValidationError("rule targets must be strings or objects")
    _check_unknown(raw, _TARGET_PROPERTIES, "target")
    values = dict(raw)
    values["context"] = values.get("context", "target")
    _require_id(values["context"], "target.context")
    values["contextType"] = values.get("contextType", "variable")
    if values["contextType"] not in ("type", "variable"):
        raise RuleIRValidationError("target.contextType must be 'type' or 'variable'")
    variable = values.get("variable")
    if variable is not None:
        _require_id(variable, "target.variable")
    transform = values.get("transform")
    raw_params = values.pop("parameters", values.pop("parameter", []))
    if transform is not None and transform not in MAPPING_TRANSFORMS:
        raise RuleIRValidationError(
            f"unsupported StructureMap transform {transform!r}"
        )
    if raw_params and transform is None:
        raise RuleIRValidationError("target parameters require a transform")
    if not isinstance(raw_params, list):
        raise RuleIRValidationError("target parameters must be a list")
    if raw_params:
        values["parameter"] = [_compile_parameter(item) for item in raw_params]

    raw_modes = values.pop("targetListMode", values.pop("listMode", None))
    if raw_modes is not None:
        modes = _as_string_list(raw_modes, "targetListMode")
        invalid = sorted(set(modes) - TARGET_LIST_MODES)
        if invalid:
            raise RuleIRValidationError(
                f"targetListMode contains unsupported modes {invalid}"
            )
        if len(set(modes)) != len(modes):
            raise RuleIRValidationError("targetListMode must not contain duplicates")
        values["listMode"] = modes
    list_rule_id = values.get("listRuleId")
    if list_rule_id is not None:
        _require_id(list_rule_id, "target.listRuleId")
        if "share" not in (values.get("listMode") or []):
            raise RuleIRValidationError(
                "target.listRuleId requires targetListMode 'share'"
            )
    return StructureMapGroupRuleTarget.model_construct(**values)


def _compile_dependent(raw: Any) -> StructureMapGroupRuleDependent:
    if not isinstance(raw, dict):
        raise RuleIRValidationError("dependent declarations must be objects")
    _check_unknown(raw, {"name", "variables", "variable"}, "dependent")
    name = _require_id(raw.get("name"), "dependent.name")
    variables = raw.get("variables", raw.get("variable"))
    variables = _as_string_list(variables, "dependent.variables")
    return StructureMapGroupRuleDependent.model_construct(
        name=name, variable=variables
    )


_RULE_PROPERTIES = {
    "name",
    "sources",
    "source",
    "targets",
    "target",
    "rules",
    "dependent",
    "documentation",
    "profiles",
    "resourceTypes",
}


def compile_typed_rule(raw: Any, default_name: str = "typed-rule"):
    if not isinstance(raw, dict):
        raise RuleIRValidationError("typed rules must be objects")
    _check_unknown(raw, _RULE_PROPERTIES, "rule")
    name = _require_id(raw.get("name", default_name), "rule.name")

    raw_sources = raw.get("sources", raw.get("source"))
    if raw_sources is None:
        raw_sources = [{}]
    elif not isinstance(raw_sources, list):
        raw_sources = [raw_sources]
    if not raw_sources:
        raise RuleIRValidationError("a StructureMap rule requires at least one source")
    sources = [
        _compile_source(source, index)
        for index, source in enumerate(raw_sources, start=1)
    ]

    raw_targets = raw.get("targets", raw.get("target"))
    if raw_targets is None:
        raw_targets = []
    elif not isinstance(raw_targets, list):
        raw_targets = [raw_targets]
    targets = [_compile_target(target) for target in raw_targets]

    nested = [
        compile_typed_rule(child, f"{name}-nested-{index}")
        for index, child in enumerate(raw.get("rules") or [], start=1)
    ]
    dependent = [
        _compile_dependent(item) for item in (raw.get("dependent") or [])
    ]
    if not targets and not nested and not dependent:
        raise RuleIRValidationError(
            "a typed rule requires a target, nested rule, or dependent group"
        )
    return StructureMapGroupRule.model_construct(
        name=name,
        source=sources,
        target=targets or None,
        rule=nested or None,
        dependent=dependent or None,
        documentation=raw.get("documentation"),
    )


_GROUP_PROPERTIES = {
    "name",
    "typeMode",
    "extends",
    "inputs",
    "rules",
    "documentation",
    "profiles",
    "resourceTypes",
}


def _compile_group(raw: Any) -> StructureMapGroup:
    if not isinstance(raw, dict):
        raise RuleIRValidationError("group declarations must be objects")
    _check_unknown(raw, _GROUP_PROPERTIES, "group")
    name = _require_id(raw.get("name"), "group.name")
    type_mode = raw.get("typeMode", "none")
    if type_mode not in GROUP_TYPE_MODES:
        raise RuleIRValidationError(
            f"group.typeMode must be one of {sorted(GROUP_TYPE_MODES)}"
        )
    inputs = []
    for item in raw.get("inputs") or []:
        if not isinstance(item, dict):
            raise RuleIRValidationError("group inputs must be objects")
        _check_unknown(item, {"name", "mode", "type", "documentation"}, "group input")
        input_name = _require_id(item.get("name"), "group input.name")
        mode = item.get("mode")
        if mode not in ("source", "target"):
            raise RuleIRValidationError(
                "group input.mode must be 'source' or 'target'"
            )
        inputs.append(
            StructureMapGroupInput.model_construct(
                name=input_name,
                mode=mode,
                type=item.get("type"),
                documentation=item.get("documentation"),
            )
        )
    if not inputs:
        raise RuleIRValidationError("a typed group requires at least one input")
    rules = [
        compile_typed_rule(item, f"{name}-rule-{index}")
        for index, item in enumerate(raw.get("rules") or [], start=1)
    ]
    if not rules:
        raise RuleIRValidationError("a typed group requires at least one rule")
    extends = raw.get("extends")
    if extends is not None:
        _require_id(extends, "group.extends")
    return StructureMapGroup.model_construct(
        name=name,
        typeMode=type_mode,
        extends=extends,
        input=inputs,
        rule=rules,
        documentation=raw.get("documentation"),
    )


def _local_source_element(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _reference_target_location(raw: dict, resource_type: str) -> tuple[str, str]:
    context = raw.get("targetContext", "target")
    _require_id(context, "reference.targetContext")
    element = raw.get("element")
    path = raw.get("path")
    if not element:
        if not isinstance(path, str) or not path:
            raise RuleIRValidationError("reference.path must be a non-empty string")
        relative = (
            path[len(resource_type) + 1 :]
            if path.startswith(f"{resource_type}.")
            else path
        )
        if (
            raw.get("strategy", "direct") != "bundled"
            and "." in relative
            and "targetContext" not in raw
        ):
            raise RuleIRValidationError(
                "a nested reference path requires targetContext and element"
            )
        element = relative.rsplit(".", 1)[-1]
        if element.endswith("[x]"):
            element = element[: -len("[x]")] + "Reference"
    if not isinstance(element, str) or not element:
        raise RuleIRValidationError(
            "reference.element must be a non-empty string"
        )
    return context, element


def _copy_child(
    name: str,
    source_context: str,
    target_context: str,
    target_element: str,
    *,
    source_element: str = None,
    literal: str = None,
    transform: str = "copy",
    prefix: str = None,
) -> StructureMapGroupRule:
    source = StructureMapGroupRuleSource.model_construct(context=source_context)
    source_var = f"src-{name}"
    if source_element:
        source.element = source_element
        source.variable = source_var
    target = StructureMapGroupRuleTarget.model_construct(
        context=target_context,
        contextType="variable",
        element=target_element,
        transform=transform,
    )
    if transform == "append":
        params = []
        if prefix is not None:
            params.append(
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString=prefix
                )
            )
        if source_element:
            params.append(
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueId=source_var
                )
            )
        elif literal is not None:
            params.append(
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString=literal
                )
            )
        target.parameter = params
    elif literal is not None:
        target.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString=literal
            )
        ]
    elif source_element:
        target.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(valueId=source_var)
        ]
    return StructureMapGroupRule.model_construct(
        name=name, source=[source], target=[target]
    )


_REFERENCE_PROPERTIES = {
    "name",
    "path",
    "element",
    "targetContext",
    "strategy",
    "source",
    "displaySource",
    "literal",
    "targetType",
    "targetTypes",
    "targetProfile",
    "identifierSystem",
    "match",
    "sourceKey",
    "targetKey",
    "referenceMode",
    "containedType",
    "containedIdSource",
    "dependentGroup",
    "sourceListMode",
    "targetListMode",
    "listRuleId",
    "profiles",
    "resourceTypes",
}


def _compile_reference(
    raw: Any, resource_type: str, index: int
) -> tuple[StructureMapGroupRule, str]:
    if not isinstance(raw, dict):
        raise RuleIRValidationError("reference declarations must be objects")
    _check_unknown(raw, _REFERENCE_PROPERTIES, "reference")
    strategy = raw.get("strategy", "direct")
    if strategy not in REFERENCE_STRATEGIES:
        raise RuleIRValidationError(
            f"reference.strategy must be one of {sorted(REFERENCE_STRATEGIES)}"
        )
    context, element = _reference_target_location(raw, resource_type)
    path = raw.get("path") or element
    relative_path = (
        path[len(resource_type) + 1 :]
        if isinstance(path, str) and path.startswith(f"{resource_type}.")
        else path
    )
    name = _require_id(
        raw.get("name", f"reference-{index}-{re.sub(r'[^A-Za-z0-9.-]', '-', element)}"),
        "reference.name",
    )
    source_path = raw.get("source")
    literal = raw.get("literal")
    if source_path is not None and (
        not isinstance(source_path, str) or not source_path
    ):
        raise RuleIRValidationError("reference.source must be a non-empty string")
    if literal is not None and not isinstance(literal, str):
        raise RuleIRValidationError("reference.literal must be a string")
    if strategy in ("direct", "conditional", "canonical", "contained") and not (
        source_path or literal
    ):
        raise RuleIRValidationError(
            f"reference strategy {strategy!r} requires source or literal"
        )
    target_types = raw.get("targetTypes", raw.get("targetType"))
    if target_types is not None:
        target_types = _as_string_list(target_types, "reference.targetTypes")
    else:
        target_types = []
    target_profile = raw.get("targetProfile")
    if target_profile is not None and (
        not isinstance(target_profile, str) or not target_profile
    ):
        raise RuleIRValidationError(
            "reference.targetProfile must be a non-empty string"
        )
    for key in ("displaySource", "identifierSystem", "containedIdSource"):
        value = raw.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise RuleIRValidationError(
                f"reference.{key} must be a non-empty string"
            )

    if strategy == "bundled":
        ignored_list_options = [
            key
            for key in ("sourceListMode", "targetListMode", "listRuleId")
            if raw.get(key) is not None
        ]
        if ignored_list_options:
            raise RuleIRValidationError(
                "bundled references use the explicit match policy instead of "
                + ", ".join(ignored_list_options)
            )
        match = raw.get("match", "byOrder")
        if match not in REFERENCE_MATCH_POLICIES:
            raise RuleIRValidationError(
                f"reference.match must be one of {sorted(REFERENCE_MATCH_POLICIES)}"
            )
        if not target_types:
            raise RuleIRValidationError(
                "bundled references require targetType or targetTypes"
            )
        if match == "identifier" and not (
            isinstance(raw.get("sourceKey"), str)
            and isinstance(raw.get("targetKey"), str)
        ):
            raise RuleIRValidationError(
                "identifier reference matching requires sourceKey and targetKey"
            )
        reference_mode = raw.get("referenceMode", "urn")
        if reference_mode not in REFERENCE_VALUE_MODES:
            raise RuleIRValidationError(
                f"referenceMode must be one of {sorted(REFERENCE_VALUE_MODES)}"
            )
        contract = {
            "sourceType": resource_type,
            "path": relative_path,
            "targetTypes": target_types,
            "targetProfile": target_profile,
            "match": match,
            "sourceKey": raw.get("sourceKey"),
            "targetKey": raw.get("targetKey"),
            "referenceMode": reference_mode,
        }
        documentation = "FHIRBRIDGE_REFERENCE:" + json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        )
        rule = StructureMapGroupRule.model_construct(
            name=f"TODO-resolve-reference-{name}",
            source=[
                StructureMapGroupRuleSource.model_construct(context="source")
            ],
            documentation=documentation,
        )
        return rule, relative_path

    if strategy == "canonical":
        source = StructureMapGroupRuleSource.model_construct(context="source")
        source_var = "src-canonical"
        if source_path:
            source.element = _local_source_element(source_path)
            source.variable = source_var
        target = StructureMapGroupRuleTarget.model_construct(
            context=context,
            contextType="variable",
            element=element,
            transform="copy",
            parameter=[
                StructureMapGroupRuleTargetParameter.model_construct(
                    **(
                        {"valueId": source_var}
                        if source_path
                        else {"valueString": literal}
                    )
                )
            ],
        )
        return (
            StructureMapGroupRule.model_construct(
                name=name, source=[source], target=[target]
            ),
            relative_path,
        )

    source_context = "source"
    outer_source = StructureMapGroupRuleSource.model_construct(
        context=source_context, variable=f"src-{name}-context"
    )
    target_var = f"tgt-{name}"
    reference_target = StructureMapGroupRuleTarget.model_construct(
        context=context,
        contextType="variable",
        element=element,
        variable=target_var,
        transform="create",
        parameter=[
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString="Reference"
            )
        ],
    )
    raw_target_modes = raw.get("targetListMode")
    if raw_target_modes is not None:
        modes = _as_string_list(raw_target_modes, "reference.targetListMode")
        invalid = sorted(set(modes) - TARGET_LIST_MODES)
        if invalid:
            raise RuleIRValidationError(
                f"reference.targetListMode contains unsupported modes {invalid}"
            )
        reference_target.listMode = modes
    if raw.get("listRuleId"):
        reference_target.listRuleId = _require_id(
            raw["listRuleId"], "reference.listRuleId"
        )
        if "share" not in (reference_target.listMode or []):
            raise RuleIRValidationError(
                "reference.listRuleId requires targetListMode 'share'"
            )
    if raw.get("sourceListMode"):
        if raw["sourceListMode"] not in SOURCE_LIST_MODES:
            raise RuleIRValidationError(
                f"reference.sourceListMode must be one of {sorted(SOURCE_LIST_MODES)}"
            )
        outer_source.listMode = raw["sourceListMode"]

    reference_source_path = source_path
    if strategy == "contained" and raw.get("containedIdSource"):
        reference_source_path = raw["containedIdSource"]
    source_element = (
        _local_source_element(reference_source_path)
        if reference_source_path
        else None
    )
    if strategy == "conditional":
        if not target_types:
            raise RuleIRValidationError(
                "conditional references require targetType or targetTypes"
            )
        prefix = f"{target_types[0]}?identifier="
        if raw.get("identifierSystem"):
            prefix += f"{raw['identifierSystem']}|"
        reference_child = _copy_child(
            f"{name}-reference",
            outer_source.variable,
            target_var,
            "reference",
            source_element=source_element,
            literal=literal,
            transform="append",
            prefix=prefix,
        )
    else:
        reference_child = _copy_child(
            f"{name}-reference",
            outer_source.variable,
            target_var,
            "reference",
            source_element=source_element,
            literal=literal,
            transform="append" if strategy == "contained" else "copy",
            prefix="#" if strategy == "contained" else None,
        )
    nested = [reference_child]
    display_source = raw.get("displaySource")
    if display_source:
        nested.append(
            _copy_child(
                f"{name}-display",
                outer_source.variable,
                target_var,
                "display",
                source_element=_local_source_element(display_source),
            )
        )

    targets = [reference_target]
    dependents = []
    if strategy == "contained":
        contained_type = raw.get("containedType")
        if not isinstance(contained_type, str) or not contained_type:
            raise RuleIRValidationError(
                "contained references require containedType"
            )
        contained_var = f"tgt-{name}-contained"
        targets.insert(
            0,
            StructureMapGroupRuleTarget.model_construct(
                context=context,
                contextType="variable",
                element="contained",
                variable=contained_var,
                transform="create",
                parameter=[
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueString=contained_type
                    )
                ],
            ),
        )
        nested.insert(
            0,
            _copy_child(
                f"{name}-contained-id",
                outer_source.variable,
                contained_var,
                "id",
                source_element=_local_source_element(
                    raw.get("containedIdSource") or source_path
                )
                if source_path
                else None,
                literal=literal,
            ),
        )
        dependent_group = raw.get("dependentGroup")
        if dependent_group:
            dependents.append(
                StructureMapGroupRuleDependent.model_construct(
                    name=_require_id(
                        dependent_group, "reference.dependentGroup"
                    ),
                    variable=[outer_source.variable, contained_var],
                )
            )
    return (
        StructureMapGroupRule.model_construct(
            name=name,
            source=[outer_source],
            target=targets,
            rule=nested,
            dependent=dependents or None,
            documentation=f"Explicit {strategy} reference for {resource_type}.{relative_path}",
        ),
        relative_path,
    )


def compile_rule_document(
    mapping_table: Any,
    *,
    profile_id: str,
    profile_url: str,
    resource_type: str,
) -> CompiledRuleDocument:
    """Compile the optional document-level typed envelope.

    Legacy ``source -> target`` entries and collection-rule objects are ignored
    here and continue through the existing field mapper.
    """
    result = CompiledRuleDocument()
    if not isinstance(mapping_table, dict):
        return result

    def _diagnose(section: str, index: int, exc: Exception):
        result.diagnostics.append(
            {
                "code": "invalid-typed-rule",
                "message": str(exc),
                "section": section,
                "index": index,
                "severity": "error",
            }
        )

    def _section(name: str) -> list:
        raw = mapping_table.get(name, [])
        if raw is None:
            return []
        if not isinstance(raw, list):
            _diagnose(
                name,
                0,
                RuleIRValidationError(f"{name} must be a list"),
            )
            return []
        return raw

    imports = mapping_table.get("$imports", [])
    if imports:
        try:
            result.imports = list(dict.fromkeys(_as_string_list(imports, "$imports")))
        except RuleIRValidationError as exc:
            _diagnose("$imports", 0, exc)

    for index, raw in enumerate(_section("$structures")):
        try:
            if not isinstance(raw, dict):
                raise RuleIRValidationError("structure declarations must be objects")
            _check_unknown(
                raw,
                {
                    "url",
                    "mode",
                    "alias",
                    "documentation",
                    "profiles",
                    "resourceTypes",
                },
                "structure",
            )
            if not _scope_matches(raw, profile_id, profile_url, resource_type):
                continue
            url = raw.get("url")
            mode = raw.get("mode")
            if not isinstance(url, str) or not url:
                raise RuleIRValidationError("structure.url must be a non-empty string")
            if mode not in STRUCTURE_MODES:
                raise RuleIRValidationError(
                    f"structure.mode must be one of {sorted(STRUCTURE_MODES)}"
                )
            result.structures.append(
                StructureMapStructure.model_construct(
                    url=url,
                    mode=mode,
                    alias=raw.get("alias"),
                    documentation=raw.get("documentation"),
                )
            )
        except (RuleIRValidationError, TypeError) as exc:
            _diagnose("$structures", index, exc)

    for index, raw in enumerate(_section("$rules")):
        try:
            if not isinstance(raw, dict):
                raise RuleIRValidationError("typed rules must be objects")
            if not _scope_matches(raw, profile_id, profile_url, resource_type):
                continue
            result.primary_rules.append(
                compile_typed_rule(raw, f"typed-rule-{index + 1}")
            )
        except (RuleIRValidationError, TypeError) as exc:
            _diagnose("$rules", index, exc)

    for index, raw in enumerate(_section("$groups")):
        try:
            if not isinstance(raw, dict):
                raise RuleIRValidationError("group declarations must be objects")
            if not _scope_matches(raw, profile_id, profile_url, resource_type):
                continue
            result.groups.append(_compile_group(raw))
        except (RuleIRValidationError, TypeError) as exc:
            _diagnose("$groups", index, exc)

    for index, raw in enumerate(_section("$references")):
        try:
            if not isinstance(raw, dict):
                raise RuleIRValidationError("reference declarations must be objects")
            if not _scope_matches(raw, profile_id, profile_url, resource_type):
                continue
            rule, path = _compile_reference(raw, resource_type, index + 1)
            result.primary_rules.append(rule)
            result.reference_paths.add(path)
        except (RuleIRValidationError, TypeError) as exc:
            _diagnose("$references", index, exc)

    return result
