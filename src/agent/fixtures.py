"""fixed dummy input values for in-the-loop matchbox validation"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class FixtureKind(str, Enum):
    SYNTHETIC_FILLED = "synthetic-filled"
    SYNTHETIC_EMPTY_OPTIONAL = "synthetic-empty-optional"
    EXAMPLE_DERIVED = "example-derived"


TYPE_DEFAULTS: Dict[str, Any] = {
    "string": "1",
    "code": "1",
    "id": "1",
    "markdown": "1",
    "integer": 1,
    "positiveInt": 1,
    "unsignedInt": 1,
    "decimal": 1.5,
    "boolean": True,
    "date": "2024-01-15",
    "dateTime": "2024-01-15T10:30:00+01:00",
    "instant": "2024-01-15T10:30:00.000+01:00",
    "time": "10:30:00",
    "uri": "http://example.org/value",
    "url": "http://example.org/value",
    "canonical": "http://example.org/value",
    "oid": "urn:oid:1.2.3.4",
    "uuid": "urn:uuid:9d8f2a3e-0d0f-4f6a-9c1a-1f0a2b3c4d5e",
}

_EMPTYABLE_TYPES = frozenset(
    {"string", "code", "id", "markdown", "uri", "url", "canonical",
     "date", "dateTime", "instant", "time"}
)


def translate_source_fields(documents: Iterable[Mapping[str, Any]]) -> Set[str]:
    """Source fields whose value is handed to a translate()"""

    wanted: Set[str] = set()
    bindings: Dict[str, Set[str]] = {}

    def walk(rules: Any) -> None:
        for rule in rules or []:
            if not isinstance(rule, Mapping):
                continue
            for source in rule.get("source") or []:
                if not isinstance(source, Mapping) or not source.get("variable"):
                    continue
                variable = str(source["variable"])
                if source.get("element"):
                    bindings.setdefault(variable, set()).add(str(source["element"]))
                    continue
                inherited = bindings.get(str(source.get("context") or ""))
                if inherited:
                    bindings.setdefault(variable, set()).update(inherited)
            for target in rule.get("target") or []:
                if not isinstance(target, Mapping) or target.get("transform") != "translate":
                    continue
                for parameter in target.get("parameter") or []:
                    if isinstance(parameter, Mapping) and parameter.get("valueId"):
                        wanted.add(str(parameter["valueId"]))
            walk(rule.get("rule"))

    for document in documents:
        for group in document.get("group") or []:
            if isinstance(group, Mapping):
                walk(group.get("rule"))

    return {element for name in wanted for element in bindings.get(name, ())}


_TEMPORAL_TYPES = frozenset({"date", "dateTime", "instant", "time"})


class SourceFixture(BaseModel):
    """One executable input, with everything needed to judge its output."""

    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    kind: FixtureKind
    label: str
    instance: Dict[str, Any]
    
    required: bool = True

    verified_fields: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)

    @property
    def gate_relevant(self) -> bool:
        return self.required


def _content_id(prefix: str, payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def fixtures_digest(fixtures: Sequence[SourceFixture]) -> str:
    """go over everything about a fixture that can change a finding"""

    return _content_id(
        "fx",
        [
            {
                "kind": fixture.kind.value,
                "instance": fixture.instance,
                "required": fixture.required,
                "verified_fields": sorted(fixture.verified_fields),
            }
            for fixture in fixtures
        ],
    )

def source_field_specs(source_model: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """flat field specs from a source logical StructureDefinition

    :return: [{"name", "min", "max", "type"}] in declaration order.
    """

    elements = (
        (source_model.get("snapshot") or {}).get("element")
        or (source_model.get("differential") or {}).get("element")
        or []
    )
    specs: List[Dict[str, Any]] = []
    seen = set()
    for element in elements:
        path = str(element.get("path") or "")
        if "." not in path:
            continue
        name = path.split(".", 1)[1]
        if "." in name or name in seen:
            continue
        seen.add(name)
        types = [
            str(entry.get("code"))
            for entry in (element.get("type") or [])
            if isinstance(entry, dict) and entry.get("code")
        ]
        try:
            minimum = int(element.get("min") or 0)
        except (TypeError, ValueError):
            minimum = 0
        specs.append(
            {
                "name": name,
                "min": minimum,
                "max": str(element.get("max") or "1"),
                "type": types[0] if types else "string",
            }
        )
    return specs

def _walk_rules(document: Mapping[str, Any], visit) -> None:
    """Call visit(rule, source_vars, target_vars) for every rule in scope"""

    def walk(rules, source_vars, target_vars):
        for rule in rules or []:
            if not isinstance(rule, dict):
                continue
            local_sources = dict(source_vars)
            local_targets = dict(target_vars)
            for entry in rule.get("source") or []:
                if not isinstance(entry, dict) or not entry.get("variable"):
                    continue
                variable = str(entry["variable"])
                if entry.get("element"):
                    local_sources[variable] = str(entry["element"])
                elif entry.get("context") in local_sources:
                    local_sources[variable] = local_sources[str(entry["context"])]
            visit(rule, local_sources, local_targets)
            for entry in rule.get("target") or []:
                if not isinstance(entry, dict):
                    continue
                variable = entry.get("variable")
                element = entry.get("element")
                if not variable:
                    continue
                parent = local_targets.get(str(entry.get("context") or ""), "")
                local_targets[str(variable)] = (
                    f"{parent}.{element}" if parent and element else str(element or "")
                )
            walk(rule.get("rule"), local_sources, local_targets)

    for group in document.get("group") or []:
        if not isinstance(group, dict):
            continue
        target_vars = {
            str(entry["name"]): str(entry.get("type") or "")
            for entry in group.get("input") or []
            if isinstance(entry, dict) and entry.get("name") and entry.get("mode") == "target"
        }
        walk(group.get("rule"), {}, target_vars)


def _parameters(target: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """get parameter field from given mapping"""
    return [p for p in (target.get("parameter") or []) if isinstance(p, Mapping)]


def _value_id(parameters: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """get value id from mapping"""
    return next((str(p["valueId"]) for p in parameters if p.get("valueId")), None)


def temporal_cast_overrides(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Format-valid values for source fields the map cast() to a date/time"""

    overrides: Dict[str, Any] = {}

    def visit(rule, source_vars, _target_vars):
        for target in rule.get("target") or []:
            if not isinstance(target, dict) or target.get("transform") != "cast":
                continue
            parameters = _parameters(target)
            variable = _value_id(parameters)
            cast_type = next(
                (
                    str(p["valueString"])
                    for p in parameters
                    if str(p.get("valueString") or "") in _TEMPORAL_TYPES
                ),
                None,
            )
            if variable and cast_type and variable in source_vars:
                overrides[source_vars[variable].rsplit(".", 1)[-1]] = TYPE_DEFAULTS[
                    cast_type
                ]

    _walk_rules(document, visit)
    return overrides


def unemptyable_source_fields(
    documents: Iterable[Mapping[str, Any]], target_tree=None
) -> Set[str]:
    """source fields whose value reaches a type that cannot represent ""

    :param documents: the maps the fixture will be run against.
    :param target_tree: the profile view, when available
    """

    wanted: Set[str] = set()

    def visit(rule, source_vars, target_vars):
        for target in rule.get("target") or []:
            if not isinstance(target, dict):
                continue
            if target.get("transform") not in ("copy", "cast"):
                continue
            variable = _value_id(_parameters(target))
            if not variable or variable not in source_vars:
                continue

            types = [
                str(p["valueString"])
                for p in _parameters(target)
                if p.get("valueString")
            ]
            element = target.get("element")
            if target_tree is not None and element:
                parent = target_vars.get(str(target.get("context") or ""), "")
                path = f"{parent}.{element}" if parent else str(element)
                from agent.validation import resolve_tree_types  # noqa: PLC0415

                types.extend(resolve_tree_types(target_tree, path))

            if any(
                item in TYPE_DEFAULTS and item not in _EMPTYABLE_TYPES for item in types
            ):
                wanted.add(source_vars[variable].rsplit(".", 1)[-1])

    for document in documents:
        _walk_rules(document, visit)
    return wanted


def target_type_overrides(document: Mapping[str, Any], target_tree) -> Dict[str, Any]:
    """values valid for the target element a source field is copied into"""

    if target_tree is None:
        return {}

    overrides: Dict[str, Any] = {}

    def visit(rule, source_vars, target_vars):
        for target in rule.get("target") or []:
            if not isinstance(target, dict):
                continue
            if target.get("transform") not in ("copy", "cast"):
                continue
            element = target.get("element")
            if not element:
                continue
            parent = target_vars.get(str(target.get("context") or ""), "")
            path = f"{parent}.{element}" if parent else str(element)
            from agent.validation import resolve_tree_types  # noqa: PLC0415

            types = resolve_tree_types(target_tree, path)
            if not types:
                continue
            default = TYPE_DEFAULTS.get(str(types[0]))
            
            if default is None or default == "1":
                continue
            variable = _value_id(_parameters(target))
            if variable and variable in source_vars:
                overrides[source_vars[variable].rsplit(".", 1)[-1]] = default

    _walk_rules(document, visit)
    return overrides


def translate_overrides(
    document: Mapping[str, Any],
    concept_map_codes: Mapping[str, str],
) -> Dict[str, Any]:
    """Valid source codes for fields the map runs through translate().

    :param concept_map_codes: {ConceptMap canonical: a source code it maps}.
    """

    if not concept_map_codes:
        return {}

    overrides: Dict[str, Any] = {}

    def visit(rule, source_vars, _target_vars):
        for target in rule.get("target") or []:
            if not isinstance(target, dict) or target.get("transform") != "translate":
                continue
            parameters = _parameters(target)
            variable = _value_id(parameters)
            canonical = next(
                (str(p["valueString"]) for p in parameters if p.get("valueString")),
                None,
            )
            if not variable or not canonical or variable not in source_vars:
                continue
            code = concept_map_codes.get(canonical)
            if code:
                overrides[source_vars[variable].rsplit(".", 1)[-1]] = code

    _walk_rules(document, visit)
    return overrides


def concept_map_source_codes(concept_maps: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """{canonical: first mapped source code} for the given ConceptMaps"""

    codes: Dict[str, str] = {}
    for concept_map in concept_maps or []:
        url = str(concept_map.get("url") or "")
        if not url or url in codes:
            continue
        for group in concept_map.get("group") or []:
            code = next(
                (
                    str(element["code"])
                    for element in (group.get("element") or [])
                    if isinstance(element, Mapping) and element.get("code")
                ),
                None,
            )
            if code:
                codes[url] = code
                break
    return codes


def expansion_overrides(
    target_tree,
    mapping_table: Optional[Mapping[str, Any]],
    expand: Optional[Callable[[str], Optional[Sequence[str]]]],
) -> Dict[str, Any]:
    """Valid codes for fields copied into a required bound element"""

    if target_tree is None or expand is None or not mapping_table:
        return {}

    from mapping.rule_ir import mapping_target_path  # noqa: PLC0415

    overrides: Dict[str, Any] = {}
    for source_field, value in mapping_table.items():
        target = mapping_target_path(value)
        if not target:
            continue
        from agent.validation import resolve_tree_node  # noqa: PLC0415

        node = resolve_tree_node(target_tree, str(target))
        binding = getattr(node, "binding", None) if node is not None else None
        if not isinstance(binding, Mapping) or binding.get("strength") != "required":
            continue
        value_set = binding.get("valueSet")
        if not value_set:
            continue
        codes = expand(str(value_set))
        if codes:
            overrides[str(source_field).rsplit(".", 1)[-1]] = str(codes[0])
    return overrides

def build_synthetic_fixtures(
    source_model: Mapping[str, Any],
    document: Mapping[str, Any],
    **kwargs: Any,
) -> List[SourceFixture]:
    """Build the filled and empty-optional variants for one map (wrapper)"""

    return build_shared_fixtures(source_model, [document], **kwargs)


def build_shared_fixtures(
    source_model: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    *,
    concept_maps: Iterable[Mapping[str, Any]] = (),
    target_tree=None,
    mapping_table: Optional[Mapping[str, Any]] = None,
    expand: Optional[Callable[[str], Optional[Sequence[str]]]] = None,
    resource_type: Optional[str] = None,
) -> List[SourceFixture]:
    """Build one fixture set valid for every revision in documents

    :param source_model: the source logical StructureDefinition
    :param documents: every revision that will be executed with these fixtures.
    :param concept_maps: ConceptMaps available to the engine.
    :param target_tree: target profile tree, enabling target-type overrides.
    :param mapping_table: routes source fields to bound target elements
    :param expand: valueSet -> codes for required bindings, if available.
    :param resource_type: value for resourceType on the instance
    :return: two fixtures, both gate-relevant.
    """

    specs = source_field_specs(source_model)
    if not specs:
        logger.warning(
            "Source model %s declares no fields — no synthetic fixture can be built.",
            source_model.get("url"),
        )
        return []

    codes = concept_map_source_codes(concept_maps)
    expanded = expansion_overrides(target_tree, mapping_table, expand)
    translated: Dict[str, Any] = {}
    derived: Dict[str, Any] = {}
    for document in documents:
        translated.update(translate_overrides(document, codes))
        derived.update(target_type_overrides(document, target_tree))
        derived.update(temporal_cast_overrides(document))

    overrides: Dict[str, Any] = {**expanded, **translated, **derived}
    verified = sorted(set(translated) | set(expanded))
    translated_fields = translate_source_fields(documents)
    unemptyable_fields = unemptyable_source_fields(documents, target_tree)

    root_type = resource_type or source_model.get("type")
    fixtures: List[SourceFixture] = []
    for kind, empty_optional in (
        (FixtureKind.SYNTHETIC_FILLED, False),
        (FixtureKind.SYNTHETIC_EMPTY_OPTIONAL, True),
    ):
        instance: Dict[str, Any] = {}
        if root_type:
            instance["resourceType"] = str(root_type)
        for spec in specs:
            name = spec["name"]
            type_code = str(spec["type"])
            if empty_optional and spec["min"] < 1:
                if (
                    type_code in _EMPTYABLE_TYPES
                    and name not in translated_fields
                    and name not in unemptyable_fields
                ):
                    instance[name] = ""
                continue
            if name in overrides:
                instance[name] = overrides[name]
                continue
            instance[name] = TYPE_DEFAULTS.get(type_code, "1")
        fixtures.append(
            SourceFixture(
                fixture_id=_content_id(kind.value, instance),
                kind=kind,
                label=kind.value,
                instance=instance,
                required=True,
                verified_fields=verified,
                provenance={
                    "source_model": source_model.get("url"),
                    "overrides": sorted(overrides),
                },
            )
        )
    return fixtures


def build_example_fixtures(
    examples: Iterable[Mapping[str, Any]],
    mapping_table: Optional[Mapping[str, Any]],
    *,
    profile_ids: Optional[Iterable[str]] = None,
    resource_type: Optional[str] = None,
    limit: int = 5,
) -> List[SourceFixture]:
    """Reverse-extract flat source instances from real example resources.

    :param examples: FHIR resources found in the project.
    :param mapping_table: the table to invert.
    :param profile_ids: profile ids whose rooted target paths also apply.
    :param limit: cap, so a project full of examples cannot dominate a run.
    :return: at most *limit* fixtures, none of them gate-relevant.
    """

    if not mapping_table:
        return []

    from data_handling.instance_validation import reverse_extract  # noqa: PLC0415

    fixtures: List[SourceFixture] = []
    ids = set(profile_ids or ())
    for resource in examples or []:
        if len(fixtures) >= limit:
            break
        if not isinstance(resource, Mapping) or not resource.get("resourceType"):
            continue
        flat = reverse_extract(dict(resource), dict(mapping_table), profile_ids=ids)
        if not flat:
            continue
        if resource_type:
            flat = {"resourceType": str(resource_type), **flat}
        label = f"example:{resource.get('resourceType')}/{resource.get('id')}"
        fixtures.append(
            SourceFixture(
                fixture_id=_content_id(FixtureKind.EXAMPLE_DERIVED.value, flat),
                kind=FixtureKind.EXAMPLE_DERIVED,
                label=label,
                instance=flat,
                required=False,
                verified_fields=sorted(flat),
                provenance={
                    "resource": label,
                    "extraction": "reverse_extract",
                    "status": "experimental",
                },
            )
        )
    return fixtures
