"""Construct-level and optional Matchbox execution checks for authored rules.

The offline tests are always executed.  They prove that every capability exposed
by the selected R4/R4B specification context can pass through the authored-rule
compiler, the shared R4B carrier, and JSON serialization.  The execution tests
reuse those exact resources when ``MATCHBOX_URL`` (or the documented localhost
default) points to a running Matchbox instance.

Resource validity is deliberately not asserted here.  A transform can be
accepted by the Mapping Language engine while producing a resource that fails a
profile; those are separate conformance layers.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from typing import Any
import urllib.error
import urllib.request

import pytest
from fhir.resources.R4B.structuremap import StructureMap

from controller.external_services.matchbox_controller import MatchboxController
from mapping.fml_creator.fml_factory import FMLRuleFactory
from mapping.rule_ir import (
    CollectionRuleSpec,
    CONTEXT_TYPES,
    GROUP_TYPE_MODES,
    INPUT_MODES,
    MAPPING_TRANSFORMS,
    PARAMETER_VALUE_TYPES,
    SOURCE_LIST_MODES,
    STRUCTURE_MODES,
    TARGET_LIST_MODES,
    compile_rule_document,
    parse_collection_rule,
)

pytestmark = pytest.mark.integration

_MAP_ROOT = "http://example.org/fhirbridge/StructureMap/conformance"
_PATIENT_SD = "http://hl7.org/fhir/StructureDefinition/Patient"
_OBSERVATION_SD = "http://hl7.org/fhir/StructureDefinition/Observation"
_MATCHBOX_BASE = os.environ.get(
    "MATCHBOX_URL", "http://localhost:8080/matchboxv3"
).rstrip("/")


def _parameter(value_id: str | None = None, **literal: Any) -> dict:
    if value_id is not None:
        return {"valueId": value_id}
    return literal


def _simple_rule(
    name: str,
    transform: str,
    parameters: list[dict],
    *,
    source_element: str = "gender",
    target_element: str = "gender",
) -> dict:
    return {
        "name": name,
        "source": {
            "element": source_element,
            "variable": "value",
        },
        "target": {
            "element": target_element,
            "transform": transform,
            "parameters": parameters,
        },
    }


def _nested_string_rule(name: str, transform: str, parameters: list[dict]) -> dict:
    return {
        "name": name,
        "source": {
            "element": "name",
            "variable": "source-name",
            "listMode": "first",
        },
        "target": {
            "element": "name",
            "variable": "target-name",
            "transform": "create",
            "parameters": [{"valueString": "HumanName"}],
        },
        "rules": [
            {
                "name": f"{name}-value",
                "source": {
                    "context": "source-name",
                    "element": "family",
                    "variable": "value",
                },
                "target": {
                    "context": "target-name",
                    "element": "family",
                    "transform": transform,
                    "parameters": parameters,
                },
            }
        ],
    }


def _transform_documents() -> dict[str, dict]:
    """One executable-shaped authored document for each R4 transform code."""

    return {
        "create": {
            "$rules": [
                {
                    "name": "use-create",
                    "source": {},
                    "target": {
                        "element": "name",
                        "transform": "create",
                        "parameters": [{"valueString": "HumanName"}],
                    },
                }
            ]
        },
        "copy": {
            "$rules": [
                _simple_rule(
                    "use-copy", "copy", [_parameter(value_id="value")]
                )
            ]
        },
        "truncate": {
            "$rules": [
                _nested_string_rule(
                    "use-truncate",
                    "truncate",
                    [
                        _parameter(value_id="value"),
                        {"valueInteger": 4},
                    ],
                )
            ]
        },
        "escape": {
            "$rules": [
                _nested_string_rule(
                    "use-escape",
                    "escape",
                    [
                        _parameter(value_id="value"),
                        {"valueString": "json"},
                        {"valueString": "json"},
                    ],
                )
            ]
        },
        "cast": {
            "$rules": [
                _nested_string_rule(
                    "use-cast",
                    "cast",
                    [
                        _parameter(value_id="value"),
                        {"valueString": "string"},
                    ],
                )
            ]
        },
        "append": {
            "$rules": [
                _nested_string_rule(
                    "use-append",
                    "append",
                    [
                        {"valueString": "prefix-"},
                        _parameter(value_id="value"),
                    ],
                )
            ]
        },
        "translate": {
            "$rules": [
                _simple_rule(
                    "use-translate",
                    "translate",
                    [
                        _parameter(value_id="value"),
                        {"valueString": "http://example.org/ConceptMap/gender"},
                        {"valueString": "code"},
                    ],
                )
            ]
        },
        "reference": {
            "$rules": [
                _simple_rule(
                    "use-reference",
                    "reference",
                    [_parameter(value_id="value")],
                    source_element="managingOrganization",
                    target_element="managingOrganization",
                )
            ]
        },
        "dateOp": {
            "$rules": [
                _simple_rule(
                    "use-date-op",
                    "dateOp",
                    [_parameter(value_id="value")],
                    source_element="birthDate",
                    target_element="birthDate",
                )
            ]
        },
        "uuid": {
            "$rules": [
                {
                    "name": "use-uuid",
                    "source": {},
                    "target": {
                        "element": "id",
                        "transform": "uuid",
                        "parameters": [],
                    },
                }
            ]
        },
        "pointer": {
            "$rules": [
                _simple_rule(
                    "use-pointer",
                    "pointer",
                    [_parameter(value_id="value")],
                    source_element="managingOrganization",
                    target_element="managingOrganization",
                )
            ]
        },
        "evaluate": {
            "$rules": [
                _simple_rule(
                    "use-evaluate",
                    "evaluate",
                    [
                        _parameter(value_id="value"),
                        {"valueString": "$this"},
                    ],
                )
            ]
        },
        "cc": {
            "$rules": [
                _simple_rule(
                    "use-cc",
                    "cc",
                    [
                        {"valueString": "http://example.org/system"},
                        {"valueString": "M"},
                        {"valueString": "Married"},
                    ],
                    target_element="maritalStatus",
                )
            ]
        },
        "c": {
            "$rules": [
                {
                    "name": "use-c",
                    "source": {},
                    "target": {
                        "element": "meta",
                        "variable": "target-meta",
                        "transform": "create",
                        "parameters": [{"valueString": "Meta"}],
                    },
                    "rules": [
                        {
                            "name": "use-c-value",
                            "source": {"context": "source"},
                            "target": {
                                "context": "target-meta",
                                "element": "tag",
                                "transform": "c",
                                "parameters": [
                                    {"valueString": "http://example.org/system"},
                                    {"valueString": "fixture"},
                                    {"valueString": "Fixture"},
                                ],
                            },
                        }
                    ],
                }
            ]
        },
        "qty": {
            "$rules": [
                _simple_rule(
                    "use-qty",
                    "qty",
                    [{"valueString": "12 mg"}],
                    source_element="gender",
                    target_element="value",
                )
            ],
            "_target_type": "Observation",
        },
        "id": {
            "$rules": [
                _simple_rule(
                    "use-id",
                    "id",
                    [
                        {"valueString": "http://example.org/id"},
                        {"valueString": "123"},
                    ],
                    target_element="identifier",
                )
            ]
        },
        "cp": {
            "$rules": [
                _simple_rule(
                    "use-cp",
                    "cp",
                    [{"valueString": "+49-555-0100"}],
                    target_element="telecom",
                )
            ]
        },
    }


TRANSFORM_DOCUMENTS = _transform_documents()


def _core_structure_url(resource_type: str) -> str:
    return (
        _OBSERVATION_SD if resource_type == "Observation" else _PATIENT_SD
    )


def _compiled_map(
    case_id: str,
    document: dict,
    *,
    source_type: str = "Patient",
    target_type: str = "Patient",
) -> dict:
    document = deepcopy(document)
    document.pop("_target_type", None)
    compiled = compile_rule_document(
        document,
        profile_id=f"Fixture{target_type}",
        profile_url=_core_structure_url(target_type),
        resource_type=target_type,
    )
    assert compiled.diagnostics == [], (case_id, compiled.diagnostics)

    groups = []
    if compiled.primary_rules:
        groups.append(
            {
                "name": "Main",
                "typeMode": "none",
                "input": [
                    {"name": "source", "mode": "source", "type": source_type},
                    {"name": "target", "mode": "target", "type": target_type},
                ],
                "rule": [
                    rule.model_dump(exclude_none=True, by_alias=True)
                    for rule in compiled.primary_rules
                ],
            }
        )
    groups.extend(
        group.model_dump(exclude_none=True, by_alias=True)
        for group in compiled.groups
    )
    assert groups, f"{case_id}: fixture requires at least one group"

    structures = [
        {
            "url": _core_structure_url(source_type),
            "mode": "source",
            "alias": f"Source{source_type}",
        },
        {
            "url": _core_structure_url(target_type),
            "mode": "target",
            "alias": f"Target{target_type}",
        },
    ]
    structures.extend(
        structure.model_dump(exclude_none=True, by_alias=True)
        for structure in compiled.structures
    )
    resource = {
        "resourceType": "StructureMap",
        "id": f"fixture-{case_id}".lower(),
        "url": f"{_MAP_ROOT}/{case_id}",
        "name": f"Fixture{case_id.replace('-', '').title()}",
        "status": "draft",
        "structure": structures,
        "group": groups,
    }
    if compiled.imports:
        resource["import"] = compiled.imports
    return StructureMap.model_validate(resource).model_dump(
        mode="json", exclude_none=True, by_alias=True
    )


def _map_from_rules(case_id: str, rules: list[Any]) -> dict:
    resource = {
        "resourceType": "StructureMap",
        "id": f"fixture-{case_id}".lower(),
        "url": f"{_MAP_ROOT}/{case_id}",
        "name": f"Fixture{case_id.replace('-', '').title()}",
        "status": "draft",
        "structure": [
            {"url": _PATIENT_SD, "mode": "source", "alias": "SourcePatient"},
            {"url": _PATIENT_SD, "mode": "target", "alias": "TargetPatient"},
        ],
        "group": [
            {
                "name": "Main",
                "typeMode": "none",
                "input": [
                    {"name": "source", "mode": "source", "type": "Patient"},
                    {"name": "target", "mode": "target", "type": "Patient"},
                ],
                "rule": [
                    rule.model_dump(exclude_none=True, by_alias=True)
                    for rule in rules
                ],
            }
        ],
    }
    return StructureMap.model_validate(resource).model_dump(
        mode="json", exclude_none=True, by_alias=True
    )


def _field(path: str, field_type: str, maximum: str = "1", children=None) -> dict:
    return {
        "path": path,
        "type": field_type,
        "cardinality": {"min": 0, "max": maximum},
        "children": children or [],
    }


def _collection_rule(correlation: str | dict | None):
    factory = object.__new__(FMLRuleFactory)
    factory.source_field_types = {}
    factory.source_field_max = {}
    factory.collection_rules = {}
    factory.diagnostics = []
    factory._current_profile_id = None
    factory._current_profile_sd = None
    factory._target_tree_cache = (None, None)

    raw_spec: dict[str, Any] = {"target": "Patient.identifier"}
    if correlation is not None:
        raw_spec["correlation"] = correlation
    spec = parse_collection_rule("Patient.identifier", raw_spec)
    assert isinstance(spec, CollectionRuleSpec)
    factory.collection_rules["Patient.identifier"] = spec
    target = _field(
        "Patient.identifier",
        "Identifier",
        "*",
        [
            _field("Patient.identifier.system", "uri"),
            _field("Patient.identifier.value", "string"),
        ],
    )
    return factory.create_mappable_field_rule(
        target,
        "Patient",
        "source",
        "target",
        automapped_mappings={
            "Patient.identifier": "Patient.identifier",
            "Patient.identifier.system": "Patient.identifier.system",
            "Patient.identifier.value": "Patient.identifier.value",
        },
    )


def _source_list_document(mode: str) -> dict:
    return {
        "$rules": [
            {
                "name": f"source-list-{mode.replace('_', '-')}",
                "source": {
                    "element": "name",
                    "variable": "value",
                    "listMode": mode,
                },
                "target": {
                    "element": "name",
                    "transform": "copy",
                    "parameters": [{"valueId": "value"}],
                },
            }
        ]
    }


def _target_list_document(mode: str) -> dict:
    target = {
        "element": "name",
        "transform": "copy",
        "parameters": [{"valueId": "value"}],
        "targetListMode": mode,
    }
    if mode == "share":
        target["listRuleId"] = "shared-name"
    return {
        "$rules": [
            {
                "name": f"target-list-{mode}",
                "source": {"element": "name", "variable": "value"},
                "target": target,
            }
        ]
    }


GROUP_DOCUMENT = {
    "$groups": [
        {
            "name": "Base",
            "inputs": [
                {"name": "source", "mode": "source", "type": "Patient"},
                {"name": "target", "mode": "target", "type": "Patient"},
            ],
            "rules": [
                {
                    "name": "base-active",
                    "source": {"context": "source"},
                    "target": {
                        "context": "target",
                        "element": "active",
                        "transform": "copy",
                        "parameters": [{"valueBoolean": True}],
                    },
                }
            ],
        },
        {
            "name": "Derived",
            "extends": "Base",
            "inputs": [
                {"name": "source", "mode": "source", "type": "Patient"},
                {"name": "target", "mode": "target", "type": "Patient"},
            ],
            "rules": [
                {
                    "name": "derived-gender",
                    "source": {
                        "context": "source",
                        "element": "gender",
                        "variable": "gender",
                    },
                    "target": {
                        "context": "target",
                        "element": "gender",
                        "transform": "copy",
                        "parameters": [{"valueId": "gender"}],
                    },
                }
            ],
        },
    ],
    "$rules": [
        {
            "name": "call-derived",
            "source": {"context": "source"},
            "dependent": [
                {"name": "Derived", "variables": ["source", "target"]}
            ],
        }
    ],
}


IMPORT_DOCUMENT = {
    "$imports": [f"{_MAP_ROOT}/imported"],
    "$rules": [
        {
            "name": "call-imported",
            "source": {"context": "source"},
            "dependent": [
                {"name": "Imported", "variables": ["source", "target"]}
            ],
        }
    ],
}


IMPORTED_GROUP_DOCUMENT = {
    "$groups": [
        {
            "name": "Imported",
            "inputs": [
                {"name": "source", "mode": "source", "type": "Patient"},
                {"name": "target", "mode": "target", "type": "Patient"},
            ],
            "rules": [
                {
                    "name": "imported-active",
                    "source": {"context": "source"},
                    "target": {
                        "context": "target",
                        "element": "active",
                        "transform": "copy",
                        "parameters": [{"valueBoolean": True}],
                    },
                }
            ],
        }
    ]
}


CONTROL_DOCUMENT = {
    "$structures": [
        {
            "url": "http://example.org/StructureDefinition/queried",
            "mode": "queried",
            "alias": "Queried",
        },
        {
            "url": "http://example.org/StructureDefinition/produced",
            "mode": "produced",
            "alias": "Produced",
        },
    ],
    "$rules": [
        {
            "name": "controls-and-nesting",
            "sources": [
                {
                    "element": "name",
                    "variable": "first-name",
                    "listMode": "first",
                    "condition": "family.exists()",
                    "check": "family.count() = 1",
                    "logMessage": "'copying name'",
                },
                {
                    "element": "gender",
                    "variable": "gender",
                    "defaultValueCode": "unknown",
                },
            ],
            "targets": [
                {
                    "element": "name",
                    "variable": "target-name",
                    "transform": "create",
                    "parameters": [{"valueString": "HumanName"}],
                },
                {
                    "element": "gender",
                    "transform": "copy",
                    "parameters": [{"valueId": "gender"}],
                },
            ],
            "rules": [
                {
                    "name": "nested-family",
                    "source": {
                        "context": "first-name",
                        "element": "family",
                        "variable": "family",
                    },
                    "target": {
                        "context": "target-name",
                        "element": "family",
                        "transform": "copy",
                        "parameters": [{"valueId": "family"}],
                    },
                }
            ],
        }
    ],
}


@pytest.mark.parametrize("transform", sorted(TRANSFORM_DOCUMENTS))
def test_transform_fixture_compiles_and_round_trips(transform):
    target_type = TRANSFORM_DOCUMENTS[transform].get("_target_type", "Patient")
    resource = _compiled_map(
        f"transform-{transform.lower()}",
        TRANSFORM_DOCUMENTS[transform],
        target_type=target_type,
    )
    assert StructureMap.model_validate_json(json.dumps(resource)).url == (
        f"{_MAP_ROOT}/transform-{transform.lower()}"
    )


def test_fixture_matrix_covers_every_selected_release_capability():
    assert set(TRANSFORM_DOCUMENTS) == set(MAPPING_TRANSFORMS)
    assert set(SOURCE_LIST_MODES) == {
        "first",
        "not_first",
        "last",
        "not_last",
        "only_one",
    }
    assert set(TARGET_LIST_MODES) == {"first", "share", "last", "collate"}
    assert set(STRUCTURE_MODES) == {"source", "queried", "target", "produced"}
    assert set(GROUP_TYPE_MODES) == {"none", "types", "type-and-types"}
    assert set(INPUT_MODES) == {"source", "target"}
    assert set(CONTEXT_TYPES) == {"type", "variable"}
    assert set(PARAMETER_VALUE_TYPES) == {
        "id",
        "string",
        "boolean",
        "integer",
        "decimal",
    }


@pytest.mark.parametrize("mode", sorted(SOURCE_LIST_MODES))
def test_source_list_mode_fixture_compiles_and_round_trips(mode):
    _compiled_map(f"source-list-{mode.replace('_', '-')}", _source_list_document(mode))


@pytest.mark.parametrize("mode", sorted(TARGET_LIST_MODES))
def test_target_list_mode_fixture_compiles_and_round_trips(mode):
    _compiled_map(f"target-list-{mode}", _target_list_document(mode))


@pytest.mark.parametrize(
    ("case_id", "document"),
    [
        ("controls", CONTROL_DOCUMENT),
        ("groups", GROUP_DOCUMENT),
        ("imports", IMPORT_DOCUMENT),
        ("imported", IMPORTED_GROUP_DOCUMENT),
    ],
)
def test_cross_construct_fixture_compiles_and_round_trips(case_id, document):
    resource = _compiled_map(case_id, document)
    assert resource["group"]


@pytest.mark.parametrize(
    ("case_id", "correlation", "expected_check"),
    [
        ("collection-repetition", None, None),
        ("collection-same-key", "system", "system.count() = 1"),
        (
            "collection-explicit-keys",
            {"sourceKey": "system", "targetKey": "value"},
            "system.count() = 1",
        ),
    ],
)
def test_generated_collection_fixture_round_trips(
    case_id, correlation, expected_check
):
    rule = _collection_rule(correlation)
    assert rule.source[0].check == expected_check
    resource = _map_from_rules(case_id, [rule])
    assert resource["group"][0]["rule"][0]["target"][0]["element"] == "identifier"


def _matchbox_available() -> bool:
    try:
        with urllib.request.urlopen(
            f"{_MATCHBOX_BASE}/fhir/metadata", timeout=3
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


@pytest.fixture(scope="module")
def matchbox():
    if not _matchbox_available():
        pytest.skip(
            f"Matchbox is not reachable at {_MATCHBOX_BASE}; set MATCHBOX_URL "
            "to execute the StructureMap fixture corpus"
        )
    return MatchboxController({"url": _MATCHBOX_BASE})


def _upsert_map(matchbox: MatchboxController, resource: dict) -> None:
    response = matchbox.mc.send_request(
        f"StructureMap/{resource['id']}",
        "PUT",
        data=json.dumps(resource),
    )
    assert response is not None, f"Matchbox rejected {resource['url']}"


def _upsert_resource(matchbox: MatchboxController, resource: dict) -> None:
    response = matchbox.mc.send_request(
        f"{resource['resourceType']}/{resource['id']}",
        "PUT",
        data=json.dumps(resource),
    )
    assert response is not None, (
        f"Matchbox rejected {resource['resourceType']}/{resource['id']}"
    )


def _gender_concept_map() -> dict:
    return {
        "resourceType": "ConceptMap",
        "id": "fixture-gender",
        "url": "http://example.org/ConceptMap/gender",
        "status": "active",
        "sourceUri": "http://hl7.org/fhir/ValueSet/administrative-gender",
        "targetUri": "http://hl7.org/fhir/ValueSet/administrative-gender",
        "group": [
            {
                "source": "http://hl7.org/fhir/administrative-gender",
                "target": "http://hl7.org/fhir/administrative-gender",
                "element": [
                    {
                        "code": "male",
                        "target": [
                            {"code": "male", "equivalence": "equivalent"}
                        ],
                    }
                ],
            }
        ],
    }


def _assert_transform_succeeded(result: dict | None, case_id: str) -> None:
    assert result is not None, f"{case_id}: Matchbox returned no result"
    assert result.get("resourceType") != "OperationOutcome", (
        case_id,
        result,
    )


@pytest.mark.parametrize("transform", sorted(TRANSFORM_DOCUMENTS))
def test_matchbox_executes_transform_fixture(matchbox, transform):
    target_type = TRANSFORM_DOCUMENTS[transform].get("_target_type", "Patient")
    case_id = f"transform-{transform.lower()}"
    resource = _compiled_map(
        case_id,
        TRANSFORM_DOCUMENTS[transform],
        target_type=target_type,
    )
    if transform == "translate":
        _upsert_resource(matchbox, _gender_concept_map())
    _upsert_map(matchbox, resource)
    result = matchbox.transform_data(
        {
            "resourceType": "Patient",
            "gender": "male",
            "birthDate": "2000-01-01",
            "name": [{"family": 'O"Neil'}, {"family": "Second"}],
            "managingOrganization": {"reference": "Organization/example"},
        },
        resource["url"],
    )
    _assert_transform_succeeded(result, case_id)


@pytest.mark.parametrize("mode", sorted(SOURCE_LIST_MODES))
def test_matchbox_executes_source_list_mode_fixture(matchbox, mode):
    case_id = f"source-list-{mode.replace('_', '-')}"
    resource = _compiled_map(case_id, _source_list_document(mode))
    _upsert_map(matchbox, resource)
    names = [{"family": "Only"}]
    if mode != "only_one":
        names.append({"family": "Second"})
    result = matchbox.transform_data(
        {"resourceType": "Patient", "name": names},
        resource["url"],
    )
    _assert_transform_succeeded(result, case_id)


@pytest.mark.parametrize("mode", sorted(TARGET_LIST_MODES))
def test_matchbox_executes_target_list_mode_fixture(matchbox, mode):
    case_id = f"target-list-{mode}"
    resource = _compiled_map(case_id, _target_list_document(mode))
    _upsert_map(matchbox, resource)
    result = matchbox.transform_data(
        {
            "resourceType": "Patient",
            "name": [{"family": "First"}, {"family": "Second"}],
        },
        resource["url"],
    )
    _assert_transform_succeeded(result, case_id)


@pytest.mark.parametrize(
    ("case_id", "correlation"),
    [
        ("collection-repetition", None),
        ("collection-same-key", "system"),
        (
            "collection-explicit-keys",
            {"sourceKey": "system", "targetKey": "value"},
        ),
    ],
)
def test_matchbox_executes_generated_collection_fixture(
    matchbox, case_id, correlation
):
    resource = _map_from_rules(case_id, [_collection_rule(correlation)])
    _upsert_map(matchbox, resource)
    result = matchbox.transform_data(
        {
            "resourceType": "Patient",
            "identifier": [
                {"system": "http://example.org/a", "value": "1"},
                {"system": "http://example.org/b", "value": "2"},
            ],
        },
        resource["url"],
    )
    _assert_transform_succeeded(result, case_id)


def test_matchbox_executes_dependent_and_inherited_groups(matchbox):
    resource = _compiled_map("groups", GROUP_DOCUMENT)
    _upsert_map(matchbox, resource)
    result = matchbox.transform_data(
        {"resourceType": "Patient", "gender": "female"},
        resource["url"],
    )
    _assert_transform_succeeded(result, "groups")


def test_matchbox_executes_imported_group(matchbox):
    imported = _compiled_map("imported", IMPORTED_GROUP_DOCUMENT)
    main = _compiled_map("imports", IMPORT_DOCUMENT)
    _upsert_map(matchbox, imported)
    _upsert_map(matchbox, main)
    result = matchbox.transform_data(
        {"resourceType": "Patient"},
        main["url"],
    )
    _assert_transform_succeeded(result, "imports")


def test_matchbox_negative_check_fixture(matchbox):
    document = {
        "$rules": [
            {
                "name": "failing-check",
                "source": {
                    "element": "name",
                    "variable": "name",
                    "check": "family.exists()",
                },
                "target": {
                    "element": "name",
                    "transform": "copy",
                    "parameters": [{"valueId": "name"}],
                },
            }
        ]
    }
    resource = _compiled_map("negative-check", document)
    _upsert_map(matchbox, resource)
    result = matchbox.transform_data(
        {"resourceType": "Patient", "name": [{"given": ["No family"]}]},
        resource["url"],
    )
    assert result is None or result.get("resourceType") == "OperationOutcome"
