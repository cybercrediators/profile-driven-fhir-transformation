"""Typed StructureMap rule IR and explicit reference strategies."""

import json

import pytest

from mapping.rule_ir import MAPPING_TRANSFORMS, compile_rule_document

pytestmark = pytest.mark.unit


def _compile(table, resource_type="Observation"):
    return compile_rule_document(
        table,
        profile_id="ExampleObservation",
        profile_url="http://example.org/StructureDefinition/ExampleObservation",
        resource_type=resource_type,
    )


def test_document_compiles_imports_structures_groups_and_full_rule_shape():
    document = _compile(
        {
            "$imports": [
                "http://example.org/StructureMap/common",
                "http://example.org/StructureMap/common",
            ],
            "$structures": [
                {
                    "url": "http://example.org/StructureDefinition/Lookup",
                    "mode": "queried",
                    "alias": "Lookup",
                },
                {
                    "url": "http://example.org/StructureDefinition/Produced",
                    "mode": "produced",
                },
            ],
            "$groups": [
                {
                    "name": "Correlate",
                    "typeMode": "types",
                    "extends": "BaseCorrelate",
                    "inputs": [
                        {"name": "left", "mode": "source", "type": "Observation"},
                        {"name": "right", "mode": "source", "type": "Patient"},
                        {"name": "out", "mode": "target", "type": "Observation"},
                    ],
                    "rules": [
                        {
                            "name": "correlated",
                            "sources": [
                                {
                                    "context": "left",
                                    "element": "identifier",
                                    "variable": "left-id",
                                    "condition": "left-id.exists()",
                                    "check": "left-id.value.exists()",
                                    "logMessage": "'correlating'",
                                    "listMode": "first",
                                },
                                {
                                    "context": "right",
                                    "element": "identifier",
                                    "variable": "right-id",
                                    "listMode": "only_one",
                                },
                            ],
                            "targets": [
                                {
                                    "context": "out",
                                    "element": "identifier",
                                    "variable": "out-id",
                                    "transform": "copy",
                                    "parameters": [{"valueId": "left-id"}],
                                    "targetListMode": ["share"],
                                    "listRuleId": "shared-id",
                                },
                                {
                                    "context": "out",
                                    "element": "status",
                                    "transform": "copy",
                                    "parameters": [{"valueString": "final"}],
                                },
                            ],
                            "rules": [
                                {
                                    "name": "nested",
                                    "source": {"context": "left-id"},
                                    "target": {
                                        "context": "out-id",
                                        "element": "use",
                                        "transform": "copy",
                                        "parameters": [{"valueString": "usual"}],
                                    },
                                }
                            ],
                            "dependent": [
                                {
                                    "name": "Finish",
                                    "variables": ["left", "right", "out"],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    assert document.diagnostics == []
    assert document.imports == ["http://example.org/StructureMap/common"]
    assert [item.mode for item in document.structures] == ["queried", "produced"]
    group = document.groups[0]
    assert [item.mode for item in group.input] == ["source", "source", "target"]
    rule = group.rule[0]
    assert len(rule.source) == 2
    assert len(rule.target) == 2
    assert rule.rule[0].name == "nested"
    assert rule.dependent[0].variable == ["left", "right", "out"]


@pytest.mark.parametrize("transform", sorted(MAPPING_TRANSFORMS))
def test_every_r4b_structuremap_transform_is_accepted(transform):
    rule = {
        "name": f"use-{transform.lower()}",
        "source": {"element": "value", "variable": "value"},
        "target": {
            "element": "value",
            "transform": transform,
            "parameters": [{"valueId": "value"}],
        },
    }
    document = _compile({"$rules": [rule]})
    assert document.diagnostics == []
    assert document.primary_rules[0].target[0].transform == transform


def test_default_value_choice_and_invalid_rule_diagnostic():
    document = _compile(
        {
            "$rules": [
                {
                    "name": "defaulted",
                    "source": {
                        "element": "status",
                        "defaultValueString": "unknown",
                    },
                    "target": {
                        "element": "status",
                        "transform": "copy",
                        "parameters": [{"valueString": "unknown"}],
                    },
                },
                {
                    "name": "bad",
                    "source": {"element": "status"},
                    "target": {"element": "status", "transform": "not-a-transform"},
                },
            ]
        }
    )
    assert len(document.primary_rules) == 1
    assert document.primary_rules[0].source[0].defaultValueString == "unknown"
    assert document.diagnostics[0]["code"] == "invalid-typed-rule"


def test_malformed_section_and_non_numeric_decimal_are_diagnosed():
    document = _compile(
        {
            "$groups": {"name": "not-a-list"},
            "$rules": [
                {
                    "name": "bad-decimal",
                    "source": {"element": "value"},
                    "target": {
                        "element": "value",
                        "transform": "qty",
                        "parameters": [{"valueDecimal": "not-a-number"}],
                    },
                }
            ],
        }
    )
    assert document.primary_rules == []
    assert {item["section"] for item in document.diagnostics} == {
        "$groups",
        "$rules",
    }


def test_bundled_reference_rejects_structuremap_list_options():
    document = _compile(
        {
            "$references": [
                {
                    "path": "Observation.performer",
                    "strategy": "bundled",
                    "targetType": "Practitioner",
                    "targetListMode": "share",
                }
            ]
        }
    )
    assert document.primary_rules == []
    assert document.diagnostics[0]["code"] == "invalid-typed-rule"


@pytest.mark.parametrize(
    "declaration,expected_transform,expected_literal",
    [
        (
            {
                "name": "direct",
                "path": "Observation.subject",
                "strategy": "direct",
                "source": "patientId",
            },
            "copy",
            None,
        ),
        (
            {
                "name": "conditional",
                "path": "Observation.subject",
                "strategy": "conditional",
                "source": "patientId",
                "targetType": "Patient",
                "identifierSystem": "http://example.org/mrn",
            },
            "append",
            "Patient?identifier=http://example.org/mrn|",
        ),
        (
            {
                "name": "contained",
                "path": "Observation.subject",
                "strategy": "contained",
                "literal": "contained-patient",
                "containedType": "Patient",
            },
            "append",
            "#",
        ),
    ],
)
def test_explicit_reference_rule_strategies(
    declaration, expected_transform, expected_literal
):
    document = _compile({"$references": [declaration]})
    assert document.diagnostics == []
    rule = document.primary_rules[0]
    child = next(item for item in rule.rule if item.name.endswith("-reference"))
    assert child.target[0].transform == expected_transform
    strings = [
        parameter.valueString
        for parameter in child.target[0].parameter or []
        if parameter.valueString is not None
    ]
    if expected_literal:
        assert strings[0] == expected_literal
    if declaration["strategy"] == "contained":
        assert [target.element for target in rule.target] == [
            "contained",
            "subject",
        ]
        assert strings == ["#", "contained-patient"]


def test_canonical_and_nested_bundled_references():
    document = _compile(
        {
            "$references": [
                {
                    "name": "definition",
                    "path": "Observation.definition",
                    "strategy": "canonical",
                    "literal": "http://example.org/PlanDefinition/example",
                },
                {
                    "name": "performer",
                    "path": "component.performer",
                    "strategy": "bundled",
                    "targetTypes": ["Practitioner", "Organization"],
                    "match": "all",
                    "referenceMode": "relative",
                },
            ]
        }
    )
    assert document.diagnostics == []
    canonical, bundled = document.primary_rules
    assert canonical.target[0].transform == "copy"
    marker = "FHIRBRIDGE_REFERENCE:"
    assert bundled.documentation.startswith(marker)
    contract = json.loads(bundled.documentation[len(marker) :])
    assert contract["path"] == "component.performer"
    assert contract["targetTypes"] == ["Practitioner", "Organization"]


def test_profile_scoping_omits_nonmatching_declarations():
    document = _compile(
        {
            "$rules": [
                {
                    "name": "patient-only",
                    "resourceTypes": ["Patient"],
                    "source": {"element": "id"},
                    "target": {
                        "element": "id",
                        "transform": "copy",
                        "parameters": [{"valueString": "x"}],
                    },
                }
            ]
        }
    )
    assert document.primary_rules == []
    assert document.diagnostics == []
