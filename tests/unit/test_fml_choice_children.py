"""Unit tests for addressing children of an unsliced multi-type choice element.

A profile may leave ``value[x]`` unsliced while admitting many types (us-core
smoking-status: 11). The parser expands every candidate type's children, but they
used to be dropped for multi-type fields, so a mapping table could not target
anything inside the choice — ``value[x].coding.code`` had nothing to bind to and
the emitted rule created an empty element. Children are now registered under both
the ``[x]`` form and the concrete form (``valueCodeableConcept.coding.code``).
"""

from types import SimpleNamespace

import pytest

from mapping.fml_creator.fml_helper import conv_mappable
from mapping.fml_map import StructureMapGenerator

pytestmark = pytest.mark.unit


def _child(path):
    return {"path": path, "type": [{"code": "string"}], "cardinality": {"min": 0, "max": "1"}}


def _choice_field():
    """Observation.value[x] with CodeableConcept / Quantity candidates, as the parser builds it."""
    return {
        "path": "Observation.value[x]",
        "cardinality": {"min": 1, "max": "1"},
        "type": [
            {
                "code": "CodeableConcept",
                "type_structure": [
                    {
                        "path": "Observation.value[x].coding",
                        "type": [{"code": "Coding"}],
                        "type_structure": [
                            _child("Observation.value[x].coding.system"),
                            _child("Observation.value[x].coding.code"),
                        ],
                    },
                    _child("Observation.value[x].text"),
                ],
            },
            {
                "code": "Quantity",
                "type_structure": [
                    _child("Observation.value[x].value"),
                    _child("Observation.value[x].unit"),
                ],
            },
        ],
    }


def _generator():
    gen = object.__new__(StructureMapGenerator)
    gen.factory = SimpleNamespace(_choice_suffix=lambda ct: ct[0].upper() + ct[1:])
    return gen


def _resolve(table):
    """Run the mapping table against the choice field; returns target_path -> source id."""
    gen = _generator()
    field = conv_mappable(None, _choice_field())
    source_fields = [{"id": src.split(".")[-1], "path": src} for src in table]
    _, mappings = gen._apply_custom_mapping_table(
        table, source_fields, [field], "Observation", "obs-profile"
    )
    return mappings


def test_conv_mappable_keeps_every_candidates_children():
    conv = conv_mappable(None, _choice_field())
    assert conv["type"] == "choice"
    assert set(conv["choice_structures"]) == {"CodeableConcept", "Quantity"}


def test_conv_mappable_still_lifts_a_single_types_children():
    single = {
        "path": "Observation.code",
        "type": [{"code": "CodeableConcept", "type_structure": [_child("Observation.code.text")]}],
    }
    conv = conv_mappable(None, single)
    assert conv["type"] == "CodeableConcept"
    assert conv["type_structure"] == [_child("Observation.code.text")]
    assert "choice_structures" not in conv


def test_choice_child_is_addressable_via_the_x_form():
    mappings = _resolve({"src.smokCode": "Observation.value[x].coding.code"})
    assert mappings.get("Observation.value[x].coding.code") == "smokCode"


def test_choice_child_is_addressable_via_the_concrete_form():
    mappings = _resolve({"src.smokCode": "Observation.valueCodeableConcept.coding.code"})
    assert (
        mappings.get("Observation.value[x]:valueCodeableConcept.coding.code")
        == "smokCode"
    )


def test_a_second_candidate_types_children_resolve_too():
    mappings = _resolve({"src.weight": "Observation.valueQuantity.value"})
    assert mappings.get("Observation.value[x]:valueQuantity.value") == "weight"


def test_concrete_primitive_choice_root_preserves_selected_type():
    gen = _generator()
    field = conv_mappable(
        None,
        {
            "path": "Observation.value[x]",
            "type": [{"code": "boolean"}, {"code": "CodeableConcept"}],
        },
    )
    _, mappings = gen._apply_custom_mapping_table(
        {"src.flag": "Observation.valueBoolean"},
        [{"id": "flag", "path": "src.flag"}],
        [field],
        "Observation",
        "obs-profile",
    )
    assert mappings.get("Observation.value[x]:valueBoolean") == "flag"


def test_documented_x_colon_concrete_choice_root_is_addressable():
    gen = _generator()
    field = {
        "path": "Observation.effective[x]",
        "type": [{"code": "dateTime"}, {"code": "Period"}],
        "cardinality": {"min": 1, "max": "1"},
    }

    _, mappings = gen._apply_custom_mapping_table(
        {"src.when": "Observation.effective[x]:effectiveDateTime"},
        [{"id": "when", "path": "src.when"}],
        [field],
        "Observation",
        "observation-profile",
    )

    assert mappings["Observation.effective[x]:effectiveDateTime"] == "when"
    assert gen.mapping_diagnostics == []


def test_unknown_x_colon_concrete_choice_type_is_rejected():
    gen = _generator()
    field = {
        "path": "Observation.effective[x]",
        "type": [{"code": "dateTime"}, {"code": "Period"}],
    }

    paths, mappings = gen._apply_custom_mapping_table(
        {"src.when": "Observation.effective[x]:effectiveTiming"},
        [{"id": "when", "path": "src.when"}],
        [field],
        "Observation",
        "observation-profile",
    )

    assert paths == set()
    assert mappings == {}
    assert gen.mapping_diagnostics[0]["code"] == "mapping-target-path-not-found"


def test_slice_qualified_descendant_uses_exact_snapshot_identity():
    gen = _generator()
    coding = {
        "path": "Medication.code.coding",
        "type": [{"code": "Coding"}],
        "slices": [
            {
                "path": "Medication.code.coding",
                "id": "Medication.code.coding:Pharmazentralnummer",
                "slice_identity": "Medication.code.coding:Pharmazentralnummer",
                "sliceName": "Pharmazentralnummer",
                "type": [{"code": "Coding"}],
                "children": [
                    {
                        "path": "Medication.code.coding.code",
                        "id": (
                            "Medication.code.coding:Pharmazentralnummer.code"
                        ),
                        "type": [{"code": "code"}],
                    }
                ],
            }
        ],
    }
    target = {
        "path": "Medication.code",
        "type": [{"code": "CodeableConcept"}],
        "children": [coding],
    }

    _, mappings = gen._apply_custom_mapping_table(
        {
            "src.pzn": (
                "Medication.code.coding:Pharmazentralnummer.code"
            )
        },
        [{"id": "pzn", "path": "src.pzn"}],
        [target],
        "Medication",
        "medication-profile",
    )

    assert mappings[
        "Medication.code.coding:Pharmazentralnummer.code"
    ] == "pzn"
    assert gen.mapping_diagnostics == []


def test_unknown_slice_qualified_descendant_is_rejected():
    gen = _generator()
    target = {
        "path": "Medication.code.coding",
        "type": [{"code": "Coding"}],
        "slices": [
            {
                "path": "Medication.code.coding",
                "id": "Medication.code.coding:known",
                "slice_identity": "Medication.code.coding:known",
                "sliceName": "known",
                "type": [{"code": "Coding"}],
                "children": [
                    {
                        "path": "Medication.code.coding.code",
                        "id": "Medication.code.coding:known.code",
                        "type": [{"code": "code"}],
                    }
                ],
            }
        ],
    }

    paths, mappings = gen._apply_custom_mapping_table(
        {"src.code": "Medication.code.coding:missing.code"},
        [{"id": "code", "path": "src.code"}],
        [target],
        "Medication",
        "medication-profile",
    )

    assert paths == set()
    assert mappings == {}
    assert gen.mapping_diagnostics[0]["code"] == "mapping-target-path-not-found"


def test_nested_raw_choice_indexes_every_candidate_not_only_first():
    gen = _generator()
    nested = {
        "path": "Claim.diagnosis.diagnosis[x]",
        "type": [{"code": "CodeableConcept"}, {"code": "Reference"}],
    }
    _, mappings = gen._apply_custom_mapping_table(
        {"src.ref": "Claim.diagnosis.diagnosisReference"},
        [{"id": "ref", "path": "src.ref"}],
        [nested],
        "Claim",
        "claim-profile",
    )

    assert mappings["Claim.diagnosis.diagnosis[x]:diagnosisReference"] == "ref"


def test_duplicate_target_assignments_are_diagnosed(caplog):
    gen = _generator()
    field = conv_mappable(None, _choice_field())
    _, mappings = gen._apply_custom_mapping_table(
        {
            "src.first": "Observation.value[x].text",
            "src.second": "Observation.value[x].text",
        },
        [
            {"id": "first", "path": "src.first"},
            {"id": "second", "path": "src.second"},
        ],
        [field],
        "Observation",
        "obs-profile",
    )

    assert mappings["Observation.value[x].text"] == "second"
    assert gen.mapping_diagnostics == [
        {
            "code": "duplicate-target-assignment",
            "target": "Observation.value[x].text",
            "previous_source": "first",
            "source": "second",
        }
    ]
    assert "Duplicate custom mapping target" in caplog.text


def test_unknown_child_of_a_known_base_is_rejected():
    gen = _generator()
    field = conv_mappable(None, _choice_field())
    paths, mappings = gen._apply_custom_mapping_table(
        {"src.bogus": "Observation.value[x].notAChild"},
        [{"id": "bogus", "path": "src.bogus"}],
        [field],
        "Observation",
        "obs-profile",
    )

    assert paths == set()
    assert mappings == {}
    assert gen.mapping_diagnostics[0]["code"] == "mapping-target-path-not-found"
    assert gen.mapping_diagnostics[0]["target"] == "Observation.value[x].notAChild"


def test_unknown_source_path_is_rejected():
    gen = _generator()
    field = conv_mappable(None, _choice_field())
    paths, mappings = gen._apply_custom_mapping_table(
        {"src.missing": "Observation.value[x].text"},
        [{"id": "present", "path": "src.present"}],
        [field],
        "Observation",
        "obs-profile",
    )

    assert paths == set()
    assert mappings == {}
    assert gen.mapping_diagnostics[0]["code"] == "mapping-source-path-not-found"
    assert gen.mapping_diagnostics[0]["source"] == "src.missing"


def test_unique_root_relative_source_path_is_accepted():
    gen = _generator()
    field = conv_mappable(None, _choice_field())
    paths, mappings = gen._apply_custom_mapping_table(
        {"value": "Observation.value[x].text"},
        [{"id": "Source.value", "path": "Source.value"}],
        [field],
        "Observation",
        "obs-profile",
    )

    assert "Observation.value[x].text" in paths
    assert mappings["Observation.value[x].text"] == "Source.value"
    assert gen.mapping_diagnostics == []


def test_ambiguous_root_relative_source_path_requires_full_path():
    gen = _generator()
    field = conv_mappable(None, _choice_field())
    paths, mappings = gen._apply_custom_mapping_table(
        {"value": "Observation.value[x].text"},
        [
            {"id": "First.value", "path": "First.value"},
            {"id": "Second.value", "path": "Second.value"},
        ],
        [field],
        "Observation",
        "obs-profile",
    )

    assert paths == set()
    assert mappings == {}
    assert gen.mapping_diagnostics[0]["code"] == "mapping-source-path-ambiguous"


@pytest.mark.parametrize("source", ["TODO_MAP_VALUE", "TODO-MAP-VALUE"])
def test_todo_source_scaffolds_remain_valid(source):
    gen = _generator()
    field = conv_mappable(None, _choice_field())
    paths, mappings = gen._apply_custom_mapping_table(
        {source: "Observation.value[x].text"},
        [],
        [field],
        "Observation",
        "obs-profile",
    )

    assert "Observation.value[x].text" in paths
    assert mappings["Observation.value[x].text"] == source
    assert gen.mapping_diagnostics == []


def test_unknown_slice_is_rejected_even_when_unsliced_base_exists():
    gen = _generator()
    target = {
        "path": "Observation.component",
        "type": "BackboneElement",
        "cardinality": {"min": 0, "max": "*"},
    }
    paths, mappings = gen._apply_custom_mapping_table(
        {"src.value": "Observation.component:not-a-slice"},
        [{"id": "value", "path": "src.value"}],
        [target],
        "Observation",
        "obs-profile",
    )

    assert paths == set()
    assert mappings == {}
    assert gen.mapping_diagnostics[0]["code"] == "mapping-target-path-not-found"
