"""Bundle assembly coverage for BundleService: create_bundle end-to-end wiring,
ifNoneExist conditional-create keys, SM-spec parsing, array-path collection,
auto-wiring of unresolved required references, co-occurrence dropping, and the
low-level _set_nested/_get_nested/normalize_list_cardinality helpers.

test_bundle_references.py already covers _wire_references / _iter_reference_fields /
_collect_todo_refs / _build_profile_type_map in isolation; this file exercises the
surrounding assembly machinery that those unit tests don't reach.
"""

from types import SimpleNamespace

import pytest

from controller.bundle_service import BundleService
from data_handling.registry.registry_object import RegistryObject

pytestmark = pytest.mark.unit


def make_registry(objects):
    """objects: list of (url, RegistryObject)"""
    return SimpleNamespace(registry_objects=dict(objects))


def root_obj(url, ftype, mappable_fields):
    obj = RegistryObject(data=SimpleNamespace(url=url, type=ftype), res_type=ftype)
    obj.mappable_fields = mappable_fields
    obj.set_root()
    return obj


# --------------------------------------------------------------------------- #
# create_bundle: end-to-end assembly
# --------------------------------------------------------------------------- #


def test_create_bundle_basic_structure():
    pat = {"resourceType": "Patient", "name": [{"family": "X"}]}
    bundle = BundleService.create_bundle([pat])
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    assert "id" in bundle
    assert len(bundle["entry"]) == 1
    entry = bundle["entry"][0]
    assert entry["resource"] is pat
    assert entry["request"] == {"method": "POST", "url": "Patient"}
    assert entry["fullUrl"].startswith("urn:uuid:")


def test_create_bundle_skips_falsy_and_typeless_resources():
    pat = {"resourceType": "Patient"}
    bundle = BundleService.create_bundle([pat, None, {}, {"no": "type"}])
    assert len(bundle["entry"]) == 1
    assert bundle["entry"][0]["resource"] is pat


def test_create_bundle_sets_if_none_exist_from_identifier():
    pat = {
        "resourceType": "Patient",
        "identifier": [{"system": "http://sys", "value": "a b"}],
    }
    bundle = BundleService.create_bundle([pat])
    assert (
        bundle["entry"][0]["request"]["ifNoneExist"]
        == "identifier=http://sys|a%20b"
    )


def test_create_bundle_no_if_none_exist_without_business_key():
    obs = {"resourceType": "Observation", "status": "final"}
    bundle = BundleService.create_bundle([obs])
    assert "ifNoneExist" not in bundle["entry"][0]["request"]


def test_create_bundle_resolve_references_false_skips_wiring_and_urn():
    obs = {"resourceType": "Observation"}
    pat = {"resourceType": "Patient"}
    sm = {
        "group": [
            {
                "rule": [
                    {
                        "name": "TODO-resolve-reference-Observation-subject",
                        "documentation": "Reference<Observation.subject> → Patient",
                    }
                ]
            }
        ]
    }
    bundle = BundleService.create_bundle(
        [obs, pat], structure_maps=[sm], resolve_references=False
    )
    for entry in bundle["entry"]:
        assert "fullUrl" not in entry
    assert "subject" not in obs


def test_create_bundle_wires_via_structure_map_todo_rules():
    obs = {"resourceType": "Observation"}
    pat = {"resourceType": "Patient"}
    sm = {
        "group": [
            {
                "rule": [
                    {
                        "name": "TODO-resolve-reference-Observation-subject",
                        "documentation": "Reference<Observation.subject> → Patient",
                    }
                ]
            }
        ]
    }
    bundle = BundleService.create_bundle([obs, pat], structure_maps=[sm])
    pat_urn = next(
        e["fullUrl"] for e in bundle["entry"] if e["resource"] is pat
    )
    assert obs["subject"] == {"reference": pat_urn}


def test_create_bundle_applies_declared_external_reference_when_target_absent():
    obs = {"resourceType": "Observation"}
    defaults = [
        {
            "path": "Observation.subject",
            "reference": "Patient/eval-patient-1",
        }
    ]

    BundleService.create_bundle(
        [obs], external_reference_defaults=defaults
    )

    assert obs["subject"] == {
        "reference": "Patient/eval-patient-1",
        "type": "Patient",
    }


def test_external_reference_does_not_overwrite_mapped_reference():
    obs = {
        "resourceType": "Observation",
        "subject": {"reference": "Patient/from-source"},
    }
    defaults = [
        {
            "source_type": "Observation",
            "path": "subject",
            "reference": "Patient/external-default",
        }
    ]

    BundleService.create_bundle(
        [obs], external_reference_defaults=defaults
    )

    assert obs["subject"] == {"reference": "Patient/from-source"}


def test_external_reference_respects_prohibited_and_required_children():
    profile = "http://example.org/StructureDefinition/display-only-procedure"
    fields = [
        {
            "path": "Procedure.subject",
            "cardinality": {"min": 1, "max": "1"},
            "type": [{"code": "Reference"}],
            "children": [
                {
                    "path": "Procedure.subject.reference",
                    "cardinality": {"min": 0, "max": "0"},
                    "type": [{"code": "string"}],
                },
                {
                    "path": "Procedure.subject.type",
                    "cardinality": {"min": 0, "max": "0"},
                    "type": [{"code": "uri"}],
                },
                {
                    "path": "Procedure.subject.display",
                    "cardinality": {"min": 1, "max": "1"},
                    "type": [{"code": "string"}],
                },
            ],
        }
    ]
    obj = root_obj(profile, "Procedure", fields)
    obj.prohibited_paths = {"subject.reference", "subject.type"}
    registry = make_registry([(profile, obj)])
    procedure = {
        "resourceType": "Procedure",
        "meta": {"profile": [profile]},
    }

    BundleService.create_bundle(
        [procedure],
        registry=registry,
        external_reference_defaults=[
            {
                "path": "Procedure.subject",
                "value": {
                    "reference": "Patient/external",
                    "type": "Patient",
                },
            }
        ],
    )

    assert "subject" not in procedure


def test_external_reference_accepts_required_allowed_display():
    profile = "http://example.org/StructureDefinition/display-only-procedure"
    fields = [
        {
            "path": "Procedure.subject",
            "cardinality": {"min": 1, "max": "1"},
            "type": [{"code": "Reference"}],
            "children": [
                {
                    "path": "Procedure.subject.reference",
                    "cardinality": {"min": 0, "max": "0"},
                    "type": [{"code": "string"}],
                },
                {
                    "path": "Procedure.subject.display",
                    "cardinality": {"min": 1, "max": "1"},
                    "type": [{"code": "string"}],
                },
            ],
        }
    ]
    obj = root_obj(profile, "Procedure", fields)
    obj.prohibited_paths = {"subject.reference"}
    registry = make_registry([(profile, obj)])
    procedure = {
        "resourceType": "Procedure",
        "meta": {"profile": [profile]},
    }

    BundleService.create_bundle(
        [procedure],
        registry=registry,
        external_reference_defaults=[
            {
                "path": "Procedure.subject",
                "value": {
                    "reference": "Patient/external",
                    "display": "External patient",
                },
            }
        ],
    )

    assert procedure["subject"] == {
        "display": "External patient",
        "type": "Patient",
    }


def test_bundle_wired_reference_takes_precedence_over_external_default():
    obs = {"resourceType": "Observation"}
    pat = {"resourceType": "Patient"}
    sm = {
        "group": [
            {
                "rule": [
                    {
                        "name": "TODO-resolve-reference-Observation-subject",
                        "documentation": "Reference<Observation.subject> → Patient",
                    }
                ]
            }
        ]
    }
    defaults = [
        {
            "path": "Observation.subject",
            "reference": "Patient/external-default",
        }
    ]

    bundle = BundleService.create_bundle(
        [obs, pat],
        structure_maps=[sm],
        external_reference_defaults=defaults,
    )

    pat_urn = next(
        entry["fullUrl"] for entry in bundle["entry"] if entry["resource"] is pat
    )
    assert obs["subject"] == {"reference": pat_urn}


def test_external_reference_can_be_scoped_to_source_profile():
    target_profile = "http://example.org/StructureDefinition/molgen-observation"
    molgen = {
        "resourceType": "Observation",
        "meta": {"profile": [target_profile]},
    }
    other = {
        "resourceType": "Observation",
        "meta": {"profile": ["http://example.org/StructureDefinition/other"]},
    }
    defaults = [
        {
            "source_type": "Observation",
            "source_profile": "molgen-observation",
            "path": "subject",
            "value": {
                "reference": "Patient/eval-patient-1",
                "type": "Patient",
            },
        }
    ]

    BundleService.create_bundle(
        [molgen, other], external_reference_defaults=defaults
    )

    assert molgen["subject"]["reference"] == "Patient/eval-patient-1"
    assert "subject" not in other


def test_external_identifier_reference_needs_no_target_resource():
    procedure = {"resourceType": "Procedure"}
    defaults = [
        {
            "path": "Procedure.subject",
            "value": {
                "type": "Patient",
                "identifier": {
                    "system": "urn:example:patient-id",
                    "value": "patient-1",
                },
            },
        }
    ]

    BundleService.create_bundle(
        [procedure], external_reference_defaults=defaults
    )

    assert procedure["subject"] == defaults[0]["value"]
    assert procedure["subject"] is not defaults[0]["value"]


def test_external_reference_does_not_materialize_optional_parent():
    patient = {"resourceType": "Patient"}
    defaults = [
        {
            "path": "Patient.link.other",
            "reference": "RelatedPerson/external",
            "type": "RelatedPerson",
        }
    ]

    BundleService.create_bundle(
        [patient], external_reference_defaults=defaults
    )

    assert "link" not in patient


def test_resolve_references_false_also_disables_external_defaults():
    obs = {"resourceType": "Observation"}
    defaults = [
        {
            "path": "Observation.subject",
            "reference": "Patient/eval-patient-1",
        }
    ]

    BundleService.create_bundle(
        [obs],
        resolve_references=False,
        external_reference_defaults=defaults,
    )

    assert "subject" not in obs


# --------------------------------------------------------------------------- #
# _if_none_exist
# --------------------------------------------------------------------------- #


def test_if_none_exist_identifier_quotes_value():
    res = {
        "resourceType": "Patient",
        "identifier": [{"system": "http://sys", "value": "ab cd"}],
    }
    assert BundleService._if_none_exist(res) == "identifier=http://sys|ab%20cd"


def test_if_none_exist_medication_uses_code():
    res = {
        "resourceType": "Medication",
        "code": {"coding": [{"system": "http://atc", "code": "A01"}]},
    }
    assert BundleService._if_none_exist(res) == "code=http://atc|A01"


def test_if_none_exist_none_without_identifier_or_medication_code():
    assert BundleService._if_none_exist({"resourceType": "Patient"}) is None
    assert BundleService._if_none_exist({"resourceType": "Medication"}) is None


def test_if_none_exist_identifier_missing_system_or_value_returns_none():
    res = {"resourceType": "Patient", "identifier": [{"system": "http://sys"}]}
    assert BundleService._if_none_exist(res) is None


# --------------------------------------------------------------------------- #
# _specs_from_structure_maps / profile-name -> type resolution
# --------------------------------------------------------------------------- #


def test_specs_from_structure_maps_resolves_profile_target_via_group_input():
    sm = {
        "group": [
            {
                "input": [{"mode": "target", "type": "Condition"}],
                "rule": [
                    {
                        "name": "TODO-resolve-reference-Observation-subject",
                        "documentation": (
                            "Reference<Observation.subject> → "
                            "http://example.org/StructureDefinition/MinimalCondition3"
                        ),
                    }
                ],
            }
        ],
        "structure": [
            {
                "mode": "target",
                "url": "http://example.org/StructureDefinition/MinimalCondition3",
            }
        ],
    }
    specs = BundleService._specs_from_structure_maps([sm])
    # 4th element = the declaring map's target profile canonical, used to scope wiring.
    assert specs == [
        (
            "Observation",
            "subject",
            "Condition",
            "http://example.org/StructureDefinition/MinimalCondition3",
        )
    ]


def test_specs_from_structure_maps_dedups_and_recurses_nested_rules():
    sm = {
        "group": [
            {
                "rule": [
                    {
                        "name": "outer",
                        "rule": [
                            {
                                "name": "TODO-resolve-reference-a",
                                "documentation": "Reference<A.b> → Patient",
                            },
                            {
                                "name": "TODO-resolve-reference-a-dup",
                                "documentation": "Reference<A.b> → Patient",
                            },
                        ],
                    }
                ]
            }
        ]
    }
    specs = BundleService._specs_from_structure_maps([sm])
    # No target structure declared → source_profile is None (4th element).
    assert specs == [("A", "b", "Patient", None)]


# --------------------------------------------------------------------------- #
# _build_array_paths / _collect_array_paths
# --------------------------------------------------------------------------- #


def test_build_array_paths_collects_multi_cardinality_and_nested_children():
    fields = [
        {
            "path": "Encounter.diagnosis",
            "cardinality": {"max": "*"},
            "children": [
                {"path": "Encounter.diagnosis.condition", "cardinality": {"max": "1"}}
            ],
        },
        {"path": "Encounter.status", "cardinality": {"max": "1"}},
        {
            "path": "Encounter.identifier",
            "cardinality": {"max": "2"},
            "slices": [
                {"path": "Encounter.identifier.extra", "cardinality": {"max": "5"}}
            ],
        },
    ]
    obj = root_obj("http://example.org/enc", "Encounter", fields)
    registry = make_registry([("http://example.org/enc", obj)])
    result = BundleService._build_array_paths(registry)
    assert result["Encounter"] == {"diagnosis", "identifier", "identifier.extra"}


def test_build_array_paths_empty_without_registry():
    assert BundleService._build_array_paths(None) == {}


def test_build_array_paths_skips_objects_without_type_or_fields():
    obj = RegistryObject(data=SimpleNamespace(url="u"), res_type=None)
    obj.mappable_fields = []
    registry = make_registry([("u", obj)])
    assert BundleService._build_array_paths(registry) == {}


# --------------------------------------------------------------------------- #
# _resolve_unresolved_required: auto-wiring & warn-only paths
# --------------------------------------------------------------------------- #


def _ref_field(path, target, min_=1, ftype=None):
    return {
        "path": path,
        "reference_target": target,
        "cardinality": {"min": min_},
        "type": ftype or [{"code": "Reference"}],
    }


def test_resolve_unresolved_required_auto_wires_single_candidate():
    obj = root_obj(
        "http://example.org/ms",
        "MedicationStatement",
        [_ref_field("MedicationStatement.medication", "Medication")],
    )
    registry = make_registry([("http://example.org/ms", obj)])
    ms = {"resourceType": "MedicationStatement"}
    med = {"resourceType": "Medication"}
    urn_map = {id(ms): "urn:uuid:ms", id(med): "urn:uuid:med"}
    BundleService._resolve_unresolved_required([ms, med], registry, [], urn_map)
    assert ms["medication"] == {"reference": "urn:uuid:med"}


def test_resolve_unresolved_required_skips_already_wired_field():
    obj = root_obj(
        "http://example.org/ms",
        "MedicationStatement",
        [_ref_field("MedicationStatement.medication", "Medication")],
    )
    registry = make_registry([("http://example.org/ms", obj)])
    ms = {"resourceType": "MedicationStatement", "medication": {"reference": "Medication/123"}}
    med = {"resourceType": "Medication"}
    urn_map = {id(ms): "urn:uuid:ms", id(med): "urn:uuid:med"}
    BundleService._resolve_unresolved_required([ms, med], registry, [], urn_map)
    # untouched -- still points at the pre-existing manual reference
    assert ms["medication"] == {"reference": "Medication/123"}


def test_resolve_unresolved_required_warns_when_multiple_candidates(caplog):
    obj = root_obj(
        "http://example.org/ms",
        "MedicationStatement",
        [_ref_field("MedicationStatement.medication", "Medication")],
    )
    registry = make_registry([("http://example.org/ms", obj)])
    ms = {"resourceType": "MedicationStatement"}
    med1 = {"resourceType": "Medication"}
    med2 = {"resourceType": "Medication"}
    urn_map = {}
    with caplog.at_level("WARNING"):
        BundleService._resolve_unresolved_required(
            [ms, med1, med2], registry, [], urn_map
        )
    assert "medication" not in ms
    assert any("has no value and no TODO rule" in r.message for r in caplog.records)


def test_resolve_unresolved_required_self_reference_left_unwired():
    obj = root_obj(
        "http://example.org/acc",
        "Account",
        [_ref_field("Account.partOf", "Account")],
    )
    registry = make_registry([("http://example.org/acc", obj)])
    acc = {"resourceType": "Account"}
    urn_map = {id(acc): "urn:uuid:acc"}
    BundleService._resolve_unresolved_required([acc], registry, [], urn_map)
    assert "partOf" not in acc


def test_resolve_unresolved_required_ignores_non_root_objects():
    obj = RegistryObject(
        data=SimpleNamespace(url="u", type="MedicationStatement"), res_type="MedicationStatement"
    )
    obj.mappable_fields = [_ref_field("MedicationStatement.medication", "Medication")]
    # not marked as root
    registry = make_registry([("u", obj)])
    ms = {"resourceType": "MedicationStatement"}
    med = {"resourceType": "Medication"}
    BundleService._resolve_unresolved_required([ms, med], registry, [], {})
    assert "medication" not in ms


# --------------------------------------------------------------------------- #
# _collect_unresolvable_required / co-occurrence drop
# --------------------------------------------------------------------------- #


def test_collect_unresolvable_required_drops_resource_with_missing_driver_ref():
    obj = root_obj(
        "http://example.org/ms",
        "MedicationStatement",
        [_ref_field("MedicationStatement.medicationReference", "Medication")],
    )
    registry = make_registry([("http://example.org/ms", obj)])
    ms = {"resourceType": "MedicationStatement"}
    drop = BundleService._collect_unresolvable_required([ms], registry)
    assert drop == {id(ms)}


def test_collect_unresolvable_required_keeps_resource_when_target_present():
    obj = root_obj(
        "http://example.org/ms",
        "MedicationStatement",
        [_ref_field("MedicationStatement.medicationReference", "Medication")],
    )
    registry = make_registry([("http://example.org/ms", obj)])
    ms = {"resourceType": "MedicationStatement"}
    med = {"resourceType": "Medication"}
    drop = BundleService._collect_unresolvable_required([ms, med], registry)
    assert drop == set()


def test_collect_unresolvable_required_context_types_never_drop():
    obj = root_obj(
        "http://example.org/obs",
        "Observation",
        [_ref_field("Observation.subject", "Patient")],
    )
    registry = make_registry([("http://example.org/obs", obj)])
    obs = {"resourceType": "Observation"}
    drop = BundleService._collect_unresolvable_required([obs], registry)
    assert drop == set()


def test_collect_unresolvable_required_skips_optional_fields():
    obj = root_obj(
        "http://example.org/ms",
        "MedicationStatement",
        [_ref_field("MedicationStatement.medicationReference", "Medication", min_=0)],
    )
    registry = make_registry([("http://example.org/ms", obj)])
    ms = {"resourceType": "MedicationStatement"}
    drop = BundleService._collect_unresolvable_required([ms], registry)
    assert drop == set()


def test_collect_unresolvable_required_skips_nested_sliced_required_ref():
    """A required reference nested inside a backbone/slice the resource does not
    instantiate must NOT condemn it. The profile parser flattens a sliced
    requirement (Patient.identifier:versichertenId.assigner→Organization, required
    only for the *versichertenId* slice) to a bare path (identifier.assigner); a
    Patient carrying only a `pid` identifier (whose assigner is optional) validates
    with zero errors standalone, so it must be kept even with no Organization in
    the bundle. Only *top-level driver* refs drive the drop."""
    obj = root_obj(
        "http://example.org/patient",
        "Patient",
        [_ref_field("Patient.identifier.assigner", "Organization")],
    )
    registry = make_registry([("http://example.org/patient", obj)])
    # The resource HAS an identifier (so the bare parent path is present) but no
    # assigner — the exact shape that previously triggered the false-positive drop.
    pat = {"resourceType": "Patient", "identifier": [{"value": "PID-1"}]}
    drop = BundleService._collect_unresolvable_required([pat], registry)
    assert drop == set()


def test_collect_unresolvable_required_skips_mixed_choice_ref():
    # Not ref_only: a choice permitting both Reference and CodeableConcept is
    # satisfiable without the reference, so it must not cause a drop.
    field = _ref_field(
        "MedicationStatement.medicationReference",
        "Medication",
        ftype=[{"code": "Reference"}, {"code": "CodeableConcept"}],
    )
    obj = root_obj("http://example.org/ms", "MedicationStatement", [field])
    registry = make_registry([("http://example.org/ms", obj)])
    ms = {"resourceType": "MedicationStatement"}
    drop = BundleService._collect_unresolvable_required([ms], registry)
    assert drop == set()


def test_create_bundle_end_to_end_drops_co_occurrence_resource():
    obj = root_obj(
        "http://example.org/ms",
        "MedicationStatement",
        [_ref_field("MedicationStatement.medicationReference", "Medication")],
    )
    registry = make_registry([("http://example.org/ms", obj)])
    ms = {"resourceType": "MedicationStatement"}
    pat = {"resourceType": "Patient"}
    bundle = BundleService.create_bundle([ms, pat], registry=registry)
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert types == ["Patient"]


def test_external_reference_prevents_co_occurrence_drop():
    obj = root_obj(
        "http://example.org/ms",
        "MedicationStatement",
        [_ref_field("MedicationStatement.medicationReference", "Medication")],
    )
    registry = make_registry([("http://example.org/ms", obj)])
    medication_statement = {"resourceType": "MedicationStatement"}

    bundle = BundleService.create_bundle(
        [medication_statement],
        registry=registry,
        external_reference_defaults=[
            {
                "path": "MedicationStatement.medicationReference",
                "reference": "Medication/external-medication-1",
            }
        ],
    )

    assert bundle["entry"][0]["resource"]["medicationReference"] == {
        "reference": "Medication/external-medication-1",
        "type": "Medication",
    }


# --------------------------------------------------------------------------- #
# _set_nested / _get_nested
# --------------------------------------------------------------------------- #


def test_set_nested_creates_list_backbone_for_array_path():
    obj = {}
    BundleService._set_nested(obj, ["a", "b"], {"reference": "x"}, {"a"})
    assert obj == {"a": [{"b": {"reference": "x"}}]}


def test_set_nested_merges_into_existing_list_first_element():
    obj = {"a": [{"b": "old", "c": "keep"}]}
    BundleService._set_nested(obj, ["a", "b"], "new", {"a"})
    assert obj == {"a": [{"b": "new", "c": "keep"}]}


def test_set_nested_replaces_non_dict_intermediate():
    obj = {"a": "scalar"}
    BundleService._set_nested(obj, ["a", "b"], "v")
    assert obj == {"a": {"b": "v"}}


def test_set_nested_single_segment_direct_assignment():
    obj = {}
    BundleService._set_nested(obj, ["subject"], {"reference": "urn:uuid:1"})
    assert obj == {"subject": {"reference": "urn:uuid:1"}}


def test_get_nested_descends_into_list_first_element():
    assert BundleService._get_nested({"a": [{"b": 1}]}, ["a", "b"]) == 1


def test_get_nested_empty_list_returns_none():
    assert BundleService._get_nested({"a": []}, ["a", "b"]) is None


def test_get_nested_missing_key_returns_none():
    assert BundleService._get_nested({}, ["a", "b"]) is None


# --------------------------------------------------------------------------- #
# normalize_list_cardinality / _normalize_obj / _resolve_model_class
# --------------------------------------------------------------------------- #


def test_normalize_list_cardinality_wraps_bare_object_as_list():
    res = {
        "resourceType": "Patient",
        "identifier": {"system": "sys", "value": "v"},
    }
    out = BundleService.normalize_list_cardinality(res)
    assert out["identifier"] == [{"system": "sys", "value": "v"}]


def test_normalize_list_cardinality_recurses_into_nested_backbones():
    res = {
        "resourceType": "Patient",
        "name": {"family": "X", "given": "John"},
    }
    out = BundleService.normalize_list_cardinality(res)
    assert out["name"] == [{"family": "X", "given": ["John"]}]


def test_normalize_list_cardinality_unknown_resource_type_returns_unchanged():
    res = {"resourceType": "TotallyNotAFhirType", "foo": "bar"}
    out = BundleService.normalize_list_cardinality(res)
    assert out == {"resourceType": "TotallyNotAFhirType", "foo": "bar"}


def test_normalize_list_cardinality_non_dict_input_returned_as_is():
    assert BundleService.normalize_list_cardinality("not-a-dict") == "not-a-dict"


def test_normalize_list_cardinality_missing_resource_type_returned_as_is():
    res = {"foo": "bar"}
    assert BundleService.normalize_list_cardinality(res) == {"foo": "bar"}


def test_resolve_model_class_non_string_returns_none():
    assert BundleService._resolve_model_class(None) is None


def test_resolve_model_class_unknown_type_returns_none():
    assert BundleService._resolve_model_class("NotARealFhirType") is None
