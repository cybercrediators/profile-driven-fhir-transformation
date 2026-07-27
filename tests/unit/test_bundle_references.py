"""Reference wiring in BundleService, incl. the profile-typed-target fix.

A `Reference(SomeProfile)` carries the *profile id* as the target type in the
StructureMap rule documentation, but the present resource is stamped with the
base `resourceType`. `_wire_references` must resolve the profile id → base type
(via the registry-derived profile_type_map) or the reference is silently skipped.
"""

import pytest

from controller.bundle_service import BundleService

pytestmark = pytest.mark.unit


def _obs_pat():
    obs = {"resourceType": "Observation", "status": "final"}
    pat = {"resourceType": "Patient", "name": [{"family": "X"}]}
    return obs, pat


def test_profile_typed_reference_wires_with_profile_map():
    obs, pat = _obs_pat()
    urn = {id(obs): "urn:uuid:obs", id(pat): "urn:uuid:pat"}
    BundleService._wire_references(
        [obs, pat], [("Observation", "subject", "synth-l1-patient")], urn, {},
        {"synth-l1-patient": "Patient"},
    )
    assert obs.get("subject") == {"reference": "urn:uuid:pat"}


def test_profile_typed_reference_skipped_without_map():
    # Without the profile→type map the profile id matches no resourceType.
    obs, pat = _obs_pat()
    urn = {id(obs): "urn:uuid:obs", id(pat): "urn:uuid:pat"}
    BundleService._wire_references(
        [obs, pat], [("Observation", "subject", "synth-l1-patient")], urn, {}, {}
    )
    assert "subject" not in obs


def test_base_type_reference_still_wires():
    obs, pat = _obs_pat()
    urn = {id(obs): "urn:uuid:obs", id(pat): "urn:uuid:pat"}
    BundleService._wire_references(
        [obs, pat], [("Observation", "subject", "Patient")], urn, {}, {}
    )
    assert obs.get("subject") == {"reference": "urn:uuid:pat"}


def test_choice_target_picks_present_type():
    # "Patient|Group" — only Patient present, profile map empty.
    obs, pat = _obs_pat()
    urn = {id(obs): "urn:uuid:obs", id(pat): "urn:uuid:pat"}
    BundleService._wire_references(
        [obs, pat], [("Observation", "subject", "Patient|Group")], urn, {}, {}
    )
    assert obs.get("subject") == {"reference": "urn:uuid:pat"}


def test_build_profile_type_map_empty_without_registry():
    assert BundleService._build_profile_type_map(None) == {}


def test_reference_into_absent_optional_backbone_is_skipped():
    # Patient.link.other → Patient, but the transform produced no `link` backbone.
    # Wiring must NOT materialise an (incomplete, required-field-missing) link entry.
    pat = {"resourceType": "Patient", "name": [{"family": "X"}]}
    other = {"resourceType": "Patient", "name": [{"family": "Y"}]}
    urn = {id(pat): "urn:uuid:pat", id(other): "urn:uuid:other"}
    BundleService._wire_references(
        [pat, other], [("Patient", "link.other", "Patient")], urn, {}, {}
    )
    assert "link" not in pat  # backbone not created


def test_reference_into_present_backbone_still_wires():
    # If the backbone already exists, the nested reference wires normally.
    pat = {"resourceType": "Patient", "link": [{"type": "seealso"}]}
    rp = {"resourceType": "RelatedPerson"}
    urn = {id(pat): "urn:uuid:pat", id(rp): "urn:uuid:rp"}
    BundleService._wire_references(
        [pat, rp], [("Patient", "link.other", "RelatedPerson")], urn,
        {"Patient": {"link"}}, {},
    )
    assert pat["link"][0].get("other") == {"reference": "urn:uuid:rp"}
    assert pat["link"][0].get("type") == "seealso"  # existing content preserved


def test_choice_reference_field_expands_x_to_typed_name():
    # A choice element `code[x]` holding a Reference value must serialise as
    # `codeReference` (FHIR poly-property naming), not the literal `code[x]`.
    # `_iter_reference_fields` is the single point that must expand the leaf.
    fields = [
        {
            "path": "SomeResource.code[x]",
            "reference_target": "http://example.org/StructureDefinition/some-profile",
            "cardinality": {"min": 1, "max": "1"},
            "type": [{"code": "Reference"}, {"code": "CodeableConcept"}],
        }
    ]
    out = list(BundleService._iter_reference_fields(fields))
    assert len(out) == 1
    relative, raw, is_required, ref_only = out[0]
    assert relative == "codeReference"  # not "code[x]"
    assert raw == "some-profile"
    assert is_required is True
    # mixed choice (Reference|CodeableConcept) → satisfiable without the reference
    assert ref_only is False


def test_ref_only_choice_still_expands_x():
    # A choice narrowed to only Reference (e.g. a `foo[x] only Reference(Bar)`) is
    # ref_only=True and must also expand to the typed property name.
    fields = [
        {
            "path": "SomeResource.thing[x]",
            "reference_target": "Bar",
            "cardinality": {"min": 1, "max": "1"},
            "type": [{"code": "Reference"}],
        }
    ]
    relative, raw, is_required, ref_only = next(
        iter(BundleService._iter_reference_fields(fields))
    )
    assert relative == "thingReference"
    assert ref_only is True


def test_collect_todo_refs_expands_choice_element_to_typed_reference():
    # The SM-declared wiring path parses the field path from the rule documentation;
    # a choice element (medication[x]) must expand to its typed form there too, or
    # _set_nested writes the literal `[x]` key (rejected by HAPI-1809) and the two
    # wiring paths (_iter_reference_fields vs TODO specs) disagree about the field.
    rule = {
        "name": "TODO-resolve-reference-MedicationStatement-medication",
        "documentation": (
            "Reference<MedicationStatement.medication[x]> → Medication "
            "— resolve via bundle assembler"
        ),
    }
    specs = []
    BundleService._collect_todo_refs(rule, specs)
    # Specs carry the declaring map's target profile (None when not supplied) so
    # _wire_references can scope a reference to the profile that declared it.
    assert specs == [("MedicationStatement", "medicationReference", "Medication", None)]
