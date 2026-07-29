"""Unit tests for structure_definition_parser: slice helpers, class lookup, the processed
guard, baseDefinition dependency recording, and mappable-field hierarchy building.
"""

from types import SimpleNamespace

import pytest

from fhir.resources.R4B.structuredefinition import (
    StructureDefinition,
    StructureDefinitionSnapshot,
)
from fhir.resources.R4B.elementdefinition import ElementDefinition, ElementDefinitionType

from data_handling.registry.registry_object import RegistryObject
from parser.resource_parser import structure_definition_parser as sdp

pytestmark = pytest.mark.unit


def _elem(path, **kw):
    kw.setdefault("id", path)
    return ElementDefinition(path=path, **kw)


def _sd_ro(elements, *, url="http://example.org/sd/Test", base="http://hl7.org/fhir/StructureDefinition/Patient"):
    snap = StructureDefinitionSnapshot(element=elements)
    sd = StructureDefinition(
        url=url, name="Test", status="active", kind="resource",
        abstract=False, type="Patient", baseDefinition=base, snapshot=snap,
    )
    return RegistryObject(sd, "StructureDefinition")


def _app_state():
    used_by = []
    registry = SimpleNamespace(
        get_obj_by_name=lambda u: None,
        add_to_used_by=lambda src, tgt: used_by.append((src, tgt)),
    )
    return SimpleNamespace(registry=registry, used_by=used_by)


# --------------------------------------------------------------------------- #
# slice helpers
# --------------------------------------------------------------------------- #


def test_extract_slices_returns_only_named_slices():
    a = _elem("Patient.identifier", sliceName="mrn")
    b = _elem("Patient.identifier")
    assert sdp.extract_slices([a, b]) == [a]


def test_get_slices_matches_root_path_and_named():
    a = _elem("Patient.identifier", sliceName="mrn")
    b = _elem("Patient.name", sliceName="official")
    assert sdp.get_slices([a, b], "Patient.identifier") == [a]


def test_get_slices_uses_exact_path_not_substring():
    exact = _elem(
        "Patient.identifier",
        id="Patient.identifier:mrn",
        sliceName="mrn",
    )
    collision = _elem(
        "Patient.identifierHistory",
        id="Patient.identifierHistory:old",
        sliceName="old",
    )
    assert sdp.get_slices([exact, collision], "Patient.identifier") == [exact]


def test_multiple_discriminators_do_not_duplicate_slices():
    elements = [
        _elem(
            "Patient.identifier",
            min=0,
            max="*",
            type=[ElementDefinitionType(code="Identifier")],
            slicing={
                "discriminator": [
                    {"type": "value", "path": "system"},
                    {"type": "value", "path": "type"},
                ],
                "ordered": True,
                "rules": "closed",
            },
        ),
        _elem(
            "Patient.identifier",
            id="Patient.identifier:mrn",
            sliceName="mrn",
            min=0,
            max="1",
            type=[ElementDefinitionType(code="Identifier")],
        ),
    ]
    ro = _sd_ro(elements)
    sdp.parse_structure_definition(ro, _app_state())
    identifier = next(f for f in ro.mappable_fields if f["path"] == "Patient.identifier")

    assert len(identifier["slices"]) == 1
    assert identifier["slicing"] == {
        "discriminators": [
            {"path": "system", "type": "value"},
            {"path": "type", "type": "value"},
        ],
        "ordered": True,
        "rules": "closed",
    }


def test_reslices_are_nested_by_element_definition_identity():
    elements = [
        _elem(
            "Patient.identifier",
            min=0,
            max="*",
            type=[ElementDefinitionType(code="Identifier")],
            slicing={
                "discriminator": [{"type": "value", "path": "system"}],
                "rules": "open",
            },
        ),
        _elem(
            "Patient.identifier",
            id="Patient.identifier:national",
            sliceName="national",
            min=0,
            max="*",
            type=[ElementDefinitionType(code="Identifier")],
        ),
        _elem(
            "Patient.identifier",
            id="Patient.identifier:national/ssn",
            sliceName="national/ssn",
            min=0,
            max="1",
            type=[ElementDefinitionType(code="Identifier")],
        ),
        _elem(
            "Patient.identifier.system",
            id="Patient.identifier:national.system",
            min=1,
            max="1",
            type=[ElementDefinitionType(code="uri")],
            fixedUri="urn:national",
        ),
        _elem(
            "Patient.identifier.value",
            id="Patient.identifier:national/ssn.value",
            min=1,
            max="1",
            type=[ElementDefinitionType(code="string")],
        ),
    ]
    ro = _sd_ro(elements)
    sdp.parse_structure_definition(ro, _app_state())
    identifier = next(f for f in ro.mappable_fields if f["path"] == "Patient.identifier")

    assert [s["slice_identity"] for s in identifier["slices"]] == [
        "Patient.identifier:national"
    ]
    assert [s["slice_identity"] for s in identifier["slices"][0]["slices"]] == [
        "Patient.identifier:national/ssn"
    ]
    ssn = identifier["slices"][0]["slices"][0]
    assert {
        child["id"] for child in ssn["children"]
    } == {
        "Patient.identifier:national/ssn.system",
        "Patient.identifier:national/ssn.value",
    }
    inherited_system = next(
        child for child in ssn["children"] if child["path"].endswith(".system")
    )
    assert inherited_system["fixed_value"] == "urn:national"


# --------------------------------------------------------------------------- #
# slice descendants (sub-elements under a slice — discriminator / value types)
# --------------------------------------------------------------------------- #


def test_extract_slice_descendants_detects_subelements_not_roots():
    # slice ROOT carries sliceName; its sub-elements do not, and their id has a `:name.`
    root = _elem("DeviceDefinition.property", id="DeviceDefinition.property:weight", sliceName="weight")
    typ = _elem("DeviceDefinition.property.type", id="DeviceDefinition.property:weight.type")
    val = _elem("DeviceDefinition.property.valueQuantity", id="DeviceDefinition.property:weight.valueQuantity")
    base = _elem("DeviceDefinition.property", id="DeviceDefinition.property")
    desc = sdp.extract_slice_descendants([root, typ, val, base])
    assert typ in desc and val in desc
    assert root not in desc and base not in desc  # slice root + unsliced element excluded


def test_parse_attaches_slice_descendants_as_slice_children():
    # A sliced backbone whose discriminator + value live in sub-elements (id-distinguished,
    # path-colliding with the unsliced element) must nest under the slice, not the base.
    elements = [
        _elem("Patient.identifier", min=0, max="*",
              type=[ElementDefinitionType(code="Identifier")],
              slicing={"discriminator": [{"type": "pattern", "path": "system"}], "rules": "open"}),
        _elem("Patient.identifier", id="Patient.identifier:mrn", sliceName="mrn",
              min=0, max="1", type=[ElementDefinitionType(code="Identifier")]),
        _elem("Patient.identifier.system", id="Patient.identifier:mrn.system",
              min=1, max="1", type=[ElementDefinitionType(code="uri")],
              fixedUri="http://example.org/mrn"),
    ]
    ro = _sd_ro(elements)
    sdp.parse_structure_definition(ro, _app_state())
    ident = next(f for f in ro.mappable_fields if f["path"] == "Patient.identifier")
    assert len(ident["slices"]) == 1
    mrn = ident["slices"][0]
    child_paths = {c["path"] for c in mrn.get("children", [])}
    assert "Patient.identifier.system" in child_paths
    # the descendant must NOT also bind to the unsliced identifier field's children
    assert all("system" not in c.get("path", "") for c in ident.get("children", []))


def test_parse_retains_prohibited_slice_child_for_exists_discriminator():
    elements = [
        _elem(
            "Patient.identifier",
            min=0,
            max="*",
            type=[ElementDefinitionType(code="Identifier")],
            slicing={
                "discriminator": [{"type": "exists", "path": "period"}],
                "rules": "open",
            },
        ),
        _elem(
            "Patient.identifier",
            id="Patient.identifier:without-period",
            sliceName="without-period",
            min=0,
            max="1",
            type=[ElementDefinitionType(code="Identifier")],
        ),
        _elem(
            "Patient.identifier.period",
            id="Patient.identifier:without-period.period",
            min=0,
            max="0",
            type=[ElementDefinitionType(code="Period")],
        ),
    ]
    ro = _sd_ro(elements)

    sdp.parse_structure_definition(ro, _app_state())

    identifier = next(
        field for field in ro.mappable_fields if field["path"] == "Patient.identifier"
    )
    period = identifier["slices"][0]["children"][0]
    assert period["id"] == "Patient.identifier:without-period.period"
    assert period["cardinality"]["max"] == "0"
    assert period["is_prohibited"] is True


def test_parse_indexes_extension_slice_below_backbone_slice():
    """A slice below a sliced backbone child belongs to that child, not the root."""

    canonical = "http://example.org/StructureDefinition/contact-role"
    elements = [
        _elem(
            "Patient.contact",
            min=0,
            max="*",
            type=[ElementDefinitionType(code="BackboneElement")],
            slicing={
                "discriminator": [{"type": "value", "path": "extension.url"}],
                "rules": "open",
            },
        ),
        _elem(
            "Patient.contact",
            id="Patient.contact:research",
            sliceName="research",
            min=1,
            max="1",
            type=[ElementDefinitionType(code="BackboneElement")],
        ),
        _elem(
            "Patient.contact.extension",
            id="Patient.contact:research.extension",
            min=1,
            max="*",
            type=[ElementDefinitionType(code="Extension")],
            slicing={
                "discriminator": [{"type": "value", "path": "url"}],
                "rules": "open",
            },
        ),
        _elem(
            "Patient.contact.extension",
            id="Patient.contact:research.extension:role",
            sliceName="role",
            min=1,
            max="1",
            type=[
                ElementDefinitionType(code="Extension", profile=[canonical])
            ],
        ),
        _elem(
            "Patient.contact.extension.value[x]",
            id="Patient.contact:research.extension:role.value[x]",
            min=1,
            max="1",
            type=[ElementDefinitionType(code="string")],
        ),
    ]
    ro = _sd_ro(elements)

    sdp.parse_structure_definition(ro, _app_state())

    contact = next(
        field for field in ro.mappable_fields if field["path"] == "Patient.contact"
    )
    research = contact["slices"][0]
    extension = next(
        child
        for child in research["children"]
        if child["id"] == "Patient.contact:research.extension"
    )
    assert extension["slicing"]["discriminators"] == [
        {"path": "url", "type": "value"}
    ]
    assert [item["slice_identity"] for item in extension["slices"]] == [
        "Patient.contact:research.extension:role"
    ]
    role = extension["slices"][0]
    assert role["extension_url"] == canonical
    assert [child["id"] for child in role["children"]] == [
        "Patient.contact:research.extension:role.value[x]"
    ]
    assert all(
        child["id"] != "Patient.contact:research.extension:role.value[x]"
        for child in research["children"]
    )


# --------------------------------------------------------------------------- #
# minimal-mode prune keeps a slice's mapped descendants (marker/[x]-tolerant)
# --------------------------------------------------------------------------- #


def test_prune_keeps_slice_mapped_descendant_marker_and_choice_tolerant():
    from mapping.fml_map import StructureMapGenerator

    gen = object.__new__(StructureMapGenerator)
    # slice child paths drop the slice marker and use value[x]; the mapped path keeps the
    # marker and uses `value` — pruning must keep `answer` (ancestor) and `answer.value[x]`.
    field = {
        "path": "QuestionnaireResponse.item",
        "children": [
            {"path": "QuestionnaireResponse.item.linkId", "fixed_value": "1.1"},
            {
                "path": "QuestionnaireResponse.item.answer",
                "children": [
                    {"path": "QuestionnaireResponse.item.answer.value[x]"},
                ],
            },
        ],
    }
    mapped = {"QuestionnaireResponse.item:hfall.answer.value"}
    gen._prune_nested_components(field, allowed_paths=set(), mapped_paths=mapped)
    kept = {c["path"] for c in field["children"]}
    assert "QuestionnaireResponse.item.answer" in kept  # ancestor backbone survived
    answer = next(c for c in field["children"] if c["path"].endswith(".answer"))
    assert any("value" in g["path"] for g in answer.get("children", []))


def test_prune_keeps_mapped_child_in_embedded_type_structure():
    """Plain complex datatypes retain mapped leaves in their raw parser shape."""
    from mapping.fml_map import StructureMapGenerator

    gen = object.__new__(StructureMapGenerator)
    field = {
        "path": "Patient.identifier",
        "type": [
            {
                "code": "Identifier",
                "type_structure": [
                    {"path": "Patient.identifier.system", "type": "uri"},
                    {"path": "Patient.identifier.value", "type": "string"},
                    {"path": "Patient.identifier.use", "type": "code"},
                ],
            }
        ],
    }

    gen._prune_nested_components(
        field,
        allowed_paths={"Patient", "Patient.identifier", "Patient.identifier.value"},
        mapped_paths={"Patient.identifier.value"},
    )

    retained = field["type"][0]["type_structure"]
    assert [child["path"] for child in retained] == ["Patient.identifier.value"]
    assert "type_structure" not in field  # pruning does not mutate the representation


# --------------------------------------------------------------------------- #
# get_fhir_class
# --------------------------------------------------------------------------- #


def test_get_fhir_class_known():
    assert sdp.get_fhir_class("Patient") is not None


def test_get_fhir_class_unknown_returns_none():
    assert sdp.get_fhir_class("NotARealResource") is None


# --------------------------------------------------------------------------- #
# parse_structure_definition
# --------------------------------------------------------------------------- #


def test_parse_already_processed_is_noop():
    ro = _sd_ro([_elem("Patient.active", min=0, max="1", type=[ElementDefinitionType(code="boolean")])])
    ro.set_processed()
    out, _ = sdp.parse_structure_definition(ro, _app_state())
    assert out is ro
    assert ro.mappable_fields == []  # not parsed


def test_parse_marks_processed_and_records_base_definition_used_by():
    ro = _sd_ro([_elem("Patient.active", min=0, max="1", type=[ElementDefinitionType(code="boolean")])])
    state = _app_state()
    sdp.parse_structure_definition(ro, state)
    assert ro.is_processed()
    assert (ro.data.baseDefinition, ro.data.url) in state.used_by


def test_parse_rejects_profile_without_usable_snapshot():
    sd = StructureDefinition(
        url="http://example.org/sd/DifferentialOnly",
        name="DifferentialOnly",
        status="active",
        kind="resource",
        abstract=False,
        type="Patient",
        baseDefinition="http://hl7.org/fhir/StructureDefinition/Patient",
    )
    ro = RegistryObject(sd, "StructureDefinition")

    with pytest.raises(ValueError, match="Compile/expand snapshots"):
        sdp.parse_structure_definition(ro, _app_state())

    assert not ro.is_processed()


def test_parse_builds_mappable_field_hierarchy():
    elements = [
        _elem("Patient.active", min=0, max="1", type=[ElementDefinitionType(code="boolean")]),
        _elem("Patient.contact", min=0, max="*", type=[ElementDefinitionType(code="BackboneElement")]),
        _elem("Patient.contact.gender", min=0, max="1", type=[ElementDefinitionType(code="code")]),
    ]
    ro = _sd_ro(elements)
    sdp.parse_structure_definition(ro, _app_state())

    top_paths = {f["path"] for f in ro.mappable_fields}
    assert top_paths == {"Patient.active", "Patient.contact"}
    contact = next(f for f in ro.mappable_fields if f["path"] == "Patient.contact")
    child_paths = {c["path"] for c in contact["children"]}
    assert "Patient.contact.gender" in child_paths


def test_content_reference_inherits_and_rebases_target_children():
    elements = [
        _elem(
            "Patient.contact",
            min=0,
            max="*",
            type=[ElementDefinitionType(code="BackboneElement")],
        ),
        _elem(
            "Patient.contact.gender",
            min=0,
            max="1",
            type=[ElementDefinitionType(code="code")],
        ),
        _elem(
            "Patient.link",
            min=0,
            max="*",
            contentReference="#Patient.contact",
        ),
    ]
    ro = _sd_ro(elements)
    sdp.parse_structure_definition(ro, _app_state())
    link = next(f for f in ro.mappable_fields if f["path"] == "Patient.link")

    assert link["content_reference_resolved"] is True
    assert link["content_reference_target"] == "Patient.contact"
    assert link["type"][0]["code"] == "BackboneElement"
    assert link["children"][0]["path"] == "Patient.link.gender"
    assert link["children"][0]["id"] == "Patient.link.gender"
