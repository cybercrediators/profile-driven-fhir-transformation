"""Unit tests for mapping/derived_resources.py.

The module turns a Reference-valued extension whose target no declared root
satisfies into an additional map target. These cover the decisions it makes on its
own — what counts as a candidate, which value element the source fills, whether a
definition from the wrong FHIR release may be used — with plain dicts and
SimpleNamespace stand-ins, so no registry, cache or network is involved.
"""

from types import SimpleNamespace

import pytest

from mapping import derived_resources as dr

pytestmark = pytest.mark.unit


def _element(element_id, min_card=0, type_code=None, **extra):
    return SimpleNamespace(
        id=element_id,
        min=min_card,
        type=[SimpleNamespace(code=type_code)] if type_code else [],
        fixedString=None,
        fixedCode=None,
        fixedUri=None,
        patternString=None,
        **extra,
    )


def _bodyweight_sd(fhir_version="4.0.1"):
    """The shape that matters: a required decimal value plus unpinned unit/code."""
    return SimpleNamespace(
        url="http://hl7.org/fhir/StructureDefinition/bodyweight",
        kind="resource",
        type="Observation",
        fhirVersion=fhir_version,
        snapshot=SimpleNamespace(
            element=[
                _element("Observation.status", 1, "code"),
                _element("Observation.value[x]", 0, "Quantity"),
                _element("Observation.value[x]:valueQuantity.value", 1, "decimal"),
                _element("Observation.value[x]:valueQuantity.unit", 1, "string"),
                _element("Observation.value[x]:valueQuantity.code", 1, "code"),
                _element("Observation.component.value[x]", 0, "Quantity"),
            ]
        ),
    )


# ── reference_target_profiles ─────────────────────────────────────────────────
def test_reference_target_read_from_the_parsed_extension_definition():
    field = {
        "path": "Patient.extension",
        "id": "Patient.extension:bodyweight",
        "type": [
            {
                "code": "Extension",
                "profile": [
                    [
                        {"id": "Extension.url", "type": [{"code": "uri"}]},
                        {
                            "id": "Extension.value[x]",
                            "type": [
                                {
                                    "code": "Reference",
                                    "targetProfile": [
                                        "http://hl7.org/fhir/StructureDefinition/bodyweight"
                                    ],
                                }
                            ],
                        },
                    ]
                ],
            }
        ],
    }
    assert dr.reference_target_profiles(field) == [
        "http://hl7.org/fhir/StructureDefinition/bodyweight"
    ]


def test_non_reference_extension_yields_no_target():
    field = {
        "type": [
            {
                "code": "Extension",
                "profile": [[{"id": "Extension.value[x]", "type": [{"code": "code"}]}]],
            }
        ]
    }
    assert dr.reference_target_profiles(field) == []


def test_unparsed_extension_definition_yields_no_target():
    # Only the canonical is present, so there is nothing to read offline.
    field = {"type": [{"code": "Extension", "profile": ["http://example.org/ext"]}]}
    assert dr.reference_target_profiles(field) == []


# ── infer_value_element ───────────────────────────────────────────────────────
def test_decimal_source_resolves_to_the_quantity_value():
    path, leaf_type, ambiguous = dr.infer_value_element(
        _bodyweight_sd(), "decimal", "Observation"
    )
    assert path == "value[x]:valueQuantity.value"
    assert leaf_type == "decimal"
    assert ambiguous == []


def test_integer_source_widens_to_a_decimal_value():
    path, _, _ = dr.infer_value_element(_bodyweight_sd(), "integer", "Observation")
    assert path == "value[x]:valueQuantity.value"


def test_string_source_resolves_to_unit_not_code():
    # `code` is FHIR `code`, which a plain string does not widen into, so `unit` is
    # the only candidate and the choice is forced rather than guessed.
    path, leaf_type, ambiguous = dr.infer_value_element(
        _bodyweight_sd(), "string", "Observation"
    )
    assert path == "value[x]:valueQuantity.unit"
    assert leaf_type == "string"
    assert ambiguous == []


def test_two_compatible_required_value_leaves_are_refused_as_ambiguous():
    # With no way to tell which leaf is meant, picking one would silently invent
    # intent; the caller reports the gap instead.
    sd = SimpleNamespace(
        snapshot=SimpleNamespace(
            element=[
                _element("Observation.value[x]:valueQuantity.value", 1, "decimal"),
                _element("Observation.value[x]:valueRatio.numerator", 1, "decimal"),
            ]
        )
    )
    path, _, ambiguous = dr.infer_value_element(sd, "decimal", "Observation")
    assert path is None
    assert ambiguous == [
        "value[x]:valueQuantity.value",
        "value[x]:valueRatio.numerator",
    ]


def test_source_with_no_compatible_value_element_is_refused():
    path, _, ambiguous = dr.infer_value_element(
        _bodyweight_sd(), "boolean", "Observation"
    )
    assert path is None
    assert ambiguous == []


def test_optional_value_elements_are_never_candidates():
    sd = SimpleNamespace(
        snapshot=SimpleNamespace(
            element=[_element("Observation.value[x]:valueQuantity.value", 0, "decimal")]
        )
    )
    path, _, _ = dr.infer_value_element(sd, "decimal", "Observation")
    assert path is None


# ── pin_quantity_unit ─────────────────────────────────────────────────────────
def test_declared_unit_is_pinned_on_unit_and_code():
    sd = _bodyweight_sd()
    pinned = dr.pin_quantity_unit(
        sd, "value[x]:valueQuantity.value", "kg", "Observation"
    )
    assert sorted(pinned) == [
        "Observation.value[x]:valueQuantity.code",
        "Observation.value[x]:valueQuantity.unit",
    ]
    by_id = {e.id: e for e in sd.snapshot.element}
    assert by_id["Observation.value[x]:valueQuantity.unit"].fixedString == "kg"
    assert by_id["Observation.value[x]:valueQuantity.code"].fixedCode == "kg"


def test_a_unit_the_profile_already_pins_is_left_alone():
    sd = _bodyweight_sd()
    by_id = {e.id: e for e in sd.snapshot.element}
    by_id["Observation.value[x]:valueQuantity.code"].fixedCode = "[lb_av]"
    dr.pin_quantity_unit(sd, "value[x]:valueQuantity.value", "kg", "Observation")
    assert by_id["Observation.value[x]:valueQuantity.code"].fixedCode == "[lb_av]"


def test_no_unit_declared_pins_nothing():
    sd = _bodyweight_sd()
    assert dr.pin_quantity_unit(sd, "value[x]:valueQuantity.value", None, "Observation") == []


# ── _declared_unit ────────────────────────────────────────────────────────────
def test_ucum_unit_is_read_from_the_standard_allowed_units_extension():
    field = {
        "path": "Source.gewicht",
        "extension": [
            {
                "url": dr._ALLOWED_UNITS_URL,
                "valueCodeableConcept": {
                    "coding": [{"system": dr._UCUM_SYSTEM, "code": "kg"}]
                },
            }
        ],
    }
    assert dr._declared_unit(field) == "kg"


def test_a_non_ucum_coding_is_not_treated_as_a_unit():
    field = {
        "extension": [
            {
                "url": dr._ALLOWED_UNITS_URL,
                "valueCodeableConcept": {
                    "coding": [{"system": "http://example.org/units", "code": "kg"}]
                },
            }
        ]
    }
    assert dr._declared_unit(field) is None


# ── release handling ──────────────────────────────────────────────────────────
def test_release_key_collapses_patch_versions():
    assert dr._release_key("4.0.1") == dr._release_key("4.0")
    assert dr._release_key("5.0.0") != dr._release_key("4.0.1")


def test_wrong_release_definition_is_rejected_in_favour_of_the_pinned_package(
    monkeypatch,
):
    # The cache can hold an R5 profile for an R4 project; generating a resource from
    # it would emit elements R4 does not define.
    r5 = _bodyweight_sd("5.0.0")
    r4 = {"resourceType": "StructureDefinition", "fhirVersion": "4.0.1"}
    monkeypatch.setattr(dr, "resolve_url", lambda url, state: r5)
    monkeypatch.setattr(dr, "get_resource_from_local_package", lambda url, conf: r4)
    monkeypatch.setattr(
        dr.utils, "json_to_obj", lambda raw, kind: SimpleNamespace(**raw)
    )
    sd, mismatched = dr._resolve_for_release(
        SimpleNamespace(conf={}), "http://x", "4.0"
    )
    assert mismatched is None
    assert sd.fhirVersion == "4.0.1"


def test_wrong_release_with_no_local_fallback_is_reported_not_used(monkeypatch):
    monkeypatch.setattr(dr, "resolve_url", lambda url, state: _bodyweight_sd("5.0.0"))
    monkeypatch.setattr(dr, "get_resource_from_local_package", lambda url, conf: None)
    sd, mismatched = dr._resolve_for_release(
        SimpleNamespace(conf={}), "http://x", "4.0"
    )
    assert sd is None
    assert mismatched == "5.0"


def test_matching_release_is_used_directly(monkeypatch):
    r4 = _bodyweight_sd("4.0.1")
    monkeypatch.setattr(dr, "resolve_url", lambda url, state: r4)
    sd, mismatched = dr._resolve_for_release(
        SimpleNamespace(conf={}), "http://x", "4.0"
    )
    assert sd is r4 and mismatched is None


# ── mapping overlay ───────────────────────────────────────────────────────────
def _spec(**over):
    base = dict(
        target_profile="http://hl7.org/fhir/StructureDefinition/bodyweight",
        target_type="Observation",
        target_identity="bodyweight",
        owner_profile_url="http://example.org/WAVESPatient",
        owner_res_type="Patient",
        owner_path="Patient.extension:bodyweight",
        extension_url="http://example.org/StructureDefinition/body-weight-extension",
        source_field="Source.gewicht",
        value_path="value[x]:valueQuantity.value",
        value_type="decimal",
    )
    base.update(over)
    return dr.DerivedResource(**base)


def test_overlay_rebinds_the_source_without_touching_the_authored_table():
    # The shared table is one target per source, and this source is already spent on
    # the extension that references the derived resource — so the inferred binding
    # is scoped to the derived profile's own map.
    base = {"Source.gewicht": "Patient.extension:bodyweight"}
    overlay = dr.mapping_overlay(base, _spec())
    assert overlay["Source.gewicht"] == "bodyweight.value[x]:valueQuantity.value"
    assert base["Source.gewicht"] == "Patient.extension:bodyweight"


def test_overlay_leaves_the_table_alone_when_no_value_was_inferred():
    base = {"Source.gewicht": "Patient.extension:bodyweight"}
    overlay = dr.mapping_overlay(base, _spec(value_path=None, value_type=None))
    assert overlay == base


# ── _authored_source ──────────────────────────────────────────────────────────
def test_authored_source_matches_the_profile_identity_form():
    table = {"Source.gewicht": "WAVESPatient.extension:bodyweight"}
    assert (
        dr._authored_source(
            table, "Patient.extension:bodyweight", "Patient", "WAVESPatient"
        )
        == "Source.gewicht"
    )


def test_authored_source_matches_the_resource_type_form():
    table = {"Source.gewicht": "Patient.extension:bodyweight"}
    assert (
        dr._authored_source(
            table, "Patient.extension:bodyweight", "Patient", "WAVESPatient"
        )
        == "Source.gewicht"
    )


def test_unbound_extension_has_no_authored_source():
    table = {"Source.other": "Patient.gender"}
    assert (
        dr._authored_source(
            table, "Patient.extension:bodyweight", "Patient", "WAVESPatient"
        )
        is None
    )
