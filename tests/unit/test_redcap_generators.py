"""Unit tests for REDCap -> FHIR ConceptMap generation (plugins/redcap/generators.py)."""

import pytest

from plugins.redcap.generators import build_concept_map

pytestmark = pytest.mark.unit


NUMERIC_CHOICES = [
    {"code": "1", "label": "Ja"},
    {"code": "2", "label": "Nein"},
    {"code": "3", "label": "Unbekannt"},
]

NON_NUMERIC_CHOICES = [
    {"code": "male", "label": "männlich"},
    {"code": "female", "label": "weiblich"},
]


# --------------------------------------------------------------------------- #
# build_concept_map — no choices
# --------------------------------------------------------------------------- #


def test_build_concept_map_returns_none_without_choices():
    assert build_concept_map("field", [], "http://example.org") is None


# --------------------------------------------------------------------------- #
# build_concept_map — non-numeric codes => identity mapping
# --------------------------------------------------------------------------- #


def test_build_concept_map_non_numeric_codes_identity_mapping():
    cm = build_concept_map("geschlecht", NON_NUMERIC_CHOICES, "http://example.org")

    assert cm["resourceType"] == "ConceptMap"
    assert cm["id"] == "redcap-geschlecht"
    assert cm["url"] == "http://example.org/ConceptMap/redcap-geschlecht"
    group = cm["group"][0]
    assert group["source"] == "SourceData"
    assert group["target"] == "SourceData"
    elements = {e["code"]: e["target"][0]["code"] for e in group["element"]}
    assert elements == {"male": "male", "female": "female"}


def test_build_concept_map_non_numeric_uses_annotation_target_system():
    fhir_annotation = {"primaryElementSystem": "http://hl7.org/fhir/administrative-gender"}
    cm = build_concept_map(
        "geschlecht", NON_NUMERIC_CHOICES, "http://example.org", fhir_annotation=fhir_annotation
    )
    group = cm["group"][0]
    assert group["source"] == "http://hl7.org/fhir/administrative-gender"
    assert group["target"] == "http://hl7.org/fhir/administrative-gender"


# --------------------------------------------------------------------------- #
# build_concept_map — numeric codes without explicit mapping => label mapping
# --------------------------------------------------------------------------- #


def test_build_concept_map_numeric_codes_without_mapping_uses_label_as_target():
    cm = build_concept_map("diabetes", NUMERIC_CHOICES, "http://example.org")
    group = cm["group"][0]
    assert group["source"] == "SourceData"
    assert group["target"] == "SourceData"
    elements = {e["code"]: e["target"][0]["code"] for e in group["element"]}
    assert elements == {"1": "Ja", "2": "Nein", "3": "Unbekannt"}


# --------------------------------------------------------------------------- #
# build_concept_map — explicit mapping table takes priority
# --------------------------------------------------------------------------- #


def test_build_concept_map_explicit_mapping_by_code():
    explicit = {"1": "Y", "2": "N", "3": "U"}
    cm = build_concept_map(
        "diabetes", NUMERIC_CHOICES, "http://example.org", explicit_mapping=explicit
    )
    group = cm["group"][0]
    elements = {e["code"]: e["target"][0]["code"] for e in group["element"]}
    assert elements == {"1": "Y", "2": "N", "3": "U"}
    # The REDCap label glosses the SOURCE code, so it belongs on the element. It used
    # to be written as the TARGET's display, which both mislabelled the target and —
    # since a string-valued answer stores whatever `translate` is asked for — stored
    # the unmapped source text in the answer.
    ja_elem = next(e for e in group["element"] if e["code"] == "1")
    assert ja_elem["display"] == "Ja"
    assert ja_elem["target"][0]["display"] == "Y"


def test_build_concept_map_explicit_mapping_by_label_fallback():
    # KDS-style mapping keyed on the REDCap LABEL text rather than the code
    choices = [{"code": "1", "label": "Melanom"}, {"code": "2", "label": "Leukämie"}]
    explicit = {"Melanom": "363346000", "Leukämie": "87163000"}
    cm = build_concept_map("tumor_art", choices, "http://example.org", explicit_mapping=explicit)
    group = cm["group"][0]
    elements = {e["code"]: e["target"][0]["code"] for e in group["element"]}
    assert elements == {"1": "363346000", "2": "87163000"}


def test_build_concept_map_explicit_mapping_label_match_is_case_insensitive():
    choices = [{"code": "1", "label": "MELANOM"}]
    explicit = {"melanom": "363346000"}
    cm = build_concept_map("tumor_art", choices, "http://example.org", explicit_mapping=explicit)
    elements = cm["group"][0]["element"]
    assert elements[0]["target"][0]["code"] == "363346000"


def test_build_concept_map_explicit_mapping_no_match_falls_back_to_label():
    choices = [{"code": "9", "label": "Sonstiges"}]
    explicit = {"1": "Y"}  # no entry for code "9" or label "Sonstiges"
    cm = build_concept_map("field", choices, "http://example.org", explicit_mapping=explicit)
    elements = cm["group"][0]["element"]
    assert elements[0]["target"][0]["code"] == "Sonstiges"


# --------------------------------------------------------------------------- #
# build_concept_map — merging into an existing ConceptMap
# --------------------------------------------------------------------------- #


def test_build_concept_map_merges_new_entries_into_existing_group():
    existing_cm = {
        "resourceType": "ConceptMap",
        "id": "redcap-diabetes",
        "group": [
            {
                "source": "SourceData",
                "target": "SourceData",
                "element": [
                    {"code": "1", "target": [{"code": "Ja", "equivalence": "equivalent"}]}
                ],
            }
        ],
    }
    cm = build_concept_map(
        "diabetes", NUMERIC_CHOICES, "http://example.org", existing_cm=existing_cm
    )
    group = cm["group"][0]
    codes = [e["code"] for e in group["element"]]
    # existing "1" entry preserved once, new "2" and "3" merged in
    assert codes.count("1") == 1
    assert "2" in codes and "3" in codes
    assert len(group["element"]) == 3


def test_build_concept_map_merge_does_not_mutate_existing_cm_argument():
    existing_cm = {
        "resourceType": "ConceptMap",
        "group": [{"source": "SourceData", "target": "SourceData", "element": []}],
    }
    build_concept_map("diabetes", NUMERIC_CHOICES, "http://example.org", existing_cm=existing_cm)
    # generators.py deep-copies existing_cm before mutating -> original stays empty
    assert existing_cm["group"][0]["element"] == []


def test_build_concept_map_merge_creates_new_group_for_new_source_system():
    existing_cm = {
        "resourceType": "ConceptMap",
        "group": [
            {
                "source": "http://other-system",
                "target": "http://other-system",
                "element": [{"code": "x", "target": [{"code": "x", "equivalence": "equivalent"}]}],
            }
        ],
    }
    cm = build_concept_map(
        "diabetes", NUMERIC_CHOICES, "http://example.org", existing_cm=existing_cm
    )
    assert len(cm["group"]) == 2
    sources = {g["source"] for g in cm["group"]}
    assert sources == {"http://other-system", "SourceData"}
