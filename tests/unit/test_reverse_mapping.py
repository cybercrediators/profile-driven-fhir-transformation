"""Unit tests for the reverse-mapping PoC helpers (data_handling.reverse_mapping).

Reverse mapping interprets the forward mapping table as an extraction spec over
transformed FHIR output and inverts generated ConceptMaps for coded fields.
These tests pin the pure helpers: value simplification, inverse-ConceptMap
loading/lookup/translation, payload shapes, record assembly, and the
round-trip report classification.
"""

import json

import pytest

import data_handling.reverse_mapping as rm

pytestmark = pytest.mark.unit


# ── fixtures ─────────────────────────────────────────────────────────────────

MAPPING_TABLE = {
    "givenName": "Patient.name.given",
    "familyName": "Patient.name.family",
    "id": "Patient.identifier.value",
    "gender": "Patient.gender",
    "observationCode": "Observation.code.coding.code",
    "diagnosisCode": "Condition.code",  # CodeableConcept-level mapping
}

PATIENT = {
    "resourceType": "Patient",
    "identifier": [{"value": "1"}],
    "name": [{"family": "Smith", "given": ["Darcy"]}],
    "gender": "male",
}

OBSERVATION = {
    "resourceType": "Observation",
    "status": "final",
    "code": {"coding": [{"code": "TEST-A"}]},
}

CONDITION = {
    "resourceType": "Condition",
    "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006"}]},
}

BUNDLE = {
    "resourceType": "Bundle",
    "type": "transaction",
    "entry": [{"resource": PATIENT}, {"resource": OBSERVATION}, {"resource": CONDITION}],
}

# inverse of a generated ConceptMap: output code -> [source codes]
INVERSE_CMS = {
    "gender": {"male": ["M"], "female": ["F"]},
    # non-injective forward map (0 and 1 both -> in-progress)
    "status": {"in-progress": ["0", "1"], "completed": ["2"]},
}


# ── simplify_value ───────────────────────────────────────────────────────────

def test_simplify_value_codeable_concept_single_coding():
    assert rm.simplify_value({"coding": [{"code": "44054006"}]}) == "44054006"


def test_simplify_value_codeable_concept_multiple_codings():
    value = {"coding": [{"code": "a"}, {"code": "b"}]}
    assert rm.simplify_value(value) == ["a", "b"]


def test_simplify_value_bare_coding():
    assert rm.simplify_value({"system": "http://x", "code": "y"}) == "y"


def test_simplify_value_passthrough_scalars_and_plain_dicts():
    assert rm.simplify_value("final") == "final"
    assert rm.simplify_value(5) == 5
    period = {"start": "2020-01-01"}
    assert rm.simplify_value(period) is period


# ── inverse ConceptMap loading / lookup / translation ────────────────────────

def test_load_inverse_concept_maps_both_naming_conventions(tmp_path):
    (tmp_path / "cm-gender-aaaa.json").write_text(json.dumps({
        "resourceType": "ConceptMap",
        "name": "ConceptMap_gender",
        "group": [{"element": [
            {"code": "M", "target": [{"code": "male"}]},
            {"code": "F", "target": [{"code": "female"}]},
        ]}],
    }))
    (tmp_path / "cm-status-bbbb.json").write_text(json.dumps({
        "resourceType": "ConceptMap",
        "name": "REDCap_waves_complete",
        "group": [{"element": [
            {"code": "0", "target": [{"code": "in-progress"}]},
            {"code": "1", "target": [{"code": "in-progress"}]},
            {"code": "2", "target": [{"code": "completed"}]},
        ]}],
    }))
    inverse = rm.load_inverse_concept_maps(tmp_path)
    assert inverse["gender"] == {"male": ["M"], "female": ["F"]}
    assert sorted(inverse["waves_complete"]["in-progress"]) == ["0", "1"]


def test_load_inverse_concept_maps_missing_dir():
    assert rm.load_inverse_concept_maps(None) == {}
    assert rm.load_inverse_concept_maps("/nonexistent/path") == {}


def test_inverse_for_prefers_target_leaf_then_source_field():
    cms = {"gender": {"male": ["M"]}, "sex": {"male": ["X"]}}
    # target leaf 'gender' wins over source field 'sex'
    assert rm.inverse_for("sex", "Patient.gender", cms)["male"] == ["M"]
    # falls back to the source field when no leaf key exists
    assert rm.inverse_for("sex", "Patient.other", cms)["male"] == ["X"]
    # Sourcedefinition_ prefixed fields resolve via their leaf
    assert rm.inverse_for("Sourcedefinition_p.sex", "Patient.other", cms)["male"] == ["X"]


def test_translate_back_unambiguous_ambiguous_and_inapplicable():
    assert rm.translate_back("male", INVERSE_CMS["gender"]) == ("M", None)
    value, candidates = rm.translate_back("in-progress", INVERSE_CMS["status"])
    assert value == "in-progress"  # left untouched: non-injective forward map
    assert candidates == ["0", "1"]
    assert rm.translate_back("unmapped", INVERSE_CMS["gender"]) == ("unmapped", None)
    assert rm.translate_back(5, INVERSE_CMS["gender"]) == (5, None)


# ── payload shapes ───────────────────────────────────────────────────────────

def test_iter_resource_sets_bundle_yields_one_set():
    sets = list(rm.iter_resource_sets(BUNDLE))
    assert len(sets) == 1
    assert len(sets[0]) == 3


def test_iter_resource_sets_list_of_bundles_yields_one_set_each():
    sets = list(rm.iter_resource_sets([BUNDLE, BUNDLE]))
    assert len(sets) == 2


def test_iter_resource_sets_single_resource_and_resource_list():
    assert list(rm.iter_resource_sets(PATIENT)) == [[PATIENT]]
    assert list(rm.iter_resource_sets([PATIENT, OBSERVATION])) == [[PATIENT, OBSERVATION]]


# ── record assembly ──────────────────────────────────────────────────────────

def test_reverse_map_resources_extracts_all_field_kinds():
    record = rm.reverse_map_resources(
        [PATIENT, OBSERVATION, CONDITION], MAPPING_TABLE, inverse_cms=INVERSE_CMS
    )
    assert record["givenName"] == "Darcy"
    assert record["familyName"] == "Smith"
    assert record["id"] == "1"
    assert record["gender"] == "M"  # translated back via inverse ConceptMap
    assert record["observationCode"] == "TEST-A"
    assert record["diagnosisCode"] == "44054006"  # CC-level -> coding code


def test_reverse_map_resources_strips_sourcedefinition_prefix():
    table = {"Sourcedefinition_proj.familyName": "Patient.name.family"}
    record = rm.reverse_map_resources([PATIENT], table)
    assert record == {"familyName": "Smith"}


def test_reverse_map_resources_profile_rooted_path_needs_profile_ids():
    table = {"code": "MyObsProfile.code.coding.code"}
    # without profile context the profile-rooted path cannot be resolved
    assert rm.reverse_map_resources([OBSERVATION], table) == {}
    record = rm.reverse_map_resources(
        [OBSERVATION], table, profile_ids_map={"Observation": {"MyObsProfile"}}
    )
    assert record == {"code": "TEST-A"}


def test_reverse_map_resources_flags_ambiguous_inverse():
    table = {"status": "QuestionnaireResponse.status"}
    qr = {"resourceType": "QuestionnaireResponse", "status": "in-progress"}
    record = rm.reverse_map_resources([qr], table, inverse_cms=INVERSE_CMS)
    assert record["status"] == "in-progress"
    assert record["_reverse_mapping"]["status"] == {
        "ambiguous_inverse_candidates": ["0", "1"]
    }


def test_reverse_map_resources_flags_multiple_candidates():
    table = {"code": "Condition.code"}
    other = {"resourceType": "Condition", "code": {"coding": [{"code": "E11.9"}]}}
    record = rm.reverse_map_resources([CONDITION, other], table)
    assert sorted(record["code"]) == ["44054006", "E11.9"]
    assert "multiple_candidates" in record["_reverse_mapping"]["code"]


def test_reverse_map_payload_list_of_bundles():
    records = rm.reverse_map_payload([BUNDLE, BUNDLE], MAPPING_TABLE)
    assert len(records) == 2
    assert records[0]["familyName"] == "Smith"


# ── profile_ids_by_type ──────────────────────────────────────────────────────

def test_profile_ids_by_type_collects_id_and_name():
    sds = [
        {"resourceType": "StructureDefinition", "type": "Observation",
         "id": "my-obs", "name": "MyObsProfile"},
        {"resourceType": "StructureDefinition", "type": "Patient", "id": "my-pat"},
        {"resourceType": "ValueSet", "id": "not-an-sd"},
    ]
    result = rm.profile_ids_by_type(sds)
    assert result["Observation"] == {"my-obs", "MyObsProfile"}
    assert result["Patient"] == {"my-pat"}
    assert "not-an-sd" not in str(result)


# ── round-trip report ────────────────────────────────────────────────────────

def test_roundtrip_report_classification():
    records = [{"familyName": "Smith", "id": "1", "gender": "X"}]
    sources = [{
        "familyName": "Smith",   # exact
        "id": 1,                 # cast (int -> "1")
        "gender": "M",           # recovered_differs
        "givenName": "Darcy",    # missing (mapped but not extracted)
        "extraField": "zzz",     # not_mapped (absent from table)
    }]
    report = rm.roundtrip_report(records, sources, MAPPING_TABLE)
    fields = report["records"][0]
    assert fields["familyName"]["status"] == "exact"
    assert fields["id"]["status"] == "cast"
    assert fields["gender"]["status"] == "recovered_differs"
    assert fields["givenName"]["status"] == "missing"
    assert fields["extraField"]["status"] == "not_mapped"
    assert report["summary"] == {
        "exact": 1, "cast": 1, "recovered_differs": 1, "missing": 1, "not_mapped": 1
    }


def test_roundtrip_report_accepts_single_source_dict():
    report = rm.roundtrip_report([{"familyName": "Smith"}], {"familyName": "Smith"},
                                 MAPPING_TABLE)
    assert report["summary"] == {"exact": 1}
