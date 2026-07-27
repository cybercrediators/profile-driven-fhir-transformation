"""Unit tests for the pure instance-based-validation helpers.

Uses projects/example_project2 as a fixture: its converted_data/expected_output.json is a
list of Bundles of *produced* FHIR resources (perfect stand-in ground-truth instances),
alongside source_data/mapping_table.json and source_data/source_data.json.
"""
import json
from pathlib import Path

import pytest

from data_handling.data_io import is_definition_resource
from data_handling.instance_validation import (
    extract_path,
    reverse_extract,
    compare_instance,
    synthesize_from_element_examples,
    discover_examples,
    strip_version,
)

PROJECT = Path(__file__).resolve().parents[2] / "projects" / "example_project2"


@pytest.fixture
def mapping_table():
    return json.loads((PROJECT / "source_data" / "mapping_table.json").read_text())


@pytest.fixture
def source_rows():
    return json.loads((PROJECT / "source_data" / "source_data.json").read_text())


@pytest.fixture
def bundles():
    return json.loads((PROJECT / "converted_data" / "expected_output.json").read_text())


def _resources_of(bundle, resource_type):
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == resource_type]


# --- extract_path -----------------------------------------------------------

def test_extract_path_handles_nested_arrays():
    patient = {"resourceType": "Patient", "name": [{"given": ["Ada", "May"], "family": "Lovelace"}]}
    assert extract_path(patient, "Patient.name.given") == ["Ada", "May"]
    assert extract_path(patient, "Patient.name.family") == ["Lovelace"]


def test_extract_path_descends_into_coding():
    obs = {"resourceType": "Observation", "code": {"coding": [{"code": "X"}, {"code": "Y"}]}}
    assert extract_path(obs, "Observation.code.coding.code") == ["X", "Y"]


def test_extract_path_strips_slice_qualifier():
    # MII/US-Core mapping tables use slice syntax; navigating an instance ignores the slice name
    patient = {
        "resourceType": "Patient",
        "identifier": [{"value": "pid-1"}],
        "name": [{"family": "Mustermann", "given": ["Erika"]}],
    }
    assert extract_path(patient, "Patient.identifier:pid.value") == ["pid-1"]
    assert extract_path(patient, "Patient.name:name.family") == ["Mustermann"]


def test_extract_path_missing_returns_empty():
    assert extract_path({"resourceType": "Patient"}, "Patient.gender") == []
    assert extract_path({}, "Patient.gender") == []


def test_extract_path_strips_leading_resource_type_regardless_of_match():
    # leading capitalised segment is always treated as the resource type
    obs = {"resourceType": "Observation", "status": "final"}
    assert extract_path(obs, "Observation.status") == ["final"]


# --- reverse_extract --------------------------------------------------------

def test_reverse_extract_observation_round_trips_to_source(mapping_table, source_rows, bundles):
    obs = _resources_of(bundles[0], "Observation")[0]
    flat = reverse_extract(obs, mapping_table)
    # only Observation-targeted fields should be present
    assert set(flat) == {"observationId", "observationStatus", "observationCode"}
    row = source_rows[0]
    for key in flat:
        assert str(flat[key]) == str(row[key])


def test_reverse_extract_patient_fields(mapping_table, bundles):
    patient = _resources_of(bundles[0], "Patient")[0]
    flat = reverse_extract(patient, mapping_table)
    assert flat["givenName"] == "Darcy"
    assert flat["familyName"] == "Smith"
    assert str(flat["id"]) == "1"          # identifier.value is "1" in the produced resource
    assert "observationId" not in flat       # Observation-targeted entries excluded


# --- compare_instance -------------------------------------------------------

def test_compare_identical_full_coverage(mapping_table, bundles):
    obs = _resources_of(bundles[0], "Observation")[0]
    result = compare_instance(obs, obs, mapping_table)
    assert result["coverage"] == 1.0
    assert result["diffs"] == []
    assert result["total"] == 3  # three Observation-targeted mapped paths present


def test_compare_detects_mismatch(mapping_table, bundles):
    obs = _resources_of(bundles[0], "Observation")[0]
    perturbed = json.loads(json.dumps(obs))
    perturbed["status"] = "amended"
    result = compare_instance(obs, perturbed, mapping_table)
    assert result["coverage"] < 1.0
    statuses = {d["status"] for d in result["diffs"]}
    assert statuses == {"mismatch"}
    assert any(d["path"] == "Observation.status" for d in result["diffs"])


def test_compare_detects_missing_in_output(mapping_table, bundles):
    obs = _resources_of(bundles[0], "Observation")[0]
    stripped = {"resourceType": "Observation", "status": obs["status"], "code": obs["code"]}
    result = compare_instance(obs, stripped, mapping_table)
    assert any(d["status"] == "missing_in_output" and d["path"] == "Observation.identifier.value"
               for d in result["diffs"])


# --- profile-aware target matching ------------------------------------------

def test_strip_version():
    assert strip_version("http://x/StructureDefinition/Patient|2025.0.1") == "http://x/StructureDefinition/Patient"
    assert strip_version("http://x/StructureDefinition/Patient") == "http://x/StructureDefinition/Patient"


def test_reverse_extract_profile_id_rooted_target():
    # mapping target rooted at a profile id (not the base type) — how the generator routes
    # fields to a specific profile among several sharing a base resource type
    cond = {"resourceType": "Condition", "code": {"coding": [{"code": "C50"}]}}
    table = {"tumorCode": "anamnese-tumor.code.coding.code"}
    # without the profile id, the entry does not apply
    assert reverse_extract(cond, table) == {}
    # with the profile id known, it resolves against the base resource
    assert reverse_extract(cond, table, profile_ids={"anamnese-tumor"}) == {"tumorCode": "C50"}


def test_compare_instance_profile_id_rooted_target():
    cond = {"resourceType": "Condition", "code": {"coding": [{"code": "C50"}]}}
    table = {"tumorCode": "anamnese-tumor.code.coding.code"}
    # not applicable without profile id -> nothing counted
    assert compare_instance(cond, cond, table)["total"] == 0
    # applicable with profile id -> full coverage, diff keeps the profile-rooted path
    result = compare_instance(cond, cond, table, profile_ids={"anamnese-tumor"})
    assert result["coverage"] == 1.0 and result["total"] == 1


# --- classifier -------------------------------------------------------------

@pytest.mark.parametrize("rt,expected", [
    ("StructureDefinition", True),
    ("ValueSet", True),
    ("CodeSystem", True),
    ("ImplementationGuide", True),
    ("Questionnaire", True),
    ("Patient", False),
    ("Observation", False),
    ("QuestionnaireResponse", False),
    ("Bundle", False),
])
def test_is_definition_resource(rt, expected):
    assert is_definition_resource(rt) is expected


# --- discover_examples ------------------------------------------------------

def _ig(*entries):
    return {"resourceType": "ImplementationGuide", "definition": {"resource": list(entries)}}


def test_discover_examples_ig_driven_attaches_profile():
    patient = {"resourceType": "Patient", "id": "p1", "meta": {"profile": ["http://x/meta"]}}
    org = {"resourceType": "Organization", "id": "o1"}
    ig = _ig(
        {"reference": {"reference": "Patient/p1"}, "exampleCanonical": "http://x/PatientProfile"},
        {"reference": {"reference": "Organization/o1"}, "exampleBoolean": True},
        {"reference": {"reference": "StructureDefinition/sd1"}},  # not an example
    )
    records = discover_examples([("p.json", patient), ("o.json", org)], ig_resource=ig)
    by_id = {r["resource"]["id"]: r for r in records}
    assert by_id["p1"]["profile_url"] == "http://x/PatientProfile"  # exampleCanonical wins over meta
    assert by_id["p1"]["source"] == "ig"
    assert by_id["o1"]["profile_url"] is None                        # exampleBoolean, no canonical
    assert by_id["o1"]["source"] == "ig"


def test_discover_examples_file_fallback_and_dedup():
    patient = {"resourceType": "Patient", "id": "p1", "meta": {"profile": ["http://x/meta"]}}
    sd = {"resourceType": "StructureDefinition", "id": "sd1", "url": "http://x/sd"}
    # IG references the same patient -> should not be tested twice
    ig = _ig({"reference": {"reference": "Patient/p1"}, "exampleCanonical": "http://x/P"})
    records = discover_examples([("p.json", patient), ("sd.json", sd)], ig_resource=ig)
    assert len(records) == 1                       # SD skipped, patient not duplicated
    assert records[0]["source"] == "ig"


def test_discover_examples_no_ig_uses_meta_profile():
    patient = {"resourceType": "Patient", "id": "p1", "meta": {"profile": ["http://x/meta"]}}
    sd = {"resourceType": "StructureDefinition", "id": "sd1", "url": "http://x/sd"}
    records = discover_examples([("p.json", patient), ("sd.json", sd)])
    assert len(records) == 1
    assert records[0]["profile_url"] == "http://x/meta"
    assert records[0]["source"] == "file"
    assert records[0]["file"] == "p.json"


# --- synthesize_from_element_examples ---------------------------------------

def test_synthesize_from_element_examples():
    sd = {
        "resourceType": "StructureDefinition",
        "type": "Patient",
        "url": "http://x/PatientProfile",
        "snapshot": {"element": [
            {"path": "Patient.gender", "example": [{"label": "e", "valueCode": "female"}]},
            {"path": "Patient.birthDate", "example": [{"label": "e", "valueDate": "1990-01-01"}]},
            {"path": "Patient.name", "min": 1},  # no example -> ignored
        ]},
    }
    instance = synthesize_from_element_examples(sd)
    assert instance["resourceType"] == "Patient"
    assert extract_path(instance, "Patient.gender") == ["female"]
    assert extract_path(instance, "Patient.birthDate") == ["1990-01-01"]


def test_synthesize_returns_none_without_examples():
    sd = {"resourceType": "StructureDefinition", "type": "Patient",
          "snapshot": {"element": [{"path": "Patient.gender"}]}}
    assert synthesize_from_element_examples(sd) is None
