import json
from types import SimpleNamespace

import pytest

import data_handling.instance_validation as iv_mod
from controller.pipeline_controller.pipeline_instance_validator import (
    PipelineInstanceValidator,
)


pytestmark = pytest.mark.unit


def make_validator(
    make_app_state, state, *, dataIO=None, cache=None, matchbox=None,
    validation_service=None, transform_service=None, conf=None,
):
    app_state = make_app_state(dataIO=dataIO, cache=cache, conf=conf or {})
    return PipelineInstanceValidator(
        app_state, state, matchbox,
        validation_service or SimpleNamespace(),
        transform_service or SimpleNamespace(),
    )


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


def test_first_of_type_picks_matching(make_app_state, state):
    v = make_validator(make_app_state, state)
    produced = [{"resourceType": "Observation"}, {"resourceType": "Patient"}]
    assert v._first_of_type(produced, "Patient") == {"resourceType": "Patient"}


def test_first_of_type_falls_back_to_first(make_app_state, state):
    v = make_validator(make_app_state, state)
    assert v._first_of_type([{"resourceType": "Observation"}], "Patient") == {
        "resourceType": "Observation"
    }


def test_first_of_type_none_for_empty(make_app_state, state):
    v = make_validator(make_app_state, state)
    assert v._first_of_type(None, "Patient") is None


def test_summarize_instance_report_counts(make_app_state, state):
    v = make_validator(make_app_state, state)
    instances = [
        {"direct_validate": "PASS", "roundtrip_validate": "PASS", "coverage": 1.0},
        {"direct_validate": "FAIL", "roundtrip_validate": None, "coverage": 0.5},
        {"direct_validate": None, "roundtrip_validate": "FAIL", "coverage": None},
    ]
    s = v._summarize_instance_report(instances)
    assert s == {
        "n": 3,
        "direct_total": 2,
        "direct_pass": 1,
        "roundtrip_total": 2,
        "roundtrip_pass": 1,
        "mean_coverage": pytest.approx(0.75),
    }


def _sm_with_target_type(target_type, *structure_urls):
    return {
        "resourceType": "StructureMap",
        "url": "http://x/sm-" + target_type,
        "group": [{"input": [{"mode": "target", "type": target_type}]}],
        "structure": [{"mode": "target", "url": u} for u in structure_urls],
    }


def test_find_structure_map_for_type(make_app_state, state, fake_dataio, write_json):
    sm = write_json("sm.json", _sm_with_target_type("Patient", "http://x/P"))
    v = make_validator(make_app_state, state, dataIO=fake_dataio(sm_files=[sm]))
    assert v._find_structure_map_for_type("Patient") == "http://x/sm-Patient"
    assert v._find_structure_map_for_type("Observation") is None


def test_find_structure_map_for_profile(make_app_state, state, fake_dataio, write_json):
    sm = write_json("sm.json", _sm_with_target_type("Patient", "http://x/MyPatient"))
    v = make_validator(make_app_state, state, dataIO=fake_dataio(sm_files=[sm]))
    assert v._find_structure_map_for_profile("http://x/MyPatient") == "http://x/sm-Patient"
    assert v._find_structure_map_for_profile("http://x/Other") is None


# --------------------------------------------------------------------------- #
# run_instance_validation
# --------------------------------------------------------------------------- #


def test_run_instance_validation_direct_only(
    make_app_state, state, fake_dataio, fake_matchbox, monkeypatch, tmp_path
):
    records = [
        {
            "resource": {"resourceType": "Patient", "id": "p1"},
            "profile_url": "http://x/P",
            "file": "p1.json",
            "source": "examples",
        }
    ]
    monkeypatch.setattr(iv_mod, "discover_examples", lambda *a, **k: records)
    monkeypatch.setattr(iv_mod, "strip_version", lambda u: u)

    val_svc = SimpleNamespace(
        find_target_profile_for_type=lambda rt: None,
        summarize_outcome=lambda outcome: {"status": "PASS"},
    )
    mb = fake_matchbox(validate_outcome={"resourceType": "OperationOutcome", "issue": []})
    report_path = tmp_path / "report.json"

    v = make_validator(
        make_app_state, state, dataIO=fake_dataio(), matchbox=mb, validation_service=val_svc
    )
    report = v.run_instance_validation(direct_only=True, report_path=str(report_path))

    assert report["summary"]["n"] == 1
    assert report["summary"]["direct_pass"] == 1
    assert report["summary"]["roundtrip_total"] == 0
    assert report["instances"][0]["direct_validate"] == "PASS"
    # report persisted to disk
    assert json.loads(report_path.read_text())["summary"]["n"] == 1


def test_run_instance_validation_roundtrip(
    make_app_state, state, fake_dataio, fake_matchbox, monkeypatch, tmp_path
):
    state.custom_mapping_table = {"src": "tgt"}
    records = [
        {
            "resource": {"resourceType": "Patient", "id": "p1"},
            "profile_url": "http://x/P",
            "file": "p1.json",
            "source": "examples",
        }
    ]
    monkeypatch.setattr(iv_mod, "discover_examples", lambda *a, **k: records)
    monkeypatch.setattr(iv_mod, "strip_version", lambda u: u)
    monkeypatch.setattr(iv_mod, "reverse_extract", lambda *a, **k: {"field": 1})
    monkeypatch.setattr(
        iv_mod, "compare_instance", lambda *a, **k: {"coverage": 1.0, "diffs": []}
    )

    val_svc = SimpleNamespace(
        find_target_profile_for_type=lambda rt: None,
        summarize_outcome=lambda outcome: {"status": "PASS"},
    )
    transform_svc = SimpleNamespace(
        transform_data=lambda flat, structure_map_url=None: [{"resourceType": "Patient"}]
    )
    mb = fake_matchbox(validate_outcome={"resourceType": "OperationOutcome", "issue": []})

    v = make_validator(
        make_app_state, state, dataIO=fake_dataio(), matchbox=mb,
        validation_service=val_svc, transform_service=transform_svc,
    )
    monkeypatch.setattr(v, "_find_structure_map_for_profile", lambda u: "http://x/sm")

    report = v.run_instance_validation(report_path=str(tmp_path / "r.json"))
    entry = report["instances"][0]
    assert entry["roundtrip_validate"] == "PASS"
    assert entry["coverage"] == 1.0
    assert report["summary"]["roundtrip_pass"] == 1


def test_run_instance_validation_no_records(
    make_app_state, state, fake_dataio, fake_matchbox, monkeypatch
):
    monkeypatch.setattr(iv_mod, "discover_examples", lambda *a, **k: [])
    monkeypatch.setattr(iv_mod, "strip_version", lambda u: u)
    v = make_validator(make_app_state, state, dataIO=fake_dataio(), matchbox=fake_matchbox())
    report = v.run_instance_validation(direct_only=True)
    assert report == {"summary": {"n": 0}, "instances": []}
