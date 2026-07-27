from types import SimpleNamespace

import pytest

from controller.pipeline_controller.pipeline_validation_service import (
    PipelineValidationService,
)


pytestmark = pytest.mark.unit


def make_service(
    make_app_state, state, *, dataIO=None, cache=None, matchbox=None, matchbox_sync=None, conf=None
):
    app_state = make_app_state(dataIO=dataIO, cache=cache, conf=conf or {})
    return PipelineValidationService(app_state, state, matchbox, matchbox_sync)


def _outcome(*issues):
    return {"resourceType": "OperationOutcome", "issue": list(issues)}


# --------------------------------------------------------------------------- #
# summarize_outcome
# --------------------------------------------------------------------------- #


def test_summarize_outcome_pass_with_warnings(make_app_state, state):
    svc = make_service(make_app_state, state)
    result = svc.summarize_outcome(
        _outcome(
            {"severity": "warning", "diagnostics": "w1"},
            {"severity": "information", "diagnostics": "i"},
        )
    )
    assert result == {"status": "PASS", "errors": [], "warnings": ["w1"]}


def test_summarize_outcome_fail_collects_errors(make_app_state, state):
    svc = make_service(make_app_state, state)
    result = svc.summarize_outcome(
        _outcome(
            {"severity": "error", "diagnostics": "e1"},
            {"severity": "fatal", "details": {"text": "e2"}},
            {"severity": "warning", "diagnostics": "w1"},
        )
    )
    assert result["status"] == "FAIL"
    assert result["errors"] == ["e1", "e2"]
    assert result["warnings"] == ["w1"]


def test_summarize_outcome_non_outcome_is_pass(make_app_state, state):
    svc = make_service(make_app_state, state)
    assert svc.summarize_outcome(None) == {"status": "PASS", "errors": [], "warnings": []}


# --------------------------------------------------------------------------- #
# find_target_profile_for_type
# --------------------------------------------------------------------------- #


def _sm(target_type, *structure_urls):
    return {
        "resourceType": "StructureMap",
        "group": [{"input": [{"mode": "target", "type": target_type}]}],
        "structure": [{"mode": "target", "url": u} for u in structure_urls],
    }


def test_find_target_profile_prefers_specific_over_base(
    make_app_state, state, fake_dataio, write_json
):
    sm = write_json(
        "sm.json",
        _sm(
            "Patient",
            "http://hl7.org/fhir/StructureDefinition/Patient",
            "http://x/StructureDefinition/MyPatient",
        ),
    )
    svc = make_service(make_app_state, state, dataIO=fake_dataio(sm_files=[sm]))
    assert svc.find_target_profile_for_type("Patient") == "http://x/StructureDefinition/MyPatient"


def test_find_target_profile_falls_back_to_base_only(make_app_state, state, fake_dataio, write_json):
    sm = write_json("sm.json", _sm("Patient", "http://hl7.org/fhir/StructureDefinition/Patient"))
    svc = make_service(make_app_state, state, dataIO=fake_dataio(sm_files=[sm]))
    assert (
        svc.find_target_profile_for_type("Patient")
        == "http://hl7.org/fhir/StructureDefinition/Patient"
    )


def test_find_target_profile_none_when_no_match(make_app_state, state, fake_dataio, write_json):
    sm = write_json("sm.json", _sm("Observation", "http://x/StructureDefinition/Obs"))
    svc = make_service(make_app_state, state, dataIO=fake_dataio(sm_files=[sm]))
    assert svc.find_target_profile_for_type("Patient") is None


# --------------------------------------------------------------------------- #
# validate_data / validate_local_files
# --------------------------------------------------------------------------- #


def test_validate_data_delegates_to_matchbox(make_app_state, state, fake_matchbox):
    mb = fake_matchbox(validate_outcome=_outcome())
    svc = make_service(make_app_state, state, matchbox=mb)
    out = svc.validate_data({"resourceType": "Patient"}, "http://x/p")
    assert out == _outcome()
    assert mb.validate_calls == [({"resourceType": "Patient"}, "http://x/p")]


def test_validate_local_files_true_when_all_present(make_app_state, state, fake_dataio):
    svc = make_service(make_app_state, state, dataIO=fake_dataio(processed=True))
    assert svc.validate_local_files() is True


def test_validate_local_files_false_when_missing(make_app_state, state, fake_dataio):
    svc = make_service(make_app_state, state, dataIO=fake_dataio(processed=False))
    assert svc.validate_local_files() is False


# --------------------------------------------------------------------------- #
# validate_setup (composes matchbox_sync)
# --------------------------------------------------------------------------- #


def test_validate_setup_true_when_local_and_matchbox_ok(make_app_state, state, fake_dataio):
    sync = SimpleNamespace(prepare_matchbox_setup=lambda read_only=True: True)
    svc = make_service(make_app_state, state, dataIO=fake_dataio(processed=True), matchbox_sync=sync)
    assert svc.validate_setup(read_only=True) is True


def test_validate_setup_false_when_local_files_missing(make_app_state, state, fake_dataio):
    called = {"prepare": False}

    def _prepare(read_only=True):
        called["prepare"] = True
        return True

    sync = SimpleNamespace(prepare_matchbox_setup=_prepare)
    svc = make_service(make_app_state, state, dataIO=fake_dataio(processed=False), matchbox_sync=sync)
    assert svc.validate_setup() is False
    # short-circuits: matchbox not consulted when local files fail
    assert called["prepare"] is False


def test_validate_setup_false_when_matchbox_incomplete(make_app_state, state, fake_dataio):
    sync = SimpleNamespace(prepare_matchbox_setup=lambda read_only=True: False)
    svc = make_service(make_app_state, state, dataIO=fake_dataio(processed=True), matchbox_sync=sync)
    assert svc.validate_setup() is False


# --------------------------------------------------------------------------- #
# run_validate
# --------------------------------------------------------------------------- #


def test_run_validate_derives_profile_and_passes(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    resource = write_json("res.json", {"resourceType": "Patient"})
    sm = write_json("sm.json", _sm("Patient", "http://x/StructureDefinition/MyPatient"))
    mb = fake_matchbox(validate_outcome=_outcome({"severity": "warning", "diagnostics": "w"}))
    svc = make_service(make_app_state, state, dataIO=fake_dataio(sm_files=[sm]), matchbox=mb)

    result = svc.run_validate(str(resource))
    assert result["status"] == "PASS"
    # profile was derived from the SM and used for validation
    assert mb.validate_calls[0][1] == "http://x/StructureDefinition/MyPatient"


def test_run_validate_missing_profile_returns_none(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    resource = write_json("res.json", {"resourceType": "Patient"})
    mb = fake_matchbox()
    # no SMs -> no profile derivable, no -p given
    svc = make_service(make_app_state, state, dataIO=fake_dataio(), matchbox=mb)
    assert svc.run_validate(str(resource)) is None
    assert mb.validate_calls == []
