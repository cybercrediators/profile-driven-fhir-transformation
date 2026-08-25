"""Unit tests for MatchboxController (controller/external_services/matchbox_controller.py).

MatchboxController wraps MatchboxConnector.send_request. We construct a real
MatchboxController (cheap -- no network at construction) and replace its
underlying connector's send_request with a small recording fake, so no HTTP/
requests traffic ever leaves the process.
"""


import pytest

from controller.external_services.matchbox_controller import MatchboxController

pytestmark = pytest.mark.unit


class FakeConnector:
    """Stand-in for MatchboxConnector.send_request; records calls, returns presets."""

    def __init__(self, response=None, responses=None):
        self.calls = []
        self._response = response
        self._responses = list(responses) if responses is not None else None

    def send_request(self, endpoint, method, params=None, headers=None, **kwargs):
        self.calls.append(
            {"endpoint": endpoint, "method": method, "params": params, "headers": headers, **kwargs}
        )
        if self._responses is not None:
            return self._responses.pop(0)
        return self._response


def make_controller(response=None, responses=None):
    controller = MatchboxController(matchbox_con={"url": "http://fake.example"})
    controller.mc = FakeConnector(response=response, responses=responses)
    return controller


# --------------------------------------------------------------------------- #
# get_capability_statement
# --------------------------------------------------------------------------- #


def test_get_capability_statement_success():
    controller = make_controller(response={"resourceType": "CapabilityStatement"})
    result = controller.get_capability_statement()
    assert result == {"resourceType": "CapabilityStatement"}
    assert controller.mc.calls[0] == {
        "endpoint": "metadata",
        "method": "GET",
        "params": None,
        "headers": None,
    }


def test_get_capability_statement_failure_returns_none():
    controller = make_controller(response=None)
    assert controller.get_capability_statement() is None


# --------------------------------------------------------------------------- #
# install_npm_package
# --------------------------------------------------------------------------- #


def test_install_npm_package_no_name_or_version_returns_none():
    controller = make_controller()
    assert controller.install_npm_package(package_name=None, package_version=None) is None
    assert controller.mc.calls == []


def test_install_npm_package_with_path_reads_bytes(tmp_path):
    pkg = tmp_path / "pkg.tgz"
    pkg.write_bytes(b"binary-content")
    controller = make_controller(response={"ok": True})
    result = controller.install_npm_package(
        package_name="my.ig", package_version="1.0.0", package_path=pkg
    )
    assert result == {"ok": True}
    call = controller.mc.calls[0]
    assert call["endpoint"] == "$install-npm-package"
    assert call["method"] == "POST"
    assert call["data"] == b"binary-content"
    assert call["headers"]["content-type"] == "application/octet-stream"


def test_install_npm_package_with_url():
    controller = make_controller(response={"ok": True})
    controller.install_npm_package(
        package_name="my.ig", package_version="1.0.0", package_url="http://example.org/pkg"
    )
    call = controller.mc.calls[0]
    assert call["params"] == {
        "name": "my.ig",
        "version": "1.0.0",
        "url": "http://example.org/pkg",
    }
    assert "data" not in call


# --------------------------------------------------------------------------- #
# get_resource_by_id / get_resource_by_url
# --------------------------------------------------------------------------- #


def test_get_resource_by_id():
    controller = make_controller(response={"resourceType": "Patient", "id": "123"})
    result = controller.get_resource_by_id("Patient", "123")
    assert result == {"resourceType": "Patient", "id": "123"}
    assert controller.mc.calls[0]["endpoint"] == "Patient/123"


def test_get_resource_by_url_found():
    response = {
        "resourceType": "Bundle",
        "entry": [{"resource": {"resourceType": "Patient", "id": "1"}}],
    }
    controller = make_controller(response=response)
    result = controller.get_resource_by_url("Patient", "http://example.org/patient")
    assert result == {"resourceType": "Patient", "id": "1"}


def test_get_resource_by_url_not_found_empty_entry():
    controller = make_controller(response={"resourceType": "Bundle", "entry": []})
    assert controller.get_resource_by_url("Patient", "http://example.org/patient") is None


def test_get_resource_by_url_not_found_no_response():
    controller = make_controller(response=None)
    assert controller.get_resource_by_url("Patient", "http://example.org/patient") is None


# --------------------------------------------------------------------------- #
# upload_* helpers
# --------------------------------------------------------------------------- #


def test_upload_resource_string_infers_resource_type():
    controller = make_controller(response={"id": "abc"})
    result = controller.upload_resource_string({"resourceType": "Observation"})
    assert result == {"id": "abc"}
    assert controller.mc.calls[0]["endpoint"] == "Observation"
    assert controller.mc.calls[0]["method"] == "POST"


def test_upload_structure_map_serialises_json():
    controller = make_controller(response={"id": "sm1"})
    sm = {"resourceType": "StructureMap", "url": "http://x/sm"}
    controller.upload_structure_map(sm)
    call = controller.mc.calls[0]
    assert call["endpoint"] == "StructureMap"
    import json

    assert json.loads(call["data"]) == sm


def test_upload_concept_map():
    controller = make_controller(response={"id": "cm1"})
    cm = {"resourceType": "ConceptMap"}
    controller.upload_concept_map(cm)
    assert controller.mc.calls[0]["endpoint"] == "ConceptMap"


def test_upload_structure_definition_normal_profile_no_warning(caplog):
    controller = make_controller(response={"id": "sd1"})
    sd = {
        "resourceType": "StructureDefinition",
        "url": "http://example.org/StructureDefinition/my-custom-profile",
        "derivation": "constraint",
    }
    with caplog.at_level("WARNING"):
        controller.upload_structure_definition(sd)
    assert not any("shadowing" in r.message for r in caplog.records)
    assert controller.mc.calls[0]["endpoint"] == "StructureDefinition"


def test_upload_structure_definition_core_type_shadowing_guard_warns(caplog):
    controller = make_controller(response={"id": "sd1"})
    sd = {
        "resourceType": "StructureDefinition",
        "url": "http://example.org/StructureDefinition/Patient",
        "derivation": "constraint",
    }
    with caplog.at_level("WARNING"):
        controller.upload_structure_definition(sd)
    assert any("shadowing" in r.message for r in caplog.records)
    # upload still proceeds despite the warning
    assert controller.mc.calls[0]["endpoint"] == "StructureDefinition"


def test_upload_structure_definition_specialization_no_shadow_check():
    # derivation != "constraint" -> guard branch is not entered at all.
    controller = make_controller(response={"id": "sd1"})
    sd = {
        "resourceType": "StructureDefinition",
        "url": "http://example.org/StructureDefinition/Patient",
        "derivation": "specialization",
    }
    controller.upload_structure_definition(sd)
    assert controller.mc.calls[0]["endpoint"] == "StructureDefinition"


# --------------------------------------------------------------------------- #
# check_implementation_guide_installed
# --------------------------------------------------------------------------- #


def test_check_ig_installed_no_url_or_id_returns_false():
    controller = make_controller()
    assert controller.check_implementation_guide_installed() is False
    assert controller.mc.calls == []


def test_check_ig_installed_by_url_found():
    response = {"resourceType": "Bundle", "entry": [{"resource": {"packageId": "x"}}]}
    controller = make_controller(response=response)
    assert controller.check_implementation_guide_installed(ig_url="http://ig") is True


def test_check_ig_installed_by_id_matching_package_id():
    response = {
        "resourceType": "Bundle",
        "entry": [{"resource": {"packageId": "my.ig"}}],
    }
    controller = make_controller(response=response)
    assert controller.check_implementation_guide_installed(ig_id="my.ig") is True


def test_check_ig_installed_by_id_no_match_returns_false():
    response = {
        "resourceType": "Bundle",
        "entry": [{"resource": {"packageId": "other.ig"}}],
    }
    controller = make_controller(response=response)
    assert controller.check_implementation_guide_installed(ig_id="my.ig") is False


def test_check_ig_installed_empty_entries_returns_false():
    controller = make_controller(response={"resourceType": "Bundle", "entry": []})
    assert controller.check_implementation_guide_installed(ig_url="http://ig") is False


def test_check_ig_installed_no_response_returns_false():
    controller = make_controller(response=None)
    assert controller.check_implementation_guide_installed(ig_url="http://ig") is False


# --------------------------------------------------------------------------- #
# transform_data / validate_fhir_resources
# --------------------------------------------------------------------------- #


def test_transform_data_success():
    controller = make_controller(response={"resourceType": "Bundle"})
    result = controller.transform_data({"resourceType": "Foo"}, "http://x/sm")
    assert result == {"resourceType": "Bundle"}
    call = controller.mc.calls[0]
    assert call["endpoint"] == "StructureMap/$transform"
    assert call["params"] == {"source": "http://x/sm"}


def test_transform_data_failure_returns_none():
    controller = make_controller(response=None)
    assert controller.transform_data({"resourceType": "Foo"}, "http://x/sm") is None


def test_validate_fhir_resources_success():
    outcome = {"resourceType": "OperationOutcome", "issue": []}
    controller = make_controller(response=outcome)
    result = controller.validate_fhir_resources({"resourceType": "Patient"}, "http://profile")
    assert result == outcome
    assert controller.mc.calls[0]["endpoint"] == "$validate"
    assert controller.mc.calls[0]["params"] == {"profile": "http://profile"}
