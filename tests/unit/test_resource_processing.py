"""Unit tests for resource_processing: on-disk naming of processed resources.

``StructureDefinition.id`` is optional in FHIR (the canonical url is the identity)
and published IGs do ship profiles without one, so the processed-resource filename
must fall back to the canonical instead of ``None``.
"""

from types import SimpleNamespace

import pytest

from parser.resource_processing import _result_filename

pytestmark = pytest.mark.unit


def _val(**kw):
    return SimpleNamespace(data=SimpleNamespace(**kw))


def test_id_is_used_when_present():
    val = _val(id="ca-on-ps-profile-patient", url="http://example.org/sd/PatientPSON")
    assert _result_filename("http://example.org/sd/PatientPSON", val) == "ca-on-ps-profile-patient"


def test_missing_id_falls_back_to_canonical_tail():
    val = _val(id=None, url="http://ontariohealth.ca/fhir/StructureDefinition/ext-loinc-ontology-axis")
    assert _result_filename("http://ontariohealth.ca/fhir/x", val) == "ext-loinc-ontology-axis"


def test_versioned_canonical_drops_the_version_suffix():
    val = _val(id=None, url="https://fake-acme.org/fhir/StructureDefinition/ACME-base-smoking-status|4.1.7")
    assert _result_filename("", val) == "ACME-base-smoking-status"


def test_registry_key_is_used_when_the_object_carries_no_url():
    val = _val(id=None, url=None)
    assert _result_filename("http://example.org/fhir/StructureDefinition/Thing", val) == "Thing"


def test_trailing_slash_does_not_produce_an_empty_name():
    val = _val(id=None, url="http://example.org/fhir/StructureDefinition/Thing/")
    assert _result_filename("", val) == "Thing"


def test_unresolvable_identity_gets_a_placeholder_rather_than_none():
    val = _val(id=None, url="")
    assert _result_filename("", val) == "unnamed_resource"
