"""Unit tests for utils.resource_identity.

``StructureDefinition.id`` is optional in FHIR (the canonical url is the identity) and
published IGs do ship profiles without one — acme.base.r4's smoking-status profile and
all three de.kvtelematik.eterminservice.r4 profiles, for instance. Identity is used for
the processed-resource filename, the StructureMap name/id/alias, the transform group
name, the coverage-report key, and the mapping-table target prefix, so it must never
be ``None``.
"""

from types import SimpleNamespace

import pytest

from helpers.utils import resource_identity

pytestmark = pytest.mark.unit


def _data(**kw):
    kw.setdefault("id", None)
    kw.setdefault("url", None)
    return SimpleNamespace(**kw)


def test_id_wins_when_present():
    data = _data(id="acme-base-observation-lab", url="https://fake-acme.org/fhir/StructureDefinition/X")
    assert resource_identity(data) == "acme-base-observation-lab"


def test_falls_back_to_canonical_tail():
    data = _data(url="https://fake-acme.org/fhir/StructureDefinition/ACME-base-smoking-status")
    assert resource_identity(data) == "ACME-base-smoking-status"


def test_version_suffix_is_stripped():
    data = _data(url="https://fhir.kbv.de/StructureDefinition/74_PR_ETS_Appointment|1.0.0")
    assert resource_identity(data) == "74_PR_ETS_Appointment"


def test_registry_url_is_used_when_the_resource_carries_none():
    data = _data()
    assert resource_identity(data, "http://example.org/StructureDefinition/Thing") == "Thing"


def test_trailing_slash_does_not_yield_an_empty_identity():
    data = _data(url="http://example.org/StructureDefinition/Thing/")
    assert resource_identity(data) == "Thing"


def test_placeholder_when_nothing_identifies_the_resource():
    assert resource_identity(_data()) == "unnamed_resource"


def test_two_same_type_profiles_get_distinct_identities():
    """The acme case: without this, both Observation profiles collapse to one prefix."""
    lab = _data(id="acme-base-observation-lab")
    smoking = _data(url="https://fake-acme.org/fhir/StructureDefinition/ACME-base-smoking-status")
    assert resource_identity(lab) != resource_identity(smoking)


def test_fhir_id_token_replaces_characters_a_fhir_id_cannot_hold():
    """Capable.repository names its profiles `Communication_Profile`; `_` is not legal in an id."""
    from helpers.utils import fhir_id_token
    assert fhir_id_token("Communication_Profile") == "Communication-Profile"


def test_fhir_id_token_leaves_a_legal_identity_untouched():
    from helpers.utils import fhir_id_token
    assert fhir_id_token("mii-pr-diagnose-condition") == "mii-pr-diagnose-condition"
    assert fhir_id_token("ca-on-ps-profile-patient.v2") == "ca-on-ps-profile-patient.v2"


def test_fhir_id_token_truncates_to_the_id_budget():
    from helpers.utils import fhir_id_token
    assert len(fhir_id_token("x" * 120)) == 50
