"""Which canonical the e16 harness validates each generated resource against.

This is measurement code, not product code, but a bug here is worse than a bug in
the generator: it silently changes the reported score. The original implementation
keyed the project's canonicals by their last path segment, which let MII's
`…/StructureDefinition/LogicalModel/Diagnose` shadow the Condition profile
`…/StructureDefinition/Diagnose`. Every real element of a valid Condition then came
back as "Unrecognized property" and the project scored 0/1 instead of 1/1.
"""

import json

import pytest

from eval.e16_run import _project_profile_canonicals, _validation_target

pytestmark = pytest.mark.unit

_MII = "https://www.medizininformatik-initiative.de/fhir/core/modul-diagnose"
_CONDITION = f"{_MII}/StructureDefinition/Diagnose"
_LOGICAL = f"{_MII}/StructureDefinition/LogicalModel/Diagnose"


def _project(tmp_path, *definitions):
    profiles = tmp_path / "input_profile"
    profiles.mkdir()
    for name, body in definitions:
        (profiles / name).write_text(json.dumps(body))
    return tmp_path


def _sd(url, kind="resource", type_="Condition"):
    return {"resourceType": "StructureDefinition", "url": url,
            "kind": kind, "type": type_}


def test_logical_model_never_shadows_a_profile_with_the_same_tail(tmp_path):
    # Sorted globbing puts `mii-lm-…` first, so the logical model wins any
    # last-segment keying purely by filename order.
    pdir = _project(
        tmp_path,
        ("StructureDefinition-mii-lm-diagnose.json",
         _sd(_LOGICAL, kind="logical", type_=_LOGICAL)),
        ("StructureDefinition-mii-pr-diagnose-condition.json", _sd(_CONDITION)),
    )
    canonicals = _project_profile_canonicals(pdir)

    assert _LOGICAL not in canonicals.values()
    assert _validation_target(_CONDITION, "Condition", canonicals) == _CONDITION


def test_scheme_only_mismatch_resolves_to_the_registered_canonical(tmp_path):
    """evo13 pins the http:// form of a canonical published as https://."""
    published = "https://fhir.element44.de/E44_EVO13_PR_CorrectionRequest"
    pinned = "http://fhir.element44.de/E44_EVO13_PR_CorrectionRequest"
    pdir = _project(tmp_path, ("evo13.json", _sd(published, type_="Communication")))

    canonicals = _project_profile_canonicals(pdir)

    assert _validation_target(pinned, "Communication", canonicals) == published


def test_a_foreign_profile_is_still_honoured(tmp_path):
    """Only the project's own canonicals may be substituted."""
    foreign = "http://hl7.org/fhir/StructureDefinition/vitalsigns"
    pdir = _project(tmp_path, ("own.json", _sd(_CONDITION)))

    canonicals = _project_profile_canonicals(pdir)

    assert _validation_target(foreign, "Observation", canonicals) == foreign


def test_a_resource_without_a_stated_profile_falls_back_to_its_type(tmp_path):
    pdir = _project(tmp_path, ("own.json", _sd(_CONDITION)))
    canonicals = _project_profile_canonicals(pdir)

    assert _validation_target(None, "Condition", canonicals) == "Condition"


def test_a_differing_tail_is_not_treated_as_a_match(tmp_path):
    """Two unrelated profiles sharing a last segment must stay distinct."""
    ours = "http://example.org/fhir/StructureDefinition/Patient"
    theirs = "http://other.example/fhir/StructureDefinition/Patient"
    pdir = _project(tmp_path, ("own.json", _sd(ours, type_="Patient")))

    canonicals = _project_profile_canonicals(pdir)

    assert _validation_target(theirs, "Patient", canonicals) == theirs
