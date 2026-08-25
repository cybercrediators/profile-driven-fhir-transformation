"""Engine-grounded agent validation against a real Matchbox (WP6, layers 5-6).

The unit suite proves the validator's *logic* against a stand-in. Only these
tests prove the part that cannot be faked: that Matchbox actually accepts the
maps this tool generates, actually runs the synthetic fixtures, and actually
reports the failure shapes the classification table is keyed on. A change in
Matchbox's ``OperationOutcome`` codes would leave every unit test green and turn
real failures into ``unclassified`` findings — these are what would notice.

They skip cleanly when no server is reachable. The offline gates never skip.
"""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Any
import urllib.error
import urllib.request

import pytest

from agent.engine import EngineSession, scratch_identity
from agent.fixtures import build_shared_fixtures, source_field_specs
from agent.validation import (
    ActionOwner,
    Producer,
    Stage,
    build_target_tree,
    decide_acceptance,
    merge_engine_result,
    validate_offline,
)
from controller.external_services.matchbox_controller import MatchboxController

pytestmark = pytest.mark.integration

_MATCHBOX_BASE = os.environ.get(
    "MATCHBOX_URL", "http://localhost:8080/matchboxv3"
).rstrip("/")

_SOURCE_URL = "http://example.org/StructureDefinition/agent-validation-source"
_PROFILE_URL = "http://example.org/StructureDefinition/AgentValidationPatient"
_MAP_URL = "http://example.org/StructureMap/agent-validation"


def _matchbox_available() -> bool:
    try:
        with urllib.request.urlopen(
            f"{_MATCHBOX_BASE}/fhir/metadata", timeout=3
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


@pytest.fixture(scope="module")
def matchbox() -> MatchboxController:
    if not _matchbox_available():
        pytest.skip(
            f"Matchbox is not reachable at {_MATCHBOX_BASE}; set MATCHBOX_URL to "
            "run the engine-grounded agent validation checks"
        )
    return MatchboxController({"url": _MATCHBOX_BASE})


# --- the corpus ---------------------------------------------------------------


def source_model() -> dict[str, Any]:
    """A flat logical model, the shape the tool consumes as a source contract."""

    def element(path, minimum, type_code):
        return {
            "id": path,
            "path": path,
            "min": minimum,
            "max": "1",
            "type": [{"code": type_code}],
        }

    return {
        "resourceType": "StructureDefinition",
        "id": "agent-validation-source",
        "url": _SOURCE_URL,
        "name": "AgentValidationSource",
        "status": "draft",
        "kind": "logical",
        "abstract": False,
        "type": "AgentValidationSource",
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Element",
        "derivation": "specialization",
        "differential": {
            "element": [
                {
                    "id": "AgentValidationSource",
                    "path": "AgentValidationSource",
                    "min": 0,
                    "max": "*",
                    "type": [{"code": "Element"}],
                },
                element("AgentValidationSource.givenName", 1, "string"),
                element("AgentValidationSource.familyName", 0, "string"),
                element("AgentValidationSource.born", 0, "string"),
            ]
        },
    }


def profile() -> dict[str, Any]:
    """A Patient profile that requires ``name``, so a gap is observable."""

    def element(path, minimum, maximum, type_code):
        return {
            "id": path,
            "path": path,
            "min": minimum,
            "max": maximum,
            "type": [{"code": type_code}],
        }

    return {
        "resourceType": "StructureDefinition",
        "id": "AgentValidationPatient",
        "url": _PROFILE_URL,
        "name": "AgentValidationPatient",
        "status": "draft",
        "kind": "resource",
        "abstract": False,
        "type": "Patient",
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Patient",
        "derivation": "constraint",
        "snapshot": {
            "element": [
                element("Patient", 0, "*", "Patient"),
                element("Patient.meta", 0, "1", "Meta"),
                element("Patient.identifier", 0, "*", "Identifier"),
                element("Patient.name", 1, "*", "HumanName"),
                element("Patient.birthDate", 0, "1", "date"),
            ]
        },
    }


def structure_map() -> dict[str, Any]:
    return {
        "resourceType": "StructureMap",
        "id": "agent-validation",
        "url": _MAP_URL,
        "name": "AgentValidation",
        "status": "draft",
        "structure": [
            {"url": _SOURCE_URL, "mode": "source", "alias": "Source"},
            {"url": _PROFILE_URL, "mode": "target", "alias": "Target"},
        ],
        "group": [
            {
                "name": "TransformPatient",
                "typeMode": "none",
                "input": [
                    {"name": "source", "type": "AgentValidationSource", "mode": "source"},
                    {"name": "target", "type": "Patient", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "map-birthDate",
                        "source": [
                            {"context": "source", "element": "born", "variable": "src-born"}
                        ],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "birthDate",
                                "transform": "cast",
                                "parameter": [
                                    {"valueId": "src-born"},
                                    {"valueString": "date"},
                                ],
                            }
                        ],
                    },
                    {
                        "name": "map-name",
                        "source": [{"context": "source", "variable": "src-name"}],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "name",
                                "variable": "tgt-name",
                                "transform": "create",
                                "parameter": [{"valueString": "HumanName"}],
                            }
                        ],
                        "rule": [
                            {
                                "name": "map-family",
                                "source": [
                                    {
                                        "context": "source",
                                        "element": "familyName",
                                        "variable": "src-family",
                                    }
                                ],
                                "target": [
                                    {
                                        "context": "tgt-name",
                                        "contextType": "variable",
                                        "element": "family",
                                        "transform": "copy",
                                        "parameter": [{"valueId": "src-family"}],
                                    }
                                ],
                            },
                            {
                                "name": "map-given",
                                "source": [
                                    {
                                        "context": "source",
                                        "element": "givenName",
                                        "variable": "src-given",
                                    }
                                ],
                                "target": [
                                    {
                                        "context": "tgt-name",
                                        "contextType": "variable",
                                        "element": "given",
                                        "transform": "copy",
                                        "parameter": [{"valueId": "src-given"}],
                                    }
                                ],
                            },
                        ],
                    },
                ],
            }
        ],
    }


MAPPING_TABLE = {
    "givenName": "Patient.name.given",
    "familyName": "Patient.name.family",
    "born": "Patient.birthDate",
}


@pytest.fixture(scope="module")
def tree():
    return build_target_tree(profile())


@pytest.fixture(scope="module")
def session(matchbox) -> EngineSession:
    engine = EngineSession(matchbox)
    findings = engine.bootstrap(source_model=source_model(), profiles=[profile()])
    assert findings == [], [f.message for f in findings]
    return engine


def fixtures_for(*documents):
    """One fixture set valid for every revision under comparison.

    Deriving fixtures separately per revision would run the baseline and the
    candidate on different inputs, and a comparison across different inputs
    establishes nothing.
    """

    return build_shared_fixtures(
        source_model(), list(documents), mapping_table=MAPPING_TABLE
    )


def full_report(session, tree, document, fixtures=None):
    """Layers 1-6 for one revision, exactly as an agent run would collect them."""

    report = validate_offline(
        document,
        target_tree=tree,
        mapping_table=MAPPING_TABLE,
        source_fields={spec["name"] for spec in source_field_specs(source_model())},
        profile_url=_PROFILE_URL,
    )
    return merge_engine_result(
        report,
        session.validate_map(
            document,
            fixtures if fixtures is not None else fixtures_for(document),
            target_tree=tree,
            profile_url=_PROFILE_URL,
            mapping_table=MAPPING_TABLE,
        ),
    )


def compare(session, tree, baseline_doc, candidate_doc):
    """Report both revisions against one shared fixture set, then decide."""

    shared = fixtures_for(baseline_doc, candidate_doc)
    baseline = full_report(session, tree, baseline_doc, shared)
    candidate = full_report(session, tree, candidate_doc, shared)
    return baseline, candidate, decide_acceptance(baseline, candidate)


# --- the checks ---------------------------------------------------------------


def test_a_correct_map_passes_every_layer(session, tree):
    report = full_report(session, tree, structure_map())
    assert [
        (f.code, f.message) for f in report.findings if f.blocking
    ] == []
    assert len(report.executed_fixtures) == 2


def test_the_synthetic_fixtures_actually_transform(session, tree):
    result = session.validate_map(
        structure_map(),
        fixtures_for(structure_map()),
        target_tree=tree,
        profile_url=_PROFILE_URL,
        mapping_table=MAPPING_TABLE,
    )
    assert len(result.outputs) == 2
    for output in result.outputs.values():
        assert output["resourceType"] == "Patient"
    filled = result.outputs[fixtures_for(structure_map())[0].fixture_id]
    # The cast override gave `born` a real date; a generic placeholder would
    # have failed the cast instead.
    assert filled["birthDate"] == "2024-01-15"
    assert filled["name"][0]["family"] == "1"


def test_a_bogus_target_element_is_caught_by_the_engine(session, tree):
    document = structure_map()
    document["id"] = "agent-validation-bogus"
    document["group"][0]["rule"][1]["rule"][0]["target"][0]["element"] = "notAnElement"
    report = full_report(session, tree, document)
    codes = {f.code for f in report.findings}
    # Offline addressing catches it, and so does the engine — the two layers are
    # independent evidence for the same defect.
    assert "target-path-not-found" in codes
    engine_failures = [
        f
        for f in report.findings
        if f.producer is Producer.ENGINE and f.stage is Stage.TRANSFORM
    ]
    assert engine_failures
    assert all(f.blocking for f in engine_failures)
    assert all(f.action_owner is ActionOwner.MAP_FIXABLE for f in engine_failures)


def test_the_engine_failure_codes_are_the_ones_the_table_knows(session, tree):
    """The classification table is keyed on Matchbox's issue codes.

    If Matchbox changes them, every unit test still passes and real failures
    silently become ``unclassified``. This is the test that would notice.
    """

    document = structure_map()
    document["id"] = "agent-validation-unclassified-probe"
    document["group"][0]["rule"][1]["rule"][0]["target"][0]["element"] = "notAnElement"
    report = full_report(session, tree, document)
    engine_failures = [
        f
        for f in report.findings
        if f.producer is Producer.ENGINE and f.stage is Stage.TRANSFORM
    ]
    assert engine_failures, "expected the engine to reject this map"
    unclassified = [
        f for f in engine_failures if f.action_owner is ActionOwner.UNCLASSIFIED
    ]
    assert not unclassified, (
        "Matchbox reported issue codes the classification table does not know: "
        f"{sorted({f.code for f in unclassified})}"
    )


def test_a_missing_required_element_is_reported_with_its_path(session, tree):
    document = structure_map()
    document["id"] = "agent-validation-no-name"
    document["group"][0]["rule"] = [document["group"][0]["rule"][0]]
    report = full_report(session, tree, document)
    located = [f for f in report.findings if f.code == "output-required-missing"]
    assert [f.path for f in located] == ["Patient.name"]
    # Matchbox reports the same gap, but puts the failing path only in prose —
    # this is why the located finding is derived rather than parsed.
    engine_structure = [f for f in report.findings if f.code == "validate:structure"]
    assert engine_structure
    assert engine_structure[0].path is None


def test_a_regression_is_rejected_and_the_baseline_is_not(session, tree):
    baseline = full_report(session, tree, structure_map())
    assert decide_acceptance(baseline, baseline).accepted

    broken = structure_map()
    broken["id"] = "agent-validation-regression"
    broken["group"][0]["rule"][1]["rule"][1]["target"][0]["element"] = "nonsense"
    _base, _cand, decision = compare(session, tree, structure_map(), broken)
    assert not decision.accepted
    failed = {item.name for item in decision.failures}
    assert "engine-executes-required-fixtures" in failed
    assert "no-obligation-loss" in failed


def test_both_revisions_run_on_one_shared_fixture_set(session, tree):
    # WP6 requires identical fixtures on both sides. Deriving them per revision
    # would compare two runs on different inputs, which establishes nothing.
    broken = structure_map()
    broken["id"] = "agent-validation-shared-fixtures"
    broken["group"][0]["rule"][1]["rule"][1]["target"][0]["element"] = "nonsense"
    baseline, candidate, _decision = compare(session, tree, structure_map(), broken)
    assert baseline.executed_fixtures == candidate.executed_fixtures
    assert baseline.executed_fixtures


def test_an_improvement_over_a_broken_baseline_is_accepted(session, tree):
    # The direction that matters: a project whose baseline is already broken must
    # still be able to accept a repair.
    broken = structure_map()
    broken["id"] = "agent-validation-broken-baseline"
    broken["group"][0]["rule"][1]["rule"][1]["target"][0]["element"] = "nonsense"
    baseline = full_report(session, tree, broken)
    assert baseline.blocking_findings

    repaired = structure_map()
    repaired["id"] = "agent-validation-repaired"
    candidate = full_report(session, tree, repaired)
    decision = decide_acceptance(baseline, candidate)
    assert decision.accepted, [item.detail for item in decision.failures]
    assert decision.resolved_blocking_ids


def test_a_candidate_is_never_uploaded_over_the_baseline(session, matchbox):
    # Validating a revision must not replace the project's own map on a shared
    # server: the next person to run a transform would silently get the
    # candidate.
    document = structure_map()
    document["id"] = "agent-validation-scratch"
    _scratch_id, scratch_url = scratch_identity(document)
    session.validate_map(document, fixtures_for(document))

    assert matchbox.get_resource_by_url("StructureMap", scratch_url) is not None
    stored = matchbox.get_resource_by_url("StructureMap", _MAP_URL)
    if stored is not None:
        assert stored["url"] == _MAP_URL
        assert stored["id"] != _scratch_id


def test_the_verdict_cache_survives_a_deep_copy(session, tree):
    document = structure_map()
    document["id"] = "agent-validation-cache"
    first = session.validate_map(document, fixtures_for(document), target_tree=tree)
    again = session.validate_map(
        deepcopy(document), fixtures_for(document), target_tree=tree
    )
    assert first.cached is False
    assert again.cached is True
