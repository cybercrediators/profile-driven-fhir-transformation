"""Public StructureMap generation-result and diagnostic contracts."""

import json

import pytest

from mapping.generation_result import (
    COVERAGE_REPORT_VERSION,
    CoverageReport,
    DiagnosticActionability,
    MappingDiagnostic,
    classify_diagnostic,
    diagnostic_id,
)

pytestmark = pytest.mark.unit


def _raw_report(diagnostics=None):
    return {
        "report_version": 2,
        "map": "structure_map_example",
        "note": "Coverage describes providers, not successful validation.",
        "summary": {
            "profiles": 1,
            "required_total": 1,
            "required_mapped": 0,
            "required_unmapped": 1,
            "static_required_total": 2,
            "latent_required_total": 1,
            "required_coverage_pct": 0.0,
        },
        "profiles": {
            "ExamplePatient": {
                "resource_type": "Patient",
                "required_total": 1,
                "required_mapped": 0,
                "required_unmapped": 1,
                "unmapped_required_paths": ["Patient.identifier"],
                "static_required_total": 2,
                "latent_required_total": 1,
                "latent_required_paths": ["Patient.contact.name"],
                "requirement_manifest": [
                    {
                        "id": "Patient.identifier",
                        "path": "Patient.identifier",
                        "min": 1,
                        "max": "*",
                        "active": True,
                        "provider": "missing",
                        "status": "unmapped",
                    }
                ],
            }
        },
        "mapping_diagnostics": diagnostics or [],
    }


def test_actionability_is_a_deterministic_reviewed_registry():
    assert classify_diagnostic("unmaterialized-nested-target") == (
        DiagnosticActionability.MAP_FIXABLE
    )
    assert classify_diagnostic("mapping-source-path-not-found") == (
        DiagnosticActionability.MAPPING_INPUT_REQUIRED
    )
    assert classify_diagnostic("derived-resource-unresolvable") == (
        DiagnosticActionability.ENVIRONMENT
    )
    assert classify_diagnostic("target-fhirpath-constraint") == (
        DiagnosticActionability.ADVISORY
    )
    # A new code cannot authorize an agent edit until a developer classifies it.
    assert classify_diagnostic("future-unreviewed-code") == (
        DiagnosticActionability.ADVISORY
    )


def test_external_actionability_is_ignored_and_recomputed():
    diagnostic = MappingDiagnostic.from_raw(
        {
            "code": "mapping-source-path-not-found",
            "message": "missing",
            "source": "source.unknown",
            "actionability": "map-fixable",
        }
    )

    assert diagnostic.actionability == DiagnosticActionability.MAPPING_INPUT_REQUIRED


def test_diagnostic_id_is_stable_across_message_wording_changes():
    first = {
        "code": "mapping-target-path-not-found",
        "profile": "ExamplePatient",
        "source": "src.value",
        "target": "Patient.unknown",
        "message": "first wording",
    }
    second = {**first, "message": "revised wording"}
    moved = {**second, "target": "Patient.otherUnknown"}

    assert diagnostic_id(first) == diagnostic_id(second)
    assert diagnostic_id(first) != diagnostic_id(moved)


def test_diagnostic_id_distinguishes_multiple_findings_at_the_same_location():
    first = {
        "code": "target-fhirpath-constraint",
        "profile": "ExampleObservation",
        "path": "Observation.value[x]",
        "expression": "value.exists()",
        "severity": "error",
    }
    second = {**first, "expression": "dataAbsentReason.exists()"}
    reclassified = {**first, "severity": "warning", "actionability": "map-fixable"}

    assert diagnostic_id(first) != diagnostic_id(second)
    assert diagnostic_id(first) == diagnostic_id(reclassified)


def test_v2_report_loads_without_rewriting_its_version_and_is_enriched():
    raw = _raw_report(
        [
            {
                "code": "mapping-source-path-not-found",
                "message": "missing source",
                "profile": "ExamplePatient",
                "source": "src.unknown",
            }
        ]
    )

    report = CoverageReport.from_raw(raw)
    dumped = json.loads(report.model_dump_json(exclude_none=True))

    assert report.report_version == 2
    assert dumped["profiles"] == raw["profiles"]
    diagnostic = dumped["mapping_diagnostics"][0]
    assert diagnostic["code"] == "mapping-source-path-not-found"
    assert diagnostic["diagnostic_id"].startswith("diag-")
    assert diagnostic["actionability"] == "mapping-input-required"


def test_current_report_schema_is_machine_readable():
    schema = CoverageReport.model_json_schema()

    assert COVERAGE_REPORT_VERSION == 3
    assert "mapping_diagnostics" in schema["properties"]
    assert "profiles" in schema["properties"]
