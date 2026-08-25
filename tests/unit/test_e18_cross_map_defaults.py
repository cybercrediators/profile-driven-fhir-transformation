"""Whether the E18 harness applies the project's external reference defaults.

Measurement code, and a bug here silently changes the reported score rather than
breaking anything. `measure` ran the cross-map recheck without
`external_reference_defaults`, so a deferred reference the operator had already
supplied counted as a gap in the assembled set. On one five-project subset that
was 18 of 22 cross-map findings — HMB 4/4, KDS Onkologie 3/7, Nictiz CIO 11/11 —
inflating reported blocking findings from roughly 104 to 122.

It inflates every arm equally, so it did not manufacture a regression; it did
make every project's outcome look worse than the pipeline's own rules make it.
"""

import pytest

from agent.validation import recheck_cross_map_references

pytestmark = pytest.mark.unit

_SUBJECT = "Observation.subject"


def _map_with_deferred_subject():
    return {
        "resourceType": "StructureMap",
        "id": "sm-observation",
        "url": "http://example.org/StructureMap/sm-observation",
        "structure": [
            {
                "url": "http://example.org/StructureDefinition/Observation",
                "mode": "target",
                "alias": "Observation",
            }
        ],
        "group": [
            {
                "name": "TransformObservation",
                "input": [
                    {"name": "source", "mode": "source"},
                    {"name": "target", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "TODO-resolve-reference-Observation-subject",
                        # The generator's marker for a reference it deliberately
                        # left for the bundle assembler to wire.
                        "documentation": (
                            "FHIRBRIDGE_REFERENCE:"
                            '{"sourceType": "Observation", "path": "subject"}'
                        ),
                        "source": [{"context": "source", "element": "subject"}],
                        "target": [
                            {
                                "context": "target",
                                "element": "subject",
                                "transform": "copy",
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _profile_requiring_a_patient_subject():
    return {
        "resourceType": "StructureDefinition",
        "url": "http://example.org/StructureDefinition/Observation",
        "type": "Observation",
        "snapshot": {
            "element": [
                {
                    "id": _SUBJECT,
                    "path": _SUBJECT,
                    "min": 1,
                    "type": [
                        {
                            "code": "Reference",
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Patient"
                            ],
                        }
                    ],
                }
            ]
        },
    }


def _assembled():
    return [(_map_with_deferred_subject(), _profile_requiring_a_patient_subject())]


def test_without_defaults_a_lone_map_reports_an_unsatisfied_reference():
    """The precondition: nothing in the set produces the referenced resource."""
    findings = recheck_cross_map_references(_assembled())

    assert [f.code for f in findings] == ["cross-map-reference-unsatisfied"]


def test_a_path_the_configuration_supplies_is_not_a_gap_in_the_set():
    """What the harness was failing to pass through."""
    findings = recheck_cross_map_references(
        _assembled(),
        external_defaults=[{"path": _SUBJECT, "reference": "Patient/example"}],
    )

    assert findings == []


def test_the_e18_harness_passes_the_projects_defaults():
    """Pinned on `measure` itself, since the bug was a dropped argument.

    Asserting on `recheck_cross_map_references` alone would have stayed green
    throughout: the function always honoured `external_defaults`, and the defect
    was entirely that its one caller in the harness never supplied them.
    """
    import inspect

    from eval import e18_agent_modes

    source = inspect.getsource(e18_agent_modes.measure)

    assert "external_defaults=" in source
    assert 'conf.get("external_reference_defaults")' in source
