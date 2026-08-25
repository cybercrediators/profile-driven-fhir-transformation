"""Routing a finding to an edit scope, and who owns an unresolvable canonical.

These are regression tests for a batch in which 89 of 118 maps stopped at "no
deterministic edit scope could be resolved" without a single provider call. The
findings were classified ``map-fixable`` and were genuinely on the worklist; the
routing in front of the model simply had no way to turn any of them into a
pointer, so the run measured the plumbing rather than the model.

Each test below names one link in that chain.
"""

import pytest

from agent.context import build_context, finding_path
from agent.engine import unresolvable_canonical
from agent.fixtures import unemptyable_source_fields
from agent.validation import (
    ActionOwner,
    GateStatus,
    Producer,
    Stage,
    ValidationFinding,
    classify,
)


def structure_map():
    return {
        "resourceType": "StructureMap",
        "id": "sm-scope",
        "url": "http://example.org/StructureMap/sm-scope",
        "name": "SmScope",
        "status": "draft",
        "structure": [
            {"url": "http://example.org/StructureDefinition/src", "mode": "source"},
            {
                "url": "http://example.org/StructureDefinition/TestObservation",
                "mode": "target",
            },
        ],
        "group": [
            {
                "name": "TransformObservation",
                "typeMode": "none",
                "input": [
                    {"name": "source", "type": "SrcModel", "mode": "source"},
                    {"name": "target", "type": "Observation", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "map-code",
                        "source": [{"context": "source", "variable": "s"}],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "code",
                                "variable": "tgt-code",
                                "transform": "create",
                                "parameter": [{"valueString": "CodeableConcept"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def cardinality_finding():
    """What Matchbox actually returns: prose, no structured path."""
    return ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:structure",
        "Observation.status: minimum required = 1, but only found 0 "
        "(from http://example.org/StructureDefinition/TestObservation)",
        fixture_id="synthetic-filled",
        evidence={"path_hint": "Observation.status"},
    )


# -- the hint is a path ------------------------------------------------------


def test_a_recovered_hint_is_read_as_the_finding_path():
    assert finding_path(cardinality_finding()) == "Observation.status"


def test_a_structured_path_still_wins_over_a_hint():
    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:structure",
        "…",
        path="Observation.code",
        evidence={"path_hint": "Observation.status"},
    )
    assert finding_path(finding) == "Observation.code"


def test_prose_that_matched_no_pattern_contributes_no_path():
    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:invariant",
        "Constraint obs-7 failed",
    )
    assert finding_path(finding) == ""


# -- and a path is an edit scope --------------------------------------------


def test_an_engine_cardinality_finding_authorizes_an_insertion_point():
    """The case that stranded 268 findings: no pointer, no path, no scope."""
    context = build_context(structure_map(), [cardinality_finding()])
    authorized = [p.pointer for p in context.pointers if p.focused] + [
        point.pointer for point in context.insertion_points
    ]
    assert authorized, "a map-fixable finding must authorize somewhere to write"
    assert "/group/0/rule/-" in authorized


def test_a_missing_required_output_authorizes_an_insertion_point():
    # The owner is assigned at production time by `required_gap_owner`, not
    # through CLASSIFICATION — a gap with no source field is the mapping's, and
    # only the map-fixable ones reach a prompt at all.
    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "output-required-missing",
        "The transform output has no Observation.status, which the profile "
        "requires (min = 1).",
        owner=ActionOwner.MAP_FIXABLE,
        path="Observation.status",
        fixture_id="synthetic-filled",
    )
    context = build_context(structure_map(), [finding])
    assert [point.pointer for point in context.insertion_points] == ["/group/0/rule/-"]


def test_a_hint_naming_an_emitted_element_focuses_that_rule_instead():
    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:structure",
        "Observation.code: minimum required = 1, but only found 0",
        evidence={"path_hint": "Observation.code"},
    )
    context = build_context(structure_map(), [finding])
    assert "/group/0/rule/0" in [p.pointer for p in context.pointers if p.focused]


# -- an unresolvable canonical is the environment's ---------------------------


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "HAPI-0389: Failed to call access method: Error setting coding on "
            "CodeableConcept for rule Map_x|Transform-x|add-category-coding to "
            "value Coding=Coding[null]: Unable to find definition "
            "'http://hl7.org/fhir/uv/ips/StructureDefinition/CodeableConcept-uv-ips|2.0.1' "
            "for type 'CodeableConcept' for name 'category' on property "
            "Condition.category",
            "http://hl7.org/fhir/uv/ips/StructureDefinition/CodeableConcept-uv-ips|2.0.1",
        ),
        (
            "Slicing cannot be evaluated: Unable to resolve profile "
            "CanonicalType[https://fhir.kbv.de/StructureDefinition/74_EX_ETS_Erster]",
            "https://fhir.kbv.de/StructureDefinition/74_EX_ETS_Erster",
        ),
        (
            "No definition could be found for URL value "
            "'https://fhir.kbv.de/identifierer/74_ID_ETS_Vermittlungscode'",
            "https://fhir.kbv.de/identifierer/74_ID_ETS_Vermittlungscode",
        ),
    ],
)
def test_the_engine_saying_it_cannot_resolve_a_url_is_recognized(message, expected):
    assert unresolvable_canonical(message) == expected


def test_an_ordinary_map_defect_is_not_mistaken_for_a_missing_definition():
    message = (
        "The property coding must be a JSON Array, not an Object (at "
        "Condition.category)"
    )
    assert unresolvable_canonical(message) is None


# -- the catch-all buckets are reviewed, not unknown --------------------------


def test_matchboxs_catch_all_validate_codes_are_classified():
    for code in ("validate:invalid", "validate:processing"):
        owner, gate = classify(Producer.ENGINE, code)
        assert owner is ActionOwner.MAP_FIXABLE
        assert gate is GateStatus.BLOCKING


def test_an_unreviewed_code_is_still_unclassified():
    owner, _gate = classify(Producer.ENGINE, "validate:something-new")
    assert owner is ActionOwner.UNCLASSIFIED


# -- the empty-optional fixture must not feed "" to a cast --------------------


def cast_map(cast_type="positiveInt"):
    return {
        "group": [
            {
                "name": "g",
                "input": [
                    {"name": "source", "type": "SrcModel", "mode": "source"},
                    {"name": "target", "type": "Observation", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "map-frequency",
                        "source": [
                            {
                                "context": "source",
                                "element": "frequency",
                                "variable": "src-frequency",
                            }
                        ],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "frequency",
                                "transform": "cast",
                                "parameter": [
                                    {"valueId": "src-frequency"},
                                    {"valueString": cast_type},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_a_field_cast_to_an_integer_is_never_emptied():
    assert unemptyable_source_fields([cast_map()]) == {"frequency"}


def test_a_field_cast_to_a_string_type_may_still_be_emptied():
    assert unemptyable_source_fields([cast_map("code")]) == set()


def test_an_unrecognized_cast_type_is_left_alone():
    """Fail open here: silently dropping fields from every fixture is worse."""
    assert unemptyable_source_fields([cast_map("SomeNewType")]) == set()


# -- a binding failure on the fixture's own placeholder is not the map's ------


def test_a_profile_rooted_mapping_target_still_matches_the_engines_expression():
    """The two sides root the same path differently; comparing them whole never
    matched, so the fixture's generic "1" was reported as a map defect."""
    from agent.engine import EngineSession
    from agent.fixtures import FixtureKind, SourceFixture

    fixture = SourceFixture(
        fixture_id="filled",
        kind=FixtureKind.SYNTHETIC_FILLED,
        label="synthetic-filled",
        instance={"resourceType": "SrcModel", "lrvStatus": "1"},
        verified_fields=[],
    )
    unverified = EngineSession._unverified_target_paths(
        [fixture],
        {"Sourcedefinition_x.lrvStatus": "hddt-lung-reference-value.status"},
    )
    assert EngineSession._placeholder_expression(["Observation.status"], unverified)


def test_a_verified_field_is_not_excused():
    from agent.engine import EngineSession
    from agent.fixtures import FixtureKind, SourceFixture

    fixture = SourceFixture(
        fixture_id="filled",
        kind=FixtureKind.SYNTHETIC_FILLED,
        label="synthetic-filled",
        instance={"resourceType": "SrcModel", "lrvStatus": "final"},
        verified_fields=["lrvStatus"],
    )
    unverified = EngineSession._unverified_target_paths(
        [fixture],
        {"Sourcedefinition_x.lrvStatus": "hddt-lung-reference-value.status"},
    )
    assert EngineSession._placeholder_expression(["Observation.status"], unverified) is None


def test_stripping_the_root_does_not_collapse_distinct_elements():
    from agent.engine import EngineSession

    unverified = EngineSession._unverified_target_paths(
        [],
        {"Sourcedefinition_x.f": "profile-id.value.status"},
    )
    assert EngineSession._placeholder_expression(["Observation.status"], unverified) is None


def test_a_container_binding_matches_only_placeholder_fed_descendants():
    from agent.engine import EngineSession

    unverified = {
        "code.coding.system",
        "code.coding.code",
    }
    assert EngineSession._placeholder_expression(
        ["Observation.code"], unverified, set()
    ) == "Observation.code"


def test_a_verified_descendant_keeps_a_container_binding_blocking():
    from agent.engine import EngineSession

    assert (
        EngineSession._placeholder_expression(
            ["Observation.code"],
            {"code.coding.system"},
            {"code.coding.code"},
        )
        is None
    )


# -- the package a project is given must be closed over what it references ----


def test_the_closure_adds_supporting_definitions_but_never_a_mapping_target():
    """A staged resource profile would become a map, changing a drawn selection."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from bootstrap_simplifier_project import dependency_closure

    patient = {
        "resourceType": "StructureDefinition",
        "kind": "resource",
        "derivation": "constraint",
        "url": "http://x/SD/Patient-x",
        "snapshot": {
            "element": [
                {
                    "id": "Patient.communication",
                    "type": [{"code": "CodeableConcept", "profile": ["http://x/SD/CC-x"]}],
                },
                {
                    "id": "Patient.managingOrganization",
                    "type": [
                        {"code": "Reference", "targetProfile": ["http://x/SD/Org-x"]}
                    ],
                },
            ]
        },
    }
    datatype = {
        "resourceType": "StructureDefinition",
        "kind": "complex-type",
        "derivation": "constraint",
        "url": "http://x/SD/CC-x",
        "snapshot": {"element": [{"id": "CodeableConcept.coding", "binding": {
            "valueSet": "http://x/VS/codes"}}]},
    }
    organization = {
        "resourceType": "StructureDefinition",
        "kind": "resource",
        "derivation": "constraint",
        "url": "http://x/SD/Org-x",
    }
    value_set = {"resourceType": "ValueSet", "url": "http://x/VS/codes"}

    resources = [
        ("p.json", patient),
        ("cc.json", datatype),
        ("org.json", organization),
        ("vs.json", value_set),
    ]
    closure = {r.get("url") for _, r in dependency_closure([("p.json", patient)], resources)}

    assert "http://x/SD/CC-x" in closure, "the datatype profile is why 500s happened"
    assert "http://x/VS/codes" in closure, "the closure is transitive"
    assert "http://x/SD/Org-x" not in closure, "staging this would add a map"


# -- the structured locator beats the sentence --------------------------------
#
# `$validate` states most issues against a resource-rooted FHIRPath and puts
# nothing usable in the message. Reading only the prose left 155 of 253 findings
# in one batch with no element, unroutable, and blocking their map without a
# provider call.


@pytest.mark.parametrize(
    "expressions, expected",
    [
        (["Communication.basedOn[0].identifier.system"], "Communication.basedOn.identifier.system"),
        (["Observation.category[0]"], "Observation.category"),
        (["MedicationRequest.dosageInstruction[0].maxDosePerPeriod"],
         "MedicationRequest.dosageInstruction.maxDosePerPeriod"),
        (["Appointment.participant[0].type[0].text"], "Appointment.participant.type.text"),
    ],
)
def test_an_expression_yields_the_element_it_addresses(expressions, expected):
    from agent.engine import expression_path

    assert expression_path(expressions) == expected


def test_a_root_only_expression_is_not_a_location():
    """It says no more than the map already says about itself."""
    from agent.engine import expression_path

    assert expression_path(["Observation"]) is None
    assert expression_path([]) is None


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "Slice 'Observation.derivedFrom:measurement': a matching slice is "
            "required, but not found",
            "Observation.derivedFrom",
        ),
        (
            "Slicing cannot be evaluated: Could not match discriminator (coding) "
            "for slice Observation.category:obstetrics in profile",
            "Observation.category",
        ),
        (
            "Observation.status: minimum required = 1, but only found 0",
            "Observation.status",
        ),
    ],
)
def test_prose_shapes_that_name_an_element_are_recovered(message, expected):
    from agent.engine import _hints

    assert _hints(message).get("path_hint") == expected


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "ValueSet 'https://gematik.de/fhir/hddt/ValueSet/hddt-device-type' not found",
            "https://gematik.de/fhir/hddt/ValueSet/hddt-device-type",
        ),
        (
            "Profile reference 'http://fhir.element44.de/E44_EVO13_PR_CorrectionRequest' "
            "has not been checked because it could not be resolved",
            "http://fhir.element44.de/E44_EVO13_PR_CorrectionRequest",
        ),
    ],
)
def test_a_binding_or_profile_the_server_lacks_is_the_environments(message, expected):
    """The URL comes from the profile, never the map; no rule edit reaches it."""
    assert unresolvable_canonical(message) == expected


def test_a_choice_type_slice_is_recovered_too():
    """`effective[x]` carries the indexer in its name; the class must allow it."""
    from agent.engine import _hints

    message = (
        "Slice 'Observation.effective[x]:effectiveDateTime': a matching slice "
        "is required, but not found"
    )
    assert _hints(message).get("path_hint") == "Observation.effective[x]"


def test_a_resource_level_invariant_has_no_single_element_and_stays_unlocated():
    """`app-2` is about a relationship between elements, not one of them."""
    from agent.engine import _hints, expression_path

    message = (
        "Constraint failed: app-2: 'Either start and end are specified, or "
        "neither' (defined in http://hl7.org/fhir/StructureDefinition/Appointment)"
    )
    assert _hints(message).get("path_hint") is None
    assert expression_path(["Appointment"]) is None


# -- a diagnostic belongs to the map whose profile it names --------------------


def diagnostic(profile, code="unmaterialized-nested-target"):
    return {
        "diagnostic_id": f"diag-{profile}-{code}",
        "code": code,
        "message": f"{code} on {profile}",
        "severity": "error",
        "profile": profile,
        "path": "Observation.component.value[x]",
    }


def test_a_diagnostic_reaches_only_the_map_that_owns_its_profile():
    """One profile's defect was attached to all ten maps of a project."""
    from agent.validation import findings_from_diagnostics

    diags = [diagnostic("anamnese-raucher"), diagnostic("WAVESPatient")]
    mine = findings_from_diagnostics(diags, profile={"anamnese-raucher"})
    assert [f.message for f in mine] == ["unmaterialized-nested-target on anamnese-raucher"]


def test_a_canonical_url_and_an_id_name_the_same_profile():
    from agent.validation import findings_from_diagnostics

    diags = [diagnostic("anamnese-raucher")]
    by_url = findings_from_diagnostics(
        diags, profile={"http://example.org/StructureDefinition/anamnese-raucher|1.0"}
    )
    assert len(by_url) == 1


def test_a_diagnostic_naming_no_profile_is_kept():
    """It is about the project, not one map; dropping it would lose it entirely."""
    from agent.validation import findings_from_diagnostics

    diags = [dict(diagnostic("x"), profile=None)]
    assert len(findings_from_diagnostics(diags, profile={"anamnese-raucher"})) == 1


def test_without_a_profile_every_diagnostic_is_kept():
    from agent.validation import findings_from_diagnostics

    diags = [diagnostic("a"), diagnostic("b")]
    assert len(findings_from_diagnostics(diags)) == 2


# -- a reference the configuration supplies is not a gap in the set ------------


def _map_with_deferred(url, target_profile):
    return {
        "resourceType": "StructureMap",
        "id": url.rsplit("/", 1)[-1],
        "url": url,
        "structure": [{"url": target_profile, "mode": "target"}],
        "group": [
            {
                "name": "g",
                "input": [{"name": "target", "type": "MedicationRequest", "mode": "target"}],
                "rule": [
                    {
                        "name": "TODO-resolve-reference-MedicationRequest-subject",
                        "source": [{"context": "source"}],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "subject",
                                "transform": "create",
                                "parameter": [{"valueString": "Reference"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _profile_requiring_patient(url):
    return {
        "resourceType": "StructureDefinition",
        "url": url,
        "id": url.rsplit("/", 1)[-1],
        "type": "MedicationRequest",
        "snapshot": {
            "element": [
                {
                    "id": "MedicationRequest.subject",
                    "path": "MedicationRequest.subject",
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


def test_an_unsatisfied_cross_map_reference_is_still_reported():
    from agent.validation import recheck_cross_map_references

    profile = _profile_requiring_patient("http://example.org/SD/MedReq")
    assembled = [(_map_with_deferred("http://example.org/SM/a", profile["url"]), profile)]
    findings = recheck_cross_map_references(assembled)
    assert [f.code for f in findings] == ["cross-map-reference-unsatisfied"]


def test_a_configured_external_default_satisfies_it():
    """81 of 179 project findings in one batch named a configured path."""
    from agent.validation import recheck_cross_map_references

    profile = _profile_requiring_patient("http://example.org/SD/MedReq")
    assembled = [(_map_with_deferred("http://example.org/SM/a", profile["url"]), profile)]
    findings = recheck_cross_map_references(
        assembled,
        external_defaults=[
            {
                "path": "MedicationRequest.subject",
                "reference": "Patient/medikation-patient-example",
                "type": "Patient",
            }
        ],
    )
    assert findings == []


def test_a_relative_external_default_satisfies_it():
    """Project configs use the same source_type + relative path form as BundleService."""
    from agent.validation import recheck_cross_map_references

    profile = _profile_requiring_patient("http://example.org/SD/MedReq")
    assembled = [(_map_with_deferred("http://example.org/SM/a", profile["url"]), profile)]
    findings = recheck_cross_map_references(
        assembled,
        external_defaults=[
            {
                "source_type": "MedicationRequest",
                "path": "subject",
                "reference": "Patient/example",
            }
        ],
    )
    assert findings == []


def test_a_profile_derived_from_patient_satisfies_a_patient_reference():
    from agent.validation import recheck_cross_map_references

    medication = _profile_requiring_patient("http://example.org/SD/MedReq")
    patient_profile = {
        "resourceType": "StructureDefinition",
        "url": "http://example.org/SD/ProjectPatient",
        "id": "ProjectPatient",
        "kind": "resource",
        "type": "Patient",
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Patient",
        "snapshot": {"element": []},
    }
    patient_map = _map_with_deferred(
        "http://example.org/SM/patient", patient_profile["url"]
    )
    patient_map["group"][0]["rule"] = []
    assembled = [
        (_map_with_deferred("http://example.org/SM/med", medication["url"]), medication),
        (patient_map, patient_profile),
    ]

    assert recheck_cross_map_references(assembled) == []


def test_a_default_for_a_different_path_does_not_satisfy_it():
    from agent.validation import recheck_cross_map_references

    profile = _profile_requiring_patient("http://example.org/SD/MedReq")
    assembled = [(_map_with_deferred("http://example.org/SM/a", profile["url"]), profile)]
    findings = recheck_cross_map_references(
        assembled,
        external_defaults=[{"path": "MedicationRequest.requester", "type": "Practitioner"}],
    )
    assert [f.code for f in findings] == ["cross-map-reference-unsatisfied"]


# -- the prompt must say which variables a rule may reference ------------------


def test_variables_in_scope_follow_engine_scoping():
    """19 candidates in one batch referenced a context nothing had bound."""
    from agent.context import variables_in_scope

    document = {
        "group": [
            {
                "name": "g",
                "input": [
                    {"name": "source", "type": "Src", "mode": "source"},
                    {"name": "target", "type": "Observation", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "outer",
                        "source": [{"context": "source", "variable": "s"}],
                        "target": [
                            {"context": "target", "element": "code", "variable": "vCC"}
                        ],
                        "rule": [
                            {
                                "name": "inner",
                                "source": [{"context": "s", "variable": "s2"}],
                                "target": [
                                    {
                                        "context": "vCC",
                                        "element": "coding",
                                        "variable": "vCoding",
                                    }
                                ],
                            }
                        ],
                    },
                    {"name": "sibling", "source": [{"context": "source"}], "target": []},
                ],
            }
        ]
    }
    scopes = variables_in_scope(document)

    assert scopes["/group/0/rule/0"]["target"] == ["target", "vCC"]
    # the child sees its ancestor's bindings and its own
    assert scopes["/group/0/rule/0/rule/0"]["target"] == ["target", "vCC", "vCoding"]
    assert scopes["/group/0/rule/0/rule/0"]["source"] == ["s", "s2", "source"]
    # a sibling never sees them
    assert "vCC" not in scopes["/group/0/rule/1"]["target"]


def test_described_rules_carry_their_scope():
    from agent.context import describe_rules

    rules = describe_rules(structure_map())
    assert rules, "the fixture map has rules"
    assert all(isinstance(r.source_variables, list) for r in rules)
    assert "target" in rules[0].target_variables


# -- a finding must reach the rule that actually emits its element -------------


def choice_map():
    return {
        "resourceType": "StructureMap",
        "id": "sm-choice",
        "url": "http://example.org/StructureMap/sm-choice",
        "group": [
            {
                "name": "g",
                "input": [
                    {"name": "source", "type": "Src", "mode": "source"},
                    {"name": "target", "type": "Procedure", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "map-performed",
                        "source": [{"context": "source", "element": "d", "variable": "s"}],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "performedDateTime",
                                "transform": "copy",
                                "parameter": [{"valueId": "s"}],
                            }
                        ],
                    },
                    {
                        "name": "map-code",
                        "source": [{"context": "source"}],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "code",
                                "variable": "vCC",
                                "transform": "create",
                                "parameter": [{"valueString": "CodeableConcept"}],
                            }
                        ],
                    },
                ],
            }
        ],
    }


def finding_at(path):
    return ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:structure",
        f"{path}: minimum required = 1, but only found 0",
        owner=ActionOwner.MAP_FIXABLE,
        evidence={"path_hint": path},
    )


def test_a_choice_finding_focuses_the_rule_emitting_the_typed_element():
    """`Procedure.performed` must reach the rule emitting `performedDateTime`."""
    context = build_context(choice_map(), [finding_at("Procedure.performed")])
    focused = [p.pointer for p in context.pointers if p.focused]
    assert "/group/0/rule/0" in focused


def test_a_prefix_that_is_not_a_choice_type_does_not_match():
    """`Procedure.perform` must not capture `performedDateTime` by prefix alone."""
    context = build_context(choice_map(), [finding_at("Procedure.perform")])
    assert [p.pointer for p in context.pointers if p.focused] == []


def test_a_deep_finding_does_not_authorize_an_arbitrary_ancestor():
    """Reversed deliberately.

    This used to focus the rule building ``Procedure.code`` for a finding about
    ``Procedure.code.coding.system``. Matching on path prefix across every
    addressed path reaches back to rules that merely sit above the finding, and
    in a repeated structure that meant dozens of them — the ambiguity that took
    one prompt over its budget.

    Adding a missing child is an *insertion*, and `_insertion_points` authorizes
    exactly one pointer beneath the deepest existing ancestor for the codes that
    mean it. Scope stays narrow and deterministic either way.
    """

    context = build_context(choice_map(), [finding_at("Procedure.code.coding.system")])
    assert [p.pointer for p in context.pointers if p.focused] == []


def test_a_skeleton_rule_owns_what_its_children_emit():
    """The single largest class of out-of-scope rejections.

    A generated skeleton carries an empty ``target`` and delegates emitting to
    its children, so it appears in no path index — yet it is the rule a repair
    belongs in, and the model kept correctly finding it.
    """

    document = {
        "resourceType": "StructureMap",
        "id": "sm-skeleton",
        "url": "http://example.org/StructureMap/sm-skeleton",
        "group": [
            {
                "name": "g",
                "input": [
                    {"name": "source", "type": "Src", "mode": "source"},
                    {"name": "target", "type": "Observation", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "map-effective",
                        "source": [
                            {"context": "source", "element": "d", "variable": "src-eff"}
                        ],
                        "target": [],
                        "documentation": "Maps to Observation.effective[x] | Type: dateTime",
                        "rule": [
                            {
                                "name": "map-effective-dateTime",
                                "source": [{"context": "source", "variable": "v"}],
                                "target": [
                                    {
                                        "context": "target",
                                        "contextType": "variable",
                                        "element": "effectiveDateTime",
                                        "transform": "cast",
                                        "parameter": [
                                            {"valueId": "v"},
                                            {"valueString": "dateTime"},
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    context = build_context(document, [finding_at("Observation.effective")])
    focused = [p.pointer for p in context.pointers if p.focused]
    assert "/group/0/rule/0" in focused, "the skeleton that owns effective[x]"


def test_a_path_emitted_by_many_rules_is_ambiguous_not_focused():
    """28 rules emitting one path stopped the excerpt pruning entirely."""

    rules = [
        {
            "name": f"map-answer-{index}",
            "source": [{"context": "source", "variable": f"s{index}"}],
            "target": [
                {
                    "context": "target",
                    "contextType": "variable",
                    "element": "answer",
                    "transform": "create",
                }
            ],
        }
        for index in range(20)
    ]
    document = {
        "resourceType": "StructureMap",
        "id": "sm-many",
        "url": "http://example.org/StructureMap/sm-many",
        "group": [
            {
                "name": "g",
                "input": [
                    {"name": "source", "type": "Src", "mode": "source"},
                    {
                        "name": "target",
                        "type": "QuestionnaireResponse",
                        "mode": "target",
                    },
                ],
                "rule": rules,
            }
        ],
    }

    context = build_context(document, [finding_at("QuestionnaireResponse.answer")])
    assert [p.pointer for p in context.pointers if p.focused] == []


# -- a deferred reference means what its contract says, not what its name spells


def _deferred(name, contract=None):
    rule = {"name": name, "source": [{"context": "source"}]}
    if contract is not None:
        import json as _json

        rule["documentation"] = "FHIRBRIDGE_REFERENCE:" + _json.dumps(contract)
    return {"group": [{"name": "g", "rule": [rule]}]}


def test_the_contract_path_wins_over_the_rule_name():
    """Hyphens inside a slice name made every extension reference mis-addressed."""
    from agent.validation import deferred_reference_paths

    document = _deferred(
        "TODO-resolve-reference-Observation-extension-genomic-risk-assessment",
        {"sourceType": "Observation", "path": "extension.valueReference"},
    )
    assert deferred_reference_paths(document) == {"Observation.extension.valueReference"}


def test_a_rule_without_a_contract_still_falls_back_to_its_name():
    from agent.validation import deferred_reference_paths

    document = _deferred("TODO-resolve-reference-Observation-subject")
    assert deferred_reference_paths(document) == {"Observation.subject"}


def test_an_unparseable_contract_falls_back_rather_than_dropping_the_rule():
    from agent.validation import deferred_reference_paths

    document = {
        "group": [
            {
                "name": "g",
                "rule": [
                    {
                        "name": "TODO-resolve-reference-Observation-subject",
                        "documentation": "FHIRBRIDGE_REFERENCE:{not json",
                    }
                ],
            }
        ]
    }
    assert deferred_reference_paths(document) == {"Observation.subject"}


# -- generated rule names must fit FHIR's id limit -----------------------------


def test_an_overlong_rule_name_is_shortened_deterministically():
    """87 findings in one batch were rule names over the 64-character limit."""
    from mapping.fml_creator.fml_helper import fit_rule_name

    name = "TODO-resolve-reference-DiagnosticReport-extension-genomic-risk-assessment"
    assert len(name) > 64
    short = fit_rule_name(name)
    assert len(short) <= 64
    assert short == fit_rule_name(name), "stable across runs"
    assert short.startswith("TODO-resolve-reference-")


def test_a_short_name_is_returned_unchanged():
    from mapping.fml_creator.fml_helper import fit_rule_name

    assert fit_rule_name("TODO-resolve-reference-Observation-subject") == (
        "TODO-resolve-reference-Observation-subject"
    )


def test_two_names_sharing_a_prefix_do_not_collide():
    from mapping.fml_creator.fml_helper import fit_rule_name

    a = "TODO-resolve-reference-DiagnosticReport-extension-genomic-risk-assessment-one"
    b = "TODO-resolve-reference-DiagnosticReport-extension-genomic-risk-assessment-two"
    assert fit_rule_name(a) != fit_rule_name(b)


# -- serialization must not be what puts a prompt over budget -----------------


def context_for(excerpt):
    from agent.context import AgentContext

    return AgentContext(map_excerpt=excerpt)


def test_a_large_excerpt_is_serialized_compactly():
    """`indent=2` on a 13-level map tripled it and cost a provider call."""
    from agent.service import PRETTY_MAP_BUDGET_CHARS, LLMProposer

    big = {
        "resourceType": "StructureMap",
        "group": [
            {
                "name": "g",
                "rule": [
                    {
                        "name": f"rule-{i}",
                        "source": [{"context": "source", "variable": f"s{i}"}],
                        "target": [
                            {"context": "target", "element": f"e{i}", "transform": "copy"}
                        ],
                    }
                    for i in range(400)
                ],
            }
        ],
    }
    context = context_for(big)
    user, _messages = LLMProposer._messages(context)
    import json as _json

    assert len(_json.dumps(big, indent=2)) > PRETTY_MAP_BUDGET_CHARS
    # the compact form is what reached the prompt
    assert _json.dumps(big, separators=(",", ":")) in user


def test_a_small_excerpt_stays_readable():
    from agent.service import LLMProposer

    small = {"resourceType": "StructureMap", "group": []}
    context = context_for(small)
    user, _messages = LLMProposer._messages(context)
    import json as _json

    assert _json.dumps(small, indent=2) in user


# -- a value the fixture invented is the fixture's problem --------------------


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("Observation.value.ofType(Quantity)", "Observation.valueQuantity"),
        ("Observation.value.ofType(Quantity).system", "Observation.valueQuantity.system"),
        ("Patient.identifier[0].system", "Patient.identifier.system"),
        ("Observation.ofType(Quantity)", "Observation"),
    ],
)
def test_an_expression_is_reduced_to_its_element_definition(expression, expected):
    """`ofType(T)` is how the validator names a choice; the map calls it valueT."""
    from agent.engine import normalize_expression

    assert normalize_expression(expression) == expected


def test_a_placeholder_behind_an_oftype_cast_is_recognised():
    """`Coding.system must be an absolute reference` on a synthetic "1"."""
    from agent.engine import EngineSession

    unverified = EngineSession._unverified_target_paths(
        [],
        {"Sourcedefinition_x.unit": "obs-profile.valueQuantity.system"},
    )
    assert EngineSession._placeholder_expression(
        ["Observation.value.ofType(Quantity).system"], unverified
    )


# -- a choice repair has one correct shape, and the tool already knows it ------


class _Tree:
    def __init__(self, types):
        self._types = types

    def effective_types(self, path):
        return self._types.get(path, [])


def test_a_choice_element_gets_a_typed_element_and_transform():
    """A source string bound for performedDateTime needs `cast`, not `copy`."""
    from agent.context import _choice_repair_hints

    hints = _choice_repair_hints(
        [finding_at("Procedure.performed")],
        _Tree({"Procedure.performed[x]": ["dateTime", "Period"]}),
        {"Sourcedefinition_x.procDate": "Procedure.performed"},
    )
    assert len(hints) == 1
    hint = hints[0]
    assert hint["source_field"] == "procDate"
    by_type = {o["type"]: o for o in hint["options"]}
    assert by_type["dateTime"]["element"] == "performed"
    assert by_type["dateTime"]["transform"] == "cast"


def test_a_normalized_choice_path_resolves_against_a_real_target_tree():
    """The real tree retains `[x]`; repair routing intentionally removes it."""
    from agent.context import _choice_repair_hints
    from mapping.target_tree import TargetTree

    tree = TargetTree.from_snapshot(
        {
            "resourceType": "StructureDefinition",
            "type": "Procedure",
            "snapshot": {
                "element": [
                    {
                        "id": "Procedure",
                        "path": "Procedure",
                        "min": 0,
                        "max": "1",
                        "type": [{"code": "Procedure"}],
                    },
                    {
                        "id": "Procedure.performed[x]",
                        "path": "Procedure.performed[x]",
                        "min": 0,
                        "max": "1",
                        "type": [{"code": "dateTime"}, {"code": "Period"}],
                    },
                ]
            },
        }
    )
    assert tree is not None

    hints = _choice_repair_hints(
        [finding_at("Procedure.performed")],
        tree,
        {
            "Sourcedefinition_x.performedDateTime": (
                "Procedure.performed[x]:performedDateTime"
            )
        },
    )

    assert len(hints) == 1
    assert hints[0]["source_field"] == "performedDateTime"
    assert {option["type"] for option in hints[0]["options"]} == {
        "dateTime",
        "Period",
    }
    assert {option["element"] for option in hints[0]["options"]} == {"performed"}


def test_a_profile_narrowed_single_type_choice_still_gets_a_hint():
    from agent.context import _choice_repair_hints

    hints = _choice_repair_hints(
        [finding_at("Observation.effective")],
        _Tree({"Observation.effective[x]": ["dateTime"]}),
        {"Sourcedefinition_x.when": "Observation.effective[x]"},
    )

    assert hints == [
        {
            "path": "Observation.effective",
            "source_field": "when",
            "options": [
                {"element": "effective", "type": "dateTime", "transform": "cast"}
            ],
        }
    ]


def test_a_stale_modifier_diagnostic_is_satisfied_by_the_current_mapping_table():
    from pathlib import Path
    from types import SimpleNamespace

    from agent.service import AgentFixService, ProjectContext

    profile_url = "http://example.org/StructureDefinition/TestProcedure"
    profile = {
        "resourceType": "StructureDefinition",
        "id": "TestProcedure",
        "url": profile_url,
        "type": "Procedure",
        "snapshot": {
            "element": [
                {
                    "id": "Procedure",
                    "path": "Procedure",
                    "min": 0,
                    "max": "1",
                    "type": [{"code": "Procedure"}],
                },
                {
                    "id": "Procedure.status",
                    "path": "Procedure.status",
                    "min": 1,
                    "max": "1",
                    "type": [{"code": "code"}],
                },
            ]
        },
    }
    document = structure_map()
    document["structure"] = [{"url": profile_url, "mode": "target"}]
    project = ProjectContext(
        project_dir=Path("."),
        conf={},
        profiles={profile_url: profile},
        mapping_table={"Source.status": "Procedure.status"},
        coverage_report=SimpleNamespace(
            mapping_diagnostics=[
                {
                    "code": "target-modifier-element",
                    "message": "Procedure.status requires explicit mapping intent.",
                    "profile": "TestProcedure",
                    "path": "Procedure.status",
                    "severity": "warning",
                }
            ]
        ),
    )
    service = AgentFixService(project=project, proposer=None)

    assert service.generator_findings(document) == []


def test_a_single_typed_element_is_not_a_choice():
    from agent.context import _choice_repair_hints

    assert (
        _choice_repair_hints(
            [finding_at("Observation.status")],
            _Tree({"Observation.status": ["code"]}),
            {},
        )
        == []
    )


def test_a_string_typed_choice_option_uses_copy():
    from agent.context import _choice_repair_hints

    hints = _choice_repair_hints(
        [finding_at("Observation.value")],
        _Tree({"Observation.value[x]": ["string", "Quantity"]}),
        {},
    )
    by_type = {o["type"]: o for o in hints[0]["options"]}
    assert by_type["string"]["transform"] == "copy"
    assert by_type["Quantity"]["transform"] == "create"
    assert by_type["Quantity"]["element"] == "value"


def test_no_target_tree_means_no_hint():
    from agent.context import _choice_repair_hints

    assert _choice_repair_hints([finding_at("Procedure.performed")], None, {}) == []


# -- an engine that cannot parse a real resource type is the environment -------


def test_a_real_resource_type_the_engine_rejects_is_environmental():
    """Matchbox called Claim, Medication and ChargeItem unknown; all are R4."""
    from agent.engine import unrecognized_resource_name

    for name in ("Claim", "Medication", "ChargeItem", "ClaimResponse"):
        message = (
            f"This content cannot be parsed (unknown or unrecognized resource "
            f"name '{name}')"
        )
        assert unrecognized_resource_name(message) == name


def test_a_name_that_is_not_a_resource_type_stays_the_maps_problem():
    from agent.engine import unrecognized_resource_name

    message = (
        "This content cannot be parsed (unknown or unrecognized resource name "
        "'Sourcedefinition')"
    )
    assert unrecognized_resource_name(message) is None


def test_the_owner_decision_uses_it():
    from agent.engine import _environment_owner
    from agent.validation import ActionOwner as AO

    owner, evidence = _environment_owner(
        "This content cannot be parsed (unknown or unrecognized resource name 'Claim')",
        None,
    )
    assert owner is AO.ENVIRONMENT
    assert evidence == {"unrecognized_resource": "Claim"}


# -- a choice element is emitted under its concrete name ----------------------


class _ChoiceTree:
    def __init__(self, types):
        self._types = types

    def effective_types(self, path):
        return self._types.get(path, [])


def test_a_required_choice_accepts_its_typed_output_name():
    """Six hddt maps carried a blocking `output-required-missing` for
    `Observation.effective` while `$transform` was returning
    `"effectiveDateTime": "2024-01-15T10:30:00+01:00"`."""
    from agent.engine import _output_path_candidates

    tree = _ChoiceTree({"Observation.effective[x]": ["dateTime"]})
    assert _output_path_candidates("Observation.effective[x]", tree) == [
        "Observation.effective",
        "Observation.effectiveDateTime",
    ]


def test_a_choice_above_the_required_element_is_expanded_too():
    """`effective[x].start` is emitted as `effectivePeriod.start`."""
    from agent.engine import _output_path_candidates

    tree = _ChoiceTree({"Observation.effective[x]": ["Period"]})
    assert _output_path_candidates("Observation.effective[x].start", tree) == [
        "Observation.effective.start",
        "Observation.effectivePeriod.start",
    ]


def test_every_permitted_type_is_offered():
    from agent.engine import _output_path_candidates

    tree = _ChoiceTree({"Observation.value[x]": ["Quantity", "string"]})
    got = _output_path_candidates("Observation.value[x]", tree)
    assert "Observation.valueQuantity" in got and "Observation.valueString" in got


def test_a_non_choice_path_is_unchanged():
    from agent.engine import _output_path_candidates

    tree = _ChoiceTree({})
    assert _output_path_candidates("Observation.status", tree) == ["Observation.status"]


# -- the coverage layer sees a concrete choice name too -----------------------


def test_a_typed_choice_write_satisfies_the_base_requirement():
    """`Procedure.extension.valueString` meets a `…extension.value` requirement."""
    from agent.validation import _coverage_findings

    class Tree:
        res_type = "Procedure"

        def required_manifest(self):
            return [
                {"id": "Procedure.extension.value[x]",
                 "path": "Procedure.extension.value[x]",
                 "min": 1, "max": "1", "active": True, "provider": None}
            ]

    findings, covered, _obligations = _coverage_findings(
        {"Procedure.extension.valueString"},
        target_tree=Tree(),
        mapping_table={},
        map_url="http://example.org/StructureMap/x",
        map_id="x",
        profile_url=None,
    )
    assert "Procedure.extension.value" in covered
    assert [f.code for f in findings if f.code == "required-path-unmapped"] == []


def test_an_unrelated_prefix_does_not_satisfy_it():
    from agent.validation import _coverage_findings

    class Tree:
        res_type = "Observation"

        def required_manifest(self):
            return [
                {"id": "Observation.value[x]", "path": "Observation.value[x]",
                 "min": 1, "max": "1", "active": True, "provider": None}
            ]

    _findings, covered, _ = _coverage_findings(
        {"Observation.valueset"},  # lower-case suffix: a different element
        target_tree=Tree(),
        mapping_table={},
        map_url="http://example.org/StructureMap/x",
        map_id="x",
        profile_url=None,
    )
    assert covered == []


# -- the concrete choice name must reach the rule writing the base ------------


def test_a_concrete_choice_finding_reaches_the_rule_writing_the_base():
    """`Observation.valueQuantity` must find a rule addressing `Observation.value`.

    Three NDHM maps stopped with no provider call on a finding that did carry an
    element: the engine names the concrete form, the map writes the base plus a
    typed transform.
    """
    from agent.context import _choice_base_matches

    by_path = {"Observation.value": ["/group/0/rule/3/target/0"]}
    assert _choice_base_matches("Observation.valueQuantity", by_path) == [
        "/group/0/rule/3/target/0"
    ]


def test_a_camel_case_element_is_not_split_into_a_base_nothing_emits():
    """`bodySite` must not be read as `body` + `Site`."""
    from agent.context import _choice_base_matches

    assert _choice_base_matches("Observation.bodySite", {"Observation.value": ["/x"]}) == []


def test_the_base_must_actually_be_addressed_by_the_map():
    from agent.context import _choice_base_matches

    assert _choice_base_matches("Observation.valueQuantity", {}) == []


# -- one element, several spellings -------------------------------------------


def test_a_choice_slice_yields_both_spellings():
    """The table writes `value[x]:valueQuantity`; `$validate` says `valueQuantity`."""
    from agent.validation import target_path_spellings

    assert target_path_spellings(
        "ObservationVitalSigns.value[x]:valueQuantity.system"
    ) == {
        "ObservationVitalSigns.value.system",
        "ObservationVitalSigns.valueQuantity.system",
    }


def test_a_path_without_a_choice_slice_has_one_spelling():
    from agent.validation import target_path_spellings

    assert target_path_spellings("Observation.bodySite") == {"Observation.bodySite"}
    assert target_path_spellings("Observation.effective[x]") == {"Observation.effective"}


def test_a_placeholder_behind_a_choice_slice_is_demoted():
    """Three NDHM maps blocked on a binding the *fixture* filled with "1"."""
    from agent.engine import EngineSession

    unverified = EngineSession._unverified_target_paths(
        [],
        {"Sourcedefinition_x.vsUnitSystem":
         "ObservationVitalSigns.value[x]:valueQuantity.system"},
    )
    assert EngineSession._placeholder_expression(
        ["Observation.value.ofType(Quantity).system"], unverified
    )


# -- the sweep: every path vocabulary meets through one helper ----------------
#
# The same choice-spelling mismatch was found in six separate places before this
# was made a shared concern. These pin the tree-lookup half of it.


def _tree_for(profile_element_id, types):
    from mapping.target_tree import TargetNode, TargetTree

    root = TargetNode(eid="Observation", path="Observation")
    node = TargetNode(
        eid=profile_element_id,
        path=profile_element_id,
        min=1,
        max="1",
        types=list(types),
        parent=root,
    )
    root.children[node.name] = node
    return TargetTree(root, {root.eid: root, node.eid: node}, "Observation")


def test_a_tree_lookup_accepts_every_spelling():
    """The tree keys on `effective[x]`; callers hold `effective` or
    `effectiveDateTime`, and a literal lookup silently answers "unknown"."""
    from agent.validation import resolve_tree_node, resolve_tree_types

    tree = _tree_for("Observation.effective[x]", ["dateTime"])
    for spelling in (
        "Observation.effective[x]",
        "Observation.effective",
        "Observation.effectiveDateTime",
    ):
        assert resolve_tree_node(tree, spelling) is not None, spelling
        assert resolve_tree_types(tree, spelling) == ["dateTime"], spelling


def test_a_tree_lookup_does_not_invent_a_choice():
    """`bodySite` must not be reduced to a `body[x]` the tree does not hold."""
    from agent.validation import resolve_tree_node

    tree = _tree_for("Observation.effective[x]", ["dateTime"])
    assert resolve_tree_node(tree, "Observation.bodySite") is None


def test_an_unknown_tree_or_path_is_simply_unknown():
    from agent.validation import resolve_tree_node, resolve_tree_types

    assert resolve_tree_node(None, "Observation.effective") is None
    assert resolve_tree_types(None, "Observation.effective") == []
    tree = _tree_for("Observation.effective[x]", ["dateTime"])
    assert resolve_tree_node(tree, "") is None


def test_required_gap_owners_are_keyed_under_every_spelling():
    """A manifest entry for `effective[x]` must be found by a hint that read
    `effectiveDateTime` off a FHIRPath expression."""
    from agent.engine import EngineSession

    tree = _tree_for("Observation.effective[x]", ["dateTime"])
    owners = EngineSession._required_gap_owners(tree, {}, "obs")
    assert "Observation.effective" in owners
    assert "Observation.effectiveDateTime" in owners


# -- a context-only alias still carries its source field ----------------------


def test_a_context_only_alias_inherits_its_source_field():
    """`{"context": "srcVal-2", "variable": "srcV"}` makes srcV another name for
    what srcVal-2 reads. Dropped, a translated field was emptied rather than
    omitted and `translate()` received null — an HTTP 500 against a correct map."""
    from agent.fixtures import translate_source_fields

    document = {
        "group": [
            {
                "name": "g",
                "rule": [
                    {
                        "name": "outer",
                        "source": [
                            {"context": "source", "element": "answerCode",
                             "variable": "srcVal-2"}
                        ],
                        "rule": [
                            {
                                "name": "set-val",
                                "source": [{"context": "srcVal-2", "variable": "srcV"}],
                                "target": [
                                    {
                                        "context": "target",
                                        "element": "code",
                                        "transform": "translate",
                                        "parameter": [
                                            {"valueId": "srcV"},
                                            {"valueString": "http://x/cm"},
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    assert translate_source_fields([document]) == {"answerCode"}


# -- a required slice cannot be matched by a placeholder ----------------------


def test_a_required_slice_failure_on_a_placeholder_is_demotable():
    """The map writes coding.code and .system; the fixture filled them with "1",
    and a slice is matched by discriminator."""
    from agent.engine import EngineSession

    unverified, verified = EngineSession._target_value_provenance(
        [],
        {
            "Src.vitalStatusCode": "Observation.value[x].coding.code",
            "Src.vitalStatusSystem": "Observation.value[x].coding.system",
        },
    )
    assert EngineSession._placeholder_expression(
        ["Observation.value.ofType(CodeableConcept)"], unverified, verified
    )


def test_a_verified_value_in_the_subtree_blocks_the_demotion():
    """A real ConceptMap-sourced code means the slice failure is the map's."""
    from agent.engine import EngineSession
    from agent.fixtures import FixtureKind, SourceFixture

    fixture = SourceFixture(
        fixture_id="f", kind=FixtureKind.SYNTHETIC_FILLED, label="synthetic-filled",
        instance={}, verified_fields=["vitalStatusCode"],
    )
    unverified, verified = EngineSession._target_value_provenance(
        [fixture],
        {"Src.vitalStatusCode": "Observation.value[x].coding.code"},
    )
    assert EngineSession._placeholder_expression(
        ["Observation.value.ofType(CodeableConcept)"], unverified, verified
    ) is None


# -- an extension's choice types live in the extension, not the profile -------


class _NoTypes:
    def effective_types(self, path):
        return []


def test_a_choice_with_no_declared_types_is_read_off_the_output():
    """An extension's `value[x]` types are in the extension's own definition, so
    the referring profile's tree declares none — leaving only the base spelling,
    which no conformant instance ever serializes."""
    from agent.engine import _output_path_candidates

    output = {
        "resourceType": "Condition",
        "extension": [{"url": "http://x", "valueCodeableConcept": {"coding": []}}],
    }
    assert _output_path_candidates(
        "Condition.extension.value[x]", _NoTypes(), output
    ) == ["Condition.extension.value", "Condition.extension.valueCodeableConcept"]


def test_without_an_output_only_the_declared_spelling_is_offered():
    from agent.engine import _output_path_candidates

    assert _output_path_candidates("Condition.extension.value[x]", _NoTypes(), None) == [
        "Condition.extension.value"
    ]


def test_only_a_real_choice_suffix_is_accepted():
    """`url` sits beside `valueX` in every extension and must not be mistaken
    for a concrete form of `value`."""
    from agent.engine import _choice_names_in_output

    output = {"extension": [{"url": "http://x", "valueString": "a", "value": "b"}]}
    assert _choice_names_in_output(output, "extension", "value") == ["valueString"]


def test_declared_types_still_win_over_the_instance():
    from agent.engine import _output_path_candidates

    class Typed:
        def effective_types(self, path):
            return ["dateTime"]

    output = {"resourceType": "Observation", "effectiveInstant": "x"}
    got = _output_path_candidates("Observation.effective[x]", Typed(), output)
    assert got == ["Observation.effective", "Observation.effectiveDateTime"]


# -- slicing the profile never constrains is the profile's problem -------------


def test_an_underspecified_discriminator_is_environmental():
    """The profile slices on an element it never constrains.

    No instance can satisfy such a discriminator, so no edit to the map can
    resolve it. It arrived as blocking and map-fixable 1 592 times across the
    corpus, which put an unfixable finding at the top of the worklist.
    """
    from agent.engine import _environment_owner
    from agent.validation import ActionOwner as AO

    message = (
        "Slicing cannot be evaluated: Could not match discriminator (coding) for "
        "slice Observation.category:obstetrics in profile "
        "https://example.org/StructureDefinition/x - the discriminator [coding] "
        "does not have fixed value, binding or existence assertions"
    )
    owner, evidence = _environment_owner(message, None)

    assert owner is AO.ENVIRONMENT
    assert evidence == {"underspecified_discriminator": "coding"}


def test_a_discriminator_that_merely_did_not_match_stays_the_maps_problem():
    """The distinction the tail carries.

    Without the assertion tail, "could not match" means the instance lacks the
    coding a rule was supposed to write — the map-fixable case, and the reason
    this is not keyed on `Could not match discriminator` alone.
    """
    from agent.engine import _environment_owner

    message = (
        "Could not match discriminator (coding) for slice "
        "Observation.category:vital-signs"
    )
    assert _environment_owner(message, None) == (None, {})


# -- the provider is handed pointers, not arithmetic ---------------------------


def _rule_with_a_child():
    return {
        "name": "map-gender",
        "source": [{"context": "src", "element": "gender", "variable": "v"}],
        "target": [{"context": "tgt", "element": "gender", "transform": "copy"}],
        "rule": [{"name": "child", "source": [{"context": "v", "element": "x"}]}],
    }


def test_members_are_listed_with_their_current_values():
    from agent.context import _editable_members

    members = {m.pointer: m for m in _editable_members(_rule_with_a_child(), "/r")}

    assert members["/r/source/0/element"].value == "gender"
    assert members["/r/target/0/transform"].value == "copy"
    assert members["/r/target/0/transform"].ops == ["replace", "remove"]


def test_an_array_is_offered_as_an_append_not_an_index():
    """`/-` is the one index that cannot shift under an earlier operation."""
    from agent.context import _editable_members

    members = {m.pointer: m for m in _editable_members(_rule_with_a_child(), "/r")}

    assert members["/r/target/-"].ops == ["add"]
    assert not any(p.endswith("/target/1") for p in members)


def test_a_child_rules_members_are_not_attributed_to_its_parent():
    """A child has its own pointer and is listed under it, not folded in here.

    The insertion point `/r/rule/-` is a different thing and is listed: it says
    where a *new* child goes, not what an existing one contains.
    """
    from agent.context import _editable_members

    members = [m.pointer for m in _editable_members(_rule_with_a_child(), "/r")]

    assert not any(pointer.startswith("/r/rule/0") for pointer in members)
    assert "/r/rule/-" in members


def test_a_long_value_is_left_to_the_excerpt():
    from agent.context import MAX_MEMBER_VALUE_CHARS, _editable_members

    rule = {"name": "x" * (MAX_MEMBER_VALUE_CHARS + 1), "source": []}

    members = _editable_members(rule, "/r")

    assert not any(m.pointer == "/r/name" for m in members)
    # The child-rule pointer is structural, not a value, and still belongs.
    assert [m.pointer for m in members] == ["/r/rule"]


# -- the provider schema cannot express a guard -------------------------------


def test_the_provider_schema_offers_no_test_operation():
    """The prohibition binds in the schema, not only in the prompt.

    `test-failed` and `unguarded-mutation` were 39 of 58 rejections in one
    corpus run — two thirds of every refusal, none about the repair itself.
    """
    import json

    from agent.models import ProposedPatch

    schema = json.dumps(ProposedPatch.model_json_schema())

    assert '"test"' not in schema
    assert '"replace"' in schema


def test_a_proposal_widens_into_the_internal_envelope():
    from agent.models import AgentPatch, ProposedPatch

    proposed = ProposedPatch(
        schema_version=1,
        map_url="http://example.org/StructureMap/x",
        map_id="x",
        base_sha256="0" * 64,
        diagnostic_ids=["vf-1"],
        patch=[{"op": "replace", "path": "/group/0/name", "value": "G"}],
    )

    widened = proposed.as_agent_patch()

    assert isinstance(widened, AgentPatch)
    assert widened.as_rfc6902() == [
        {"op": "replace", "path": "/group/0/name", "value": "G"}
    ]


# -- an extension finding names one extension, not the element ----------------


def _map_with_two_extensions():
    def extension(name, url, source):
        return {
            "name": f"map-extension-{name}",
            "source": [{"context": "source", "element": source}],
            "target": [
                {
                    "context": "target",
                    "element": "extension",
                    "variable": f"ext-{name}",
                    "transform": "create",
                    "parameter": [{"valueString": "Extension"}],
                }
            ],
            "rule": [
                {
                    "name": f"set-extension-url-{name}",
                    "source": [{"context": "source"}],
                    "target": [
                        {
                            "context": f"ext-{name}",
                            "element": "url",
                            "transform": "copy",
                            "parameter": [{"valueString": url}],
                        }
                    ],
                }
            ],
        }

    return {
        "resourceType": "StructureMap",
        "group": [
            {
                "name": "TransformProcedure",
                "input": [
                    {"name": "source", "mode": "source"},
                    {"name": "target", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "set-fixed-status",
                        "source": [{"context": "source"}],
                        "target": [{"context": "target", "element": "status"}],
                    },
                    extension("Abbrev", "https://fhir.example.de/EX_Abbrev", "abbrev"),
                    extension("Other", "https://fhir.example.de/EX_Other", "other"),
                ],
            }
        ],
    }


def test_a_url_predicate_survives_normalization():
    """`https:` is not a slice separator, and the URL's dots are not path dots.

    Splitting on `.` cut the URL into pieces and the slice rule then truncated
    each at the `:` of `https:`, yielding
    `Procedure.extension[url='https.example.de/EX_Abbrev']` — a string no rule
    and no tree node can match.
    """
    from agent.validation import normalize_target_path

    path = "Procedure.extension[url='https://fhir.example.de/EX_Abbrev']"

    assert normalize_target_path(path) == path


def test_normalization_still_strips_slices_and_choices():
    """The behaviour the bracket-aware split must not disturb."""
    from agent.validation import normalize_target_path

    assert normalize_target_path("Observation.category:obstetrics") == (
        "Observation.category"
    )
    assert normalize_target_path("Observation.value[x]") == "Observation.value"
    assert normalize_target_path("Observation.component:systolic.value[x]") == (
        "Observation.component.value"
    )


def test_each_extension_canonical_maps_to_the_rule_that_builds_it():
    from agent.context import _extension_url_owners

    owners = _extension_url_owners(_map_with_two_extensions())

    assert owners["https://fhir.example.de/EX_Abbrev"] == [
        "/group/0/rule/1",
        "/group/0/rule/1/rule/0",
    ]
    assert owners["https://fhir.example.de/EX_Other"] == [
        "/group/0/rule/2",
        "/group/0/rule/2/rule/0",
    ]


def test_an_extension_finding_focuses_only_its_own_extension():
    """The routing gap that left a map-fixable finding with no edit scope.

    Every extension rule addresses plain `extension`, so the target-path index
    cannot tell two apart. EVO13's modifier-extension mismatch was classified
    `map-fixable` and routed nowhere: the model identified the right rule, the
    worklist authorized no pointer, the patch was rejected as out of scope, and
    the model then abstained.
    """
    from agent.context import _focus_pointers
    from agent.validation import ActionOwner, Producer, Stage, ValidationFinding

    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:structure",
        "Extension modifier mismatch",
        owner=ActionOwner.MAP_FIXABLE,
        evidence={
            "path_hint": "Procedure.extension[url='https://fhir.example.de/EX_Abbrev']"
        },
    )

    focus = _focus_pointers(_map_with_two_extensions(), [finding])

    assert "/group/0/rule/1" in focus
    # and emphatically not the other extension
    assert not any(pointer.startswith("/group/0/rule/2") for pointer in focus)


# -- the pointer the model needs must be one it was shown --------------------


def test_an_object_entry_is_addressable_not_only_its_leaves():
    """The gap behind 8 of 10 rejections in one run.

    Only scalar leaves were listed, so a model rewriting a whole target entry
    addressed `…/target/0/element` — a string field — and sent the target object
    as its value. `invalid-candidate: Input should be a valid string …
    input_type=dict`, four times unchanged, on a repair that was otherwise right.
    """
    from agent.context import _editable_members

    members = {m.pointer: m for m in _editable_members(_rule_with_a_child(), "/r")}

    assert members["/r/target/0"].ops == ["replace"]
    assert members["/r/target/0"].note
    # and the leaf is still there, with its scalar value
    assert members["/r/target/0/element"].value == "gender"


def test_a_rule_with_children_is_told_where_to_append_one():
    from agent.context import _editable_members

    members = {m.pointer: m for m in _editable_members(_rule_with_a_child(), "/r")}

    assert members["/r/rule/-"].ops == ["add"]


def test_a_childless_rule_is_told_the_array_does_not_exist_yet():
    """RFC 6902 cannot append to an array that is not there.

    A model with a child rule to add wrote `…/rule/-` on a rule with no `rule`
    member and was rejected with `invalid-pointer`; the pointer that would have
    worked, `…/rule` with an array value, appeared nowhere.
    """
    from agent.context import _editable_members

    childless = {
        "name": "map-status",
        "target": [{"context": "target", "element": "status"}],
    }
    members = {m.pointer: m for m in _editable_members(childless, "/r")}

    assert "/r/rule/-" not in members
    assert members["/r/rule"].ops == ["add"]
    assert "array" in members["/r/rule"].note


def test_the_prompt_states_where_a_child_rule_attaches():
    """The other half of the same mistake: `rule` nested inside a target."""
    from llm.prompts import AGENT_FIX_SYSTEM, render

    rendered = render(AGENT_FIX_SYSTEM)

    assert "never inside a `target`" in rendered


def test_the_child_rule_pointer_survives_a_full_member_list():
    """It is the one pointer that cannot be derived from the excerpt.

    A generated rule with two sources and a parameterized target reaches
    MAX_MEMBERS_PER_RULE on leaves alone. Appended after them, the child-rule
    pointer was silently discarded: both focused rules of one map produced
    exactly 16 members, the model saw no insertion pointer, and invented
    `…/rule/-` on a rule with no `rule` member.
    """
    from agent.context import MAX_MEMBERS_PER_RULE, _editable_members

    crowded = {
        "name": "map-participant",
        "source": [
            {"context": "source", "element": f"field{n}", "variable": f"v{n}"}
            for n in range(4)
        ],
        "target": [
            {
                "context": "target",
                "element": f"element{n}",
                "variable": f"t{n}",
                "transform": "create",
                "parameter": [{"valueString": "Extension"}],
            }
            for n in range(4)
        ],
        "rule": [{"name": "child"}],
    }

    members = _editable_members(crowded, "/r")

    assert len(members) == MAX_MEMBERS_PER_RULE
    assert members[0].pointer == "/r/rule/-"


def test_a_scalar_member_names_its_json_type():
    """The rendered value shows the shape; the type names it.

    A leaf rendered only as `= "extension"` is already an unambiguous JSON
    string, and a model still replaced it with `{"valueString": ...}` on four
    consecutive retries. Naming the type is a second, explicit channel.
    """
    from agent.context import _editable_members

    rule = {
        "name": "write-extension",
        "source": [{"context": "src", "element": "a", "variable": "v"}],
        "target": [{"context": "tgt", "element": "extension"}],
    }
    members = {m.pointer: m for m in _editable_members(rule, "/r")}

    leaf = members["/r/target/0/element"]
    assert leaf.json_type == "string"
    assert leaf.value == "extension"


def test_an_object_member_does_not_claim_a_scalar_type():
    """`json_type` describes a value the list actually shows.

    Object entries carry a note instead of a value, so labelling them with a
    scalar type would describe something the model was never shown.
    """
    from agent.context import _editable_members

    rule = {
        "name": "write-extension",
        "target": [{"context": "tgt", "element": "extension"}],
    }
    members = {m.pointer: m for m in _editable_members(rule, "/r")}

    assert members["/r/target/0"].json_type is None
    assert members["/r/target/0"].note is not None


def test_the_member_list_renders_the_type_beside_the_value():
    """The type must reach the prompt, not merely the model object."""
    from agent.context import AgentContext, PointerValue, RulePointer
    from llm.prompts import AGENT_FIX_USER, render

    pointer = RulePointer(
        pointer="/r",
        label="G › write-extension",
        name="write-extension",
        focused=True,
        members=[
            PointerValue(
                pointer="/r/target/0/element",
                value="extension",
                ops=["replace", "remove"],
                json_type="string",
            )
        ],
    )
    context = AgentContext(map_url="http://e/x", map_id="x", pointers=[pointer])

    rendered = render(AGENT_FIX_USER, context=context, map_json="{}")

    assert '    - `/r/target/0/element` (string) = "extension"\n' in rendered


def test_the_prompt_forbids_the_fhir_value_envelope():
    """The mistake the previous wording did not name.

    The prompt already said a leaf takes a string "not a target object". The
    model was not sending a target object; it sent `{"valueString": ...}`, a
    FHIR `value[x]` envelope, which that sentence never mentioned.
    """
    from llm.prompts import AGENT_FIX_SYSTEM, render

    rendered = render(AGENT_FIX_SYSTEM)

    assert '{"valueString": "modifierExtension"}' in rendered
    assert "parameter" in rendered
