"""Deterministic validation and acceptance gates (WP6).

The property under test is that a candidate is accepted only on evidence, and
that every way of *not* having evidence fails closed. In particular: an
unreachable engine is a failure, an unrecognized engine code is blocking, a
demotion is always attributable, and nothing a model says enters the decision.
"""

import copy

import pytest

from agent.engine import EngineResult, EngineSession, scratch_identity
from agent.fixtures import (
    FixtureKind,
    SourceFixture,
    build_example_fixtures,
    build_shared_fixtures,
    build_synthetic_fixtures,
    fixtures_digest,
    source_field_specs,
    temporal_cast_overrides,
    translate_overrides,
)
from agent.validation import (
    CLASSIFICATION,
    UNKNOWN_CLASSIFICATION,
    ActionOwner,
    GateStatus,
    Producer,
    Stage,
    ValidationFinding,
    ValidationReport,
    addressed_paths,
    authored_selector_obligations,
    build_target_tree,
    classify,
    decide_acceptance,
    finding_identity,
    findings_from_diagnostics,
    merge_engine_result,
    validate_offline,
)

pytestmark = pytest.mark.unit


# --- fixtures ---------------------------------------------------------------


def source_model():
    """A flat source logical model, as the tool generates them."""

    return {
        "resourceType": "StructureDefinition",
        "id": "src-model",
        "url": "http://example.org/StructureDefinition/src-model",
        "name": "SrcModel",
        "status": "draft",
        "kind": "logical",
        "abstract": False,
        "type": "SrcModel",
        "differential": {
            "element": [
                {"path": "SrcModel", "min": 0, "max": "*", "type": [{"code": "Element"}]},
                {"path": "SrcModel.givenName", "min": 1, "max": "1", "type": [{"code": "string"}]},
                {"path": "SrcModel.familyName", "min": 0, "max": "1", "type": [{"code": "string"}]},
                {"path": "SrcModel.recordNumber", "min": 0, "max": "1", "type": [{"code": "integer"}]},
                {"path": "SrcModel.born", "min": 0, "max": "1", "type": [{"code": "string"}]},
                {"path": "SrcModel.sex", "min": 0, "max": "1", "type": [{"code": "code"}]},
            ]
        },
    }


def profile():
    """A Patient profile whose snapshot stops one level down, as they do."""

    def element(path, minimum=0, maximum="1", type_code="string", **extra):
        return {
            "id": path,
            "path": path,
            "min": minimum,
            "max": maximum,
            "type": [{"code": type_code}],
            **extra,
        }

    return {
        "resourceType": "StructureDefinition",
        "id": "TestPatient",
        "url": "http://example.org/StructureDefinition/TestPatient",
        "name": "TestPatient",
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
                element("Patient.gender", 0, "1", "code"),
                element("Patient.photo", 0, "0", "Attachment"),
            ]
        },
    }


def structure_map():
    """A map that satisfies the mapping table below."""

    return {
        "resourceType": "StructureMap",
        "id": "sm-test",
        "url": "http://example.org/StructureMap/sm-test",
        "name": "SmTest",
        "status": "draft",
        "structure": [
            {"url": "http://example.org/StructureDefinition/src-model", "mode": "source"},
            {"url": "http://example.org/StructureDefinition/TestPatient", "mode": "target"},
        ],
        "group": [
            {
                "name": "TransformPatient",
                "typeMode": "none",
                "input": [
                    {"name": "source", "type": "SrcModel", "mode": "source"},
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


def mapping_table():
    return {
        "givenName": "Patient.name.given",
        "familyName": "Patient.name.family",
        "born": "Patient.birthDate",
    }


@pytest.fixture(scope="module")
def tree():
    return build_target_tree(profile())


@pytest.fixture
def source_field_names():
    return {spec["name"] for spec in source_field_specs(source_model())}


def offline(document, tree, source_field_names, **kwargs):
    return validate_offline(
        document,
        target_tree=tree,
        mapping_table=mapping_table(),
        source_fields=source_field_names,
        profile_url=profile()["url"],
        **kwargs,
    )


# --- the classification contract --------------------------------------------


def test_an_unknown_code_is_unclassified_and_blocking():
    assert classify(Producer.ENGINE, "transform:something-new") == UNKNOWN_CLASSIFICATION
    owner, gate = UNKNOWN_CLASSIFICATION
    assert owner is ActionOwner.UNCLASSIFIED
    assert gate is GateStatus.BLOCKING


def test_an_unknown_engine_failure_cannot_authorize_an_edit():
    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.TRANSFORM,
        "transform:brand-new-code",
        "something the table has never seen",
    )
    report = ValidationReport(findings=[finding])
    assert finding.action_owner is ActionOwner.UNCLASSIFIED
    assert finding.blocking
    assert report.worklist() == []


def test_classification_is_keyed_on_the_producer_not_only_the_code():
    # The same word means different things depending on who said it, so the key
    # has to carry both. This is what stops one producer's rename from silently
    # reclassifying another's findings.
    assert (Producer.COVERAGE, "required-path-unmapped") in CLASSIFICATION
    assert (Producer.ENGINE, "required-path-unmapped") not in CLASSIFICATION
    assert classify(Producer.ENGINE, "required-path-unmapped") == UNKNOWN_CLASSIFICATION


def test_ownership_and_gating_are_independent():
    # A finding nobody can act on still stops a candidate.
    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.BOOTSTRAP,
        "engine-unreachable",
        "no server",
    )
    assert finding.action_owner is ActionOwner.ENVIRONMENT
    assert finding.blocking

    # ...and a map-fixable finding can be non-blocking.
    advisory = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:structure",
        "a warning",
        gate=GateStatus.NON_BLOCKING,
    )
    assert advisory.action_owner is ActionOwner.MAP_FIXABLE
    assert not advisory.blocking


def test_every_classified_engine_code_names_a_real_stage():
    stages = {stage.value for stage in Stage}
    for producer, code in CLASSIFICATION:
        if producer is not Producer.ENGINE or ":" not in code:
            continue
        assert code.split(":", 1)[0] in stages, code


# --- finding identity --------------------------------------------------------


def test_identity_ignores_the_pointer_and_the_message():
    # A pointer moves whenever the map is edited. Keying identity on one would
    # make every candidate look like it had fixed the old finding and introduced
    # a new one, and the comparison would never mean anything.
    first = ValidationFinding.build(
        Producer.PATH_RESOLUTION,
        Stage.PATHS,
        "target-path-not-found",
        "one wording",
        path="Patient.name.family",
        pointer="/group/0/rule/1/target/0",
    )
    second = ValidationFinding.build(
        Producer.PATH_RESOLUTION,
        Stage.PATHS,
        "target-path-not-found",
        "an entirely different wording",
        path="Patient.name.family",
        pointer="/group/0/rule/7/rule/2/target/0",
    )
    assert first.finding_id == second.finding_id


def test_two_different_invariants_do_not_collapse_into_one_finding():
    # Matchbox reports every failed invariant as `issue.code = invariant` with
    # `expression` naming the containing resource, so producer/stage/code/path/
    # fixture do not tell two of them apart. Without a discriminator a candidate
    # could replace one validation defect with another and be accepted.
    def invariant(message):
        return ValidationFinding.build(
            Producer.ENGINE,
            Stage.VALIDATE,
            "validate:invariant",
            message,
            fixture_id="fx",
            profile_url=profile()["url"],
            discriminator=EngineSession._issue_discriminator(["Patient"], message),
        )

    first = invariant("pat-1: contact must have details")
    second = invariant("us-core-6: name must have a family")
    assert first.finding_id != second.finding_id

    engine = {"engine_available": True, "executed_fixtures": ["fx"], "validated_fixtures": ["fx"]}
    decision = decide_acceptance(
        clean_report(findings=[first], **engine),
        clean_report(findings=[second], **engine),
    )
    assert not decision.accepted
    assert decision.new_blocking_ids == [second.finding_id]


def test_the_discriminator_prefers_the_structured_expression():
    same_message = "an invariant failed"
    one = EngineSession._issue_discriminator(["Patient.name"], same_message)
    other = EngineSession._issue_discriminator(["Patient.telecom"], same_message)
    assert one != other


def test_identity_separates_fixtures():
    common = (Producer.ENGINE, Stage.TRANSFORM, "transform:processing")
    assert finding_identity(*common, fixture_id="a") != finding_identity(
        *common, fixture_id="b"
    )


def test_identity_separates_paths():
    common = (Producer.COVERAGE, Stage.COVERAGE, "required-path-unmapped")
    assert finding_identity(*common, path="Patient.name") != finding_identity(
        *common, path="Patient.identifier"
    )


# --- reading a map ------------------------------------------------------------


def test_addressed_paths_resolve_nested_context_variables():
    sources, targets = addressed_paths(structure_map())
    assert [entry.path for entry in targets] == [
        "Patient.birthDate",
        "Patient.name",
        "Patient.name.family",
        "Patient.name.given",
    ]
    assert {entry.path for entry in sources} == {
        "SrcModel.born",
        "SrcModel.familyName",
        "SrcModel.givenName",
    }


def test_addressed_paths_carry_the_pointer_of_the_entry():
    _sources, targets = addressed_paths(structure_map())
    by_path = {entry.path: entry for entry in targets}
    assert by_path["Patient.name.family"].pointer == "/group/0/rule/1/rule/0/target/0"
    assert by_path["Patient.name"].structural is True
    assert by_path["Patient.name.family"].structural is False


def test_a_sibling_rules_variable_is_not_visible():
    # Variable scoping follows the engine's: a rule sees its ancestors' bindings
    # and its own, never a sibling's. A leaked binding would silently resolve a
    # path the engine will refuse.
    document = structure_map()
    group = document["group"][0]
    group["rule"].append(
        {
            "name": "uses-a-sibling-variable",
            "source": [{"context": "source"}],
            "target": [
                {
                    "context": "tgt-name",
                    "contextType": "variable",
                    "element": "prefix",
                    "transform": "copy",
                    "parameter": [{"valueString": "Dr"}],
                }
            ],
        }
    )
    _sources, targets = addressed_paths(document)
    assert not any(entry.path.endswith("prefix") for entry in targets)


# --- layers 1-4 ---------------------------------------------------------------


def test_a_clean_map_produces_no_findings(tree, source_field_names):
    report = offline(structure_map(), tree, source_field_names)
    assert report.findings == []
    assert report.satisfied_obligations == [
        "Patient.birthDate",
        "Patient.name.family",
        "Patient.name.given",
    ]
    assert "Patient.name" in report.covered_required_paths


def test_a_non_structuremap_is_rejected_at_layer_one(tree, source_field_names):
    report = offline({"resourceType": "Patient", "id": "x"}, tree, source_field_names)
    assert [finding.code for finding in report.findings] == ["not-a-structure-map"]


def test_a_document_that_does_not_parse_stops_the_layer_stack(tree, source_field_names):
    document = structure_map()
    document["group"][0]["rule"][0]["target"][0]["transform"] = 17
    report = offline(document, tree, source_field_names)
    # Exactly one finding: derived checks on a document that does not parse are
    # noise, not extra evidence.
    assert [finding.code for finding in report.findings] == ["resource-invalid"]
    assert report.findings[0].blocking


def test_an_unknown_variable_is_a_semantic_finding(tree, source_field_names):
    document = structure_map()
    document["group"][0]["rule"][0]["source"][0]["context"] = "notAVariable"
    report = offline(document, tree, source_field_names)
    codes = {finding.code for finding in report.findings}
    assert "semantic-rule-invalid" in codes
    semantic = next(f for f in report.findings if f.code == "semantic-rule-invalid")
    assert semantic.pointer == "/group/0"
    assert semantic.action_owner is ActionOwner.MAP_FIXABLE


def test_a_target_the_profile_does_not_have_is_reported(tree, source_field_names):
    document = structure_map()
    document["group"][0]["rule"][1]["rule"][0]["target"][0]["element"] = "nonsense"
    report = offline(document, tree, source_field_names)
    not_found = [f for f in report.findings if f.code == "target-path-not-found"]
    assert [f.path for f in not_found] == ["Patient.name.nonsense"]
    assert not_found[0].pointer == "/group/0/rule/1/rule/0/target/0"


def test_a_prohibited_target_is_reported_as_prohibited(tree, source_field_names):
    document = structure_map()
    document["group"][0]["rule"][0]["target"][0]["element"] = "photo"
    report = offline(document, tree, source_field_names)
    codes = {f.code for f in report.findings}
    assert "target-path-prohibited" in codes
    assert "target-path-not-found" not in codes


def test_a_mapping_obligation_below_max_zero_is_an_input_problem(
    tree, source_field_names
):
    report = validate_offline(
        structure_map(),
        target_tree=tree,
        mapping_table={"attachment": "Patient.photo.data"},
        source_fields=source_field_names,
        profile_url=profile()["url"],
    )
    finding = next(
        item for item in report.findings if item.code == "mapping-target-prohibited"
    )
    assert finding.action_owner is ActionOwner.MAPPING_INPUT_REQUIRED
    assert finding.evidence["prohibited_ancestor"] == "Patient.photo"


def test_questionnaire_link_id_selectors_satisfy_their_mapping_rows():
    document = {
        "group": [
            {
                "input": [
                    {"name": "Source", "type": "SourceModel", "mode": "source"},
                    {
                        "name": "Target",
                        "type": "QuestionnaireResponse",
                        "mode": "target",
                    },
                ],
                "rule": [
                    {
                        "source": [
                            {
                                "context": "Source",
                                "element": "smoking",
                                "variable": "src-smoking",
                            }
                        ],
                        "target": [
                            {
                                "context": "Target",
                                "element": "item",
                                "variable": "item-smoking",
                            }
                        ],
                        "rule": [
                            {
                                "target": [
                                    {
                                        "context": "item-smoking",
                                        "element": "linkId",
                                        "transform": "copy",
                                        "parameter": [{"valueString": "1.1"}],
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
    }

    obligations = authored_selector_obligations(document)
    assert ("QuestionnaireResponse.item[1.1]", "smoking") in obligations
    assert (
        "QuestionnaireResponse.item[1.1]",
        "SourceModel.smoking",
    ) in obligations


def test_questionnaire_selector_matches_a_fully_qualified_mapping_key():
    document = structure_map()
    document["group"][0]["input"][1]["type"] = "QuestionnaireResponse"
    item_rule = {
        "name": "question",
        "source": [
            {"context": "source", "element": "givenName", "variable": "value"}
        ],
        "target": [
            {"context": "target", "element": "item", "variable": "item"}
        ],
        "rule": [
            {
                "name": "link-id",
                "source": [{"context": "value"}],
                "target": [
                    {
                        "context": "item",
                        "element": "linkId",
                        "transform": "copy",
                        "parameter": [{"valueString": "1.1"}],
                    }
                ],
            }
        ],
    }
    document["group"][0]["rule"] = [item_rule]

    report = validate_offline(
        document,
        mapping_table={
            "SrcModel.givenName": "QuestionnaireResponse.item[1.1]"
        },
        profile_url="http://example.org/StructureDefinition/QuestionnaireResponse",
        profile_id="QuestionnaireResponse",
    )

    assert "mapping-obligation-dropped" not in {
        item.code for item in report.findings
    }
    assert report.satisfied_obligations == ["QuestionnaireResponse.item[1.1]"]


def test_an_unexpanded_profile_level_is_not_reported_as_missing(source_field_names):
    # The snapshot stops at Patient.name. Without introspection the tree cannot
    # tell `family` (correct) from `nonsense` (not), and reporting either would
    # fail every correct map. Silence is the only honest answer.
    shallow = build_target_tree(profile(), introspect=False)
    report = offline(structure_map(), shallow, source_field_names)
    assert [f.code for f in report.findings if f.code == "target-path-not-found"] == []


def test_a_source_field_the_model_does_not_declare_is_reported(tree, source_field_names):
    document = structure_map()
    document["group"][0]["rule"][0]["source"][0]["element"] = "notAField"
    report = offline(document, tree, source_field_names)
    finding = next(f for f in report.findings if f.code == "source-path-not-found")
    assert finding.action_owner is ActionOwner.MAPPING_INPUT_REQUIRED


def test_a_dropped_mapping_obligation_is_reported(tree, source_field_names):
    document = structure_map()
    del document["group"][0]["rule"][1]["rule"][1]
    report = offline(document, tree, source_field_names)
    dropped = [f for f in report.findings if f.code == "mapping-obligation-dropped"]
    assert [f.path for f in dropped] == ["Patient.name.given"]
    assert "Patient.name.given" not in report.satisfied_obligations


def test_an_unmapped_required_path_is_reported(tree, source_field_names):
    document = structure_map()
    document["group"][0]["rule"] = [document["group"][0]["rule"][0]]
    report = offline(document, tree, source_field_names)
    gap = next(f for f in report.findings if f.code == "required-path-unmapped")
    assert gap.path == "Patient.name"
    # The mapping table routes source fields into Patient.name, so a map edit
    # can emit it — that is what makes this one map-fixable rather than an input
    # problem.
    assert gap.action_owner is ActionOwner.MAP_FIXABLE


def test_a_required_path_with_no_source_and_no_provider_is_an_input_problem(
    tree, source_field_names
):
    document = structure_map()
    document["group"][0]["rule"] = [document["group"][0]["rule"][0]]
    report = validate_offline(
        document,
        target_tree=tree,
        mapping_table={"born": "Patient.birthDate"},  # nothing routes to name
        source_fields=source_field_names,
        profile_url=profile()["url"],
    )
    gap = next(f for f in report.findings if f.code == "required-path-unmapped")
    assert gap.action_owner is ActionOwner.MAPPING_INPUT_REQUIRED
    assert gap.blocking  # non-actionable, but it still stops a candidate


def test_the_report_records_the_digest_of_the_document_it_describes():
    # The same canonical digest WP5 binds patches to, so a report and an
    # envelope can be checked against each other instead of assumed to match.
    from agent.patch import canonical_sha256

    document = structure_map()
    report = validate_offline(document)
    assert report.map_sha256 == canonical_sha256(document)

    changed = structure_map()
    changed["group"][0]["rule"][0]["name"] = "renamed"
    assert validate_offline(changed).map_sha256 != report.map_sha256


def test_omitting_a_layers_input_produces_no_findings_from_it():
    # A skipped layer must be silent, never passing. Nothing downstream may read
    # "no findings" as "checked and fine".
    report = validate_offline(structure_map())
    assert report.findings == []
    assert report.covered_required_paths == []
    assert report.engine_requested is False


# --- the WP1 bridge -----------------------------------------------------------


def test_generator_diagnostics_keep_their_identity_and_owner():
    findings = findings_from_diagnostics(
        [
            {
                "code": "unmaterialized-nested-target",
                "message": "nested target was not materialized",
                "profile": "http://example.org/StructureDefinition/TestPatient",
                "severity": "warning",
            },
            {"code": "deferred-reference", "message": "left for the bundler"},
        ],
        map_url="http://example.org/StructureMap/sm-test",
    )
    fixable, advisory = findings
    assert fixable.finding_id.startswith("diag-")
    assert fixable.action_owner is ActionOwner.MAP_FIXABLE
    assert fixable.blocking
    assert advisory.action_owner is ActionOwner.ADVISORY
    assert not advisory.blocking


def test_the_generator_registry_stays_authoritative_for_ownership():
    # Re-deciding ownership here would give one code two owners depending on
    # which module a reader asked.
    from mapping.generation_result import DIAGNOSTIC_ACTIONABILITY

    for code, actionability in DIAGNOSTIC_ACTIONABILITY.items():
        finding = findings_from_diagnostics([{"code": code, "message": code}])[0]
        assert finding.action_owner.value == actionability.value


# --- acceptance ---------------------------------------------------------------


def clean_report(**kwargs):
    defaults = {
        "map_url": "http://example.org/StructureMap/sm-test",
        "engine_available": True,
        "engine_requested": True,
        "covered_required_paths": ["Patient.name"],
        "satisfied_obligations": ["Patient.birthDate", "Patient.name.family"],
        "executed_fixtures": ["filled", "empty-optional"],
        "required_fixtures": ["filled", "empty-optional"],
        "validated_fixtures": ["filled"],
        "evaluation_context_sha256": "shared-engine-context",
    }
    defaults.update(kwargs)
    return ValidationReport(**defaults)


def test_an_unchanged_candidate_is_accepted():
    decision = decide_acceptance(clean_report(), clean_report())
    assert decision.accepted
    assert decision.failures == []


def test_engine_evidence_requires_at_least_one_gate_relevant_fixture():
    decision = decide_acceptance(
        clean_report(
            executed_fixtures=["experimental"], required_fixtures=[]
        ),
        clean_report(
            executed_fixtures=["experimental"], required_fixtures=[]
        ),
    )
    assert not decision.accepted
    assert "engine-evidence-present" in {item.name for item in decision.failures}


def test_profile_validation_evidence_cannot_be_skipped_on_both_revisions():
    baseline = clean_report(
        profile_url="http://example.org/StructureDefinition/TestPatient",
        validation_expected_fixtures=["filled"],
        validated_fixtures=[],
    )
    candidate = baseline.model_copy(deep=True)
    decision = decide_acceptance(baseline, candidate)
    assert not decision.accepted
    assert "validate-evidence-present" in {item.name for item in decision.failures}


def test_engine_context_must_match_exactly():
    decision = decide_acceptance(
        clean_report(evaluation_context_sha256="baseline-context"),
        clean_report(evaluation_context_sha256="candidate-context"),
    )
    assert not decision.accepted
    assert "engine-evaluation-context-parity" in {
        item.name for item in decision.failures
    }


def test_candidate_fixture_superset_is_not_parity():
    decision = decide_acceptance(
        clean_report(executed_fixtures=["filled"]),
        clean_report(executed_fixtures=["filled", "extra"]),
    )
    assert not decision.accepted
    assert "engine-fixture-parity" in {item.name for item in decision.failures}


def test_an_unreachable_engine_is_never_a_pass():
    decision = decide_acceptance(
        clean_report(), clean_report(engine_available=False)
    )
    assert not decision.accepted
    assert "engine-available" in {item.name for item in decision.failures}


def test_an_explicitly_offline_run_can_still_produce_a_decision():
    decision = decide_acceptance(
        clean_report(engine_available=False, engine_requested=False),
        clean_report(engine_available=False, engine_requested=False),
        require_engine=False,
    )
    assert decision.accepted
    assert "engine-available" not in {item.name for item in decision.invariants}


def test_a_new_blocking_finding_rejects():
    regression = ValidationFinding.build(
        Producer.PATH_RESOLUTION,
        Stage.PATHS,
        "target-path-not-found",
        "gone wrong",
        path="Patient.name.nonsense",
    )
    decision = decide_acceptance(clean_report(), clean_report(findings=[regression]))
    assert not decision.accepted
    assert decision.new_blocking_ids == [regression.finding_id]


def test_a_pre_existing_blocking_finding_does_not_reject():
    # Otherwise no project with an unfixable gap could ever accept an
    # improvement, and the loop would be useless exactly where it is needed.
    existing = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "required-path-unmapped",
        "no source for this one",
        path="Patient.identifier",
    )
    decision = decide_acceptance(
        clean_report(findings=[existing]), clean_report(findings=[existing])
    )
    assert decision.accepted


def test_resolving_a_finding_is_reported_but_is_not_itself_a_gate():
    existing = ValidationFinding.build(
        Producer.PATH_RESOLUTION,
        Stage.PATHS,
        "target-path-not-found",
        "was broken",
        path="Patient.name.nonsense",
    )
    decision = decide_acceptance(clean_report(findings=[existing]), clean_report())
    assert decision.accepted
    assert decision.resolved_blocking_ids == [existing.finding_id]


def test_losing_required_coverage_rejects():
    decision = decide_acceptance(
        clean_report(), clean_report(covered_required_paths=[])
    )
    assert not decision.accepted
    failure = next(
        item for item in decision.failures if item.name == "no-required-coverage-loss"
    )
    assert failure.evidence["paths"] == ["Patient.name"]


def test_trading_one_required_path_for_another_still_rejects():
    # A count would call this level. Comparing the sets is the point.
    decision = decide_acceptance(
        clean_report(), clean_report(covered_required_paths=["Patient.identifier"])
    )
    assert not decision.accepted


def test_dropping_a_mapping_obligation_rejects():
    decision = decide_acceptance(
        clean_report(), clean_report(satisfied_obligations=["Patient.birthDate"])
    )
    assert not decision.accepted
    assert "no-obligation-loss" in {item.name for item in decision.failures}


def test_a_candidate_that_executed_nothing_is_not_accepted():
    # "No execution failures" is trivially true for a candidate that was never
    # run. Evidence has to be present, not merely unrefuted.
    decision = decide_acceptance(
        clean_report(), clean_report(executed_fixtures=[], validated_fixtures=[])
    )
    assert not decision.accepted
    assert "engine-evidence-present" in {item.name for item in decision.failures}


def test_a_candidate_tested_with_fewer_fixtures_is_not_accepted():
    decision = decide_acceptance(
        clean_report(), clean_report(executed_fixtures=["filled"])
    )
    assert not decision.accepted
    failure = next(
        item for item in decision.failures if item.name == "engine-fixture-parity"
    )
    assert failure.evidence["fixtures"] == ["empty-optional"]


def test_a_skipped_validate_layer_cannot_pass_as_a_clean_one():
    # Without this, "no new $validate error" is satisfied by never running
    # $validate at all.
    decision = decide_acceptance(clean_report(), clean_report(validated_fixtures=[]))
    assert not decision.accepted
    assert "validate-coverage-parity" in {item.name for item in decision.failures}


def test_gaining_conformance_evidence_is_not_a_parity_failure():
    # The direction agent mode exists for: a baseline whose $transform fails
    # produces no output to validate, so the repair that makes one *gains*
    # evidence. A symmetric comparison here rejected exactly that case, and
    # only a live engine could show it — no unit test moved.
    baseline = clean_report(validated_fixtures=[])
    candidate = clean_report(validated_fixtures=["filled"])
    decision = decide_acceptance(baseline, candidate)
    assert "validate-coverage-parity" not in {item.name for item in decision.failures}
    assert decision.accepted


def test_a_new_validate_error_rejects():
    regression = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:invariant",
        "an invariant now fails",
        path="Patient",
        fixture_id="fx",
    )
    decision = decide_acceptance(clean_report(), clean_report(findings=[regression]))
    assert not decision.accepted
    assert "no-validate-regression" in {item.name for item in decision.failures}


def test_a_transform_failure_rejects_even_with_a_clean_baseline():
    failure = ValidationFinding.build(
        Producer.ENGINE,
        Stage.TRANSFORM,
        "transform:processing",
        "the engine refused",
        fixture_id="fx",
    )
    decision = decide_acceptance(clean_report(), clean_report(findings=[failure]))
    assert not decision.accepted
    assert "engine-executes-required-fixtures" in {
        item.name for item in decision.failures
    }


def test_a_rejected_patch_cannot_be_accepted():
    from agent.models import PatchApplication, PatchRejection

    application = PatchApplication(
        applied=False,
        rejections=[PatchRejection(code="unguarded-mutation", message="no guard")],
    )
    decision = decide_acceptance(
        clean_report(), clean_report(), application=application
    )
    assert not decision.accepted
    assert "patch-policy-passed" in {item.name for item in decision.failures}


def test_a_candidate_cannot_pass_on_a_rationale():
    # There is deliberately no channel for the model's own opinion: the decision
    # reads reports and a policy verdict, and a confident rationale attached to a
    # broken candidate changes nothing.
    from agent.models import AgentPatch, PatchApplication, PatchProvenance, operation

    proposal = AgentPatch(
        schema_version=1,
        map_url="http://example.org/StructureMap/sm-test",
        map_id="sm-test",
        base_sha256="0" * 64,
        diagnostic_ids=[],
        patch=[operation("add", "/description", "now correct")],
        rationale="This patch definitely resolves the problem.",
        provenance=PatchProvenance(provider="openai", model="a-model"),
    )
    application = PatchApplication(
        applied=True,
        candidate={"resourceType": "StructureMap"},
        normalized_patch=proposal.as_rfc6902(),
    )
    broken = ValidationFinding.build(
        Producer.MAP_SEMANTICS,
        Stage.SEMANTICS,
        "semantic-rule-invalid",
        "still wrong",
        path="TransformPatient",
    )
    decision = decide_acceptance(
        clean_report(),
        clean_report(findings=[broken]),
        application=application,
    )
    assert not decision.accepted


def test_a_non_blocking_regression_is_reported_without_gating():
    advisory = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:informational",
        "a note",
        fixture_id="fx",
    )
    decision = decide_acceptance(clean_report(), clean_report(findings=[advisory]))
    assert decision.accepted
    assert decision.advisory_regressions == [advisory.finding_id]


# --- fixtures -----------------------------------------------------------------


def test_synthetic_fixtures_type_their_values():
    filled, empty = build_synthetic_fixtures(source_model(), structure_map())
    assert filled.kind is FixtureKind.SYNTHETIC_FILLED
    # A declared integer must be a JSON number, or the instance itself fails to
    # parse against the logical model and the engine blames the map.
    assert filled.instance["recordNumber"] == 1
    assert isinstance(filled.instance["recordNumber"], int)
    assert filled.instance["resourceType"] == "SrcModel"
    assert empty.kind is FixtureKind.SYNTHETIC_EMPTY_OPTIONAL


def test_the_empty_optional_variant_keeps_required_fields():
    _filled, empty = build_synthetic_fixtures(source_model(), structure_map())
    assert empty.instance["givenName"] == "1"  # min = 1
    assert empty.instance["familyName"] == ""  # optional string
    assert "recordNumber" not in empty.instance  # optional, not emptyable


def test_expansion_asks_the_terminology_server_before_matchbox(monkeypatch):
    """Matchbox v3 answers ``not-supported`` for ``ValueSet/$expand``.

    A project that configures ``terminology_server_uri`` and is never asked gets
    placeholder codes in its fixtures, and then every required binding fails
    validation against a map that is correct — which is exactly what happened on
    a real project, twice, at the cost of provider calls.
    """

    from agent.engine import EngineSession

    class Matchbox:
        def __init__(self):
            self.asked = False
            self.mc = self

        def send_request(self, *_args, **_kwargs):
            self.asked = True
            raise AssertionError("matchbox must not be asked when terminology answers")

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        @staticmethod
        def json():
            return {"expansion": {"contains": [{"code": "AMB"}, {"code": "IMP"}]}}

    calls = {}

    def fake_get(url, **kwargs):
        calls["url"] = url
        calls["params"] = kwargs.get("params")
        return Response()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    matchbox = Matchbox()
    session = EngineSession(matchbox, terminology_url="https://tx.example/r4/")

    assert session.expand("http://hl7.org/fhir/ValueSet/encounter-status") == [
        "AMB",
        "IMP",
    ]
    assert calls["url"] == "https://tx.example/r4/ValueSet/$expand"
    assert calls["params"] == {"url": "http://hl7.org/fhir/ValueSet/encounter-status"}
    assert matchbox.asked is False


def test_expansion_falls_back_to_matchbox_and_never_raises(monkeypatch):
    """Best-effort throughout: a failure means placeholder codes, not a finding."""

    from agent.engine import EngineSession

    class Matchbox:
        def __init__(self):
            self.mc = self

        def send_request(self, *_args, **_kwargs):
            raise RuntimeError("not supported")

    import requests

    def fake_get(*_args, **_kwargs):
        raise RuntimeError("terminology server unreachable")

    monkeypatch.setattr(requests, "get", fake_get)
    session = EngineSession(Matchbox(), terminology_url="https://tx.example/r4/")

    assert session.expand("http://hl7.org/fhir/ValueSet/encounter-status") is None


def test_a_translate_fed_field_is_omitted_rather_than_emptied():
    """An empty string in a translate source tests nothing about the map.

    Matchbox does not read ``""`` as "no code" — it dereferences it and raises
    ``Cannot invoke "String.equals(Object)" because "srccode" is null``, which
    arrives as an HTTP 500 and is recorded as a blocking, map-fixable defect
    against a map that is entirely correct. Observed on a real project: the
    empty-optional fixture set an optional `string` source field to `""`, the
    map translated it through a ConceptMap, and the transform 500'd.

    Omitted for the same reason a declared integer already is.
    """

    document = structure_map()
    document["group"][0]["rule"].append(
        {
            "name": "map-sex-translate",
            "source": [{"context": "source", "element": "sex", "variable": "src-sex"}],
            "target": [
                {
                    "context": "target",
                    "contextType": "variable",
                    "element": "gender",
                    "transform": "translate",
                    "parameter": [
                        {"valueId": "src-sex"},
                        {"valueString": "http://example.org/ConceptMap/cm-sex"},
                        {"valueString": "code"},
                    ],
                }
            ],
        }
    )

    _filled, empty = build_synthetic_fixtures(source_model(), document)

    assert "sex" not in empty.instance
    # The rule that does not translate is untouched: still emptied, not omitted.
    assert empty.instance["familyName"] == ""


def test_translate_source_fields_resolves_the_variable_to_its_source_element():
    from agent.fixtures import translate_source_fields

    document = structure_map()
    document["group"][0]["rule"].append(
        {
            "name": "map-sex-translate",
            "source": [{"context": "source", "element": "sex", "variable": "src-sex"}],
            "target": [
                {
                    "context": "target",
                    "contextType": "variable",
                    "element": "gender",
                    "transform": "translate",
                    "parameter": [{"valueId": "src-sex"}],
                }
            ],
        }
    )

    assert translate_source_fields([document]) == {"sex"}
    # No translate anywhere means nothing is withheld.
    assert translate_source_fields([structure_map()]) == set()


def test_a_cast_target_gets_a_format_valid_value():
    # The map casts `born` to a date. A generic "1" would fail the cast, and the
    # defect would be the fixture's rather than the map's.
    overrides = temporal_cast_overrides(structure_map())
    assert overrides == {"born": "2024-01-15"}
    filled, _empty = build_synthetic_fixtures(source_model(), structure_map())
    assert filled.instance["born"] == "2024-01-15"


def test_a_translate_source_gets_a_code_the_conceptmap_knows():
    document = structure_map()
    document["group"][0]["rule"].append(
        {
            "name": "map-gender",
            "source": [{"context": "source", "element": "sex", "variable": "src-sex"}],
            "target": [
                {
                    "context": "target",
                    "contextType": "variable",
                    "element": "gender",
                    "transform": "translate",
                    "parameter": [
                        {"valueId": "src-sex"},
                        {"valueString": "http://example.org/ConceptMap/sex"},
                        {"valueString": "code"},
                    ],
                }
            ],
        }
    )
    concept_map = {
        "resourceType": "ConceptMap",
        "url": "http://example.org/ConceptMap/sex",
        "group": [{"element": [{"code": "M", "target": [{"code": "male"}]}]}],
    }
    assert translate_overrides(
        document, {"http://example.org/ConceptMap/sex": "M"}
    ) == {"sex": "M"}
    filled, _empty = build_synthetic_fixtures(
        source_model(), document, concept_maps=[concept_map]
    )
    assert filled.instance["sex"] == "M"
    assert "sex" in filled.verified_fields


def test_fixture_ids_are_content_addressed():
    first = build_synthetic_fixtures(source_model(), structure_map())
    second = build_synthetic_fixtures(source_model(), structure_map())
    assert [f.fixture_id for f in first] == [f.fixture_id for f in second]
    assert fixtures_digest(first) == fixtures_digest(second)

    model = source_model()
    model["differential"]["element"].append(
        {"path": "SrcModel.extra", "min": 0, "max": "1", "type": [{"code": "string"}]}
    )
    changed = build_synthetic_fixtures(model, structure_map())
    assert fixtures_digest(changed) != fixtures_digest(first)


def test_shared_fixtures_satisfy_every_revision_under_comparison():
    # Deriving fixtures per revision gives the baseline and the candidate
    # different inputs. Reusing only the baseline's would execute a candidate's
    # new cast against a value that cannot satisfy it, so it fails on a fixture
    # artifact. The union is what makes one identical set work for both.
    baseline = structure_map()
    candidate = structure_map()
    candidate["group"][0]["rule"].append(
        {
            "name": "map-deceased",
            "source": [
                {"context": "source", "element": "recordNumber", "variable": "src-rec"}
            ],
            "target": [
                {
                    "context": "target",
                    "contextType": "variable",
                    "element": "birthDate",
                    "transform": "cast",
                    "parameter": [{"valueId": "src-rec"}, {"valueString": "dateTime"}],
                }
            ],
        }
    )
    only_baseline = build_synthetic_fixtures(source_model(), baseline)[0]
    assert only_baseline.instance["recordNumber"] == 1  # no cast known yet

    shared = build_shared_fixtures(source_model(), [baseline, candidate])
    assert shared[0].instance["recordNumber"] == "2024-01-15T10:30:00+01:00"
    # ...and the baseline's own requirement is still honoured.
    assert shared[0].instance["born"] == "2024-01-15"


def test_the_fixture_digest_covers_gate_metadata():
    # `required` never changes what the engine says, but it changes how the
    # answer is classified — so a verdict cached while a fixture was
    # experimental must not be replayed once it became gate-relevant.
    optional = SourceFixture(
        fixture_id="f1",
        kind=FixtureKind.EXAMPLE_DERIVED,
        label="example",
        instance={"givenName": "1"},
        required=False,
    )
    assert fixtures_digest([optional]) != fixtures_digest(
        [optional.model_copy(update={"required": True})]
    )
    assert fixtures_digest([optional]) != fixtures_digest(
        [optional.model_copy(update={"verified_fields": ["givenName"]})]
    )


def test_example_fixtures_are_never_gate_relevant():
    examples = [
        {
            "resourceType": "Patient",
            "id": "example-1",
            "name": [{"family": "Doe", "given": ["Jane"]}],
            "birthDate": "1980-02-03",
        }
    ]
    fixtures = build_example_fixtures(examples, mapping_table())
    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.kind is FixtureKind.EXAMPLE_DERIVED
    assert fixture.required is False
    assert fixture.gate_relevant is False
    assert fixture.provenance["status"] == "experimental"
    assert fixture.instance["familyName"] == "Doe"


def test_example_fixtures_are_capped():
    examples = [
        {"resourceType": "Patient", "id": str(index), "birthDate": f"200{index}-01-01"}
        for index in range(9)
    ]
    assert len(build_example_fixtures(examples, mapping_table(), limit=3)) == 3


# --- the engine session -------------------------------------------------------


class FakeMatchbox:
    """A Matchbox stand-in that answers exactly what a test asks it to."""

    def __init__(self, *, upsert=None, transform=None, validate=None, known=()):
        self._upsert = upsert or (lambda resource: (200, resource))
        self._transform = transform or (
            lambda instance, url: (200, {"resourceType": "Patient", "id": "out"})
        )
        # A clean outcome by default: returning nothing means the call failed,
        # which is a different fact and has its own finding.
        self._validate = validate or (
            lambda resource, profile_url: {"resourceType": "OperationOutcome", "issue": []}
        )
        self._known = set(known)
        self.transform_calls = []
        self.upserted = []

    def get_capability_statement(self):
        return {"resourceType": "CapabilityStatement"}

    def upsert_resource(self, resource):
        self.upserted.append(resource)
        return self._upsert(resource)

    def transform_data_detailed(self, instance, url):
        self.transform_calls.append((instance, url))
        return self._transform(instance, url)

    def validate_fhir_resources(self, resource, profile_url):
        return self._validate(resource, profile_url)

    def get_resource_by_url(self, resource_type, url):
        return {"resourceType": resource_type} if url in self._known else None


def outcome(*issues):
    return {"resourceType": "OperationOutcome", "issue": list(issues)}


def synthetic():
    return build_synthetic_fixtures(source_model(), structure_map())


def known_canonicals():
    return {
        "http://example.org/StructureDefinition/src-model",
        "http://example.org/StructureDefinition/TestPatient",
    }


def test_an_unreachable_engine_reports_an_environment_finding():
    class Down(FakeMatchbox):
        def get_capability_statement(self):
            return None

    session = EngineSession(Down())
    result = session.validate_map(structure_map(), synthetic())
    assert [f.code for f in result.findings] == ["engine-unreachable"]
    assert result.findings[0].action_owner is ActionOwner.ENVIRONMENT
    assert result.findings[0].blocking


def test_a_map_is_uploaded_under_a_scratch_canonical():
    # Validating a candidate must not overwrite the baseline on a shared server.
    matchbox = FakeMatchbox(known=known_canonicals())
    session = EngineSession(matchbox, validate_output=False)
    document = structure_map()
    session.validate_map(document, synthetic())
    uploaded = matchbox.upserted[0]
    assert uploaded["url"] != document["url"]
    assert uploaded["url"].startswith(document["url"] + "-agent-")
    assert uploaded["id"] != document["id"]


def test_transport_loss_during_upload_is_environmental_and_marks_engine_down():
    matchbox = FakeMatchbox(upsert=lambda resource: (None, None))
    report = merge_engine_result(
        validate_offline(structure_map()),
        EngineSession(matchbox).validate_map(structure_map(), synthetic()),
    )
    finding = next(item for item in report.findings if item.code == "engine-unreachable")
    assert finding.stage is Stage.UPLOAD
    assert finding.action_owner is ActionOwner.ENVIRONMENT
    assert report.engine_requested is True
    assert report.engine_available is False


def test_transport_loss_during_transform_is_environmental_and_marks_engine_down():
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (None, None), known=known_canonicals()
    )
    report = merge_engine_result(
        validate_offline(structure_map()),
        EngineSession(matchbox).validate_map(structure_map(), synthetic()),
    )
    finding = next(item for item in report.findings if item.code == "engine-unreachable")
    assert finding.stage is Stage.TRANSFORM
    assert finding.action_owner is ActionOwner.ENVIRONMENT
    assert report.engine_available is False


def test_the_scratch_canonical_is_stable_for_identical_content():
    first = scratch_identity(structure_map())
    assert first == scratch_identity(copy.deepcopy(structure_map()))
    changed = structure_map()
    changed["group"][0]["rule"][0]["name"] = "renamed"
    assert scratch_identity(changed) != first


def test_a_refused_upload_is_classified_from_the_issue_code():
    matchbox = FakeMatchbox(
        upsert=lambda resource: (
            400,
            outcome({"severity": "error", "code": "invalid", "diagnostics": "bad map"}),
        ),
        known=known_canonicals(),
    )
    session = EngineSession(matchbox)
    result = session.validate_map(structure_map(), synthetic())
    finding = result.findings[0]
    assert finding.code == "upload:invalid"
    assert finding.issue_code == "invalid"
    assert finding.action_owner is ActionOwner.MAP_FIXABLE
    assert matchbox.transform_calls == []  # nothing runs against a refused map


def test_a_transform_failure_carries_the_fixture_and_a_rule_hint():
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (
            500,
            outcome(
                {
                    "severity": "error",
                    "code": "processing",
                    "diagnostics": (
                        'Exception executing transform on Rule "Map|Group|map-birthDate": '
                        'Invalid date/time format: "notadate"'
                    ),
                }
            ),
        ),
        known=known_canonicals(),
    )
    session = EngineSession(matchbox)
    fixtures = synthetic()
    result = session.validate_map(structure_map(), fixtures)
    assert len(result.findings) == 2  # one per fixture
    finding = result.findings[0]
    assert finding.code == "transform:processing"
    assert finding.fixture_id == fixtures[0].fixture_id
    # The hint is a locator for a prompt, never an input to classification.
    assert finding.evidence["rule_hint"] == "map-birthDate"


def test_an_unrecognized_issue_code_stays_blocking_and_unclassified():
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (
            500,
            outcome({"severity": "error", "code": "throttled", "diagnostics": "slow down"}),
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(structure_map(), synthetic())
    finding = result.findings[0]
    assert finding.code == "transform:throttled"
    assert finding.action_owner is ActionOwner.UNCLASSIFIED
    assert finding.blocking


def test_informational_issues_are_not_reported():
    # Matchbox prefixes every outcome with a multi-kilobyte package listing. It
    # describes the server, not the map.
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {"severity": "information", "code": "informational", "diagnostics": "packages..."},
            {"severity": "error", "code": "invariant", "diagnostics": "pat-1 failed"},
        ),
        known=known_canonicals(),
    )
    session = EngineSession(matchbox)
    result = session.validate_map(
        structure_map(), synthetic(), profile_url=profile()["url"]
    )
    codes = [f.code for f in result.findings]
    assert "validate:informational" not in codes
    assert "validate:invariant" in codes


def test_a_validate_warning_does_not_gate_but_says_so():
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {"severity": "warning", "code": "value", "diagnostics": "unusual but legal"}
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(), synthetic(), profile_url=profile()["url"]
    )
    warning = next(f for f in result.findings if f.code == "validate:value")
    assert not warning.blocking
    # Every weakened gate is attributable, including this one.
    assert warning.demoted_by == "engine-severity:warning"


def test_an_unknown_warning_stays_blocking():
    # Arriving at warning severity must not be a way past the table. An
    # unreviewed code is unreviewed whatever the engine calls it.
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {"severity": "warning", "code": "brand-new", "diagnostics": "unreviewed"}
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(), synthetic(), profile_url=profile()["url"]
    )
    finding = next(f for f in result.findings if f.code == "validate:brand-new")
    assert finding.action_owner is ActionOwner.UNCLASSIFIED
    assert finding.blocking
    assert finding.demoted_by is None


def test_a_binding_failure_on_a_placeholder_value_judges_the_fixture():
    # The generator had no real code to inject, so the rejected value is the
    # fixture's. Gating on it would ask an agent to hardcode a value into a map
    # that is already correct.
    fixture = SourceFixture(
        fixture_id="filled",
        kind=FixtureKind.SYNTHETIC_FILLED,
        label="filled",
        instance={"sex": "1", "givenName": "1"},
        required=True,
        verified_fields=["givenName"],
    )
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {
                "severity": "error",
                "code": "code-invalid",
                "diagnostics": "value provided ('1') was not found in the value set",
                "expression": ["Patient.gender"],
            }
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(),
        [fixture],
        profile_url=profile()["url"],
        mapping_table={"sex": "Patient.gender", "givenName": "Patient.name.given"},
    )
    finding = next(f for f in result.findings if f.code == "validate:code-invalid")
    assert not finding.blocking
    assert finding.demoted_by == "unverified-synthetic-value:Patient.gender"


def test_a_not_found_binding_failure_on_a_placeholder_judges_the_fixture():
    """Matchbox reports an unknown system for synthetic code `1` as not-found."""
    fixture = SourceFixture(
        fixture_id="filled",
        kind=FixtureKind.SYNTHETIC_FILLED,
        label="filled",
        instance={"sex": "1", "givenName": "1"},
        required=True,
        verified_fields=["givenName"],
    )
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {
                "severity": "error",
                "code": "not-found",
                "diagnostics": (
                    "The System URI could not be determined for the code '1' "
                    "in the ValueSet 'http://hl7.org/fhir/ValueSet/administrative-gender'"
                ),
                "expression": ["Patient.gender"],
            }
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(),
        [fixture],
        profile_url=profile()["url"],
        mapping_table={"sex": "Patient.gender", "givenName": "Patient.name.given"},
    )

    finding = next(f for f in result.findings if f.code == "validate:not-found")
    assert finding.action_owner is ActionOwner.ENVIRONMENT
    assert not finding.blocking
    assert finding.demoted_by == "unverified-synthetic-value:Patient.gender"


def test_a_not_found_failure_without_placeholder_provenance_still_gates():
    fixture = SourceFixture(
        fixture_id="example",
        kind=FixtureKind.EXAMPLE_DERIVED,
        label="example",
        instance={"sex": "M", "givenName": "Ada"},
        required=True,
        verified_fields=["sex", "givenName"],
    )
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {
                "severity": "error",
                "code": "not-found",
                "diagnostics": "ValueSet could not be found",
                "expression": ["Patient.gender"],
            }
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(),
        [fixture],
        profile_url=profile()["url"],
        mapping_table={"sex": "Patient.gender", "givenName": "Patient.name.given"},
    )

    finding = next(f for f in result.findings if f.code == "validate:not-found")
    assert finding.action_owner is ActionOwner.ENVIRONMENT
    assert finding.blocking
    assert finding.demoted_by is None


def test_a_container_binding_failure_on_placeholder_descendants_is_demoted():
    fixture = SourceFixture(
        fixture_id="filled",
        kind=FixtureKind.SYNTHETIC_FILLED,
        label="filled",
        instance={"sex": "1", "familyName": "1"},
        required=True,
        verified_fields=[],
    )
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {
                "severity": "error",
                "code": "code-invalid",
                "diagnostics": "None of the codings are in the required value set",
                "expression": ["Patient.maritalStatus"],
            }
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(),
        [fixture],
        profile_url=profile()["url"],
        mapping_table={
            "sex": "Patient.maritalStatus.coding.code",
            "familyName": "Patient.maritalStatus.coding.system",
        },
    )

    finding = next(f for f in result.findings if f.code == "validate:code-invalid")
    assert not finding.blocking
    assert finding.demoted_by == "unverified-synthetic-value:Patient.maritalStatus"


def test_a_binding_failure_on_a_verified_code_still_gates():
    fixture = SourceFixture(
        fixture_id="filled",
        kind=FixtureKind.SYNTHETIC_FILLED,
        label="filled",
        instance={"sex": "M", "givenName": "1"},
        required=True,
        verified_fields=["sex"],
    )
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {
                "severity": "error",
                "code": "code-invalid",
                "diagnostics": "'M' is not in the value set",
                "expression": ["Patient.gender"],
            }
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(),
        [fixture],
        profile_url=profile()["url"],
        mapping_table={"sex": "Patient.gender"},
    )
    finding = next(f for f in result.findings if f.code == "validate:code-invalid")
    assert finding.blocking
    assert finding.demoted_by is None


def test_a_structural_failure_is_never_blamed_on_the_fixture_value():
    # A cardinality or invariant failure is about shape, not about which code a
    # placeholder carried, so the placeholder demotion must not reach it.
    fixture = SourceFixture(
        fixture_id="filled",
        kind=FixtureKind.SYNTHETIC_FILLED,
        label="filled",
        instance={"sex": "1"},
        required=True,
        verified_fields=[],
    )
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: outcome(
            {
                "severity": "error",
                "code": "structure",
                "diagnostics": "Patient.gender: minimum required = 1, but only found 0",
                "expression": ["Patient.gender"],
            }
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(),
        [fixture],
        profile_url=profile()["url"],
        mapping_table={"sex": "Patient.gender"},
    )
    finding = next(f for f in result.findings if f.code == "validate:structure")
    assert finding.blocking
    assert finding.demoted_by is None


def _map_with_deferred_extension():
    document = structure_map()
    document["group"][0]["rule"].append(
        {
            "name": "TODO-resolve-reference-Patient-extension-valueReference",
            "source": [{"context": "source"}],
            "documentation": (
                'FHIRBRIDGE_REFERENCE:{"sourceType":"Patient",'
                '"path":"extension.valueReference","selectors":['
                '{"discriminator":"url","value":"http://example.org/ext/org"}]}'
            ),
        }
    )
    return document


def test_a_deferred_extension_validation_error_is_nonblocking():
    document = _map_with_deferred_extension()
    output = {
        "resourceType": "Patient",
        "id": "out",
        "extension": [{"url": "http://example.org/ext/org"}],
    }
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (200, output),
        validate=lambda resource, profile_url: outcome(
            {
                "severity": "error",
                "code": "structure",
                "diagnostics": "Extension is incomplete before bundle assembly",
                "expression": ["Patient.extension[0].valueReference"],
            }
        ),
        known=known_canonicals(),
    )

    result = EngineSession(matchbox).validate_map(
        document, synthetic(), profile_url=profile()["url"]
    )
    finding = next(item for item in result.findings if item.code == "validate:structure")
    assert finding.action_owner is ActionOwner.ADVISORY
    assert not finding.blocking
    assert finding.demoted_by == "deferred-reference:Patient.extension.valueReference"


def test_an_unrelated_extension_at_the_same_path_still_blocks():
    output = {
        "resourceType": "Patient",
        "id": "out",
        "extension": [{"url": "http://example.org/ext/not-deferred"}],
    }
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (200, output),
        validate=lambda resource, profile_url: outcome(
            {
                "severity": "error",
                "code": "structure",
                "diagnostics": "Unrelated extension is incomplete",
                "expression": ["Patient.extension[0].valueReference"],
            }
        ),
        known=known_canonicals(),
    )

    result = EngineSession(matchbox).validate_map(
        _map_with_deferred_extension(), synthetic(), profile_url=profile()["url"]
    )
    finding = next(item for item in result.findings if item.code == "validate:structure")
    assert finding.action_owner is ActionOwner.MAP_FIXABLE
    assert finding.blocking
    assert finding.demoted_by is None


def test_a_root_only_fatal_retains_evidence_but_does_not_authorize_a_patch():
    raw_issue = {
        "severity": "fatal",
        "code": "structure",
        "diagnostics": "Unknown or unrecognized resource name Bogus",
        "expression": ["Bogus"],
    }
    output = {"resourceType": "Bogus", "id": "out", "value": "synthetic"}
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (200, output),
        validate=lambda resource, profile_url: outcome(raw_issue),
        known=known_canonicals(),
    )

    result = EngineSession(matchbox).validate_map(
        structure_map(), synthetic(), profile_url=profile()["url"]
    )
    finding = next(item for item in result.findings if item.code == "validate:structure")
    assert finding.action_owner is ActionOwner.UNCLASSIFIED
    assert finding.blocking
    assert finding.evidence["raw_issue"] == raw_issue
    assert finding.evidence["synthetic_output"] == output
    assert finding.evidence["output_sha256"]


def test_missing_dependencies_make_execution_failures_environmental(tree):
    # A map whose ConceptMap is absent will fail for a reason that is not its
    # own, and no agent should be authorized to "repair" the rule.
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (
            500,
            outcome({"severity": "error", "code": "processing", "diagnostics": "boom"}),
        ),
        known=set(),
    )
    session = EngineSession(matchbox, validate_output=False)
    result = session.validate_map(structure_map(), synthetic())
    codes = {f.code for f in result.findings}
    assert "dependency-missing" in codes
    transform_findings = [f for f in result.findings if f.code == "transform:processing"]
    assert transform_findings
    assert all(f.action_owner is ActionOwner.ENVIRONMENT for f in transform_findings)


def test_a_refused_source_model_stands_the_session_down():
    matchbox = FakeMatchbox(
        upsert=lambda resource: (
            422,
            outcome({"severity": "error", "code": "invalid", "diagnostics": "nope"}),
        )
    )
    session = EngineSession(matchbox)
    findings = session.bootstrap(source_model=source_model())
    assert [f.code for f in findings] == ["dependency-missing"]
    assert session.available() is False
    # Every transform would now fail for the same reason; running them would
    # only manufacture map defects that do not exist.
    result = session.validate_map(structure_map(), synthetic())
    assert [f.code for f in result.findings] == ["engine-unreachable"]


def test_a_verdict_is_reused_only_for_identical_inputs():
    matchbox = FakeMatchbox(known=known_canonicals())
    session = EngineSession(matchbox, validate_output=False)
    fixtures = synthetic()
    session.validate_map(structure_map(), fixtures)
    again = session.validate_map(structure_map(), fixtures)
    assert again.cached is True
    assert len(matchbox.transform_calls) == 2  # one per fixture, not four


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda doc, fx, kw: (doc, fx, {**kw, "profile_url": "http://other"}), id="profile"),
        pytest.param(
            lambda doc, fx, kw: ({**doc, "id": "sm-other"}, fx, kw), id="map"
        ),
    ],
)
def test_a_changed_input_invalidates_the_verdict(mutate):
    matchbox = FakeMatchbox(known=known_canonicals())
    session = EngineSession(matchbox, validate_output=False)
    document, fixtures, kwargs = structure_map(), synthetic(), {}
    session.validate_map(document, fixtures, **kwargs)
    document, fixtures, kwargs = mutate(document, fixtures, kwargs)
    assert session.validate_map(document, fixtures, **kwargs).cached is False


def test_a_changed_mapping_table_invalidates_the_verdict():
    # The table never reaches Matchbox, but it decides who owns a required-output
    # gap and whether a coded value counts as verified. A cached verdict carries
    # that classification with it.
    matchbox = FakeMatchbox(known=known_canonicals())
    session = EngineSession(matchbox, validate_output=False)
    fixtures = synthetic()
    session.validate_map(structure_map(), fixtures, mapping_table=mapping_table())
    again = session.validate_map(
        structure_map(), fixtures, mapping_table={"born": "Patient.birthDate"}
    )
    assert again.cached is False


def test_a_changed_required_manifest_invalidates_the_verdict():
    class Manifest:
        def __init__(self, path):
            self.path = path

        def required_manifest(self):
            return [{"id": self.path, "path": self.path, "min": 1, "active": True}]

    matchbox = FakeMatchbox(known=known_canonicals())
    session = EngineSession(matchbox, validate_output=False)
    fixtures = synthetic()
    session.validate_map(structure_map(), fixtures, target_tree=Manifest("Patient.name"))
    again = session.validate_map(
        structure_map(), fixtures, target_tree=Manifest("Patient.identifier")
    )
    assert again.cached is False


def test_making_a_fixture_gate_relevant_invalidates_the_verdict():
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (
            500,
            outcome({"severity": "error", "code": "processing", "diagnostics": "boom"}),
        ),
        known=known_canonicals(),
    )
    session = EngineSession(matchbox, validate_output=False)
    experimental = SourceFixture(
        fixture_id="example-1",
        kind=FixtureKind.EXAMPLE_DERIVED,
        label="example:Patient/1",
        instance={"resourceType": "SrcModel", "givenName": "Jane"},
        required=False,
    )
    first = session.validate_map(structure_map(), [experimental])
    assert not first.findings[0].blocking

    promoted = session.validate_map(
        structure_map(), [experimental.model_copy(update={"required": True})]
    )
    assert promoted.cached is False
    assert promoted.findings[0].blocking


def test_a_scratch_id_stays_within_the_fhir_limit():
    # `Resource.id` is capped at 64 characters, and the suffix costs 19 of them.
    document = structure_map()
    document["id"] = "a" * 64
    scratch_id, scratch_url = scratch_identity(document)
    assert len(scratch_id) == 64
    assert scratch_id.endswith(scratch_url.rsplit("-agent-", 1)[-1])


def test_a_truncated_scratch_id_still_separates_revisions():
    first = structure_map()
    first["id"] = "b" * 70
    second = copy.deepcopy(first)
    second["group"][0]["rule"][0]["name"] = "renamed"
    assert scratch_identity(first)[0] != scratch_identity(second)[0]


def test_a_changed_fixture_invalidates_the_verdict():
    matchbox = FakeMatchbox(known=known_canonicals())
    session = EngineSession(matchbox, validate_output=False)
    fixtures = synthetic()
    session.validate_map(structure_map(), fixtures)
    changed = [
        fixtures[0].model_copy(update={"instance": {**fixtures[0].instance, "born": "2020-01-01"}}),
        fixtures[1],
    ]
    assert session.validate_map(structure_map(), changed).cached is False


def test_a_missing_required_element_is_located_without_parsing_prose(tree):
    # Matchbox reports the failing path only in the message text. Deriving it
    # from the output and the requirement manifest keeps a real path on the
    # finding without anybody reading an error string.
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (
            200,
            {"resourceType": "Patient", "id": "out", "birthDate": "2024-01-15"},
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(),
        synthetic(),
        target_tree=tree,
        profile_url=profile()["url"],
        mapping_table=mapping_table(),
    )
    missing = next(f for f in result.findings if f.code == "output-required-missing")
    assert missing.path == "Patient.name"
    assert missing.action_owner is ActionOwner.MAP_FIXABLE
    assert missing.blocking


def test_a_validate_call_that_answers_nothing_is_not_a_pass():
    # $validate returns 200 even for an invalid resource, so an empty answer
    # means the call failed. Reading that as "no problems found" would turn a
    # broken server into a clean bill of health.
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: None, known=known_canonicals()
    )
    result = EngineSession(matchbox).validate_map(
        structure_map(), synthetic(), profile_url=profile()["url"]
    )
    finding = next(f for f in result.findings if f.code == "validate-unavailable")
    assert finding.action_owner is ActionOwner.ENVIRONMENT
    assert finding.blocking


def test_merging_an_engine_result_records_availability():
    from agent.validation import merge_engine_result

    report = validate_offline(structure_map())
    assert report.engine_requested is False

    matchbox = FakeMatchbox(known=known_canonicals())
    merge_engine_result(
        report, EngineSession(matchbox, validate_output=False).validate_map(
            structure_map(), synthetic()
        )
    )
    assert report.engine_requested is True
    assert report.engine_available is True
    assert len(report.executed_fixtures) == 2


def test_merging_an_unreachable_engine_marks_it_unavailable():
    from agent.validation import merge_engine_result

    class Down(FakeMatchbox):
        def get_capability_statement(self):
            return None

    report = merge_engine_result(
        validate_offline(structure_map()),
        EngineSession(Down()).validate_map(structure_map(), synthetic()),
    )
    assert report.engine_requested is True
    assert report.engine_available is False
    # ...and that is a rejection, not a quiet pass.
    assert not decide_acceptance(clean_report(), report).accepted


def test_the_empty_optional_output_is_not_conformance_checked(tree):
    # Its optional-fed elements are absent by design; validating it would report
    # the fixture's own design as a map defect.
    seen = []
    matchbox = FakeMatchbox(
        validate=lambda resource, profile_url: seen.append(resource) or None,
        known=known_canonicals(),
    )
    EngineSession(matchbox).validate_map(
        structure_map(), synthetic(), profile_url=profile()["url"]
    )
    assert len(seen) == 1


def test_an_experimental_fixtures_findings_are_demoted_with_a_reason():
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (
            500,
            outcome({"severity": "error", "code": "processing", "diagnostics": "boom"}),
        ),
        known=known_canonicals(),
    )
    experimental = SourceFixture(
        fixture_id="example-1",
        kind=FixtureKind.EXAMPLE_DERIVED,
        label="example:Patient/1",
        instance={"resourceType": "SrcModel", "givenName": "Jane"},
        required=False,
    )
    result = EngineSession(matchbox).validate_map(structure_map(), [experimental])
    finding = next(f for f in result.findings if f.code == "transform:processing")
    assert not finding.blocking
    # Demotion is recorded, never silent: a reader can always see why a failure
    # did not gate.
    assert finding.demoted_by == "experimental-fixture:example-derived"


def test_a_required_fixtures_findings_are_not_demoted():
    matchbox = FakeMatchbox(
        transform=lambda instance, url: (
            500,
            outcome({"severity": "error", "code": "processing", "diagnostics": "boom"}),
        ),
        known=known_canonicals(),
    )
    result = EngineSession(matchbox).validate_map(structure_map(), synthetic())
    finding = result.findings[0]
    assert finding.blocking
    assert finding.demoted_by is None


def test_declared_dependencies_are_read_structurally():
    document = structure_map()
    document["group"][0]["rule"].append(
        {
            "name": "map-gender",
            "source": [{"context": "source", "element": "sex", "variable": "src-sex"}],
            "target": [
                {
                    "context": "target",
                    "contextType": "variable",
                    "element": "gender",
                    "transform": "translate",
                    "parameter": [
                        {"valueId": "src-sex"},
                        {"valueString": "http://example.org/ConceptMap/sex"},
                    ],
                }
            ],
        }
    )
    session = EngineSession(FakeMatchbox())
    assert session.declared_dependencies(document) == [
        "http://example.org/ConceptMap/sex",
        "http://example.org/StructureDefinition/TestPatient",
        "http://example.org/StructureDefinition/src-model",
    ]


def test_the_engine_result_is_serializable():
    result = EngineResult(
        findings=[
            ValidationFinding.build(
                Producer.ENGINE, Stage.TRANSFORM, "transform:processing", "boom"
            )
        ]
    )
    payload = result.model_dump(mode="json")
    assert payload["findings"][0]["action_owner"] == "map-fixable"


# --- acceptance against a baseline the map cannot repair ----------------------
#
# A map whose baseline already fails to transform used to be unable to accept
# any repair at all: the two engine-conformance invariants were written as
# absolute demands, so a pre-existing failure vetoed candidates that resolved
# findings and introduced none. Seven such candidates were discarded in one
# batch, every one of them correct.
#
# The exemption is deliberately narrow. It is the *environment's* failures that
# are forgiven, because those are the ones no map edit can reach.


def execution_failure(owner, fixture_id="filled"):
    return ValidationFinding.build(
        Producer.ENGINE,
        Stage.TRANSFORM,
        "transform:processing",
        "Unable to find definition 'http://example.org/StructureDefinition/Cc|1.0' "
        "for type 'CodeableConcept'",
        owner=owner,
        fixture_id=fixture_id,
    )


def test_an_environment_transform_failure_does_not_veto_a_repair():
    failure = execution_failure(ActionOwner.ENVIRONMENT)
    baseline = clean_report(findings=[failure], validated_fixtures=[])
    candidate = clean_report(findings=[failure], validated_fixtures=[])
    decision = decide_acceptance(baseline, candidate)
    assert decision.accepted, [item.name for item in decision.failures]


def test_a_map_owned_transform_failure_still_vetoes():
    failure = execution_failure(ActionOwner.MAP_FIXABLE)
    baseline = clean_report(findings=[failure], validated_fixtures=[])
    candidate = clean_report(findings=[failure], validated_fixtures=[])
    decision = decide_acceptance(baseline, candidate)
    assert not decision.accepted
    assert "engine-executes-required-fixtures" in {
        item.name for item in decision.failures
    }


def test_a_transform_failure_the_candidate_introduced_still_vetoes():
    candidate_only = execution_failure(ActionOwner.ENVIRONMENT)
    decision = decide_acceptance(
        clean_report(), clean_report(findings=[candidate_only])
    )
    assert not decision.accepted
    assert "engine-executes-required-fixtures" in {
        item.name for item in decision.failures
    }


def test_missing_validate_evidence_is_excused_only_where_the_engine_explains_it():
    """The fixture must be both unvalidated *and* named by an environment failure."""
    unexplained = clean_report(
        validation_expected_fixtures=["filled"],
        validated_fixtures=[],
        profile_url="http://example.org/StructureDefinition/TestPatient",
    )
    assert not decide_acceptance(unexplained, unexplained.model_copy(deep=True)).accepted

    failure = execution_failure(ActionOwner.ENVIRONMENT, fixture_id="filled")
    explained = clean_report(
        findings=[failure],
        validation_expected_fixtures=["filled"],
        validated_fixtures=[],
        profile_url="http://example.org/StructureDefinition/TestPatient",
    )
    assert decide_acceptance(explained, explained.model_copy(deep=True)).accepted


# --- a candidate may not invent what the source never carried --------------------
#
# `$validate` cannot catch this. A required element with a required binding is
# satisfied by *any* member of the value set, so `Observation.status = 'final'`
# on a source that never carried a status passes conformance while fabricating
# clinical metadata. Where the engine is excused there is no conformance
# evidence at all, which is when a literal `'dateTime'` copied into
# `Observation.effective` gets through.
#
# The check is comparative because generated maps are full of *correct*
# constants — a profile that fixes or slices an element is telling the map
# exactly what to write. Only what the candidate adds is the model's doing.


def map_with(targets):
    return {
        "resourceType": "StructureMap",
        "group": [
            {
                "name": "g",
                "input": [
                    {"name": "source", "type": "SrcModel", "mode": "source"},
                    {"name": "target", "type": "Observation", "mode": "target"},
                ],
                "rule": [
                    {"name": f"r{index}", "source": [{"context": "source"}], "target": [t]}
                    for index, t in enumerate(targets)
                ],
            }
        ],
    }


def literal(element, value):
    return {
        "context": "target",
        "contextType": "variable",
        "element": element,
        "transform": "copy",
        "parameter": [{"valueString": value}],
    }


def from_source(element, variable="src-x"):
    return {
        "context": "target",
        "contextType": "variable",
        "element": element,
        "transform": "copy",
        "parameter": [{"valueId": variable}],
    }


def test_a_constant_the_candidate_adds_fails_acceptance():
    decision = decide_acceptance(
        clean_report(),
        clean_report(),
        baseline_document=map_with([from_source("value")]),
        candidate_document=map_with([from_source("value"), literal("status", "final")]),
    )
    assert not decision.accepted
    assert "no-invented-target-value" in {item.name for item in decision.failures}


def test_a_constant_the_generator_already_wrote_is_not_the_models_doing():
    """781 correct constants across 210 generated maps; judging them fails."""
    both = map_with([literal("status", "final"), literal("code.coding.system", "http://loinc.org")])
    decision = decide_acceptance(
        clean_report(), clean_report(),
        baseline_document=both, candidate_document=copy.deepcopy(both),
    )
    assert decision.accepted, [item.name for item in decision.failures]


def test_a_rule_reading_a_source_field_is_not_a_constant():
    decision = decide_acceptance(
        clean_report(), clean_report(),
        baseline_document=map_with([]),
        candidate_document=map_with([from_source("effective", "src-effective")]),
    )
    assert decision.accepted, [item.name for item in decision.failures]


def test_the_invariant_is_skipped_when_the_documents_are_not_supplied():
    decision = decide_acceptance(clean_report(), clean_report())
    assert "no-invented-target-value" not in {i.name for i in decision.invariants}


# --- accepted on excused evidence is not accepted on evidence -------------------


def test_an_accept_that_rested_on_an_excused_environment_is_provisional():
    failure = execution_failure(ActionOwner.ENVIRONMENT)
    report = clean_report(findings=[failure], validated_fixtures=[])
    decision = decide_acceptance(report, report.model_copy(deep=True))
    assert decision.accepted
    assert decision.provisional
    assert "engine-executes-required-fixtures" in decision.excused_evidence
    assert decision.as_report_dict()["provisional"] is True


def test_an_ordinary_accept_is_not_provisional():
    decision = decide_acceptance(clean_report(), clean_report())
    assert decision.accepted
    assert not decision.provisional
    assert decision.excused_evidence == []


def test_a_constant_the_profile_fixes_is_the_correct_repair_not_an_invention():
    """Refusing this would block the one edit available for a fixed element."""

    class Node:
        fixed_value = "final"

    class Tree:
        def node(self, path):
            return Node() if path == "Observation.status" else None

    decision = decide_acceptance(
        clean_report(), clean_report(),
        baseline_document=map_with([]),
        candidate_document=map_with([literal("status", "final")]),
        target_tree=Tree(),
    )
    assert decision.accepted, [item.name for item in decision.failures]


def test_without_a_tree_the_invariant_stays_strict():
    decision = decide_acceptance(
        clean_report(), clean_report(),
        baseline_document=map_with([]),
        candidate_document=map_with([literal("status", "final")]),
    )
    assert not decision.accepted
