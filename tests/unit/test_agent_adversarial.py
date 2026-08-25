"""Adversarial cases for agent mode (WP9).

The other agent suites check that each piece behaves. This one checks what
happens when something *lies* — a model that invents an element, claims a
finding it was never shown, writes against a revision that no longer exists, or
edits its way out of a check rather than through it — and what happens when the
map itself uses a FHIR construct the deterministic layers cannot fully judge.

Two kinds of test live here and they are labelled as such:

* **attacks**, which must be rejected;
* **limits**, which must *not* be reported as defects, and whose real gate is
  named explicitly. A validator that guesses at a construct it cannot resolve
  hands agent mode fabricated work — before the reconciliation these tests pin,
  91% of this repository's 462 committed maps produced at least one such
  finding, all of them on correct generator output.
"""

import copy
from pathlib import Path

import pytest

from agent.graph import AgentRuntimeContext
from agent.graph.repair_graph import build_repair_graph, run_repair_graph
from agent.loop import LoopLimits, LoopOutcome, Proposal
from agent.models import AgentPatch, operation
from agent.patch import canonical_sha256
from agent.validation import (
    ActionOwner,
    Producer,
    Stage,
    ValidationFinding,
    ValidationReport,
    build_target_tree,
    decide_acceptance,
    validate_offline,
)

pytestmark = pytest.mark.unit


# --- corpus -------------------------------------------------------------------


def element(path, minimum=0, maximum="1", type_code="string", **extra):
    return {
        "id": extra.pop("eid", path),
        "path": path,
        "min": minimum,
        "max": maximum,
        "type": [
            {"code": code}
            for code in ([type_code] if isinstance(type_code, str) else type_code)
        ],
        **extra,
    }


def patient_profile():
    """A Patient profile with an optional backbone that has a required child."""

    return {
        "resourceType": "StructureDefinition",
        "id": "AdvPatient",
        "url": "http://example.org/StructureDefinition/AdvPatient",
        "name": "AdvPatient",
        "status": "draft",
        "kind": "resource",
        "abstract": False,
        "type": "Patient",
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Patient",
        "derivation": "constraint",
        "snapshot": {
            "element": [
                element("Patient", 0, "*", "Patient"),
                element("Patient.name", 1, "*", "HumanName"),
                element("Patient.gender", 0, "1", "code"),
                element("Patient.managingOrganization", 0, "1", "Reference"),
                element("Patient.contact", 0, "1", "BackboneElement"),
                element("Patient.contact.relationship", 1, "*", "CodeableConcept"),
                element("Patient.contact.telecom", 0, "*", "ContactPoint"),
            ]
        },
    }


def observation_profile(value_types=("CodeableConcept",), slices=()):
    """An Observation profile whose ``value[x]`` is narrowed to *value_types*."""

    elements = [
        element("Observation", 0, "*", "Observation"),
        element("Observation.status", 1, "1", "code"),
        element("Observation.code", 1, "1", "CodeableConcept"),
        element("Observation.value[x]", 0, "1", list(value_types)),
    ]
    for name, type_code in slices:
        elements.append(
            element(
                "Observation.value[x]",
                0,
                "1",
                type_code,
                eid=f"Observation.value[x]:{name}",
                sliceName=name,
            )
        )
    return {
        "resourceType": "StructureDefinition",
        "id": "AdvObservation",
        "url": "http://example.org/StructureDefinition/AdvObservation",
        "name": "AdvObservation",
        "status": "draft",
        "kind": "resource",
        "abstract": False,
        "type": "Observation",
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Observation",
        "derivation": "constraint",
        "snapshot": {"element": elements},
    }


def sliced_extension_profile():
    """A Condition whose ``extension`` is sliced and never expanded on the base."""

    return {
        "resourceType": "StructureDefinition",
        "id": "AdvCondition",
        "url": "http://example.org/StructureDefinition/AdvCondition",
        "name": "AdvCondition",
        "status": "draft",
        "kind": "resource",
        "abstract": False,
        "type": "Condition",
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Condition",
        "derivation": "constraint",
        "snapshot": {
            "element": [
                element("Condition", 0, "*", "Condition"),
                element("Condition.extension", 0, "*", "Extension"),
                element(
                    "Condition.extension",
                    0,
                    "1",
                    "Extension",
                    eid="Condition.extension:onset",
                    sliceName="onset",
                ),
                element(
                    "Condition.extension",
                    0,
                    "1",
                    "Extension",
                    eid="Condition.extension:severity",
                    sliceName="severity",
                ),
                element("Condition.subject", 1, "1", "Reference"),
            ]
        },
    }


def structure_map(rules=None, **extra):
    document = {
        "resourceType": "StructureMap",
        "id": "sm-adv",
        "url": "http://example.org/StructureMap/sm-adv",
        "name": "SmAdv",
        "status": "draft",
        "structure": [
            {"url": "http://example.org/StructureDefinition/src", "mode": "source"},
            {
                "url": "http://example.org/StructureDefinition/AdvPatient",
                "mode": "target",
            },
        ],
        "group": [
            {
                "name": "TransformPatient",
                "typeMode": "none",
                "input": [
                    {"name": "source", "type": "Src", "mode": "source"},
                    {"name": "target", "type": "Patient", "mode": "target"},
                ],
                "rule": rules
                if rules is not None
                else [
                    {
                        "name": "map-name",
                        "source": [
                            {
                                "context": "source",
                                "element": "fullName",
                                "variable": "sn",
                            }
                        ],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "name",
                                "transform": "copy",
                                "parameter": [{"valueId": "sn"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    document.update(extra)
    return document


def target_map(element_name, *, context="target", parameter=None, resource="Patient"):
    """A one-rule map writing *element_name* on the target root."""

    document = structure_map(
        [
            {
                "name": "map-one",
                "source": [{"context": "source", "element": "field", "variable": "s"}],
                "target": [
                    {
                        "context": context,
                        "contextType": "variable",
                        "element": element_name,
                        "transform": "copy",
                        "parameter": parameter or [{"valueId": "s"}],
                    }
                ],
            }
        ]
    )
    document["group"][0]["input"][1]["type"] = resource
    return document


SOURCE_FIELDS = {"fullName", "field", "sex", "phone"}


def offline(document, profile, *, mapping_table=None, source_fields=SOURCE_FIELDS):
    return validate_offline(
        document,
        target_tree=build_target_tree(profile),
        mapping_table=mapping_table,
        source_fields=source_fields,
        profile_url=profile["url"],
    )


def codes(report):
    return sorted(finding.code for finding in report.findings)


# --- stubs --------------------------------------------------------------------


ENGINE_DEFAULTS = {
    "engine_available": True,
    "engine_requested": True,
    "executed_fixtures": ["filled", "empty"],
    "required_fixtures": ["filled", "empty"],
    "validated_fixtures": ["filled"],
    "validation_expected_fixtures": ["filled"],
    "evaluation_context_sha256": "shared-context",
}


class OfflineEvaluator:
    """Real WP6 offline layers, with engine evidence stubbed as present.

    The point of these tests is what the deterministic layers conclude, so the
    engine fields are filled identically on both sides. Anything that gets
    accepted here was accepted by layers 1-4 on their own merits.
    """

    def __init__(self, profile, *, mapping_table=None, engine=True):
        self.profile = profile
        self.tree = build_target_tree(profile)
        self.mapping_table = mapping_table
        self.engine = engine
        self.evaluations = 0

    def evaluate(self, document):
        self.evaluations += 1
        report = validate_offline(
            document,
            target_tree=self.tree,
            mapping_table=self.mapping_table,
            source_fields=SOURCE_FIELDS,
            profile_url=self.profile["url"],
        )
        for name, value in ENGINE_DEFAULTS.items():
            setattr(report, name, copy.copy(value))
        if not self.engine:
            report.engine_available = False
            report.executed_fixtures = []
            report.required_fixtures = []
            report.validated_fixtures = []
        return report

    def evaluate_pair(self, baseline, candidate):
        return self.evaluate(baseline), self.evaluate(candidate)


class ScriptedProposer:
    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.contexts = []
        self.calls = 0

    def propose(self, context):
        self.calls += 1
        self.contexts.append(context)
        if not self.proposals:
            return Proposal(patch=None, error="nothing scripted")
        item = self.proposals.pop(0)
        return item(context) if callable(item) else item


def patch_for(document, ops, *, diagnostic_ids, base=None, rationale="repair"):
    return AgentPatch(
        schema_version=1,
        map_url=document["url"],
        map_id=document["id"],
        base_sha256=base if base is not None else canonical_sha256(document),
        diagnostic_ids=list(diagnostic_ids),
        patch=ops,
        rationale=rationale,
    )


def from_worklist(ops_builder, **kwargs):
    """A proposal that claims exactly the findings it was offered."""

    def build(context):
        document = kwargs.pop("document", None) or structure_map()
        return Proposal(
            patch=patch_for(
                document,
                ops_builder(document),
                diagnostic_ids=[finding.finding_id for finding in context.findings],
                **kwargs,
            ),
            prompt_chars=900,
        )

    return build


MAP_KEY = "map"
REPAIR_GRAPH = build_repair_graph()


class StubService:
    """The three things the repair nodes ask a service for."""

    def __init__(self, evaluator, proposer):
        self.proposer = proposer
        self.project = type(
            "StubProject",
            (),
            {
                "mapping_table": evaluator.mapping_table,
                "profile_for": staticmethod(lambda _document: None),
            },
        )()
        self._evaluator = evaluator

    def evaluator_for(self, _document):
        return self._evaluator

    def generator_findings(self, _document):
        return []


def run(document, evaluator, *proposals, limits=None, generator_findings=()):
    """Repair one map through the WP10 subgraph, with an attacking model."""

    proposer = ScriptedProposer(*proposals)
    evaluator.target_tree = evaluator.tree
    evaluator.source_field_specs = [{"name": name} for name in sorted(SOURCE_FIELDS)]
    evaluator.profile_url = evaluator.profile["url"]
    context = AgentRuntimeContext(
        service=StubService(evaluator, proposer),
        documents={MAP_KEY: (Path("map.json"), dict(document))},
        limits=limits or LoopLimits(max_attempts=2),
    )
    result = run_repair_graph(
        MAP_KEY,
        context=context,
        generator_findings=generator_findings,
        graph=REPAIR_GRAPH,
    )
    return result, proposer


# ==============================================================================
# Attacks: a model that invents structure
# ==============================================================================


def test_a_candidate_that_invents_a_target_element_is_rejected():
    document = target_map("name")
    evaluator = OfflineEvaluator(patient_profile())
    # Nothing is wrong yet, so give the loop a reason to run.
    seed = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "The mapping table routes 'sex' to Patient.gender.",
        path="Patient.gender",
        pointer="/group/0/rule/0/target/0",
    )

    result, _ = run(
        document,
        evaluator,
        from_worklist(
            lambda doc: [
                operation("test", "/group/0/rule/0/target/0/element", "name"),
                operation("replace", "/group/0/rule/0/target/0/element", "invented"),
            ],
            document=document,
        ),
        generator_findings=[seed],
    )

    assert result.outcome is not LoopOutcome.ACCEPTED
    assert result.accepted_candidate is None
    assert any(
        finding.code == "target-path-not-found"
        for attempt in result.attempts
        if attempt.validation_report
        for finding in attempt.validation_report.findings
    )


def test_a_candidate_that_invents_a_source_field_is_rejected():
    document = target_map("name")
    evaluator = OfflineEvaluator(patient_profile())
    seed = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "The mapping table routes 'sex' to Patient.gender.",
        path="Patient.gender",
        pointer="/group/0/rule/0/target/0",
    )

    result, _ = run(
        document,
        evaluator,
        from_worklist(
            lambda doc: [
                operation("test", "/group/0/rule/0/source/0/element", "field"),
                operation("replace", "/group/0/rule/0/source/0/element", "noSuchField"),
            ],
            document=document,
        ),
        generator_findings=[seed],
    )

    assert result.outcome is not LoopOutcome.ACCEPTED
    introduced = {
        finding.code
        for attempt in result.attempts
        if attempt.validation_report
        for finding in attempt.validation_report.findings
    }
    assert "source-path-not-found" in introduced


def test_removing_an_import_to_silence_a_dependent_group_is_rejected():
    """The cheapest way to make a cross-map finding disappear is to delete the
    evidence that the other map exists. Imports are outside the authorized rule
    insertion scope, so the candidate must not reach validation."""

    document = structure_map(
        [
            {
                "name": "map-delegated",
                "source": [
                    {"context": "source", "element": "fullName", "variable": "sn"}
                ],
                "target": [
                    {
                        "context": "target",
                        "contextType": "variable",
                        "element": "name",
                        "variable": "tn",
                        "transform": "create",
                        "parameter": [{"valueString": "HumanName"}],
                    }
                ],
                "dependent": [{"name": "BuildName", "variable": ["sn", "tn"]}],
            }
        ],
        **{"import": ["http://example.org/StructureMap/helper"]},
    )
    evaluator = OfflineEvaluator(patient_profile())
    assert codes(evaluator.evaluate(document)) == []

    seed = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "The mapping table routes 'sex' to Patient.gender.",
        path="Patient.gender",
    )
    result, _ = run(
        document,
        evaluator,
        from_worklist(
            lambda doc: [
                operation(
                    "test", "/import", ["http://example.org/StructureMap/helper"]
                ),
                operation("remove", "/import"),
            ],
            document=document,
        ),
        generator_findings=[seed],
    )

    assert result.outcome is not LoopOutcome.ACCEPTED
    rejections = {
        rejection["code"]
        for attempt in result.attempts
        if attempt.application
        for rejection in attempt.application["rejections"]
    }
    assert "out-of-scope-pointer" in rejections


# ==============================================================================
# Attacks: a model that misreports what it is doing
# ==============================================================================


def test_a_patch_naming_no_diagnostic_never_reaches_the_document():
    document = target_map("name")
    evaluator = OfflineEvaluator(patient_profile())
    seed = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "obligation",
        path="Patient.gender",
    )

    result, _ = run(
        document,
        evaluator,
        lambda context: Proposal(
            patch=patch_for(
                document,
                [
                    operation("test", "/group/0/rule/0/target/0/element", "name"),
                    operation("replace", "/group/0/rule/0/target/0/element", "gender"),
                ],
                # A patch that repairs something real but names nothing.
                diagnostic_ids=[],
            ),
            prompt_chars=900,
        ),
        generator_findings=[seed],
    )

    attempt = result.attempts[0]
    assert attempt.application["applied"] is False
    assert [item["code"] for item in attempt.application["rejections"]] == [
        "diagnostic-scope-mismatch"
    ]
    assert attempt.candidate_sha256 is None
    assert result.outcome is not LoopOutcome.ACCEPTED


def test_a_patch_claiming_a_finding_it_was_never_offered_is_rejected():
    """The worklist is map-fixable only. A patch that claims an input-required
    finding is claiming authority it was not given."""

    document = target_map("name")
    evaluator = OfflineEvaluator(patient_profile())
    unfixable = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "required-path-unmapped",
        "Required element Patient.identifier has no source.",
        owner=ActionOwner.MAPPING_INPUT_REQUIRED,
        path="Patient.identifier",
    )
    fixable = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "obligation",
        path="Patient.gender",
    )

    result, proposer = run(
        document,
        evaluator,
        lambda context: Proposal(
            patch=patch_for(
                document,
                [
                    operation("test", "/group/0/rule/0/target/0/element", "name"),
                    operation("replace", "/group/0/rule/0/target/0/element", "gender"),
                ],
                diagnostic_ids=[unfixable.finding_id],
            ),
            prompt_chars=900,
        ),
        generator_findings=[fixable, unfixable],
    )

    assert [finding.finding_id for finding in proposer.contexts[0].findings] == [
        fixable.finding_id
    ]
    assert result.attempts[0].application["applied"] is False
    assert result.outcome is not LoopOutcome.ACCEPTED


def test_a_patch_written_against_the_previous_candidate_is_stale():
    """Every attempt is a complete patch against the original baseline. A model
    that keeps editing its own last answer is writing against a revision the
    loop never adopted."""

    document = target_map("name")
    evaluator = OfflineEvaluator(patient_profile())
    seed = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "obligation",
        path="Patient.gender",
    )

    result, _ = run(
        document,
        evaluator,
        lambda context: Proposal(
            patch=patch_for(
                document,
                [
                    operation("test", "/group/0/rule/0/target/0/element", "name"),
                    operation("replace", "/group/0/rule/0/target/0/element", "gender"),
                ],
                diagnostic_ids=[finding.finding_id for finding in context.findings],
                base="0" * 64,
            ),
            prompt_chars=900,
        ),
        generator_findings=[seed],
    )

    rejections = [item["code"] for item in result.attempts[0].application["rejections"]]
    assert "stale-base" in rejections
    assert result.outcome is not LoopOutcome.ACCEPTED


def test_a_patch_over_the_loop_operation_budget_is_rejected():
    document = target_map("name")
    evaluator = OfflineEvaluator(patient_profile())
    seed = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "obligation",
        path="Patient.gender",
    )

    def flood(context):
        ops = []
        for index in range(6):
            ops.append(
                operation(
                    "add",
                    "/group/0/rule/-",
                    {
                        "name": f"filler-{index}",
                        "source": [{"context": "source", "element": "field"}],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": "gender",
                                "transform": "copy",
                                "parameter": [{"valueString": "other"}],
                            }
                        ],
                    },
                )
            )
        return Proposal(
            patch=patch_for(
                document,
                ops,
                diagnostic_ids=[finding.finding_id for finding in context.findings],
            ),
            prompt_chars=900,
        )

    result, _ = run(
        document,
        evaluator,
        flood,
        limits=LoopLimits(max_attempts=1, max_operations=3),
        generator_findings=[seed],
    )

    rejections = [item["code"] for item in result.attempts[0].application["rejections"]]
    assert "too-many-operations" in rejections
    assert result.outcome is not LoopOutcome.ACCEPTED


# ==============================================================================
# Attacks: the engine is not there
# ==============================================================================


def test_a_candidate_is_never_accepted_without_engine_evidence():
    document = target_map("name")
    evaluator = OfflineEvaluator(patient_profile(), engine=False)
    seed = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "obligation",
        path="Patient.gender",
        pointer="/group/0/rule/0/target/0",
    )

    result, _ = run(
        document,
        evaluator,
        from_worklist(
            lambda doc: [
                operation("test", "/group/0/rule/0/target/0/element", "name"),
                operation("replace", "/group/0/rule/0/target/0/element", "gender"),
            ],
            document=document,
        ),
        generator_findings=[seed],
    )

    assert result.outcome is not LoopOutcome.ACCEPTED
    failed = {
        item["name"]
        for attempt in result.attempts
        if attempt.decision
        for item in attempt.decision["invariants"]
        if not item["passed"]
    }
    assert "engine-available" in failed


# ==============================================================================
# Limits: constructs the offline layers must not guess about
# ==============================================================================


def test_a_choice_addressed_by_its_base_name_is_not_a_missing_element():
    """FML writes `Observation.value`; the snapshot says `Observation.value[x]`.
    They are the same element."""

    document = target_map("value", resource="Observation")
    report = offline(document, observation_profile(("CodeableConcept",)))
    assert "target-path-not-found" not in codes(report)


def test_a_concrete_choice_variant_resolves_against_the_snapshots_x_form():
    document = target_map("valueCodeableConcept", resource="Observation")
    report = offline(document, observation_profile(("CodeableConcept",)))
    assert "target-path-not-found" not in codes(report)


def test_a_choice_variant_the_profile_excludes_is_a_type_error_not_a_missing_element():
    """The distinction matters: the element exists and the repair is to change
    the type, not to invent a path."""

    document = target_map("valueQuantity", resource="Observation")
    report = offline(document, observation_profile(("CodeableConcept",)))

    finding = next(
        item
        for item in report.findings
        if item.code == "target-choice-type-not-allowed"
    )
    assert finding.action_owner is ActionOwner.MAP_FIXABLE
    assert finding.evidence["allowed_variants"] == ["valueCodeableConcept"]


def test_a_choice_the_profile_still_leaves_open_is_not_judged_below():
    """`Observation.value` on a profile that permits several types names a real
    element, but nothing under it can be checked: the path never says which
    type was created."""

    profile = observation_profile(("Quantity", "CodeableConcept"))
    document = target_map("value", context="target", resource="Observation")
    document["group"][0]["rule"][0]["target"][0]["variable"] = "v"
    document["group"][0]["rule"][0]["rule"] = [
        {
            "name": "map-under-choice",
            "source": [{"context": "source", "element": "field", "variable": "s2"}],
            "target": [
                {
                    "context": "v",
                    "contextType": "variable",
                    "element": "whateverThisIs",
                    "transform": "copy",
                    "parameter": [{"valueId": "s2"}],
                }
            ],
        }
    ]
    report = offline(document, profile)
    assert "target-path-not-found" not in codes(report)


def test_an_element_expanded_only_under_slices_is_not_a_missing_element():
    """A profile that slices `Condition.extension` expands `url` under each
    slice and never on the base. The map addresses the base."""

    document = target_map("extension", resource="Condition")
    document["group"][0]["rule"][0]["target"][0]["variable"] = "ext"
    document["group"][0]["rule"][0]["target"][0]["transform"] = "create"
    document["group"][0]["rule"][0]["target"][0]["parameter"] = [
        {"valueString": "Extension"}
    ]
    document["group"][0]["rule"][0]["rule"] = [
        {
            "name": "set-url",
            "source": [{"context": "source", "element": "field", "variable": "s2"}],
            "target": [
                {
                    "context": "ext",
                    "contextType": "variable",
                    "element": "url",
                    "transform": "copy",
                    "parameter": [{"valueString": "http://example.org/x"}],
                }
            ],
        }
    ]
    report = offline(document, sliced_extension_profile())
    assert [code for code in codes(report) if code.startswith("target-")] == []


def test_a_path_below_a_sliced_element_is_not_attributed_to_one_slice():
    """Two extension slices produce the same dotted path. Judging the path
    against whichever slice happens to sort first would report correct maps."""

    document = target_map("extension", resource="Condition")
    document["group"][0]["rule"][0]["target"][0]["variable"] = "ext"
    document["group"][0]["rule"][0]["rule"] = [
        {
            "name": "set-value",
            "source": [{"context": "source", "element": "field", "variable": "s2"}],
            "target": [
                {
                    "context": "ext",
                    "contextType": "variable",
                    "element": "valueDateTime",
                    "transform": "copy",
                    "parameter": [{"valueId": "s2"}],
                }
            ],
        }
    ]
    report = offline(document, sliced_extension_profile())
    assert [code for code in codes(report) if code.startswith("target-")] == []


def test_two_unrelated_elements_sharing_a_path_stay_ambiguous():
    """The reconciliation is about slices of *one* element. Genuine ambiguity
    still has to be reported, or it would be silently unaddressable."""

    profile = patient_profile()
    profile["snapshot"]["element"].extend(
        [
            element(
                "Patient.contact.telecom.system",
                0,
                "1",
                "code",
                eid="Patient.contact.telecom:home.system",
            ),
            element(
                "Patient.contact.telecom.system",
                0,
                "1",
                "code",
                eid="Patient.other.telecom.system",
            ),
        ]
    )
    from agent.validation import _resolve_target_path  # noqa: PLC0415

    # Without introspection, so that the only nodes on this path are the two
    # planted above and the ambiguity is unmistakably between them.
    tree = build_target_tree(profile, introspect=False)
    resolution = _resolve_target_path(tree, "Patient.contact.telecom.system")
    assert resolution.status == "ambiguous-path"
    assert not resolution.resolved


def test_coverage_counts_a_required_choice_written_in_its_concrete_form():
    """`Observation.effectiveDateTime` covers the manifest entry
    `Observation.effective[x]`; comparing the authored spellings would report a
    covered requirement as unmapped."""

    profile = observation_profile(("CodeableConcept",))
    profile["snapshot"]["element"].append(
        element("Observation.effective[x]", 1, "1", ["dateTime", "Period"])
    )
    document = target_map("effectiveDateTime", resource="Observation")
    report = offline(document, profile)

    unmapped = {
        finding.path
        for finding in report.findings
        if finding.code == "required-path-unmapped"
    }
    assert "Observation.effective" not in unmapped
    assert "Observation.effective" in report.covered_required_paths


def test_opening_an_optional_backbone_without_its_required_child_is_engine_work():
    """`Patient.contact.relationship` is 1..* *inside* an optional backbone, so
    it is required only once the backbone exists. That is a property of the
    produced resource, not of the map, and the requirement manifest marks it
    inactive. The gate is the engine's `$validate`, and acceptance therefore
    refuses to run without one."""

    document = target_map("contact")
    document["group"][0]["rule"][0]["target"][0]["variable"] = "c"
    document["group"][0]["rule"][0]["target"][0]["transform"] = "create"
    document["group"][0]["rule"][0]["target"][0]["parameter"] = [
        {"valueString": "BackboneElement"}
    ]
    document["group"][0]["rule"][0]["rule"] = [
        {
            "name": "set-telecom",
            "source": [{"context": "source", "element": "phone", "variable": "sp"}],
            "target": [
                {
                    "context": "c",
                    "contextType": "variable",
                    "element": "telecom",
                    "transform": "copy",
                    "parameter": [{"valueId": "sp"}],
                }
            ],
        }
    ]

    report = offline(document, patient_profile())
    assert "Patient.contact.relationship" not in {
        finding.path for finding in report.findings
    }

    tree = build_target_tree(patient_profile())
    entry = next(
        item
        for item in tree.required_manifest()
        if item["path"] == "Patient.contact.relationship"
    )
    assert entry["active"] is False


def test_an_invented_literal_reference_is_invisible_to_the_offline_layers():
    """`Organization/does-not-exist` is a well-formed value at a real path. Only
    an engine that resolves it can say otherwise, so the offline layers must not
    pretend to."""

    document = target_map("managingOrganization")
    document["group"][0]["rule"][0]["target"][0]["variable"] = "o"
    document["group"][0]["rule"][0]["target"][0]["transform"] = "create"
    document["group"][0]["rule"][0]["target"][0]["parameter"] = [
        {"valueString": "Reference"}
    ]
    document["group"][0]["rule"][0]["rule"] = [
        {
            "name": "set-reference",
            "source": [{"context": "source", "element": "field", "variable": "s2"}],
            "target": [
                {
                    "context": "o",
                    "contextType": "variable",
                    "element": "reference",
                    "transform": "copy",
                    "parameter": [{"valueString": "Organization/does-not-exist"}],
                }
            ],
        }
    ]
    report = offline(document, patient_profile())
    assert [code for code in codes(report) if code.startswith("target-")] == []


def test_a_translate_naming_an_absent_conceptmap_is_invisible_to_the_offline_layers():
    """Nothing offline resolves a ConceptMap canonical. The claim this test
    pins is the honest one: this defect is found by `$transform`, which is why
    an offline run's candidate is never applicable."""

    document = target_map(
        "gender",
        parameter=[
            {"valueId": "s"},
            {"valueString": "http://example.org/ConceptMap/absent"},
            {"valueString": "code"},
        ],
    )
    document["group"][0]["rule"][0]["target"][0]["transform"] = "translate"
    report = offline(document, patient_profile())
    assert [code for code in codes(report) if code.startswith("target-")] == []


def test_a_dependent_group_in_an_imported_map_is_not_a_defect():
    document = structure_map(
        [
            {
                "name": "map-delegated",
                "source": [
                    {"context": "source", "element": "fullName", "variable": "sn"}
                ],
                "target": [
                    {
                        "context": "target",
                        "contextType": "variable",
                        "element": "name",
                        "variable": "tn",
                        "transform": "create",
                        "parameter": [{"valueString": "HumanName"}],
                    }
                ],
                "dependent": [{"name": "BuildName", "variable": ["sn", "tn"]}],
            }
        ],
        **{"import": ["http://example.org/StructureMap/helper"]},
    )
    report = offline(document, patient_profile())
    assert "semantic-rule-invalid" not in codes(report)


def test_a_dependent_group_with_nothing_imported_is_a_defect():
    document = structure_map(
        [
            {
                "name": "map-delegated",
                "source": [
                    {"context": "source", "element": "fullName", "variable": "sn"}
                ],
                "target": [
                    {
                        "context": "target",
                        "contextType": "variable",
                        "element": "name",
                        "variable": "tn",
                        "transform": "create",
                        "parameter": [{"valueString": "HumanName"}],
                    }
                ],
                "dependent": [{"name": "BuildName", "variable": ["sn", "tn"]}],
            }
        ]
    )
    report = offline(document, patient_profile())
    assert "semantic-rule-invalid" in codes(report)


# ==============================================================================
# The gate of last resort
# ==============================================================================


def test_nothing_the_offline_layers_cannot_see_can_be_accepted_without_the_engine():
    """The three limits above share one consequence, and it is the reason
    acceptance treats a missing engine as a failure rather than as silence."""

    baseline = ValidationReport(map_sha256="a" * 64, engine_requested=True)
    candidate = ValidationReport(map_sha256="b" * 64, engine_requested=True)

    decision = decide_acceptance(baseline, candidate, require_engine=True)

    assert not decision.accepted
    assert "engine-available" in {item.name for item in decision.failures}


# ==============================================================================
# Deferred references: incomplete on purpose (WP9)
# ==============================================================================


def deferred_map():
    """A map that hands `Patient.managingOrganization` to the bundle assembler,
    exactly as the generator writes it."""

    document = target_map("name")
    document["group"][0]["rule"].append(
        {
            "name": "TODO-resolve-reference-Patient-managingOrganization",
            "source": [{"context": "source", "variable": "src-org"}],
            "documentation": (
                "Reference<Patient.managingOrganization> → Organization — "
                "resolve via bundle assembler"
            ),
        }
    )
    return document


def required_reference_profile():
    profile = patient_profile()
    for item in profile["snapshot"]["element"]:
        if item["path"] == "Patient.managingOrganization":
            item["min"] = 1
    return profile


def test_a_reference_left_to_the_bundle_assembler_is_not_an_unmapped_requirement():
    """`Observation.subject` points at a resource a *different* map produces.
    The generator says so structurally; calling it an unmapped requirement put
    467 of this repository's 651 required-path findings on maps that were
    behaving as designed."""

    report = offline(deferred_map(), required_reference_profile())

    codes_seen = codes(report)
    assert "required-path-unmapped" not in codes_seen
    finding = next(
        item for item in report.findings if item.code == "required-path-deferred"
    )
    assert finding.path == "Patient.managingOrganization"
    assert finding.action_owner is ActionOwner.ADVISORY
    assert not finding.blocking
    assert report.deferred_reference_paths == ["Patient.managingOrganization"]


def test_a_deferred_reference_gives_the_agent_no_work():
    report = offline(deferred_map(), required_reference_profile())
    assert report.worklist() == []


def test_a_required_reference_without_a_placeholder_is_still_a_gap():
    """The reconciliation keys on the map's own declaration, not on the type.
    A required reference nobody accounted for stays a finding."""

    report = offline(target_map("name"), required_reference_profile())

    finding = next(
        item for item in report.findings if item.code == "required-path-unmapped"
    )
    assert finding.path == "Patient.managingOrganization"
    assert finding.blocking


def test_the_deferred_path_is_read_from_the_rule_name_not_its_prose():
    """The documentation string is for humans. Keying on it would make a
    reworded sentence silently change what gates a candidate."""

    from agent.validation import deferred_reference_paths  # noqa: PLC0415

    document = deferred_map()
    document["group"][0]["rule"][-1]["documentation"] = "anything at all"
    assert deferred_reference_paths(document) == {"Patient.managingOrganization"}

    document["group"][0]["rule"][-1]["name"] = "map-organization"
    assert deferred_reference_paths(document) == set()


# ==============================================================================
# Mapping-table obligations belong to one map (WP9)
# ==============================================================================


def test_obligations_for_another_map_are_not_this_map_s_failure():
    """A project's mapping table covers every profile in the project. Grading
    one map against all of it reported 1 211 dropped obligations across 70
    maps — every other resource's entries, charged to whichever map was being
    validated."""

    table = {
        "src.fullName": "Patient.name",
        "src.orgActive": "Organization.active",
        "src.obsStatus": "Observation.status",
    }
    report = offline(target_map("name"), patient_profile(), mapping_table=table)

    dropped = {
        finding.path
        for finding in report.findings
        if finding.code == "mapping-obligation-dropped"
    }
    assert dropped == set()
    assert report.satisfied_obligations == ["Patient.name"]


def test_a_table_target_written_as_a_profile_id_is_the_same_obligation():
    """The generator accepts a target addressed by resource type *or* by the
    profile's own id, because the id is what disambiguates two profiles of one
    resource type. Comparing the two spellings as strings made a map's own
    obligations look dropped."""

    profile = patient_profile()
    table = {"src.fullName": f"{profile['id']}.name"}
    report = offline(target_map("name"), profile, mapping_table=table)

    assert [
        finding.code
        for finding in report.findings
        if finding.code == "mapping-obligation-dropped"
    ] == []
    assert report.satisfied_obligations == ["Patient.name"]


def test_an_obligation_this_map_really_drops_is_still_reported():
    table = {"src.sex": "Patient.gender"}
    report = offline(target_map("name"), patient_profile(), mapping_table=table)

    finding = next(
        item for item in report.findings if item.code == "mapping-obligation-dropped"
    )
    assert finding.path == "Patient.gender"
    assert finding.action_owner is ActionOwner.MAP_FIXABLE


def test_an_obligation_below_an_unnarrowed_choice_is_not_called_dropped():
    """Layer 3 stops judging below a choice the profile leaves open, so layer 4
    has to stop too — otherwise the two contradict each other and the map is
    told it dropped an obligation it demonstrably writes."""

    profile = observation_profile(("Quantity", "CodeableConcept"))
    document = target_map("value", resource="Observation")
    document["group"][0]["rule"][0]["target"][0]["variable"] = "v"
    document["group"][0]["rule"][0]["rule"] = [
        {
            "name": "map-unit",
            "source": [{"context": "source", "element": "field", "variable": "s2"}],
            "target": [
                {
                    "context": "v",
                    "contextType": "variable",
                    "element": "unit",
                    "transform": "copy",
                    "parameter": [{"valueId": "s2"}],
                }
            ],
        }
    ]
    table = {"src.unit": "Observation.value.unit"}
    report = offline(document, profile, mapping_table=table)

    assert [
        finding.path
        for finding in report.findings
        if finding.code == "mapping-obligation-dropped"
    ] == []


def test_the_engine_layer_attributes_obligations_the_same_way():
    """Layers 4 and 6 both refine required-gap ownership from the mapping
    table. If only one of them re-roots the table's paths, the same element is
    `map-fixable` offline and `mapping-input-required` on the engine's output —
    and agent mode's worklist depends on which one spoke last."""

    from agent.validation import declared_target_paths  # noqa: PLC0415

    table = {
        "src.fullName": "AdvPatient.name",
        "src.orgActive": "Organization.active",
    }
    declared = declared_target_paths(table, res_type="Patient", profile_id="AdvPatient")

    assert declared == {"Patient.name": "src.fullName"}


def test_without_a_profile_nothing_is_attributed():
    """Reversed deliberately, on evidence.

    This asserted the opposite — grade against the whole table — on the
    reasoning that silently dropping every obligation would be worse. A
    ``bodyheight`` map whose target profile the project did not ship was then
    charged 32 obligations spanning ``Patient.*`` and ``QuestionnaireResponse.*``,
    all ``map-fixable`` and blocking, and the agent spent provider calls on
    them.

    Under-attributing costs a missed obligation. Over-attributing invents work
    and pays a model to attempt it. The drop is also no longer silent: a map
    with no resolvable target profile now carries an explicit
    ``target-profile-unavailable`` finding.
    """

    from agent.validation import declared_target_paths  # noqa: PLC0415

    table = {"src.a": "Patient.name", "src.b": "Organization.active"}
    assert declared_target_paths(table) == {}


def test_a_map_with_no_resolvable_target_profile_says_so():
    """The drop must not be silent, which was the old fallback's whole defence."""

    from agent.service import MapEvaluator, ProjectContext  # noqa: PLC0415

    document = {
        "resourceType": "StructureMap",
        "id": "sm-x",
        "url": "http://example.org/StructureMap/sm-x",
        "name": "SmX",
        "status": "draft",
        "structure": [
            {"url": "http://hl7.org/fhir/StructureDefinition/bodyheight", "mode": "target"}
        ],
        "group": [],
    }
    project = ProjectContext(project_dir=Path("."), conf={})
    report = MapEvaluator(project, document, require_engine=False).evaluate(document)
    unavailable = [f for f in report.findings if f.code == "target-profile-unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0].action_owner.value == "environment"
    assert "bodyheight" in unavailable[0].message
    # and it is not offered as repair work
    assert unavailable[0].finding_id not in {f.finding_id for f in report.worklist()}


# ==============================================================================
# Cross-map references: only the assembled set can answer (WP7)
# ==============================================================================


def observation_referencing_patient():
    """An Observation profile whose required `subject` must point at a Patient."""

    profile = observation_profile(("CodeableConcept",))
    profile["snapshot"]["element"].append(
        {
            "id": "Observation.subject",
            "path": "Observation.subject",
            "min": 1,
            "max": "1",
            "type": [
                {
                    "code": "Reference",
                    "targetProfile": [
                        "http://example.org/StructureDefinition/AdvPatient"
                    ],
                }
            ],
        }
    )
    return profile


def observation_deferring_subject():
    document = target_map("status", resource="Observation")
    document["group"][0]["rule"].append(
        {
            "name": "TODO-resolve-reference-Observation-subject",
            "source": [{"context": "source", "variable": "src-subject"}],
            "documentation": "resolve via bundle assembler",
        }
    )
    document["structure"][1]["url"] = observation_referencing_patient()["url"]
    return document


def test_a_deferred_reference_nothing_in_the_set_produces_is_reported():
    """The check no single map can perform: the Observation map is complete on
    its own, and the bundle assembler still cannot wire its subject."""

    from agent.validation import recheck_cross_map_references  # noqa: PLC0415

    findings = recheck_cross_map_references(
        [(observation_deferring_subject(), observation_referencing_patient())]
    )

    assert [finding.code for finding in findings] == ["cross-map-reference-unsatisfied"]
    assert findings[0].path == "Observation.subject"
    assert findings[0].action_owner is ActionOwner.MAPPING_INPUT_REQUIRED
    assert findings[0].blocking


def test_a_sibling_map_producing_the_target_satisfies_the_reference():
    from agent.validation import recheck_cross_map_references  # noqa: PLC0415

    patient_map = target_map("name")
    patient_map["structure"][1]["url"] = patient_profile()["url"]

    findings = recheck_cross_map_references(
        [
            (observation_deferring_subject(), observation_referencing_patient()),
            (patient_map, patient_profile()),
        ]
    )

    assert findings == []


def test_per_map_validation_cannot_see_it():
    """Stated as a test because it is the reason the check exists at all: the
    same map, validated alone, is clean."""

    report = offline(observation_deferring_subject(), observation_referencing_patient())
    assert "cross-map-reference-unsatisfied" not in codes(report)
    assert "required-path-deferred" in codes(report)


def test_a_reference_the_profile_does_not_constrain_is_not_judged():
    """With no `targetProfile`, the profile does not say what the reference must
    point at, so nothing here can say the set fails to provide it."""

    from agent.validation import recheck_cross_map_references  # noqa: PLC0415

    profile = observation_referencing_patient()
    for element in profile["snapshot"]["element"]:
        if element["path"] == "Observation.subject":
            element["type"] = [{"code": "Reference"}]

    assert (
        recheck_cross_map_references([(observation_deferring_subject(), profile)]) == []
    )
