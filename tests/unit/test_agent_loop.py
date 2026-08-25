"""The bounded repair loop and its prompt context (WP7, WP10.2).

The properties under test are about *restraint*. A loop that keeps asking until
something passes will eventually pass something bad, so every test here is
really asking one of three questions: did it call a provider when it should not
have, did it stop when it should have, and can any exit path report success
without an accepted candidate.

These are also the parity tests for WP10: the bodies are unchanged from when a
plain ``for`` loop ran the stages, and only the harness below was rewritten to
drive the LangGraph subgraph instead. If a property held before the refactor and
holds now, it holds because the stages are the same code, not because the graph
happens to agree.

No code execution and no network: the model is a stub that returns whatever the
test scripts.
"""

import copy
from pathlib import Path

import pytest

from agent.context import RejectedAttempt, build_context, describe_rules
from agent.graph import AgentRuntimeContext
from agent.graph.repair_graph import build_repair_graph, run_repair_graph
from agent.loop import (
    LoopLimits,
    LoopOutcome,
    LoopStats,
    Proposal,
    unresolved_by_owner,
)
from agent.models import AgentPatch, operation
from agent.patch import canonical_sha256
from agent.validation import (
    ActionOwner,
    GateStatus,
    Producer,
    Stage,
    ValidationFinding,
    ValidationReport,
    findings_from_diagnostics,
)

pytestmark = pytest.mark.unit


# --- corpus -------------------------------------------------------------------


def structure_map():
    return {
        "resourceType": "StructureMap",
        "id": "sm-fix",
        "url": "http://example.org/StructureMap/sm-fix",
        "name": "SmFix",
        "status": "draft",
        "structure": [
            {"url": "http://example.org/StructureDefinition/src", "mode": "source"},
            {
                "url": "http://example.org/StructureDefinition/TestPatient",
                "mode": "target",
            },
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
                        "name": "map-name",
                        "source": [{"context": "source", "variable": "s"}],
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
                                        "variable": "f",
                                    }
                                ],
                                "target": [
                                    {
                                        "context": "tgt-name",
                                        "contextType": "variable",
                                        "element": "nonsense",
                                        "transform": "copy",
                                        "parameter": [{"valueId": "f"}],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def broken_finding(pointer="/group/0/rule/0/rule/0/target/0"):
    return ValidationFinding.build(
        Producer.PATH_RESOLUTION,
        Stage.PATHS,
        "target-path-not-found",
        "The profile has no element Patient.name.nonsense.",
        path="Patient.name.nonsense",
        pointer=pointer,
    )


def unfixable_finding():
    return ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "required-path-unmapped",
        "Required element Patient.identifier is not emitted and has no source.",
        owner=ActionOwner.MAPPING_INPUT_REQUIRED,
        path="Patient.identifier",
    )


def unclassified_finding():
    return ValidationFinding.build(
        Producer.ENGINE,
        Stage.TRANSFORM,
        "transform:something-new",
        "an engine failure nobody has reviewed",
        fixture_id="filled",
    )


def report(findings=(), **kwargs):
    defaults = {
        "map_url": "http://example.org/StructureMap/sm-fix",
        "map_id": "sm-fix",
        "engine_available": True,
        "engine_requested": True,
        "executed_fixtures": ["filled", "empty"],
        "required_fixtures": ["filled", "empty"],
        "validated_fixtures": ["filled"],
        "evaluation_context_sha256": "shared-engine-context",
        "covered_required_paths": ["Patient.name"],
        "satisfied_obligations": ["Patient.name.family"],
        "findings": list(findings),
    }
    defaults.update(kwargs)
    return ValidationReport(**defaults)


# --- stubs --------------------------------------------------------------------


class ScriptedEvaluator:
    """Returns a scripted report per candidate digest."""

    def __init__(self, baseline_report, candidate_reports=None):
        self.baseline_report = baseline_report
        self.candidate_reports = candidate_reports or {}
        self.default_candidate = report([broken_finding()])
        self.pair_calls = 0

    def _for(self, document):
        digest = canonical_sha256(dict(document))
        if digest == canonical_sha256(structure_map()):
            return self.baseline_report.model_copy(deep=True)
        chosen = self.candidate_reports.get(digest, self.default_candidate)
        if callable(chosen):
            chosen = chosen()
        return chosen.model_copy(deep=True)

    def evaluate(self, document):
        result = self._for(document)
        result.map_sha256 = canonical_sha256(dict(document))
        return result

    def evaluate_pair(self, baseline, candidate):
        self.pair_calls += 1
        return self.evaluate(baseline), self.evaluate(candidate)


class ScriptedProposer:
    """Yields a scripted sequence of proposals and records what it was asked."""

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


def patch_for(document, ops, *, rationale="fix", diagnostic_ids=()):
    return AgentPatch(
        schema_version=1,
        map_url=document["url"],
        map_id=document["id"],
        base_sha256=canonical_sha256(document),
        diagnostic_ids=list(diagnostic_ids),
        patch=ops,
        rationale=rationale,
    )


def repair_ops(document):
    """Guarded replacement of the bogus element with a correct one."""

    pointer = "/group/0/rule/0/rule/0/target/0/element"
    return [
        operation("test", pointer, "nonsense"),
        operation("replace", pointer, "family"),
    ]


def proposal_for(ops_builder, **kwargs):
    def build(context):
        document = structure_map()
        patch_kwargs = dict(kwargs)
        patch_kwargs.setdefault(
            "diagnostic_ids", [finding.finding_id for finding in context.findings]
        )
        return Proposal(
            patch=patch_for(document, ops_builder(document), **patch_kwargs),
            prompt_chars=1200,
        )

    return build


# --- harness ------------------------------------------------------------------
#
# The WP7 call signature, driving the WP10 subgraph. Keeping the signature is
# what makes these parity tests rather than new ones.

MAP_KEY = "map"
REPAIR_GRAPH = build_repair_graph()


class StubProject:
    def __init__(self, mapping_table=None):
        self.mapping_table = dict(mapping_table or {})

    def profile_for(self, _document):
        return None


class StubService:
    """The three things the nodes ask a service for, and nothing else."""

    def __init__(self, evaluator, proposer, mapping_table=None):
        self.proposer = proposer
        self.project = StubProject(mapping_table)
        self._evaluator = evaluator

    def evaluator_for(self, _document):
        return self._evaluator

    def generator_findings(self, _document):
        return []


def run_fix_loop(
    document,
    *,
    evaluator,
    proposer,
    limits=LoopLimits(),
    generator_findings=(),
    target_tree=None,
    mapping_table=None,
    source_fields=(),
    profile_url=None,
    require_engine=True,
):
    """Repair one map through the graph and return its ``LoopResult``."""

    # The nodes read the profile view off the evaluator, which is where the
    # production ``MapEvaluator`` carries it.
    evaluator.target_tree = target_tree
    evaluator.source_field_specs = list(source_fields)
    evaluator.profile_url = profile_url
    context = AgentRuntimeContext(
        service=StubService(evaluator, proposer, mapping_table),
        documents={MAP_KEY: (Path("map.json"), dict(document))},
        limits=limits,
        require_engine=require_engine,
    )
    return run_repair_graph(
        MAP_KEY,
        context=context,
        generator_findings=generator_findings,
        graph=REPAIR_GRAPH,
    )


# --- zero-call paths ----------------------------------------------------------


def test_a_clean_map_causes_no_provider_call():
    proposer = ScriptedProposer()
    result = run_fix_loop(
        structure_map(),
        evaluator=ScriptedEvaluator(report()),
        proposer=proposer,
    )
    assert result.outcome is LoopOutcome.CLEAN
    assert proposer.calls == 0
    assert result.provider_calls == 0
    assert result.accepted_candidate is None


def test_only_non_actionable_findings_cause_no_provider_call():
    # The damaging alternative is reporting this as success. It is not clean —
    # it is blocked, and the findings have to travel out with the result.
    proposer = ScriptedProposer()
    result = run_fix_loop(
        structure_map(),
        evaluator=ScriptedEvaluator(report([unfixable_finding()])),
        proposer=proposer,
    )
    assert result.outcome is LoopOutcome.BLOCKED
    assert proposer.calls == 0
    assert [f.code for f in result.unresolved] == ["required-path-unmapped"]
    assert unresolved_by_owner(result) == {
        "mapping-input-required": [unfixable_finding().finding_id]
    }


def test_an_unclassified_engine_failure_never_authorizes_an_edit():
    proposer = ScriptedProposer()
    result = run_fix_loop(
        structure_map(),
        evaluator=ScriptedEvaluator(report([unclassified_finding()])),
        proposer=proposer,
    )
    assert result.outcome is LoopOutcome.BLOCKED
    assert proposer.calls == 0


def test_an_engine_finding_alone_is_enough_to_start_work():
    # The agent surface is not limited to generator diagnostics: a Matchbox
    # failure classified map-fixable is a valid starting point on its own.
    engine_finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.TRANSFORM,
        "transform:processing",
        "Cannot set property nonsense on Patient",
        fixture_id="filled",
        evidence={"rule_hint": "map-family"},
    )
    evaluator = ScriptedEvaluator(report([engine_finding]))
    evaluator.default_candidate = report()
    proposer = ScriptedProposer(proposal_for(repair_ops))
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=proposer,
        limits=LoopLimits(max_attempts=1),
    )
    assert proposer.calls == 1
    assert result.attempts[0].worklist == [engine_finding.finding_id]


# --- the happy path -----------------------------------------------------------


def regression_finding():
    """A blocking finding the baseline does not have, so a candidate is worse."""

    return ValidationFinding.build(
        Producer.MAP_SEMANTICS,
        Stage.SEMANTICS,
        "semantic-rule-invalid",
        "target references unknown context 'typo'",
        path="TransformPatient",
    )


def clean_after_repair():
    """Broken baseline, clean candidates: the repair succeeds."""

    evaluator = ScriptedEvaluator(report([broken_finding()]))
    evaluator.default_candidate = report()
    return evaluator


def rejecting_evaluator():
    """Broken baseline, and every candidate makes it worse.

    Acceptance is comparative, so a candidate that merely fails to improve is
    still accepted — correctly. To keep the loop iterating a candidate has to be
    genuinely worse than what it would replace.
    """

    evaluator = ScriptedEvaluator(report([broken_finding()]))
    evaluator.default_candidate = report([broken_finding(), regression_finding()])
    return evaluator


def test_a_valid_patch_is_applied_validated_and_accepted():
    evaluator = clean_after_repair()
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=ScriptedProposer(proposal_for(repair_ops)),
    )
    assert result.outcome is LoopOutcome.ACCEPTED
    assert result.accepted_candidate is not None
    element = result.accepted_candidate["group"][0]["rule"][0]["rule"][0]["target"][0]
    assert element["element"] == "family"
    assert result.attempts[-1].accepted is True


def test_an_unrelated_edit_cannot_be_accepted_as_a_repair():
    def unrelated(document):
        return [operation("add", "/description", "looks repaired")]

    result = run_fix_loop(
        structure_map(),
        evaluator=clean_after_repair(),
        proposer=ScriptedProposer(proposal_for(unrelated)),
        limits=LoopLimits(max_attempts=1),
    )
    assert result.outcome is LoopOutcome.EXHAUSTED
    assert result.attempts[0].application["applied"] is False


def test_a_scoped_edit_that_fixes_none_of_the_worklist_is_rejected():
    def documentation_only(document):
        return [
            operation(
                "add",
                "/group/0/rule/0/rule/0/documentation",
                "This does not repair the invalid target.",
            )
        ]

    evaluator = ScriptedEvaluator(report([broken_finding()]))
    evaluator.default_candidate = report([broken_finding()])
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=ScriptedProposer(proposal_for(documentation_only)),
        limits=LoopLimits(max_attempts=1),
    )
    assert result.outcome is LoopOutcome.NO_PROGRESS
    assert result.attempts[0].decision["accepted"] is False
    assert "targeted-repair-progress" in {
        item["name"]
        for item in result.attempts[0].decision["invariants"]
        if not item["passed"]
    }


def test_generator_finding_is_not_resolved_merely_because_it_was_not_rerun():
    generator = findings_from_diagnostics(
        [
            {
                "code": "unmaterialized-nested-target",
                "message": "Patient.name.family was not materialized",
                "path": "Patient.name.family",
            }
        ],
        map_url=structure_map()["url"],
    )[0]

    def unrelated(document):
        return [
            operation(
                "add",
                "/group/0/rule/0/rule/-",
                {
                    "name": "map-unrelated-given",
                    "source": [
                        {"context": "source", "element": "familyName", "variable": "f"}
                    ],
                    "target": [
                        {
                            "context": "tgt-name",
                            "contextType": "variable",
                            "element": "given",
                            "transform": "copy",
                            "parameter": [{"valueId": "f"}],
                        }
                    ],
                },
            )
        ]

    evaluator = ScriptedEvaluator(report())
    evaluator.default_candidate = report()
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=ScriptedProposer(proposal_for(unrelated)),
        generator_findings=[generator],
        limits=LoopLimits(max_attempts=1),
    )
    assert result.outcome is LoopOutcome.NO_PROGRESS
    assert generator.finding_id in result.attempts[0].worklist
    assert generator.finding_id in {
        finding.finding_id for finding in result.attempts[0].validation_report.findings
    }


def test_an_empty_patch_is_a_terminal_abstention_not_a_policy_retry():
    document = structure_map()

    def abstain(context):
        return Proposal(
            patch=patch_for(
                document,
                [],
                rationale="The required source value is not available.",
                diagnostic_ids=[finding.finding_id for finding in context.findings],
            ),
            prompt_chars=400,
        )

    proposer = ScriptedProposer(abstain, proposal_for(repair_ops))
    result = run_fix_loop(
        document,
        evaluator=ScriptedEvaluator(report([broken_finding()])),
        proposer=proposer,
    )

    assert result.outcome is LoopOutcome.NO_PROGRESS
    assert proposer.calls == 1
    assert result.attempts[0].application["abstained"] is True
    assert "baseline was left unchanged" in result.stop_reason


def test_the_input_map_is_never_modified():
    document = structure_map()
    before = copy.deepcopy(document)
    evaluator = clean_after_repair()
    run_fix_loop(
        document,
        evaluator=evaluator,
        proposer=ScriptedProposer(proposal_for(repair_ops)),
    )
    assert document == before


def test_failure_also_leaves_the_input_untouched():
    document = structure_map()
    before = copy.deepcopy(document)
    result = run_fix_loop(
        document,
        evaluator=rejecting_evaluator(),
        proposer=ScriptedProposer(proposal_for(repair_ops), proposal_for(repair_ops)),
        limits=LoopLimits(max_attempts=2),
    )
    assert result.outcome is not LoopOutcome.ACCEPTED
    assert document == before
    assert result.accepted_candidate is None


def test_baseline_and_candidate_are_evaluated_as_a_pair():
    evaluator = clean_after_repair()
    run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=ScriptedProposer(proposal_for(repair_ops)),
    )
    # Not `evaluate` twice: the pair call is what guarantees identical fixtures.
    assert evaluator.pair_calls == 1


# --- retry with feedback ------------------------------------------------------


def test_a_rejected_patch_is_retried_with_the_rejection_named():
    def unguarded(document):
        # A replace whose pointer addresses nothing. Guards are injected from the
        # base now, so an absent `test` is no longer itself a fault — but a
        # pointer with no current value leaves nothing to assert, and WP5 policy
        # still refuses to mutate blind.
        return [
            operation("replace", "/group/0/rule/0/rule/0/target/0/absent", "family")
        ]

    evaluator = clean_after_repair()
    proposer = ScriptedProposer(proposal_for(unguarded), proposal_for(repair_ops))
    result = run_fix_loop(structure_map(), evaluator=evaluator, proposer=proposer)
    assert result.outcome is LoopOutcome.ACCEPTED
    assert len(result.attempts) == 2

    first, second = result.attempts
    assert first.application["applied"] is False
    assert "unguarded-mutation" in {
        rejection["code"] for rejection in first.application["rejections"]
    }
    # The retry prompt has to name what was wrong, not just say "try again".
    retry_context = proposer.contexts[1]
    assert retry_context.attempts
    assert retry_context.attempts[0].rejections[0]["code"] == "unguarded-mutation"
    assert retry_context.attempts[0].operations


def test_a_rejected_candidate_is_retried_with_the_failed_invariants():
    regression = ValidationFinding.build(
        Producer.MAP_SEMANTICS,
        Stage.SEMANTICS,
        "semantic-rule-invalid",
        "source references unknown context 'typo'",
        path="TransformPatient",
    )
    evaluator = clean_after_repair()
    evaluator.default_candidate = report([broken_finding(), regression])
    proposer = ScriptedProposer(proposal_for(repair_ops), proposal_for(repair_ops))
    run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=proposer,
        limits=LoopLimits(max_attempts=2),
    )
    retry = proposer.contexts[1].attempts[0]
    assert any(
        item["name"] == "candidate-semantically-valid"
        for item in retry.failed_invariants
    )
    assert [f.code for f in retry.introduced] == ["semantic-rule-invalid"]


def test_an_unreachable_provider_stops_the_loop_with_a_named_outcome():
    # Retrying cannot help, but the run must still be reportable — a traceback
    # is not an audit trail.
    proposer = ScriptedProposer(
        Proposal(patch=None, error="connection refused", fatal=True, prompt_chars=500),
        proposal_for(repair_ops),
    )
    result = run_fix_loop(
        structure_map(), evaluator=rejecting_evaluator(), proposer=proposer
    )
    assert result.outcome is LoopOutcome.PROVIDER_ERROR
    assert result.stop_reason == "connection refused"
    assert proposer.calls == 1
    assert result.accepted_candidate is None


def test_a_response_that_does_not_fit_the_envelope_is_one_failed_attempt():
    # The model being wrong is not the run being broken.
    evaluator = clean_after_repair()
    proposer = ScriptedProposer(
        Proposal(patch=None, error="missing base_sha256", prompt_chars=900),
        proposal_for(repair_ops),
    )
    result = run_fix_loop(structure_map(), evaluator=evaluator, proposer=proposer)
    assert result.outcome is LoopOutcome.ACCEPTED
    assert result.attempts[0].note == "missing base_sha256"


# --- bounds -------------------------------------------------------------------


def test_an_identical_candidate_terminates_as_no_progress():
    evaluator = rejecting_evaluator()
    proposer = ScriptedProposer(
        proposal_for(repair_ops), proposal_for(repair_ops), proposal_for(repair_ops)
    )
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=proposer,
        limits=LoopLimits(max_attempts=5),
    )
    assert result.outcome is LoopOutcome.NO_PROGRESS
    assert proposer.calls == 2  # stopped as soon as the repeat was seen
    assert "identical" in result.stop_reason


def test_two_different_candidates_with_the_same_problems_stop_as_oscillation():
    # Hash comparison alone would miss this and burn the whole budget.
    def other_repair(document):
        pointer = "/group/0/rule/0/rule/0/target/0/element"
        return [
            operation("test", pointer, "nonsense"),
            operation("replace", pointer, "given"),
        ]

    evaluator = rejecting_evaluator()  # every candidate fails the same way
    proposer = ScriptedProposer(
        proposal_for(repair_ops), proposal_for(other_repair), proposal_for(repair_ops)
    )
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=proposer,
        limits=LoopLimits(max_attempts=5),
    )
    assert result.outcome is LoopOutcome.NO_PROGRESS
    assert "oscillating" in result.stop_reason


def test_the_attempt_budget_is_enforced():
    def alternating(index):
        def build(context):
            document = structure_map()
            pointer = "/group/0/rule/0/rule/0/target/0/element"
            return Proposal(
                patch=patch_for(
                    document,
                    [
                        operation("test", pointer, "nonsense"),
                        operation("replace", pointer, f"variant{index}"),
                    ],
                    diagnostic_ids=[finding.finding_id for finding in context.findings],
                ),
                prompt_chars=500,
            )

        return build

    evaluator = ScriptedEvaluator(report([broken_finding()]))
    # A different finding set each time, so neither no-progress rule fires.
    counter = {"n": 0}

    def changing():
        counter["n"] += 1
        return report(
            [
                ValidationFinding.build(
                    Producer.PATH_RESOLUTION,
                    Stage.PATHS,
                    "target-path-not-found",
                    "still wrong",
                    path=f"Patient.name.variant{counter['n']}",
                )
            ]
        )

    evaluator.default_candidate = changing
    proposer = ScriptedProposer(*[alternating(i) for i in range(6)])
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=proposer,
        limits=LoopLimits(max_attempts=3),
    )
    assert result.outcome is LoopOutcome.EXHAUSTED
    assert len(result.attempts) == 3
    assert proposer.calls == 3


def test_the_provider_call_budget_is_enforced():
    evaluator = ScriptedEvaluator(report([broken_finding()]))
    counter = {"n": 0}

    def changing():
        counter["n"] += 1
        return report(
            [
                ValidationFinding.build(
                    Producer.PATH_RESOLUTION,
                    Stage.PATHS,
                    "target-path-not-found",
                    "still wrong",
                    path=f"Patient.name.v{counter['n']}",
                )
            ]
        )

    evaluator.default_candidate = changing

    def variant(index):
        def build(context):
            document = structure_map()
            pointer = "/group/0/rule/0/rule/0/target/0/element"
            return Proposal(
                patch=patch_for(
                    document,
                    [
                        operation("test", pointer, "nonsense"),
                        operation("replace", pointer, f"v{index}"),
                    ],
                    diagnostic_ids=[finding.finding_id for finding in context.findings],
                ),
                prompt_chars=100,
            )

        return build

    proposer = ScriptedProposer(*[variant(i) for i in range(9)])
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=proposer,
        limits=LoopLimits(max_attempts=9, max_provider_calls=2),
    )
    assert result.outcome is LoopOutcome.LIMIT_REACHED
    assert result.provider_calls == 2


def test_an_oversized_prompt_aborts_rather_than_being_truncated():
    # A shortened prompt asks a different question while looking like the same
    # one, and the cache would key it as though it were the original.
    result = run_fix_loop(
        structure_map(),
        evaluator=rejecting_evaluator(),
        proposer=ScriptedProposer(
            Proposal(patch=None, prompt_chars=999_999, error="too big")
        ),
        limits=LoopLimits(max_prompt_chars=1000),
    )
    assert result.outcome is LoopOutcome.LIMIT_REACHED
    assert "exceeds" in result.stop_reason


def test_an_oversized_map_is_pruned_for_insertion_only_work():
    document = structure_map()
    template = document["group"][0]["rule"][0]
    document["group"][0]["rule"] = []
    for index in range(80):
        rule = copy.deepcopy(template)
        rule["name"] = f"unrelated-{index}"
        rule["documentation"] = "x" * 500
        document["group"][0]["rule"].append(rule)
    finding = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "The map does not emit Patient.active.",
        path="Patient.active",
    )

    context = build_context(document, [finding], map_budget_chars=1_000)

    assert context.excerpt_pruned
    assert all(rule.get("_elided") for rule in context.map_excerpt["group"][0]["rule"])
    assert context.insertion_points


def test_prompt_preflight_stops_before_the_provider_call():
    class PreflightProposer:
        calls = 0

        def prompt_chars(self, context):
            return 10_001

        def propose(self, context):
            self.calls += 1
            raise AssertionError("the provider route must not be entered")

    proposer = PreflightProposer()
    result = run_fix_loop(
        structure_map(),
        evaluator=rejecting_evaluator(),
        proposer=proposer,
        limits=LoopLimits(max_prompt_chars=10_000),
    )
    assert result.outcome is LoopOutcome.LIMIT_REACHED
    assert proposer.calls == 0
    assert result.provider_calls == 0


def test_a_large_worklist_is_offered_in_a_bounded_batch():
    findings = [
        ValidationFinding.build(
            Producer.PATH_RESOLUTION,
            Stage.PATHS,
            "target-path-not-found",
            f"Missing Patient.field{index}",
            path=f"Patient.field{index}",
            pointer="/group/0/rule/0/rule/0/target/0/element",
        )
        for index in range(10)
    ]
    proposer = ScriptedProposer(Proposal(patch=None, error="stop", fatal=True))

    run_fix_loop(
        structure_map(),
        evaluator=ScriptedEvaluator(report(findings)),
        proposer=proposer,
        limits=LoopLimits(max_findings_per_attempt=3),
    )

    assert len(proposer.contexts) == 1
    assert len(proposer.contexts[0].findings) == 3


def test_malformed_provider_payload_is_retryable_not_fatal():
    from agent.service import LLMProposer
    from llm.errors import LLMResponseValidationError

    class MalformedClient:
        def complete(self, **kwargs):
            raise LLMResponseValidationError("not an AgentPatch", raw_text="{}")

    context = build_context(structure_map(), [broken_finding()])
    proposal = LLMProposer(MalformedClient(), object()).propose(context)
    assert proposal.patch is None
    assert proposal.fatal is False
    assert "patch envelope" in proposal.error


def test_a_cache_hit_does_not_consume_the_provider_budget():
    evaluator = clean_after_repair()

    def cached(context):
        document = structure_map()
        return Proposal(
            patch=patch_for(
                document,
                repair_ops(document),
                diagnostic_ids=[finding.finding_id for finding in context.findings],
            ),
            cache_hit=True,
            prompt_chars=100,
        )

    result = run_fix_loop(
        structure_map(), evaluator=evaluator, proposer=ScriptedProposer(cached)
    )
    assert result.cache_hits == 1
    assert result.provider_calls == 0


# --- the prompt context -------------------------------------------------------


def test_the_context_offers_only_map_fixable_findings():
    # Authorization to edit is the one thing a prompt must never widen, so the
    # filter lives in the builder rather than in a template.
    context = build_context(
        structure_map(),
        [broken_finding(), unfixable_finding(), unclassified_finding()],
        all_findings=[broken_finding(), unfixable_finding(), unclassified_finding()],
    )
    assert [finding.code for finding in context.findings] == ["target-path-not-found"]
    assert {item["owner"] for item in context.out_of_scope} == {
        "mapping-input-required",
        "unclassified",
    }


def test_the_context_carries_resolved_pointers_not_indices():
    context = build_context(structure_map(), [broken_finding()])
    pointers = {rule.pointer: rule for rule in context.pointers}
    assert "/group/0/rule/0/rule/0" in pointers
    assert pointers["/group/0/rule/0/rule/0"].label.endswith("map-family")
    assert pointers["/group/0/rule/0/rule/0"].focused is True
    assert pointers["/group/0/rule/0"].focused is False


def test_a_missing_nested_target_gets_an_exact_child_insertion_pointer():
    finding = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "The mapping table route was not emitted.",
        path="Patient.name.family",
    )

    context = build_context(structure_map(), [finding])

    assert [point.pointer for point in context.insertion_points] == [
        "/group/0/rule/0/rule/-"
    ]


def test_a_missing_top_level_target_gets_a_group_append_pointer():
    gender = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "The mapping table route was not emitted.",
        path="Patient.gender",
    )
    birth_date = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "Another mapping table route was not emitted.",
        path="Patient.birthDate",
    )

    context = build_context(structure_map(), [gender, birth_date])

    assert [point.pointer for point in context.insertion_points] == ["/group/0/rule/-"]
    assert context.insertion_points[0].target_paths == [
        "Patient.birthDate",
        "Patient.gender",
    ]


def test_a_complex_ancestor_without_a_variable_uses_a_group_append_pointer():
    document = structure_map()
    del document["group"][0]["rule"][0]["target"][0]["variable"]
    finding = ValidationFinding.build(
        Producer.COVERAGE,
        Stage.COVERAGE,
        "mapping-obligation-dropped",
        "The nested mapping table route was not emitted.",
        path="Patient.name.family",
    )

    context = build_context(document, [finding])

    assert [point.pointer for point in context.insertion_points] == ["/group/0/rule/-"]


def test_a_worklist_without_a_deterministic_scope_never_calls_the_provider():
    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.TRANSFORM,
        "transform:processing",
        "The engine failed without identifying a rule.",
        fixture_id="filled",
    )
    proposer = ScriptedProposer(proposal_for(repair_ops))

    result = run_fix_loop(
        structure_map(),
        evaluator=ScriptedEvaluator(report([finding])),
        proposer=proposer,
    )

    assert result.outcome is LoopOutcome.BLOCKED
    assert proposer.calls == 0
    # The reason names what could not be located rather than reporting an
    # internal failure: the finding carries no element, and saying so is the
    # difference between "the map cannot be repaired from here" and "the
    # harness broke".
    assert "name no element, rule or pointer" in result.stop_reason
    assert "transform:processing" in result.stop_reason


def test_a_finding_pointer_focuses_the_rule_that_contains_it():
    context = build_context(
        structure_map(),
        [broken_finding(pointer="/group/0/rule/0/rule/0/target/0")],
    )
    focused = [rule.pointer for rule in context.pointers if rule.focused]
    assert focused == ["/group/0/rule/0/rule/0"]


def test_pointer_cap_keeps_focused_rule_even_when_it_is_late():
    document = structure_map()
    original = document["group"][0]["rule"].pop(0)
    for index in range(8):
        document["group"][0]["rule"].append(
            {
                "name": f"unrelated-{index}",
                "source": [{"context": "source"}],
                "target": [],
            }
        )
    document["group"][0]["rule"].append(original)
    finding = broken_finding(pointer="/group/0/rule/8/rule/0/target/0")
    context = build_context(document, [finding], list_limit=1)
    assert context.pointers[0].pointer == "/group/0/rule/8/rule/0"
    assert context.pointers[0].focused is True


def test_an_engine_rule_hint_focuses_by_name():
    finding = ValidationFinding.build(
        Producer.ENGINE,
        Stage.TRANSFORM,
        "transform:processing",
        "boom",
        fixture_id="filled",
        evidence={"rule_hint": "map-family"},
    )
    context = build_context(structure_map(), [finding])
    assert [rule.pointer for rule in context.pointers if rule.focused] == [
        "/group/0/rule/0/rule/0"
    ]


def test_a_small_map_is_sent_whole():
    context = build_context(structure_map(), [broken_finding()])
    assert context.excerpt_pruned is False
    assert context.map_excerpt == structure_map()


def test_a_large_map_is_pruned_to_the_focus_with_markers():
    document = structure_map()
    group = document["group"][0]
    for index in range(60):
        group["rule"].append(
            {
                "name": f"filler-{index}",
                "documentation": "x" * 400,
                "source": [{"context": "source"}],
                "target": [
                    {
                        "context": "target",
                        "contextType": "variable",
                        "element": "birthDate",
                        "transform": "copy",
                        "parameter": [{"valueString": "2024-01-01"}],
                    }
                ],
            }
        )
    context = build_context(document, [broken_finding()])
    assert context.excerpt_pruned is True
    rules = context.map_excerpt["group"][0]["rule"]
    # The focused rule survives in full; the rest become addressable markers, so
    # the model can see that something is there and never mistakes an elision
    # for an array it should fill.
    assert rules[0]["name"] == "map-name"
    assert rules[1]["_elided"] is True
    assert rules[1]["_pointer"] == "/group/0/rule/1"


def test_the_context_records_identity_for_the_cache_key():
    first = build_context(structure_map(), [broken_finding()]).cache_context()
    second = build_context(structure_map(), [broken_finding()]).cache_context()
    assert first == second

    with_history = build_context(
        structure_map(),
        [broken_finding()],
        attempts=[RejectedAttempt(attempt=1)],
    ).cache_context()
    # A retry must not reuse the first attempt's answer.
    assert with_history != first


def test_describe_rules_labels_repeated_names_by_their_trail():
    document = structure_map()
    document["group"][0]["rule"].append(copy.deepcopy(document["group"][0]["rule"][0]))
    labels = [rule.label for rule in describe_rules(document)]
    assert labels.count("TransformPatient › map-name") == 2
    assert "TransformPatient › map-name › map-family" in labels


def test_the_context_lists_source_fields_and_mapping_rows():
    context = build_context(
        structure_map(),
        [broken_finding()],
        mapping_table={"familyName": "Patient.name.family"},
        source_fields=[{"name": "familyName", "type": "string", "min": 0, "max": "1"}],
    )
    assert context.mapping_rows[0]["target"] == "Patient.name.family"
    assert context.source_fields[0]["name"] == "familyName"


def test_bounded_context_keeps_relevant_rows_and_reports_truncation():
    context = build_context(
        structure_map(),
        [broken_finding()],
        mapping_table={
            "other": "Patient.birthDate",
            "familyName": {"target": "Patient.name.nonsense", "fixed_value": False},
        },
        source_fields=[
            {"name": "other", "type": "date"},
            {"name": "familyName", "type": "boolean"},
            {"name": "third", "type": "string"},
        ],
        list_limit=1,
    )

    assert context.mapping_rows == [
        {
            "source": "familyName",
            "target": "Patient.name.nonsense",
            "fixed_value": False,
        }
    ]
    assert context.source_fields[0]["name"] == "familyName"
    assert context.list_status["mapping_rows"].truncated is True
    assert context.list_status["source_fields"].truncated is True


def test_the_context_names_relevant_concept_maps_without_exporting_code_pairs():
    context = build_context(
        structure_map(),
        [broken_finding()],
        mapping_table={
            "familyName": {
                "target": "Patient.name.family",
                "concept_map_url": "http://example.org/ConceptMap/family",
            }
        },
    )
    assert context.concept_map_urls == ["http://example.org/ConceptMap/family"]
    assert "pairs" not in context.model_dump(mode="json")


# --- reporting helpers --------------------------------------------------------


def test_loop_stats_aggregate_across_maps():
    stats = LoopStats()
    accepted = clean_after_repair()
    stats.add(
        run_fix_loop(
            structure_map(),
            evaluator=accepted,
            proposer=ScriptedProposer(proposal_for(repair_ops)),
        )
    )
    stats.add(
        run_fix_loop(
            structure_map(),
            evaluator=ScriptedEvaluator(report()),
            proposer=ScriptedProposer(),
        )
    )
    summary = stats.as_dict()
    assert summary["maps"] == 2
    assert summary["accepted"] == 1
    assert summary["clean"] == 1


def test_the_result_serializes_for_a_run_report():
    evaluator = clean_after_repair()
    result = run_fix_loop(
        structure_map(),
        evaluator=evaluator,
        proposer=ScriptedProposer(proposal_for(repair_ops)),
    )
    payload = result.model_dump(mode="json")
    assert payload["outcome"] == "accepted"
    assert payload["attempts"][0]["operations"]
    assert payload["accepted_sha256"]


def test_a_gate_status_of_non_blocking_does_not_make_work():
    """A demoted finding is reported, never prompted.

    ``NON_BLOCKING`` is how the engine layer says "this is not the map's fault"
    — ``engine-severity:warning``, ``experimental-fixture:…``,
    ``unverified-synthetic-value:…``. Offering one as repair work contradicts
    the demotion that produced it and asks a model to change a correct map
    because a *fixture* carried a placeholder.

    This test previously asserted the opposite: that such a finding was worth a
    provider call. On a real three-map project that cost two of six calls, both
    spent reaching the only honest answer — an abstention.
    """

    advisory = ValidationFinding.build(
        Producer.ENGINE,
        Stage.VALIDATE,
        "validate:value",
        "unusual but legal",
        gate=GateStatus.NON_BLOCKING,
        fixture_id="filled",
        evidence={"rule_hint": "map-family"},
    )
    proposer = ScriptedProposer(proposal_for(repair_ops))
    evaluator = ScriptedEvaluator(report([advisory]))
    evaluator.default_candidate = report()

    result = run_fix_loop(structure_map(), evaluator=evaluator, proposer=proposer)

    assert result.outcome is LoopOutcome.CLEAN
    assert proposer.calls == 0
    # Reported, not repaired: it is still on the baseline report.
    assert advisory.finding_id in {
        finding.finding_id for finding in result.baseline_report.findings
    }




# --- continuation after a partial accept --------------------------------------
#
# `require_targeted_progress` asks for *one* targeted finding to disappear, so a
# candidate repairing one of several used to end the map as accepted with
# map-fixable work still open and no further attempt ever made. The repaired
# revision becomes the next baseline instead, and the loop ends on its own
# terms: nothing left to fix, an abstention, no progress, or a budget.

_ELEMENT = "/group/0/rule/0/rule/0/target/0/element"


def second_broken_finding():
    """A distinct map-fixable finding in the same rule.

    A different *path*, because finding identity deliberately ignores the
    pointer — two findings differing only in pointer are one finding.
    """

    return ValidationFinding.build(
        Producer.PATH_RESOLUTION,
        Stage.PATHS,
        "target-path-not-found",
        "The profile has no element Patient.name.bogus.",
        path="Patient.name.bogus",
        pointer="/group/0/rule/0/rule/0/target/0",
    )


def proposal_against_current(ops_builder, **kwargs):
    """Like `proposal_for`, but bound to the revision actually under repair.

    `proposal_for` signs every patch with the digest of the original document.
    That is fine while the baseline never moves; once an accepted partial repair
    becomes the next baseline, the next patch has to be written against *that*
    revision or WP5 rejects it as stale — which is exactly the guard working.
    """

    def build(context):
        patch_kwargs = dict(kwargs)
        patch_kwargs.setdefault(
            "diagnostic_ids", [finding.finding_id for finding in context.findings]
        )
        return Proposal(
            patch=AgentPatch(
                schema_version=1,
                map_url=context.map_url,
                map_id=context.map_id,
                base_sha256=context.map_sha256,
                patch=ops_builder(None),
                rationale="fix",
                **patch_kwargs,
            ),
            prompt_chars=1200,
        )

    return build


def rename_ops(was, now):
    """A guarded rename inside the rule the worklist finding focuses."""

    return lambda _document: [
        operation("test", _ELEMENT, was),
        operation("replace", _ELEMENT, now),
    ]


def _apply(document, ops):
    import jsonpatch  # noqa: PLC0415

    return jsonpatch.JsonPatch([dict(op) for op in ops]).apply(copy.deepcopy(document))


def test_an_accepted_partial_repair_continues_from_the_repaired_revision():
    base = structure_map()
    still_open = second_broken_finding()
    first = _apply(base, rename_ops("nonsense", "family")(base))
    second = _apply(first, rename_ops("family", "given")(first))

    evaluator = ScriptedEvaluator(
        report([broken_finding(), still_open]),
        candidate_reports={
            # one of the two resolved: accepted, but work remains
            canonical_sha256(first): report([still_open]),
            # the rest resolved
            canonical_sha256(second): report(),
        },
    )
    proposer = ScriptedProposer(
        proposal_against_current(rename_ops("nonsense", "family")),
        proposal_against_current(rename_ops("family", "given")),
    )

    result = run_fix_loop(base, evaluator=evaluator, proposer=proposer)

    assert result.outcome is LoopOutcome.ACCEPTED
    assert proposer.calls == 2, "the loop stopped at the first accept"
    assert canonical_sha256(result.accepted_candidate) == canonical_sha256(second)
    assert not result.accepted_report.worklist()


def test_a_fully_repaired_map_still_stops_at_the_first_accept():
    """Nothing map-fixable left means there is nothing to continue for."""

    proposer = ScriptedProposer(proposal_for(repair_ops))
    result = run_fix_loop(
        structure_map(), evaluator=clean_after_repair(), proposer=proposer
    )
    assert result.outcome is LoopOutcome.ACCEPTED
    assert proposer.calls == 1


def test_a_banked_repair_survives_a_later_failure():
    """Continuing must never end worse than stopping at the first accept."""

    base = structure_map()
    still_open = second_broken_finding()
    first = _apply(base, rename_ops("nonsense", "family")(base))

    evaluator = ScriptedEvaluator(
        report([broken_finding(), still_open]),
        candidate_reports={canonical_sha256(first): report([still_open])},
    )
    # Every candidate after the first is strictly worse, so none can be accepted.
    evaluator.default_candidate = report([still_open, unclassified_finding()])
    proposer = ScriptedProposer(
        proposal_against_current(rename_ops("nonsense", "family")),
        proposal_against_current(rename_ops("family", "given")),
        proposal_against_current(rename_ops("family", "prefix")),
        proposal_against_current(rename_ops("family", "suffix")),
    )

    result = run_fix_loop(base, evaluator=evaluator, proposer=proposer)

    assert result.outcome is LoopOutcome.ACCEPTED
    assert canonical_sha256(result.accepted_candidate) == canonical_sha256(first)
    assert "earlier candidate passed every hard invariant" in result.stop_reason
    assert [item.finding_id for item in result.unresolved] == [still_open.finding_id]
