"""The project graph: rounds, assembly, global validation, requeueing (WP10.3).

The per-map subgraph is covered in ``test_agent_loop.py``. What is only true of a
*set* is covered here, and it is the reason WP10 exists at all:

- a project is not the conjunction of its maps — every map can be individually
  clean while the assembled set cannot be wired;
- a global finding may be handed back only to the map it names, and only when
  something a map edit can do would resolve it;
- the cycle that makes that possible must terminate, on rounds, on budget, and
  on a repeated project state.
"""

from pathlib import Path

import pytest

from agent.graph import AgentRuntimeContext, ProjectLimits, run_project_graph
from agent.graph.project_graph import ProjectRun, build_project_graph
from agent.graph.state import (
    PROJECT_LIMIT_REACHED,
    PROJECT_NO_PROGRESS,
    PROJECT_OK,
    PROJECT_UNRESOLVED,
)
from agent.loop import LoopLimits, LoopOutcome, Proposal
from agent.models import AgentPatch, operation
from agent.patch import canonical_sha256
from agent.validation import (
    ActionOwner,
    Producer,
    Stage,
    ValidationFinding,
    ValidationReport,
)

pytestmark = pytest.mark.unit


# --- corpus -------------------------------------------------------------------


def structure_map(key, element="family"):
    return {
        "resourceType": "StructureMap",
        "id": f"sm-{key}",
        "url": f"http://example.org/StructureMap/{key}",
        "name": f"Sm{key.title()}",
        "status": "draft",
        "structure": [
            {"url": "http://example.org/StructureDefinition/src", "mode": "source"},
            {
                "url": f"http://example.org/StructureDefinition/{key}",
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
                        "source": [
                            {"context": "source", "element": "familyName", "variable": "f"}
                        ],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": element,
                                "transform": "copy",
                                "parameter": [{"valueId": "f"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def clean_report(document):
    return ValidationReport(
        map_url=document["url"],
        map_id=document["id"],
        map_sha256=canonical_sha256(document),
        engine_available=True,
        engine_requested=True,
        executed_fixtures=["filled"],
        required_fixtures=["filled"],
        validated_fixtures=["filled"],
        evaluation_context_sha256="shared",
    )


def broken_report(document, finding):
    report = clean_report(document)
    report.findings = [finding]
    return report


def fixable_finding(path="Patient.nonsense"):
    return ValidationFinding.build(
        Producer.PATH_RESOLUTION,
        Stage.PATHS,
        "target-path-not-found",
        f"The profile has no element {path}.",
        path=path,
        pointer="/group/0/rule/0/target/0",
    )


# --- harness ------------------------------------------------------------------


class Evaluator:
    """Scripted per-document reports, keyed by canonical digest."""

    def __init__(self, baseline_report, candidate_report=None):
        self.baseline_report = baseline_report
        self.candidate_report = candidate_report
        self.target_tree = None
        self.source_field_specs = []
        self.profile_url = None

    def _for(self, document):
        digest = canonical_sha256(dict(document))
        if digest == self.baseline_report.map_sha256:
            report = self.baseline_report
        else:
            report = self.candidate_report or clean_report(dict(document))
        report = report.model_copy(deep=True)
        report.map_sha256 = digest
        return report

    def evaluate(self, document):
        return self._for(document)

    def evaluate_pair(self, baseline, candidate):
        return self._for(baseline), self._for(candidate)


class Proposer:
    """Replaces the bad element with a good one, once per map."""

    def __init__(self):
        self.calls = 0
        self.contexts = []

    def propose(self, context):
        self.calls += 1
        self.contexts.append(context)
        pointer = "/group/0/rule/0/target/0/element"
        return Proposal(
            patch=AgentPatch(
                schema_version=1,
                map_url=context.map_url,
                map_id=context.map_id,
                base_sha256=context.map_sha256,
                diagnostic_ids=[f.finding_id for f in context.findings],
                patch=[
                    operation("test", pointer, "nonsense"),
                    operation("replace", pointer, "family"),
                ],
                rationale="use the element the profile has",
            ),
            prompt_chars=800,
        )


class Project:
    def __init__(self, documents, profiles=None):
        self.project_dir = Path("/project")
        self.coverage_report = None
        self.mapping_table = {}
        self.structure_maps = [(Path(f"{key}.json"), doc) for key, doc in documents.items()]
        self.profiles = profiles or {}

    def profile_for(self, document):
        for entry in document.get("structure") or []:
            if entry.get("mode") == "target":
                return self.profiles.get(entry.get("url"))
        return None


class Service:
    """Only what the graph nodes ask for."""

    def __init__(self, documents, evaluators, proposer, profiles=None):
        self.project = Project(documents, profiles)
        self.proposer = proposer
        self._evaluators = evaluators
        self.limits = LoopLimits(max_attempts=2)
        self.require_engine = True
        self.llm_config = {}

    def evaluator_for(self, document):
        return self._evaluators[document["id"]]

    def generator_findings(self, _document):
        return []


def runtime(documents, evaluators, proposer, *, validators=(), profiles=None, **kwargs):
    return AgentRuntimeContext(
        service=Service(documents, evaluators, proposer, profiles),
        documents={key: (Path(f"{key}.json"), doc) for key, doc in documents.items()},
        limits=kwargs.pop("limits", LoopLimits(max_attempts=2)),
        project_limits=kwargs.pop("project_limits", ProjectLimits()),
        project_validators=validators,
        **kwargs,
    )


def run(context, run_id="run-test"):
    return run_project_graph(context=context, run_id=run_id)


# --- zero-call and per-map isolation ------------------------------------------


def test_a_project_of_clean_maps_calls_no_provider_and_is_ok():
    documents = {"a": structure_map("a"), "b": structure_map("b")}
    evaluators = {
        doc["id"]: Evaluator(clean_report(doc)) for doc in documents.values()
    }
    proposer = Proposer()

    result = run(runtime(documents, evaluators, proposer))

    assert proposer.calls == 0
    assert result.outcome == PROJECT_OK
    assert {entry.result.outcome for entry in result.ordered} == {LoopOutcome.CLEAN}


def test_one_broken_map_is_repaired_while_the_clean_one_is_untouched():
    documents = {"a": structure_map("a", element="nonsense"), "b": structure_map("b")}
    evaluators = {
        documents["a"]["id"]: Evaluator(
            broken_report(documents["a"], fixable_finding()), clean_report(documents["a"])
        ),
        documents["b"]["id"]: Evaluator(clean_report(documents["b"])),
    }
    proposer = Proposer()

    result = run(runtime(documents, evaluators, proposer))

    assert proposer.calls == 1
    repaired = result.results["a"]
    assert repaired.result.outcome is LoopOutcome.ACCEPTED
    assert repaired.staged["group"][0]["rule"][0]["target"][0]["element"] == "family"
    # The clean map is still in the assembled set, and still its own baseline.
    assert result.results["b"].changed is False
    assert result.results["b"].staged == documents["b"]
    assert result.outcome == PROJECT_OK


def test_the_maps_are_assembled_in_a_stable_order_with_a_digest_per_round():
    documents = {"a": structure_map("a"), "b": structure_map("b")}
    evaluators = {doc["id"]: Evaluator(clean_report(doc)) for doc in documents.values()}

    result = run(runtime(documents, evaluators, Proposer()))

    assert result.map_order == ["a", "b"]
    assert len(result.assembled_digests) == 1
    assert result.rounds[0]["dispatched"] == ["a", "b"]


# --- global findings ----------------------------------------------------------


def unroutable_global(_assembled):
    """What ``recheck_cross_map_references`` produces: nobody can act on it."""

    return [
        ValidationFinding.build(
            Producer.COVERAGE,
            Stage.COVERAGE,
            "cross-map-reference-unsatisfied",
            "Observation.subject is deferred, but no map in this set produces a Patient.",
            owner=ActionOwner.MAPPING_INPUT_REQUIRED,
            map_url="http://example.org/StructureMap/a",
            map_id="sm-a",
            path="Observation.subject",
        )
    ]


def test_a_mapping_input_required_global_finding_makes_no_call_and_blocks_the_project():
    documents = {"a": structure_map("a")}
    evaluators = {documents["a"]["id"]: Evaluator(clean_report(documents["a"]))}
    proposer = Proposer()

    result = run(
        runtime(documents, evaluators, proposer, validators=[unroutable_global])
    )

    assert proposer.calls == 0
    assert result.outcome == PROJECT_UNRESOLVED
    assert result.globally_admissible is False
    assert result.routing == {}
    # One round only: nothing was routable, so asking again would ask nobody.
    assert len(result.rounds) == 1


def test_a_preexisting_global_blocker_does_not_make_the_staged_set_regress():
    finding = unroutable_global([])[0]
    result = ProjectRun(
        run_id="run",
        outcome=PROJECT_UNRESOLVED,
        stop_reason="still unresolved",
        global_findings=[finding],
        baseline_global_findings=[finding],
    )

    assert result.blocking_global == [finding]
    assert result.global_regressions == []
    assert result.globally_admissible is True


def test_a_new_global_blocker_still_refuses_application():
    finding = unroutable_global([])[0]
    result = ProjectRun(
        run_id="run",
        outcome=PROJECT_UNRESOLVED,
        stop_reason="regressed",
        global_findings=[finding],
    )

    assert result.global_regressions == [finding]
    assert result.globally_admissible is False


def test_a_map_fixable_global_finding_requeues_only_the_map_it_names():
    # Map `a` is fine on its own — only the assembled set shows the problem, and
    # only a second round can show that the repair worked.
    documents = {"a": structure_map("a", element="nonsense"), "b": structure_map("b")}
    evaluators = {doc["id"]: Evaluator(clean_report(doc)) for doc in documents.values()}
    proposer = Proposer()
    seen = {"rounds": 0}

    def validator(_assembled):
        seen["rounds"] += 1
        if seen["rounds"] > 1:
            return []
        return [
            ValidationFinding.build(
                Producer.COVERAGE,
                Stage.COVERAGE,
                "target-path-not-found",
                "The assembled set shows map a writes an element it should not.",
                owner=ActionOwner.MAP_FIXABLE,
                map_url="http://example.org/StructureMap/a",
                map_id="sm-a",
                path="Patient.nonsense",
                pointer="/group/0/rule/0/target/0",
            )
        ]

    result = run(runtime(documents, evaluators, proposer, validators=[validator]))

    assert len(result.rounds) == 2
    # Round one routed to `a` and to nothing else; round two dispatched only it.
    assert list(result.rounds[0]["routed"]) == ["a"]
    assert result.rounds[1]["dispatched"] == ["a"]
    assert result.results["a"].requeued_for
    # The second round's validation is clean, so the project settles.
    assert result.rounds[1]["routed"] == {}
    assert result.global_findings == []
    assert result.outcome == PROJECT_OK


def test_a_global_finding_naming_no_map_in_the_run_is_reported_not_routed():
    documents = {"a": structure_map("a")}
    evaluators = {documents["a"]["id"]: Evaluator(clean_report(documents["a"]))}

    def validator(_assembled):
        return [
            ValidationFinding.build(
                Producer.COVERAGE,
                Stage.COVERAGE,
                "target-path-not-found",
                "Something about a map that is not in this run.",
                owner=ActionOwner.MAP_FIXABLE,
                map_url="http://example.org/StructureMap/elsewhere",
                map_id="sm-elsewhere",
            )
        ]

    result = run(runtime(documents, evaluators, Proposer(), validators=[validator]))

    assert result.routing == {}
    assert len(result.rounds) == 1
    assert result.outcome == PROJECT_UNRESOLVED


# --- termination ---------------------------------------------------------------


def persistent_fixable(_assembled):
    return [
        ValidationFinding.build(
            Producer.COVERAGE,
            Stage.COVERAGE,
            "target-path-not-found",
            "Still wrong.",
            owner=ActionOwner.MAP_FIXABLE,
            map_url="http://example.org/StructureMap/a",
            map_id="sm-a",
            path="Patient.nonsense",
            pointer="/group/0/rule/0/target/0",
        )
    ]


def test_a_repeated_project_state_stops_the_loop_as_no_progress():
    """Same assembled set, same blocking findings: another round asks the same
    maps the same question."""

    documents = {"a": structure_map("a")}
    evaluators = {documents["a"]["id"]: Evaluator(clean_report(documents["a"]))}

    result = run(
        runtime(
            documents,
            evaluators,
            Proposer(),
            validators=[persistent_fixable],
            project_limits=ProjectLimits(max_project_rounds=9),
        )
    )

    assert result.outcome == PROJECT_NO_PROGRESS
    assert len(result.rounds) == 2


def test_the_project_round_budget_is_a_hard_stop():
    documents = {"a": structure_map("a")}
    calls = {"n": 0}

    def churning(_assembled):
        # A different finding id each round, so no-progress cannot fire and the
        # round budget is the only thing left to stop it.
        calls["n"] += 1
        return [
            ValidationFinding.build(
                Producer.COVERAGE,
                Stage.COVERAGE,
                "target-path-not-found",
                f"Round {calls['n']} disagrees.",
                owner=ActionOwner.MAP_FIXABLE,
                map_url="http://example.org/StructureMap/a",
                map_id="sm-a",
                path=f"Patient.round{calls['n']}",
                pointer="/group/0/rule/0/target/0",
            )
        ]

    evaluators = {documents["a"]["id"]: Evaluator(clean_report(documents["a"]))}
    result = run(
        runtime(
            documents,
            evaluators,
            Proposer(),
            validators=[churning],
            project_limits=ProjectLimits(max_project_rounds=2),
        )
    )

    assert result.outcome == PROJECT_LIMIT_REACHED
    assert len(result.rounds) == 2


def test_requeueing_does_not_reset_a_map_s_attempt_budget():
    """One attempt allowed in total, not one per round.

    The map is requeued a second time with its budget already spent, so the gate
    stops it before the provider is reached: one call, not two.
    """

    documents = {"a": structure_map("a", element="nonsense")}
    # Every candidate keeps the finding, so nothing is ever accepted.
    stuck = broken_report(documents["a"], fixable_finding())
    evaluators = {documents["a"]["id"]: Evaluator(stuck, stuck)}
    proposer = Proposer()

    result = run(
        runtime(
            documents,
            evaluators,
            proposer,
            validators=[persistent_fixable],
            limits=LoopLimits(max_attempts=1),
            project_limits=ProjectLimits(max_project_rounds=3),
        )
    )

    entry = result.results["a"]
    assert len(result.rounds) == 2
    assert entry.round == 2
    assert entry.attempts_used == 1
    assert entry.result.outcome is LoopOutcome.EXHAUSTED
    assert proposer.calls == 1


def test_a_clean_map_does_not_reserve_the_budget_the_broken_one_needs():
    """The other half of the budget property, and the one a reservation breaks.

    Reserving each map's *potential* allowance before knowing whether it needs a
    provider at all means a clean map — which will make no call — can consume
    the whole project budget on paper. Dispatching one map at a time makes the
    remaining budget a fact rather than an estimate.
    """

    documents = {
        # Dispatched first, and clean: it must reserve nothing.
        "a": structure_map("a"),
        "b": structure_map("b", element="nonsense"),
    }
    evaluators = {
        documents["a"]["id"]: Evaluator(clean_report(documents["a"])),
        documents["b"]["id"]: Evaluator(
            broken_report(documents["b"], fixable_finding()),
            clean_report(documents["b"]),
        ),
    }
    proposer = Proposer()

    result = run(
        runtime(
            documents,
            evaluators,
            proposer,
            limits=LoopLimits(max_attempts=1),
            project_limits=ProjectLimits(max_provider_calls=1),
        )
    )

    assert proposer.calls == 1
    assert result.results["b"].result.outcome is LoopOutcome.ACCEPTED
    assert result.outcome == PROJECT_OK


def test_two_broken_maps_cannot_together_exceed_a_single_call_project_budget():
    """The budget is reserved before the round runs, not sampled during it.

    Every payload for a round is built before any of them executes, so a
    snapshot of "what the project has spent so far" gives both maps the same
    allowance and the round spends twice the budget — with `max_concurrency=1`
    making no difference, because the payloads were already constructed.
    """

    documents = {
        "a": structure_map("a", element="nonsense"),
        "b": structure_map("b", element="nonsense"),
    }
    evaluators = {
        doc["id"]: Evaluator(
            broken_report(doc, fixable_finding()), broken_report(doc, fixable_finding())
        )
        for doc in documents.values()
    }
    proposer = Proposer()

    result = run(
        runtime(
            documents,
            evaluators,
            proposer,
            limits=LoopLimits(max_attempts=1),
            project_limits=ProjectLimits(max_provider_calls=1),
        )
    )

    assert proposer.calls == 1
    # The second map is stopped by its (empty) share, not by its own budget.
    stopped = [
        entry
        for entry in result.ordered
        if entry.result.outcome is LoopOutcome.LIMIT_REACHED
    ]
    assert len(stopped) == 1
    assert "share of the project provider budget" in stopped[0].result.stop_reason


def test_the_project_provider_budget_stops_a_later_round():
    documents = {"a": structure_map("a", element="nonsense")}
    stuck = broken_report(documents["a"], fixable_finding())
    evaluators = {documents["a"]["id"]: Evaluator(stuck, stuck)}
    proposer = Proposer()

    result = run(
        runtime(
            documents,
            evaluators,
            proposer,
            validators=[persistent_fixable],
            limits=LoopLimits(max_attempts=1),
            project_limits=ProjectLimits(max_project_rounds=4, max_provider_calls=1),
        )
    )

    assert proposer.calls == 1
    assert result.outcome in {PROJECT_LIMIT_REACHED, PROJECT_NO_PROGRESS}


def test_a_requeued_map_keeps_its_attempt_records_and_accumulated_time():
    """The report is an audit trail of the map, not of its last round."""

    documents = {"a": structure_map("a", element="nonsense")}
    stuck = broken_report(documents["a"], fixable_finding())
    evaluators = {documents["a"]["id"]: Evaluator(stuck, stuck)}

    result = run(
        runtime(
            documents,
            evaluators,
            Proposer(),
            validators=[persistent_fixable],
            limits=LoopLimits(max_attempts=4),
            project_limits=ProjectLimits(max_project_rounds=3),
        )
    )

    entry = result.results["a"]
    assert len(result.rounds) == 2
    # Both rounds' attempts are present, and their indices continue.
    assert [record.index for record in entry.result.attempts] == [1, 2]
    assert entry.attempts_used == 2
    # Rejection feedback and the no-progress evidence crossed the boundary too,
    # so `max_seconds` is cumulative rather than restarting every round.
    assert len(entry.rejected) >= 1
    assert len(entry.seen_candidates) >= 2
    assert entry.result.elapsed_s > 0


# --- acceptance authority -------------------------------------------------------


def test_a_project_validator_cannot_make_a_candidate_acceptable():
    """No node and no validator may accept anything. WP5/WP6 evidence does."""

    documents = {"a": structure_map("a", element="nonsense")}
    stuck = broken_report(documents["a"], fixable_finding())
    evaluators = {documents["a"]["id"]: Evaluator(stuck, stuck)}

    def approving_validator(_assembled):
        return []  # says the project is fine

    result = run(
        runtime(
            documents,
            evaluators,
            Proposer(),
            validators=[approving_validator],
            limits=LoopLimits(max_attempts=1),
        )
    )

    assert result.results["a"].result.outcome is not LoopOutcome.ACCEPTED
    assert result.results["a"].changed is False
    # No blocking global finding, but the map never settled, so the project
    # is unresolved rather than ok.
    assert result.outcome == PROJECT_UNRESOLVED


def test_the_graph_compiles_with_the_documented_nodes():
    graph = build_project_graph().get_graph()
    assert {
        "load_project",
        "begin_round",
        "repair_map",
        "assemble",
        "validate_project",
        "route_global",
        "finalize",
    } <= {node for node in graph.nodes}
