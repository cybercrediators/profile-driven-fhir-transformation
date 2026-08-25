"""Interrupting a run and continuing it (WP10.4).

The property that matters is narrow and expensive to get wrong: **a resumed run
must not pay for an answer it already has.** A checkpoint restores control
state; the content-addressed LLM cache makes re-entering a node safe. Between
them, the provider is asked once per attempt no matter how many times the
process dies.

The second property is the refusal. A checkpoint says nothing about the files it
was computed against, so resuming into an edited project would apply yesterday's
decisions to today's maps. Every input the decisions depended on is fingerprinted
in ``run_manifest.json`` and compared before the graph is re-entered.

No external service is involved: the checkpointer is a SQLite file inside the
run directory, and the provider is a stub.
"""

from argparse import Namespace
import json
from pathlib import Path

import pytest

from agent.cli import EXIT_OK, EXIT_SETUP_ERROR, run_agent_fix
from agent.loop import LoopLimits, Proposal
from agent.models import AgentPatch, operation
from agent.patch import canonical_sha256
from agent.validation import (
    Producer,
    Stage,
    ValidationFinding,
    ValidationReport,
)

pytestmark = pytest.mark.integration

POINTER = "/group/0/rule/0/target/0/element"


# --- corpus ---------------------------------------------------------------------


def structure_map(index, element="nonsense"):
    return {
        "resourceType": "StructureMap",
        "id": f"sm-{index}",
        "url": f"http://example.org/StructureMap/sm-{index}",
        "name": f"Sm{index}",
        "status": "draft",
        "structure": [
            {"url": "http://example.org/StructureDefinition/src", "mode": "source"},
            {"url": "http://example.org/StructureDefinition/tgt", "mode": "target"},
        ],
        "group": [
            {
                "name": "TransformOne",
                "typeMode": "none",
                "input": [
                    {"name": "source", "type": "Src", "mode": "source"},
                    {"name": "target", "type": "Patient", "mode": "target"},
                ],
                "rule": [
                    {
                        "name": "map-value",
                        "source": [
                            {"context": "source", "element": "v", "variable": "v"}
                        ],
                        "target": [
                            {
                                "context": "target",
                                "contextType": "variable",
                                "element": element,
                                "transform": "copy",
                                "parameter": [{"valueId": "v"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def finding():
    return ValidationFinding.build(
        Producer.PATH_RESOLUTION,
        Stage.PATHS,
        "target-path-not-found",
        "The profile has no element Patient.nonsense.",
        path="Patient.nonsense",
        pointer="/group/0/rule/0/target/0",
    )


def report(document, findings=()):
    return ValidationReport(
        map_url=document.get("url"),
        map_id=document.get("id"),
        map_sha256=canonical_sha256(dict(document)),
        engine_available=True,
        engine_requested=True,
        executed_fixtures=["filled"],
        required_fixtures=["filled"],
        validated_fixtures=["filled"],
        evaluation_context_sha256="shared",
        findings=list(findings),
    )


# --- stubs ------------------------------------------------------------------------


class Evaluator:
    """Broken until repaired — and able to fail *once*, after the answer is in."""

    def __init__(self, document, crash_on_pair=False):
        self.document = document
        self.baseline_digest = canonical_sha256(dict(document))
        self.crash_on_pair = crash_on_pair
        self.target_tree = None
        self.source_field_specs = []
        self.profile_url = None

    def _for(self, document):
        digest = canonical_sha256(dict(document))
        if digest == self.baseline_digest:
            return report(document, [finding()])
        return report(document)

    def evaluate(self, document):
        return self._for(document)

    def evaluate_pair(self, baseline, candidate):
        if self.crash_on_pair:
            # The provider has already answered and its answer is checkpointed.
            self.crash_on_pair = False
            raise RuntimeError("the process died while judging the candidate")
        return self._for(baseline), self._for(candidate)


class Proposer:
    def __init__(self):
        self.calls = 0

    def propose(self, context):
        self.calls += 1
        return Proposal(
            patch=AgentPatch(
                schema_version=1,
                map_url=context.map_url,
                map_id=context.map_id,
                base_sha256=context.map_sha256,
                diagnostic_ids=[f.finding_id for f in context.findings],
                patch=[
                    operation("test", POINTER, "nonsense"),
                    operation("replace", POINTER, "family"),
                ],
                rationale="use the element the profile has",
            ),
            prompt_chars=700,
        )


class Project:
    def __init__(self, project_dir, maps):
        self.project_dir = project_dir
        self.coverage_report = None
        self.mapping_table = {}
        self.structure_maps = maps
        self.profiles = {}

    def select(self, selector):
        return self.structure_maps[0]

    def profile_for(self, _document):
        return None


class Service:
    def __init__(self, project, proposer, evaluators):
        self.project = project
        self.proposer = proposer
        self.limits = LoopLimits(max_attempts=2)
        self.require_engine = True
        self.llm_config = {"provider": "openai-compatible", "model": "stub"}
        self._evaluators = evaluators

    def targets(self, selector=None, *, all_maps=False):
        entries = (
            list(self.project.structure_maps)
            if all_maps
            else [self.project.select(selector)]
        )
        return {Path(p).stem: (Path(p), d) for p, d in entries}

    def evaluator_for(self, document):
        return self._evaluators[document["id"]]

    def generator_findings(self, _document):
        return []


@pytest.fixture
def project(tmp_path):
    maps = []
    for index in (1, 2):
        path = tmp_path / f"{index:03d}_map.json"
        document = structure_map(index)
        path.write_text(json.dumps(document), encoding="utf-8")
        maps.append((path, document))
    return Project(tmp_path, maps)


def namespace(tmp_path, **overrides):
    args = {
        "map": None,
        "all_maps": True,
        "max_attempts": 2,
        "max_project_rounds": 2,
        "resume": None,
        "output_dir": str(tmp_path),
        "offline": False,
        "use_examples": False,
        "apply": False,
        "llm_provider": None,
        "llm_model": None,
        "llm_base_url": None,
    }
    args.update(overrides)
    return Namespace(**args)


def _stub_coverage_report():
    """A regenerated coverage report — the input the agent worklist comes from."""

    from mapping.generation_result import CoverageReport

    return CoverageReport.from_raw(
        {
            "note": "regenerated",
            "summary": {
                "profiles": 1,
                "required_total": 2,
                "required_mapped": 1,
                "required_unmapped": 1,
                "static_required_total": 2,
                "latent_required_total": 0,
                "required_coverage_pct": 50.0,
            },
            "profiles": {},
            "mapping_diagnostics": [],
        }
    )


def install(monkeypatch, service):
    monkeypatch.setattr(
        "agent.service.AgentFixService.create", lambda conf, **kwargs: service
    )


# --- the checks ---------------------------------------------------------------------


def test_resuming_after_a_crash_does_not_pay_for_the_answer_twice(
    tmp_path, project, monkeypatch
):
    evaluators = {
        project.structure_maps[0][1]["id"]: Evaluator(project.structure_maps[0][1]),
        # The second map dies after its provider call, while its candidate is
        # being judged.
        project.structure_maps[1][1]["id"]: Evaluator(
            project.structure_maps[1][1], crash_on_pair=True
        ),
    }
    proposer = Proposer()
    install(monkeypatch, Service(project, proposer, evaluators))
    conf = {"project_path": str(tmp_path)}

    with pytest.raises(RuntimeError, match="died while judging"):
        run_agent_fix(namespace(tmp_path), conf)

    run_id = next((tmp_path / "agent_output").iterdir()).name
    calls_before_resume = proposer.calls
    assert calls_before_resume == 2  # one per map, both already answered

    exit_code = run_agent_fix(namespace(tmp_path, resume=run_id), conf)

    assert exit_code == EXIT_OK
    # The heart of it: no second call for either map. The first map was not
    # re-run at all, and the second re-entered only the node that failed.
    assert proposer.calls == calls_before_resume
    report_path = tmp_path / "agent_output" / run_id / "project_report.json"
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["graph"]["resumed"] is True
    assert written["totals"]["accepted"] == 2
    assert written["totals"]["provider_calls"] == 2


def test_a_resumed_run_reuses_the_same_run_directory_and_manifest(
    tmp_path, project, monkeypatch
):
    evaluators = {
        document["id"]: Evaluator(document) for _path, document in project.structure_maps
    }
    install(monkeypatch, Service(project, Proposer(), evaluators))
    conf = {"project_path": str(tmp_path)}

    assert run_agent_fix(namespace(tmp_path), conf) == EXIT_OK
    run_id = next((tmp_path / "agent_output").iterdir()).name
    manifest = json.loads(
        (tmp_path / "agent_output" / run_id / "run_manifest.json").read_text()
    )

    assert run_agent_fix(namespace(tmp_path, resume=run_id), conf) == EXIT_OK

    # No second run directory, and the manifest is the one the run started with.
    assert [p.name for p in (tmp_path / "agent_output").iterdir()] == [run_id]
    again = json.loads(
        (tmp_path / "agent_output" / run_id / "run_manifest.json").read_text()
    )
    assert again == manifest
    assert manifest["checkpointer"]["kind"] == "sqlite"
    assert {entry["map_key"] for entry in manifest["maps"]} == {"001_map", "002_map"}


def test_resume_refuses_when_the_provider_changed(tmp_path, project, monkeypatch):
    evaluators = {
        document["id"]: Evaluator(document) for _path, document in project.structure_maps
    }
    service = Service(project, Proposer(), evaluators)
    install(monkeypatch, service)
    conf = {"project_path": str(tmp_path)}

    assert run_agent_fix(namespace(tmp_path), conf) == EXIT_OK
    run_id = next((tmp_path / "agent_output").iterdir()).name

    service.llm_config = {"provider": "openai-compatible", "model": "a-different-model"}

    assert (
        run_agent_fix(namespace(tmp_path, resume=run_id), conf) == EXIT_SETUP_ERROR
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(
            lambda service: service.project.profiles.__setitem__("p", {"changed": 1}),
            "profiles changed",
            id="a profile was edited",
        ),
        pytest.param(
            lambda service: service.project.mapping_table.__setitem__("f", "Patient.x"),
            "mapping table changed",
            id="a mapping-table row was added",
        ),
        pytest.param(
            lambda service: service.llm_config.__setitem__("temperature", 0.9),
            "'temperature' changed",
            id="the sampling temperature moved",
        ),
        pytest.param(
            lambda service: service.llm_config.__setitem__("seed", 1234),
            "'seed' changed",
            id="the seed moved",
        ),
        pytest.param(
            lambda service: service.llm_config.__setitem__("cache_enabled", False),
            "'cache_enabled' changed",
            id="the response cache was turned off",
        ),
        pytest.param(
            lambda service: setattr(
                service.project, "coverage_report", _stub_coverage_report()
            ),
            "coverage report changed",
            id="the generator's coverage report was regenerated",
        ),
    ],
)
def test_resume_refuses_when_any_decision_input_changed(
    tmp_path, project, monkeypatch, mutate, expected, caplog
):
    """The redacted configuration digest is not an execution identity.

    It is an allow-list built for publication — six keys, everything else
    hidden. A profile edit, a new mapping-table row, or a different temperature
    all change what the run would decide while leaving it untouched, so the
    manifest fingerprints them separately.
    """

    evaluators = {
        document["id"]: Evaluator(document) for _path, document in project.structure_maps
    }
    service = Service(project, Proposer(), evaluators)
    install(monkeypatch, service)
    conf = {"project_path": str(tmp_path)}

    assert run_agent_fix(namespace(tmp_path), conf) == EXIT_OK
    run_id = next((tmp_path / "agent_output").iterdir()).name

    mutate(service)

    with caplog.at_level("ERROR"):
        exit_code = run_agent_fix(namespace(tmp_path, resume=run_id), conf)

    assert exit_code == EXIT_SETUP_ERROR
    assert expected in caplog.text


def test_resume_refuses_when_a_bound_was_tightened(tmp_path, project, monkeypatch, caplog):
    """A different attempt budget is a different run, not a continuation."""

    evaluators = {
        document["id"]: Evaluator(document) for _path, document in project.structure_maps
    }
    install(monkeypatch, Service(project, Proposer(), evaluators))
    conf = {"project_path": str(tmp_path)}

    assert run_agent_fix(namespace(tmp_path), conf) == EXIT_OK
    run_id = next((tmp_path / "agent_output").iterdir()).name

    with caplog.at_level("ERROR"):
        exit_code = run_agent_fix(
            namespace(tmp_path, resume=run_id, max_attempts=1), conf
        )

    assert exit_code == EXIT_SETUP_ERROR
    assert "bounds changed" in caplog.text


def test_the_manifest_records_the_inputs_it_compares(tmp_path, project, monkeypatch):
    evaluators = {
        document["id"]: Evaluator(document) for _path, document in project.structure_maps
    }
    install(monkeypatch, Service(project, Proposer(), evaluators))

    assert run_agent_fix(namespace(tmp_path), {"project_path": str(tmp_path)}) == EXIT_OK
    run_id = next((tmp_path / "agent_output").iterdir()).name
    manifest = json.loads(
        (tmp_path / "agent_output" / run_id / "run_manifest.json").read_text()
    )

    assert manifest["execution_sha256"]
    assert set(manifest["project_inputs"]) == {
        "profiles",
        "mapping_table",
        "concept_maps",
        "source_model",
        # The worklist itself is derived from this one.
        "coverage_report",
        "examples",
    }
    assert {
        "temperature",
        "seed",
        "max_output_tokens",
        "timeout_s",
        "cache_enabled",
    } <= set(manifest["provider"])
    # Namespaced: both dataclasses carry `max_provider_calls` and
    # `max_seconds`, and flattening them would hide a tightened per-map bound.
    assert {"max_attempts", "max_provider_calls", "max_seconds"} <= set(
        manifest["limits"]["map"]
    )
    assert {"max_project_rounds", "max_provider_calls", "max_seconds"} <= set(
        manifest["limits"]["project"]
    )
    assert manifest["limits"]["map"]["max_provider_calls"] != (
        manifest["limits"]["project"]["max_provider_calls"]
    )
    # A variable name, never a value.
    assert "api_key" not in manifest["provider"]
    assert json.dumps(manifest).count("sk-") == 0


def test_resume_refuses_a_run_directory_without_a_manifest(tmp_path, project, monkeypatch):
    evaluators = {
        document["id"]: Evaluator(document) for _path, document in project.structure_maps
    }
    install(monkeypatch, Service(project, Proposer(), evaluators))
    conf = {"project_path": str(tmp_path)}

    assert run_agent_fix(namespace(tmp_path), conf) == EXIT_OK
    run_id = next((tmp_path / "agent_output").iterdir()).name
    (tmp_path / "agent_output" / run_id / "run_manifest.json").unlink()

    assert (
        run_agent_fix(namespace(tmp_path, resume=run_id), conf) == EXIT_SETUP_ERROR
    )
