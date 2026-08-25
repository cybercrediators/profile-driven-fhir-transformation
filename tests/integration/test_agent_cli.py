"""Agent mode end to end: project, engine, loop, artifacts, apply (WP7/WP8).

The unit suites prove each piece against stubs. This one wires a real project
directory to a real Matchbox and drives the whole path — baseline validation
finds a genuine defect, a scripted "model" answers with a patch, WP5 applies it,
WP6 accepts it, and WP8 writes the artifacts and replaces the map.

The model is scripted rather than real. A live provider would make this
non-deterministic and would test the provider, not the pipeline; what needs
proving here is that the deterministic machinery around it is wired correctly.
The engine, by contrast, is real — nothing else can prove Matchbox actually
accepts the repaired map.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from agent.graph import AgentRuntimeContext
from agent.graph.repair_graph import run_repair_graph
from agent.loop import LoopLimits, LoopOutcome, MapRunResult, Proposal
from agent.models import AgentPatch, operation
from agent.patch import canonical_sha256
from agent.reporting import RunDirectory, apply_project, build_report, write_run
from agent.service import AgentFixService, AgentSetupError, build_project_context

pytestmark = pytest.mark.integration

_MATCHBOX_BASE = os.environ.get(
    "MATCHBOX_URL", "http://localhost:8080/matchboxv3"
).rstrip("/")

_SOURCE_URL = "http://example.org/StructureDefinition/agent-cli-source"
_PROFILE_URL = "http://example.org/StructureDefinition/AgentCliPatient"
_MAP_URL = "http://example.org/StructureMap/agent-cli"


def _matchbox_available() -> bool:
    try:
        with urllib.request.urlopen(
            f"{_MATCHBOX_BASE}/fhir/metadata", timeout=3
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


# --- the project on disk ------------------------------------------------------


def source_model():
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
        "id": "agent-cli-source",
        "url": _SOURCE_URL,
        "name": "AgentCliSource",
        "status": "draft",
        "kind": "logical",
        "abstract": False,
        "type": "AgentCliSource",
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Element",
        "derivation": "specialization",
        "differential": {
            "element": [
                {
                    "id": "AgentCliSource",
                    "path": "AgentCliSource",
                    "min": 0,
                    "max": "*",
                    "type": [{"code": "Element"}],
                },
                element("AgentCliSource.givenName", 1, "string"),
                element("AgentCliSource.familyName", 0, "string"),
            ]
        },
    }


def profile():
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
        "id": "AgentCliPatient",
        "url": _PROFILE_URL,
        "name": "AgentCliPatient",
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
            ]
        },
    }


def broken_map():
    """A map whose leaf rule writes an element ``HumanName`` does not have.

    A real defect: the offline path resolver rejects it and Matchbox refuses to
    transform it, so the loop has genuine evidence on both sides.
    """

    return {
        "resourceType": "StructureMap",
        "id": "agent-cli",
        "url": _MAP_URL,
        "name": "AgentCli",
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
                    {"name": "source", "type": "AgentCliSource", "mode": "source"},
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
                                        "element": "notAnElement",
                                        "transform": "copy",
                                        "parameter": [{"valueId": "f"}],
                                    }
                                ],
                            },
                            {
                                "name": "map-given",
                                "source": [
                                    {
                                        "context": "source",
                                        "element": "givenName",
                                        "variable": "g",
                                    }
                                ],
                                "target": [
                                    {
                                        "context": "tgt-name",
                                        "contextType": "variable",
                                        "element": "given",
                                        "transform": "copy",
                                        "parameter": [{"valueId": "g"}],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }


MAPPING_TABLE = {
    "givenName": "Patient.name.given",
    "familyName": "Patient.name.family",
}

BROKEN_POINTER = "/group/0/rule/0/rule/0/target/0/element"


@pytest.fixture
def project(tmp_path):
    if not _matchbox_available():
        pytest.skip(
            f"Matchbox is not reachable at {_MATCHBOX_BASE}; set MATCHBOX_URL to run "
            "the end-to-end agent checks"
        )

    root = tmp_path / "agent_cli_project"
    for folder in (
        "structure_maps",
        "input_profile",
        "source_definitions",
        "source_data",
        "examples",
        "processed_resources",
        "source_data/concept_maps",
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)

    (root / "structure_maps" / "001_agent_cli.json").write_text(
        json.dumps(broken_map(), indent=2), encoding="utf-8"
    )
    (root / "input_profile" / "StructureDefinition-AgentCliPatient.json").write_text(
        json.dumps(profile(), indent=2), encoding="utf-8"
    )
    (root / "source_definitions" / "agent-cli-source.json").write_text(
        json.dumps(source_model(), indent=2), encoding="utf-8"
    )
    (root / "source_data" / "mapping_table.json").write_text(
        json.dumps(MAPPING_TABLE, indent=2), encoding="utf-8"
    )
    return root


@pytest.fixture
def conf(project):
    return {
        "project_path": str(project),
        "profile_path": str(project / "input_profile"),
        "matchbox_connection": {"url": _MATCHBOX_BASE},
        "llm": {"model": "scripted", "api_key_env": "AGENT_CLI_TEST_KEY"},
    }


# --- the scripted model -------------------------------------------------------


class ScriptedProposer:
    """Answers with a fixed sequence, recording the contexts it was given."""

    def __init__(self, *builders):
        self.builders = list(builders)
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if not self.builders:
            return Proposal(patch=None, error="nothing scripted")
        return self.builders.pop(0)(context)


def _patch(context, ops, rationale="repair"):
    return Proposal(
        patch=AgentPatch(
            schema_version=1,
            map_url=context.map_url,
            map_id=context.map_id,
            base_sha256=context.map_sha256,
            diagnostic_ids=[finding.finding_id for finding in context.findings],
            patch=ops,
            rationale=rationale,
        ),
        prompt_chars=2000,
    )


def correct_repair(context):
    return _patch(
        context,
        [
            operation("test", BROKEN_POINTER, "notAnElement"),
            operation("replace", BROKEN_POINTER, "family"),
        ],
    )


def unguarded_repair(context):
    """Missing the mandatory `test` guard — WP5 rejects it before application."""

    return _patch(context, [operation("replace", BROKEN_POINTER, "family")])


def service_with(conf, proposer, **kwargs):
    return AgentFixService(
        project=build_project_context(conf),
        proposer=proposer,
        limits=LoopLimits(max_attempts=kwargs.pop("max_attempts", 3)),
        **kwargs,
    )


MAP_KEY = "001_agent_cli"


def fix(service, document):
    """Repair one map through the WP10 subgraph, against the real engine."""

    path, _document = service.project.select(None)
    context = AgentRuntimeContext(
        service=service,
        documents={MAP_KEY: (path, dict(document))},
        limits=service.limits,
        require_engine=service.require_engine,
    )
    return run_repair_graph(
        MAP_KEY,
        context=context,
        generator_findings=service.generator_findings(document),
    )


def apply_one(path, document, result, run_dir, *, offline=False):
    """The project transaction over a project of exactly one map."""

    item = MapRunResult(
        map_key=MAP_KEY,
        map_path=str(path),
        result=result,
        baseline=dict(document),
        origin_sha256=canonical_sha256(dict(document)),
        staged_candidate=result.accepted_candidate,
        staged_sha256=result.accepted_sha256,
    )
    return apply_project(
        [item], map_dirs={MAP_KEY: run_dir}, offline=offline
    )


# --- the checks ---------------------------------------------------------------


def test_the_project_context_resolves_the_project(conf, project):
    context = build_project_context(conf)
    assert len(context.structure_maps) == 1
    assert context.source_model["url"] == _SOURCE_URL
    assert _PROFILE_URL in context.profiles
    assert context.mapping_table == MAPPING_TABLE
    assert context.matchbox_controller is not None


def test_selecting_a_map_accepts_a_path_url_or_id(conf):
    context = build_project_context(conf)
    path, document = context.structure_maps[0]
    for selector in (str(path), path.name, _MAP_URL, "agent-cli"):
        assert context.select(selector)[1]["url"] == _MAP_URL
    with pytest.raises(AgentSetupError, match="No StructureMap"):
        context.select("no-such-map")


def test_a_broken_map_is_repaired_validated_and_accepted(conf):
    proposer = ScriptedProposer(correct_repair)
    service = service_with(conf, proposer)
    _path, document = service.project.select(None)

    result = fix(service, document)

    assert result.outcome is LoopOutcome.ACCEPTED, result.stop_reason
    element = result.accepted_candidate["group"][0]["rule"][0]["rule"][0]["target"][0]
    assert element["element"] == "family"
    # The baseline really was broken on both sides, which is what makes the
    # acceptance meaningful rather than vacuous.
    baseline_codes = {f.code for f in result.baseline_report.findings}
    assert "target-path-not-found" in baseline_codes
    assert any(code.startswith("transform:") for code in baseline_codes)


def test_the_prompt_context_offers_the_real_findings(conf):
    proposer = ScriptedProposer(correct_repair)
    service = service_with(conf, proposer)
    _path, document = service.project.select(None)
    fix(service, document)

    context = proposer.contexts[0]
    assert context.resource_type == "Patient"
    assert context.source_type == "AgentCliSource"
    assert context.map_sha256 == canonical_sha256(document)
    assert "target-path-not-found" in {f.code for f in context.findings}
    focused = [rule.pointer for rule in context.pointers if rule.focused]
    assert "/group/0/rule/0/rule/0" in focused
    assert {row["source"] for row in context.mapping_rows} == set(MAPPING_TABLE)


def test_a_rejected_patch_is_retried_with_feedback_and_then_accepted(conf):
    proposer = ScriptedProposer(unguarded_repair, correct_repair)
    service = service_with(conf, proposer)
    _path, document = service.project.select(None)

    result = fix(service, document)

    assert result.outcome is LoopOutcome.ACCEPTED
    assert len(result.attempts) == 2
    assert result.attempts[0].application["applied"] is False
    assert "unguarded-mutation" in {
        rejection["code"] for rejection in result.attempts[0].application["rejections"]
    }
    # The second prompt has to carry the reason, not just repeat the question.
    assert proposer.contexts[1].attempts[0].rejections[0]["code"] == "unguarded-mutation"


def test_a_clean_map_makes_no_proposal(conf, project):
    repaired = broken_map()
    repaired["group"][0]["rule"][0]["rule"][0]["target"][0]["element"] = "family"
    (project / "structure_maps" / "001_agent_cli.json").write_text(
        json.dumps(repaired, indent=2), encoding="utf-8"
    )
    proposer = ScriptedProposer()
    service = service_with(conf, proposer)
    _path, document = service.project.select(None)

    result = fix(service, document)

    assert result.outcome is LoopOutcome.CLEAN, result.stop_reason
    assert proposer.contexts == []
    assert result.provider_calls == 0


def test_an_offline_run_produces_a_candidate_that_cannot_be_applied(conf, tmp_path):
    proposer = ScriptedProposer(correct_repair)
    service = service_with(conf, proposer, require_engine=False)
    path, document = service.project.select(None)

    result = fix(service, document)
    assert result.outcome is LoopOutcome.ACCEPTED

    run_dir = RunDirectory.create(tmp_path)
    transaction = apply_one(path, document, result, run_dir, offline=True)
    assert "offline" in transaction["refused_because"]
    assert json.loads(path.read_text())["group"][0]["rule"][0]["rule"][0]["target"][0][
        "element"
    ] == "notAnElement"


def test_a_full_run_writes_artifacts_and_leaves_the_map_alone(conf, project):
    path = project / "structure_maps" / "001_agent_cli.json"
    before = path.read_bytes()

    service = service_with(conf, ScriptedProposer(unguarded_repair, correct_repair))
    _path, document = service.project.select(None)
    result = fix(service, document)

    run_dir = RunDirectory.create(project)
    artifacts = write_run(run_dir, document=document, result=result)
    report = build_report(
        command="prog agent fix",
        conf=conf,
        result=result,
        map_path=path,
        run_dir=run_dir,
        coverage_report=service.project.coverage_report,
        artifacts=artifacts,
    )
    report_path = run_dir.write_report(report)

    assert path.read_bytes() == before  # no --apply, no write
    assert (run_dir.root / "patches" / "attempt-001.json").is_file()
    assert (run_dir.root / "patches" / "attempt-002.json").is_file()
    assert (run_dir.root / "validation" / "attempt-002.json").is_file()
    assert (run_dir.root / "accepted_candidate.json").is_file()

    written = json.loads(report_path.read_text())
    assert written["outcome"] == "accepted"
    assert written["map"]["baseline_sha256"] != written["map"]["candidate_sha256"]
    assert written["baseline_findings"]["offline"]
    assert written["baseline_findings"]["engine_transform"]
    assert written["apply"]["applied"] is False
    # The engine had to have run for this candidate to be acceptable at all.
    assert any(
        item["name"] == "engine-available" and item["passed"]
        for item in written["acceptance"]["invariants"]
    )


def test_apply_replaces_the_map_and_keeps_the_baseline(conf, project):
    path = project / "structure_maps" / "001_agent_cli.json"
    service = service_with(conf, ScriptedProposer(correct_repair))
    _path, document = service.project.select(None)
    result = fix(service, document)

    run_dir = RunDirectory.create(project)
    write_run(run_dir, document=document, result=result)
    assert apply_one(path, document, result, run_dir)["applied"] is True

    applied = json.loads(path.read_text())
    assert applied["group"][0]["rule"][0]["rule"][0]["target"][0]["element"] == "family"
    preserved = json.loads((run_dir.root / "baseline_structure_map.json").read_text())
    assert (
        preserved["group"][0]["rule"][0]["rule"][0]["target"][0]["element"]
        == "notAnElement"
    )


def test_the_applied_map_is_itself_clean(conf, project):
    # The candidate was accepted comparatively; this asserts the stronger thing
    # a user actually wants, that re-validating the written file finds nothing.
    path = project / "structure_maps" / "001_agent_cli.json"
    service = service_with(conf, ScriptedProposer(correct_repair))
    _path, document = service.project.select(None)
    result = fix(service, document)
    apply_one(path, document, result, RunDirectory.create(project))

    fresh = service_with(conf, ScriptedProposer())
    _path2, reloaded = fresh.project.select(None)
    second = fix(fresh, reloaded)
    assert second.outcome is LoopOutcome.CLEAN, second.stop_reason
