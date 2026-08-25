"""What ``agent fix`` returns and writes (WP8, WP10.3).

The command has no map loop any more: one map and every map are the same
project graph over a set of size one or size many. These tests drive
``run_agent_fix`` end to end against a stub provider and ask the questions a
caller depends on — the exit code, the artifact layout, and whether ``--apply``
distinguishes pre-existing project blockers from new regressions.
"""

from argparse import Namespace
import json
from pathlib import Path

import pytest

from agent.cli import EXIT_OK, EXIT_SETUP_ERROR, EXIT_UNRESOLVED, run_agent_fix
from agent.loop import LoopLimits, Proposal
from agent.patch import canonical_sha256
from agent.validation import ValidationReport

pytestmark = pytest.mark.unit


# --- corpus -------------------------------------------------------------------

PROFILE_URL = "http://example.org/StructureDefinition/ObsProfile"
PATIENT_URL = "http://example.org/StructureDefinition/PatientProfile"


def structure_map(index, *, target_url=PROFILE_URL, deferred=False):
    rules = [
        {
            "name": "map-value",
            "source": [{"context": "source", "element": "v", "variable": "v"}],
            "target": [
                {
                    "context": "target",
                    "contextType": "variable",
                    "element": "status",
                    "transform": "copy",
                    "parameter": [{"valueId": "v"}],
                }
            ],
        }
    ]
    if deferred:
        rules.append(
            {
                "name": "TODO-resolve-reference-Observation-subject",
                "source": [{"context": "source", "variable": "s"}],
                "target": [],
            }
        )
    return {
        "resourceType": "StructureMap",
        "id": f"sm-{index}",
        "url": f"http://example.org/StructureMap/sm-{index}",
        "name": f"Sm{index}",
        "status": "draft",
        "structure": [
            {"url": "http://example.org/StructureDefinition/src", "mode": "source"},
            {"url": target_url, "mode": "target"},
        ],
        "group": [
            {
                "name": "TransformOne",
                "typeMode": "none",
                "input": [
                    {"name": "source", "type": "Src", "mode": "source"},
                    {"name": "target", "type": "Observation", "mode": "target"},
                ],
                "rule": rules,
            }
        ],
    }


def observation_profile():
    return {
        "resourceType": "StructureDefinition",
        "id": "ObsProfile",
        "url": PROFILE_URL,
        "type": "Observation",
        "snapshot": {
            "element": [
                {"id": "Observation", "path": "Observation"},
                {
                    "id": "Observation.subject",
                    "path": "Observation.subject",
                    "min": 1,
                    "max": "1",
                    "type": [{"code": "Reference", "targetProfile": [PATIENT_URL]}],
                },
            ]
        },
    }


# --- stub service ---------------------------------------------------------------


class Evaluator:
    def __init__(self):
        self.target_tree = None
        self.source_field_specs = []
        self.profile_url = None

    def evaluate(self, document):
        return ValidationReport(
            map_url=document.get("url"),
            map_id=document.get("id"),
            map_sha256=canonical_sha256(dict(document)),
            engine_available=True,
            engine_requested=True,
        )

    def evaluate_pair(self, baseline, candidate):
        return self.evaluate(baseline), self.evaluate(candidate)


class Proposer:
    def __init__(self):
        self.calls = 0

    def propose(self, _context):
        self.calls += 1
        return Proposal(patch=None, error="the stub never proposes", fatal=True)


class Project:
    def __init__(self, project_dir, maps, profiles):
        self.project_dir = project_dir
        self.coverage_report = None
        self.mapping_table = {}
        self.structure_maps = maps
        self.profiles = profiles

    def select(self, selector):
        for path, document in self.structure_maps:
            if selector in (None, str(path), path.name, document["url"], document["id"]):
                return path, document
        raise AssertionError(f"no such map {selector!r}")

    def profile_for(self, document):
        for entry in document.get("structure") or []:
            if entry.get("mode") == "target":
                return self.profiles.get(entry.get("url"))
        return None


class Service:
    def __init__(self, project, proposer):
        self.project = project
        self.proposer = proposer
        self.limits = LoopLimits(max_attempts=1)
        self.require_engine = True
        self.llm_config = {
            "provider": "openai-compatible",
            "model": "resolved-model",
            "base_url": "https://user:secret@llm.example/v1?key=hidden",
            "api_key_env": "LLM_API_KEY",
        }
        self._evaluator = Evaluator()

    # The real service resolves keys the same way; reproduced rather than
    # imported so the stub stays a stub.
    def targets(self, selector=None, *, all_maps=False):
        entries = (
            list(self.project.structure_maps)
            if all_maps
            else [self.project.select(selector)]
        )
        return {Path(path).stem: (Path(path), document) for path, document in entries}

    def evaluator_for(self, _document):
        return self._evaluator

    def generator_findings(self, _document):
        return []


def build_project(tmp_path, documents, profiles=None):
    maps = []
    for index, document in enumerate(documents, start=1):
        path = tmp_path / f"{index:03d}_map.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        maps.append((path, document))
    return Project(tmp_path, maps, profiles or {})


def install(monkeypatch, service, captured=None):
    def create(conf, **kwargs):
        if captured is not None:
            captured["conf"] = conf
            captured["overrides"] = kwargs.get("overrides")
        return service

    monkeypatch.setattr("agent.service.AgentFixService.create", create)


def namespace(tmp_path, **overrides):
    args = {
        "map": None,
        "all_maps": False,
        "max_attempts": 1,
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


def only_run(tmp_path):
    return next((tmp_path / "agent_output").iterdir())


# --- the checks -----------------------------------------------------------------


def test_a_stray_llm_section_is_dropped_and_cli_overrides_reach_the_service(
    tmp_path, monkeypatch
):
    """LLM settings are environment-only: a leftover config section cannot take
    effect, and a per-run override is applied to the resolved settings."""

    project = build_project(tmp_path, [structure_map(1)])
    captured = {}
    install(monkeypatch, Service(project, Proposer()), captured)

    exit_code = run_agent_fix(
        namespace(tmp_path, llm_model="cli-model"),
        {"project_path": str(tmp_path), "llm": {"model": "from-conf"}},
    )

    assert exit_code == EXIT_OK
    assert "llm" not in captured["conf"]
    assert captured["overrides"]["model"] == "cli-model"


def test_a_clean_single_map_keeps_the_flat_layout_and_calls_no_provider(
    tmp_path, monkeypatch
):
    project = build_project(tmp_path, [structure_map(1)])
    proposer = Proposer()
    install(monkeypatch, Service(project, proposer))

    assert run_agent_fix(namespace(tmp_path), {"project_path": str(tmp_path)}) == EXIT_OK

    assert proposer.calls == 0
    run_root = only_run(tmp_path)
    # The per-map report stays where every existing reader looks for it.
    per_map = json.loads((run_root / "agent_report.json").read_text(encoding="utf-8"))
    assert per_map["outcome"] == "clean"
    assert per_map["configuration"]["llm"]["model"] == "resolved-model"
    assert per_map["configuration"]["llm"]["base_url"] == "https://llm.example/v1"
    project_report = json.loads(
        (run_root / "project_report.json").read_text(encoding="utf-8")
    )
    assert project_report["outcome"] == "ok"
    assert project_report["totals"]["outcomes"] == {"clean": 1}
    assert project_report["graph"]["checkpointer"]["kind"] == "sqlite"
    assert (run_root / "run_manifest.json").is_file()


def test_all_maps_writes_one_directory_per_map_and_one_project_report(
    tmp_path, monkeypatch
):
    project = build_project(tmp_path, [structure_map(1), structure_map(2)])
    install(monkeypatch, Service(project, Proposer()))

    exit_code = run_agent_fix(
        namespace(tmp_path, all_maps=True), {"project_path": str(tmp_path)}
    )

    assert exit_code == EXIT_OK
    run_root = only_run(tmp_path)
    assert {p.name for p in (run_root / "maps").iterdir()} == {"001_map", "002_map"}
    report = json.loads((run_root / "project_report.json").read_text(encoding="utf-8"))
    assert [entry["map_key"] for entry in report["maps"]] == ["001_map", "002_map"]
    assert report["totals"]["maps"] == 2
    assert report["global_findings"] == []


def test_an_unsatisfied_cross_map_reference_blocks_the_project(tmp_path, monkeypatch):
    """The finding no single map can see: nothing in the set produces the
    Patient this Observation defers to."""

    document = structure_map(1, deferred=True)
    project = build_project(tmp_path, [document], {PROFILE_URL: observation_profile()})
    proposer = Proposer()
    install(monkeypatch, Service(project, proposer))

    exit_code = run_agent_fix(namespace(tmp_path), {"project_path": str(tmp_path)})

    assert exit_code == EXIT_UNRESOLVED
    # `mapping-input-required`: no edit to this map can add a sibling resource.
    assert proposer.calls == 0
    report = json.loads(
        (only_run(tmp_path) / "project_report.json").read_text(encoding="utf-8")
    )
    codes = {finding["code"] for finding in report["global_findings"]}
    assert codes == {"cross-map-reference-unsatisfied"}
    assert report["outcome"] == "unresolved"


def test_apply_allows_an_unchanged_preexisting_project_blocker(tmp_path, monkeypatch):
    document = structure_map(1, deferred=True)
    project = build_project(tmp_path, [document], {PROFILE_URL: observation_profile()})
    install(monkeypatch, Service(project, Proposer()))
    path = project.structure_maps[0][0]
    before = path.read_bytes()

    exit_code = run_agent_fix(
        namespace(tmp_path, apply=True), {"project_path": str(tmp_path)}
    )

    assert exit_code == EXIT_UNRESOLVED
    assert path.read_bytes() == before
    report = json.loads(
        (only_run(tmp_path) / "project_report.json").read_text(encoding="utf-8")
    )
    assert report["apply"]["applied"] is True
    assert report["apply"]["refused_because"] is None
    assert report["baseline_global_findings"] == report["global_findings"]
    assert report["global_regressions"] == []


def test_resuming_a_run_that_does_not_exist_is_a_setup_error(tmp_path, monkeypatch):
    project = build_project(tmp_path, [structure_map(1)])
    install(monkeypatch, Service(project, Proposer()))

    exit_code = run_agent_fix(
        namespace(tmp_path, resume="run-nope"), {"project_path": str(tmp_path)}
    )

    assert exit_code == EXIT_SETUP_ERROR


def test_resume_refuses_when_a_selected_map_changed_on_disk(tmp_path, monkeypatch):
    """A checkpoint restores control state, not the world it was computed
    against. Continuing into an edited project would apply yesterday's
    decisions to today's files."""

    project = build_project(tmp_path, [structure_map(1)])
    install(monkeypatch, Service(project, Proposer()))
    assert run_agent_fix(namespace(tmp_path), {"project_path": str(tmp_path)}) == EXIT_OK
    run_id = only_run(tmp_path).name

    edited = structure_map(1)
    edited["name"] = "SomeoneElseEditedThis"
    project.structure_maps = [(project.structure_maps[0][0], edited)]

    exit_code = run_agent_fix(
        namespace(tmp_path, resume=run_id), {"project_path": str(tmp_path)}
    )

    assert exit_code == EXIT_SETUP_ERROR
