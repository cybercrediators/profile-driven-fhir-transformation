"""Agent-mode run artifacts and apply semantics (WP8).

A run directory is the only durable evidence that an agent edited a clinical
mapping, so these tests ask what a later reader needs: is every attempt
recorded, did any credential leak into it, and can ``--apply`` be talked into
writing something it should not.

The command surface itself is covered in ``test_cli_options.py`` and
``test_service_controller.py``, alongside the parsers and dispatcher it belongs
to.
"""

import json
from pathlib import Path

import pytest

from agent.loop import AttemptRecord, LoopOutcome, LoopResult, MapRunResult
from agent.patch import canonical_sha256
from agent.reporting import (
    NON_SECRET_CONFIG_KEYS,
    RunDirectory,
    apply_project,
    build_report,
    candidate_refusal,
    new_run_id,
    redact_config,
    write_run,
)
from agent.validation import ValidationReport

pytestmark = pytest.mark.unit



# --- corpus -------------------------------------------------------------------


def structure_map():
    return {
        "resourceType": "StructureMap",
        "id": "sm-cli",
        "url": "http://example.org/StructureMap/sm-cli",
        "name": "SmCli",
        "status": "draft",
        "group": [
            {
                "name": "TransformPatient",
                "typeMode": "none",
                "input": [
                    {"name": "source", "type": "SrcModel", "mode": "source"},
                    {"name": "target", "type": "Patient", "mode": "target"},
                ],
                "rule": [],
            }
        ],
    }


def candidate_map():
    document = structure_map()
    document["description"] = "repaired"
    return document


def accepted_result(document):
    candidate_sha = canonical_sha256(candidate_map())
    candidate_report = ValidationReport(
        map_sha256=candidate_sha,
        engine_requested=True,
        engine_available=True,
    )
    return LoopResult(
        outcome=LoopOutcome.ACCEPTED,
        map_url=document["url"],
        map_id=document["id"],
        baseline_sha256=canonical_sha256(document),
        baseline_report=ValidationReport(engine_available=True),
        accepted_candidate=candidate_map(),
        accepted_sha256=candidate_sha,
        accepted_report=candidate_report,
        attempts=[
            AttemptRecord(
                index=1,
                operations=[{"op": "add", "path": "/description", "value": "repaired"}],
                rationale="added a description",
                accepted=True,
                candidate_sha256=candidate_sha,
                decision={"accepted": True, "invariants": []},
                validation_report=candidate_report,
            )
        ],
        stop_reason="Candidate accepted on attempt 1.",
    )


def rejected_result(document):
    return LoopResult(
        outcome=LoopOutcome.EXHAUSTED,
        map_url=document["url"],
        map_id=document["id"],
        baseline_sha256=canonical_sha256(document),
        baseline_report=ValidationReport(engine_available=True),
        stop_reason="No candidate passed within 4 attempts.",
    )


# --- run artifacts ------------------------------------------------------------


def test_a_run_directory_is_unique_per_run(tmp_path):
    first = RunDirectory.create(tmp_path, "run-fixed")
    second = RunDirectory.create(tmp_path, "run-fixed")
    # Two runs in the same second must not share a directory and overwrite each
    # other's evidence.
    assert first.root != second.root
    assert (first.root / "patches").is_dir()
    assert (first.root / "validation").is_dir()


def test_the_run_id_is_sortable():
    assert new_run_id().startswith("run-")


def test_every_attempt_is_written(tmp_path):
    document = structure_map()
    run_dir = RunDirectory.create(tmp_path)
    artifacts = write_run(run_dir, document=document, result=accepted_result(document))

    assert json.loads(Path(artifacts["baseline"]).read_text()) == document
    assert json.loads(Path(artifacts["accepted_candidate"]).read_text()) == candidate_map()
    patch = json.loads((run_dir.root / "patches" / "attempt-001.json").read_text())
    assert patch["operations"][0]["op"] == "add"
    validation = json.loads(
        (run_dir.root / "validation" / "attempt-001.json").read_text()
    )
    assert validation["decision"]["accepted"] is True
    assert validation["report"]["map_sha256"] == canonical_sha256(candidate_map())


def test_the_report_records_what_a_reader_needs(tmp_path):
    from mapping.generation_result import CoverageReport

    document = structure_map()
    run_dir = RunDirectory.create(tmp_path)
    coverage = CoverageReport.from_raw(
        {
            "note": "generated",
            "summary": {
                "profiles": 1,
                "required_total": 1,
                "required_mapped": 1,
                "required_unmapped": 0,
                "static_required_total": 1,
                "latent_required_total": 0,
                "required_coverage_pct": 100.0,
            },
            "mapping_diagnostics": [
                {"code": "deferred-reference", "message": "left for the bundler"}
            ],
        }
    )
    report = build_report(
        command="prog agent fix sm-cli",
        conf={"project_path": "projects/demo"},
        result=accepted_result(document),
        map_path=Path("projects/demo/structure_maps/sm-cli.json"),
        run_dir=run_dir,
        coverage_report=coverage,
        applied=False,
    )

    assert report["command"] == "prog agent fix sm-cli"
    assert report["map"]["baseline_sha256"] != report["map"]["candidate_sha256"]
    assert report["coverage_report"]["report_version"] == coverage.report_version
    assert report["coverage_report"]["diagnostic_ids"]
    assert report["outcome"] == "accepted"
    assert report["stop_reason"]
    assert report["acceptance"]["accepted"] is True
    assert report["apply"]["applied"] is False
    assert report["attempts"][0]["cache_hit"] is False
    assert report["artifacts"]["run_directory"] == str(run_dir.root)


def test_no_credential_reaches_the_report():
    # An allow-list, so a secret added to the config later is excluded by
    # default rather than by someone remembering to deny it.
    conf = {
        "project_path": "projects/demo",
        "api_key": "sk-super-secret",
        "matchbox_connection": {
            "url": "http://user:password@localhost:8080/matchboxv3?token=hidden",
            "token": "t",
        },
        "llm": {
            "model": "a-model",
            "base_url": "https://user:password@llm.example/v1?key=hidden",
            "api_key_env": "OPENAI_API_KEY",
            "api_key": "sk-also-secret",
        },
        "some_future_secret": "leak-me",
    }
    safe = redact_config(conf)
    serialized = json.dumps(safe)
    assert "sk-super-secret" not in serialized
    assert "sk-also-secret" not in serialized
    assert "leak-me" not in serialized
    assert '"token"' not in serialized
    # The env var *name* is kept: that is what makes a run reproducible.
    assert safe["llm"]["api_key_env"] == "OPENAI_API_KEY"
    assert safe["matchbox_url"] == "http://localhost:8080/matchboxv3"
    assert safe["llm"]["base_url"] == "https://llm.example/v1"
    assert set(safe) - {"llm", "matchbox_url"} <= NON_SECRET_CONFIG_KEYS


# --- apply semantics ----------------------------------------------------------


def write_map(tmp_path: Path, document) -> Path:
    path = tmp_path / "sm-cli.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def test_a_default_run_leaves_the_source_map_untouched(tmp_path):
    document = structure_map()
    path = write_map(tmp_path, document)
    before = path.read_bytes()
    run_dir = RunDirectory.create(tmp_path)
    write_run(run_dir, document=document, result=accepted_result(document))
    assert path.read_bytes() == before


def entry(path, result, *, map_key="sm-cli", baseline=None):
    """One map's durable run outcome, as the project transaction consumes it."""

    baseline = baseline if baseline is not None else structure_map()
    return MapRunResult(
        map_key=map_key,
        map_path=str(path),
        result=result,
        baseline=baseline,
        origin_sha256=canonical_sha256(baseline),
        staged_candidate=result.accepted_candidate,
        staged_sha256=result.accepted_sha256 or canonical_sha256(baseline),
    )


def apply_one(path, result, run_dir, *, offline=False, globally_admissible=True, **kwargs):
    item = entry(path, result, **kwargs)
    return apply_project(
        [item],
        map_dirs={item.map_key: run_dir},
        offline=offline,
        globally_admissible=globally_admissible,
    )


def test_apply_writes_the_candidate_and_preserves_the_baseline(tmp_path):
    document = structure_map()
    path = write_map(tmp_path, document)
    run_dir = RunDirectory.create(tmp_path)

    transaction = apply_one(path, accepted_result(document), run_dir)

    assert transaction["applied"] is True
    assert json.loads(path.read_text())["description"] == "repaired"
    preserved = json.loads((run_dir.root / "baseline_structure_map.json").read_text())
    assert "description" not in preserved
    assert transaction["maps"][0]["status"] == "replaced"


def test_apply_refuses_a_rejected_candidate(tmp_path):
    """Nothing to write is not a refusal — but it is also not a replacement."""

    document = structure_map()
    path = write_map(tmp_path, document)
    before = path.read_bytes()

    transaction = apply_one(path, rejected_result(document), RunDirectory.create(tmp_path))

    assert transaction["maps"][0]["status"] == "unchanged"
    assert path.read_bytes() == before


def test_apply_refuses_an_offline_candidate(tmp_path):
    document = structure_map()
    path = write_map(tmp_path, document)

    transaction = apply_one(
        path, accepted_result(document), RunDirectory.create(tmp_path), offline=True
    )

    assert transaction["applied"] is False
    assert "offline" in transaction["refused_because"]
    assert "description" not in json.loads(path.read_text())


def test_apply_refuses_a_stale_candidate(tmp_path):
    # Someone edited the map while the run was in flight. The candidate was
    # written against a revision that no longer exists on disk.
    document = structure_map()
    path = write_map(tmp_path, document)
    result = accepted_result(document)

    edited = json.loads(path.read_text())
    edited["name"] = "SomeoneElseEditedThis"
    path.write_text(json.dumps(edited, indent=2), encoding="utf-8")

    transaction = apply_one(path, result, RunDirectory.create(tmp_path))

    assert "changed since validation" in transaction["refused_because"]
    assert json.loads(path.read_text())["name"] == "SomeoneElseEditedThis"


def test_apply_refuses_if_candidate_content_no_longer_matches_the_accepted_digest(tmp_path):
    document = structure_map()
    path = write_map(tmp_path, document)
    result = accepted_result(document)
    result.accepted_candidate["description"] = "tampered after validation"
    before = path.read_bytes()

    transaction = apply_one(path, result, RunDirectory.create(tmp_path))

    assert "does not match the digest" in transaction["refused_because"]
    assert path.read_bytes() == before


def test_apply_infers_offline_evidence_even_if_caller_omits_offline_flag(tmp_path):
    document = structure_map()
    path = write_map(tmp_path, document)
    result = accepted_result(document)
    result.accepted_report.engine_requested = False
    result.accepted_report.engine_available = False

    reason = candidate_refusal(entry(path, result))

    assert "engine-validation report" in reason


def test_a_blocking_global_finding_refuses_the_whole_batch(tmp_path):
    """The assembled set is the unit. Writing the individually-acceptable subset
    would leave a project no round of this run ever validated."""

    document = structure_map()
    path = write_map(tmp_path, document)
    before = path.read_bytes()

    transaction = apply_one(
        path,
        accepted_result(document),
        RunDirectory.create(tmp_path),
        globally_admissible=False,
    )

    assert transaction["applied"] is False
    assert transaction["atomic"] is True
    assert path.read_bytes() == before


def test_an_unresolved_sibling_does_not_veto_an_accepted_repair(tmp_path):
    """Atomic over the *staged set*, not conditional on the run succeeding.

    The set the project validators passed already contained the unresolved map
    at its baseline, so what lands on disk is exactly what was validated.
    Requiring the whole run to succeed would let one permanently unfixable map
    veto every good repair in a large project without making any of them less
    correct.
    """

    good_doc = structure_map()
    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps(good_doc, indent=2), encoding="utf-8")
    stuck_doc = structure_map()
    stuck_path = tmp_path / "stuck.json"
    stuck_path.write_text(json.dumps(stuck_doc, indent=2), encoding="utf-8")
    before = stuck_path.read_bytes()

    run_dir = RunDirectory.create(tmp_path)
    transaction = apply_project(
        [
            entry(good_path, accepted_result(good_doc), map_key="good"),
            entry(stuck_path, rejected_result(stuck_doc), map_key="stuck"),
        ],
        map_dirs={
            "good": RunDirectory.child(run_dir, "good"),
            "stuck": RunDirectory.child(run_dir, "stuck"),
        },
    )

    assert transaction["applied"] is True
    statuses = {row["map_key"]: row["status"] for row in transaction["maps"]}
    assert statuses == {"good": "replaced", "stuck": "unchanged"}
    assert json.loads(good_path.read_text())["description"] == "repaired"
    assert stuck_path.read_bytes() == before


def test_one_failed_precondition_leaves_every_other_map_unchanged(tmp_path):
    """Project-atomic: a stale second map means the first is not written either."""

    good_doc = structure_map()
    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps(good_doc, indent=2), encoding="utf-8")
    stale_doc = structure_map()
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale_doc, indent=2), encoding="utf-8")

    good = entry(good_path, accepted_result(good_doc), map_key="good")
    stale = entry(stale_path, accepted_result(stale_doc), map_key="stale")
    stale_path.write_text(json.dumps({"edited": True}), encoding="utf-8")

    run_dir = RunDirectory.create(tmp_path)
    transaction = apply_project(
        [good, stale],
        map_dirs={"good": run_dir, "stale": RunDirectory.child(run_dir, "stale")},
    )

    assert transaction["applied"] is False
    assert "stale" in transaction["refused_because"]
    assert "description" not in json.loads(good_path.read_text())


def test_a_failure_partway_through_restores_the_files_already_replaced(tmp_path, monkeypatch):
    """A half-written project is the one outcome worse than a refused one."""

    first_doc = structure_map()
    first = tmp_path / "a.json"
    first.write_text(json.dumps(first_doc, indent=2), encoding="utf-8")
    second_doc = structure_map()
    second = tmp_path / "b.json"
    second.write_text(json.dumps(second_doc, indent=2), encoding="utf-8")
    original = first.read_bytes()

    entries = [
        entry(first, accepted_result(first_doc), map_key="a"),
        entry(second, accepted_result(second_doc), map_key="b"),
    ]
    run_dir = RunDirectory.create(tmp_path)

    real_replace = __import__("os").replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("the disk went away mid-commit")
        return real_replace(src, dst)

    monkeypatch.setattr("agent.reporting.os.replace", flaky)
    transaction = apply_project(
        entries,
        map_dirs={"a": RunDirectory.child(run_dir, "a"), "b": RunDirectory.child(run_dir, "b")},
    )

    assert transaction["applied"] is False
    assert transaction["rolled_back"] is True
    assert first.read_bytes() == original


def test_the_refusal_reason_reaches_the_report(tmp_path):
    document = structure_map()
    run_dir = RunDirectory.create(tmp_path)
    report = build_report(
        command="prog agent fix --apply",
        conf={},
        result=rejected_result(document),
        map_path=Path("sm-cli.json"),
        run_dir=run_dir,
        applied=False,
        apply_refused="No accepted candidate to apply (outcome: exhausted).",
    )
    assert report["apply"]["applied"] is False
    assert "exhausted" in report["apply"]["refused_because"]
