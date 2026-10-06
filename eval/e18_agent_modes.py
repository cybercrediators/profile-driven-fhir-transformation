"""E18 comparative evaluation of the mapping-production arms

check wether the agent mode improves maps without regression of the existing map

  ============================  ==============================  =================
  arm                           what it does                    compared against
  ============================  ==============================  =================
  table                         generation from the project's   (root)
                                fully specified mapping table
  deterministic-automap         the deterministic automapper    table
                                chooses the table
  llm-automap                   a model reranks that            deterministic-
                                automapper's candidates         automap
  agent                         table, then agent fix           table
  ============================  ==============================  =================


each writes a report to: eval/reports/e18_<project>_<arm>.json
:mod:`eval.e18_aggregate` turns those into the per-map unchanged/improved/regressed/unresolved comparison

Usage::

    PYTHONPATH=.:src python3 -m eval.e18_agent_modes --corpus
    PYTHONPATH=.:src python3 -m eval.e18_agent_modes <project> table [--engine]
    PYTHONPATH=.:src python3 -m eval.e18_agent_modes <project> deterministic-automap --engine
    PYTHONPATH=.:src python3 -m eval.e18_agent_modes <project> llm-automap --engine
    PYTHONPATH=.:src python3 -m eval.e18_agent_modes <project> agent --engine
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
_REPORTS = Path(__file__).resolve().parent / "reports"

#: The four production paths, and the generation mode each one runs.
ARM_MODES = {
    "table": "table",
    "deterministic-automap": "automap-deterministic",
    "llm-automap": "automap-llm",
    "agent": "table",
}
ARMS = tuple(ARM_MODES)

#: What each arm is compared against, because one baseline does not fit all.
BASELINE_FOR = {
    "deterministic-automap": "table",
    "llm-automap": "deterministic-automap",
    "agent": "table",
}

#: Arms that choose their own mapping table, and so have a selection to grade.
SELECTING_ARMS = frozenset({"deterministic-automap", "llm-automap"})

#: Failure classes a case is labelled with, from the deterministic baseline.
FAILURE_CLASSES = (
    "clean",
    "parse",
    "semantics",
    "target-paths",
    "source-paths",
    "coverage-map-fixable",
    "coverage-input-required",
    "obligations",
    "cross-map",
)

_CODE_TO_CLASS = {
    "resource-invalid": "parse",
    "not-a-structure-map": "parse",
    "semantic-rule-invalid": "semantics",
    "target-path-not-found": "target-paths",
    "target-path-ambiguous": "target-paths",
    "target-path-prohibited": "target-paths",
    "target-choice-type-not-allowed": "target-paths",
    "source-path-not-found": "source-paths",
    "mapping-obligation-dropped": "obligations",
    "cross-map-reference-unsatisfied": "cross-map",
}


def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _conf_for(project: str) -> Dict[str, Any]:
    path = _REPO / "conf" / f"{project}.json"
    if not path.is_file():
        raise SystemExit(f"no conf/{project}.json — see eval/README.md")
    conf = _load(path)
    if not isinstance(conf, dict):
        raise SystemExit(f"conf/{project}.json is not an object")
    return conf


def _copy_project(project: str, workdir: Path) -> Path:
    """throwaway copy of the corresponding project to avoid modifications"""

    destination = workdir / project
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(_REPO / "projects" / project, destination)
    for relative in ("structure_maps", "source_data/concept_maps", "agent_output"):
        directory = destination / relative
        if directory.is_dir():
            shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
    return destination


def _write_conf(project: str, project_dir: Path, workdir: Path, extra=None) -> Path:
    """write a config for a specific project"""
    conf = _conf_for(project)
    conf["project_path"] = str(project_dir) + "/"
    conf.update(extra or {})
    path = workdir / f"{project}.conf.json"
    path.write_text(json.dumps(conf, indent=2))
    return path


def _generate(conf_path: Path, project_dir: Path, *, mode: str) -> Dict[str, Any]:
    """run the pipeline for one arm generation step, in a subprocess"""

    command = [
        sys.executable, "src/main.py", "-c", str(conf_path),
        "pipeline", "run", "-f", "-msm",
    ]
    if mode == "table":
        command += ["-mt", str(project_dir / "source_data" / "mapping_table.json")]
    elif mode == "automap-deterministic":
        command += ["-am", "--auto-mapping-mode", "deterministic"]
    elif mode == "automap-llm":
        command += ["-am", "--auto-mapping-mode", "llm"]
    else:
        raise ValueError(f"unknown generation mode {mode!r}")

    started = time.monotonic()
    result = subprocess.run(
        command, cwd=_REPO, capture_output=True, text=True, timeout=3600,
        env=dict(os.environ),
    )
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "elapsed_s": round(time.monotonic() - started, 2),
        "stderr_tail": result.stderr[-3000:] if result.returncode != 0 else "",
    }


def measure(conf: Dict[str, Any], *, engine: bool) -> Dict[str, Any]:
    """measure evidence for every map of a project, plus cross-map check

    :param engine: enables run of Matchbox `$transform` and `$validate`
    """

    from agent.service import MapEvaluator, build_project_context
    from agent.validation import recheck_cross_map_references

    project = build_project_context(conf)
    per_map: List[Dict[str, Any]] = []
    assembled: List[Any] = []
    for path, document in project.structure_maps:
        evaluator = MapEvaluator(project, document, require_engine=engine)
        report = evaluator.evaluate(document)
        row = _map_row(path, report, engine=engine)
        row["map_id"] = document.get("id")
        per_map.append(row)
        assembled.append((document, project.profile_for(document)))

    cross = recheck_cross_map_references(
        assembled,
        external_defaults=list(conf.get("external_reference_defaults") or []),
    )
    by_map: Dict[Any, List[Any]] = {}
    for finding in cross:
        by_map.setdefault(finding.map_id, []).append(finding)
    for row in per_map:
        found = by_map.get(row.get("map_id")) or []
        _fold_cross_map_findings(row, found)

    aggregated = _aggregate(per_map, engine=engine)
    aggregated["cross_map"] = {
        "maps_in_set": len(assembled),
        "unsatisfied": [
            {
                "map": finding.map_id,
                "path": finding.path,
                "expected_target_profiles": finding.evidence.get(
                    "expected_target_profiles"
                ),
            }
            for finding in cross
        ],
    }
    return aggregated


def _fold_cross_map_findings(row: Dict[str, Any], findings: List[Any]) -> None:
    """add assembled-set evidence without leaving a contradictory clean label"""

    row["cross_map_unsatisfied"] = [finding.path for finding in findings]
    if not findings:
        return
    row["blocking_findings"] += len(findings)
    row["finding_codes"] = sorted(
        set(row["finding_codes"]) | {"cross-map-reference-unsatisfied"}
    )
    classes = set(row["failure_classes"])
    classes.discard("clean")
    row["failure_classes"] = sorted(classes | {"cross-map"})


def _map_row(path: Path, report, *, engine: bool) -> Dict[str, Any]:
    from agent.validation import ActionOwner, Producer, Stage

    codes = [finding.code for finding in report.findings]
    producers = {finding.code: finding.producer for finding in report.findings}
    parse_failed = any(
        producers.get(code) is Producer.FHIR_MODEL for code in codes
    )
    semantics_failed = any(
        producers.get(code) is Producer.MAP_SEMANTICS for code in codes
    )
    required_gaps = [
        finding for finding in report.findings if finding.code == "required-path-unmapped"
    ]
    def engine_blocking(stage) -> List[Any]:
        return [
            finding
            for finding in report.findings
            if finding.producer is Producer.ENGINE
            and finding.stage is stage
            and finding.blocking
        ]

    transform_failed = any(
        finding.producer is Producer.ENGINE
        and finding.stage in (Stage.UPLOAD, Stage.TRANSFORM)
        and finding.blocking
        for finding in report.findings
    )
    validate_findings = engine_blocking(Stage.VALIDATE)
    validate_blocked_by_environment = any(
        finding.action_owner is ActionOwner.ENVIRONMENT for finding in validate_findings
    )
    validate_failed = any(
        finding.action_owner is not ActionOwner.ENVIRONMENT
        for finding in validate_findings
    )
    return {
        "map": path.name,
        "parses": not parse_failed,
        "semantics_ok": not semantics_failed,
        "compiler_accepted": not (parse_failed or semantics_failed),
        "covered_required": len(report.covered_required_paths),
        "required_gaps": len(required_gaps),
        "required_gaps_map_fixable": sum(
            1 for finding in required_gaps
            if finding.action_owner is ActionOwner.MAP_FIXABLE
        ),
        "satisfied_obligations": len(report.satisfied_obligations),
        "deferred_references": len(report.deferred_reference_paths),
        "blocking_findings": len(report.blocking_findings),
        "finding_codes": sorted(set(codes)),
        "worklist": len(report.worklist()),
        "engine_available": report.engine_available if engine else None,
        "transform_ok": (not transform_failed) if engine and report.engine_available else None,
        "validate_ok": (
            None
            if not (engine and report.engine_available)
            or (validate_blocked_by_environment and not validate_failed)
            else not validate_failed
        ),
        "validate_blocked_by_environment": validate_blocked_by_environment
        if engine
        else None,
        "executed_fixtures": list(report.executed_fixtures),
        "validated_fixtures": list(report.validated_fixtures),
        "failure_classes": sorted(classify_findings(report)),
    }


def classify_findings(report) -> set:
    """The failure classes one map's baseline exhibits."""

    from agent.validation import ActionOwner

    classes = set()
    for finding in report.findings:
        if finding.code == "required-path-unmapped":
            classes.add(
                "coverage-map-fixable"
                if finding.action_owner is ActionOwner.MAP_FIXABLE
                else "coverage-input-required"
            )
            continue
        mapped = _CODE_TO_CLASS.get(finding.code)
        if mapped:
            classes.add(mapped)
        elif finding.blocking:
            classes.add("engine" if finding.code.startswith(("transform:", "validate:", "upload:")) else "other")
    return classes or {"clean"}


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _aggregate(per_map: List[Dict[str, Any]], *, engine: bool) -> Dict[str, Any]:
    total = len(per_map)
    graded = [row for row in per_map if row["transform_ok"] is not None]
    validated = [row for row in per_map if row["validate_ok"] is not None]
    covered = sum(row["covered_required"] for row in per_map)
    gaps = sum(row["required_gaps"] for row in per_map)
    classes: Dict[str, int] = {}
    for row in per_map:
        for name in row["failure_classes"]:
            classes[name] = classes.get(name, 0) + 1
    return {
        "maps": total,
        "compiler_acceptance": _ratio(
            sum(1 for row in per_map if row["compiler_accepted"]), total
        ),
        "required_path_coverage": _ratio(covered, covered + gaps),
        "required_gaps": gaps,
        "required_gaps_map_fixable": sum(
            row["required_gaps_map_fixable"] for row in per_map
        ),
        "blocking_findings": sum(row["blocking_findings"] for row in per_map),
        "maps_clean": sum(1 for row in per_map if row["blocking_findings"] == 0),
        "matchbox_execution": _ratio(
            sum(1 for row in graded if row["transform_ok"]), len(graded)
        )
        if engine and graded
        else None,
        "profile_validation": _ratio(
            sum(1 for row in validated if row["validate_ok"]), len(validated)
        )
        if engine and validated
        else None,
        "engine_graded_maps": len(graded) if engine else None,
        "profile_validation_graded_maps": len(validated) if engine else None,
        "validation_blocked_by_environment": sum(
            1 for row in per_map if row.get("validate_blocked_by_environment")
        )
        if engine
        else None,
        "failure_classes": dict(sorted(classes.items())),
        "per_map": per_map,
    }


def automapper_output(project_dir: Path) -> Optional[Dict[str, Any]]:
    """the table the automapper produced, or None when it wrote none"""

    found = sorted((project_dir / "source_data").glob("*_automapping.json"))
    for path in found:
        if path.name.endswith("_llm_automapping_report.json"):
            continue
        table = _load(path)
        if isinstance(table, dict):
            return table
    return None


def llm_audit_summary(project_dir: Path) -> Optional[Dict[str, Any]]:
    """counts from the LLM automapper own audit trail, when it wrote one"""

    for path in sorted((project_dir / "source_data").glob("*_llm_automapping_report.json")):
        report = _load(path)
        if not isinstance(report, dict):
            continue
        proposed = report.get("proposed") or report.get("proposed_mapping_table") or {}
        compiled = report.get("compiled") or report.get("compiled_mapping_table") or {}
        return {
            "file": path.name,
            "proposed_entries": len(proposed) if isinstance(proposed, dict) else None,
            "compiled_entries": len(compiled) if isinstance(compiled, dict) else None,
            "compiler_diagnostics": len(report.get("compiler_diagnostics") or []),
        }
    return None


def profile_roots(project: str) -> Dict[str, str]:
    """`{profile id or url tail: resource type}` for profiles of one project"""

    roots: Dict[str, str] = {}
    for path in sorted((_REPO / "projects" / project / "input_profile").glob("*.json")):
        document = _load(path)
        if not isinstance(document, dict):
            continue
        if document.get("resourceType") != "StructureDefinition":
            continue
        res_type = document.get("type")
        if not res_type:
            continue
        for key in (document.get("id"), str(document.get("url") or "").rsplit("/", 1)[-1]):
            if key:
                roots[str(key)] = str(res_type)
    return roots


def mapping_selection(
    reference: Dict[str, Any],
    produced: Dict[str, Any],
    *,
    roots: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """precision/recall of an automapper source-to-target choices

    :param roots: `{profile id: resource type}` from :func:`profile_roots`
    """

    from agent.validation import normalize_target_path
    from mapping.rule_ir import mapping_target_path

    def canonical(target: str) -> str:
        path = normalize_target_path(str(target))
        root, _, remainder = path.partition(".")
        mapped = (roots or {}).get(root)
        return f"{mapped}.{remainder}" if mapped and remainder else path

    def pairs(table: Dict[str, Any]) -> set:
        out = set()
        for source, value in (table or {}).items():
            target = mapping_target_path(value)
            if target:
                out.add((str(source), canonical(target)))
        return out

    expected, actual = pairs(reference), pairs(produced)
    hits = expected & actual
    return {
        "reference_pairs": len(expected),
        "produced_pairs": len(actual),
        "agreeing_pairs": len(hits),
        "precision": _ratio(len(hits), len(actual)),
        "recall": _ratio(len(hits), len(expected)),
        "extra": sorted(f"{s} -> {t}" for s, t in sorted(actual - expected))[:25],
        "missed": sorted(f"{s} -> {t}" for s, t in sorted(expected - actual))[:25],
        "reference": "the project's committed mapping table, not an independent gold standard",
    }


def run_arm(project: str, arm: str, *, engine: bool, keep: Optional[Path] = None) -> Dict[str, Any]:
    """runs the actual experiment based on the chosen project and arm"""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")

    workdir = Path(keep) if keep else Path(tempfile.mkdtemp(prefix=f"e18_{arm}_"))
    workdir.mkdir(parents=True, exist_ok=True)
    reference_table = _load(
        _REPO / "projects" / project / "source_data" / "mapping_table.json"
    ) or {}

    report: Dict[str, Any] = {
        "experiment": "E18",
        "project": project,
        "arm": arm,
        "engine": engine,
        "provenance": _provenance(),
        "workdir": str(workdir),
    }

    project_dir = _copy_project(project, workdir)
    conf_path = _write_conf(project, project_dir, workdir)

    report["baseline_arm"] = BASELINE_FOR.get(arm)
    report["generation"] = _generate(conf_path, project_dir, mode=ARM_MODES[arm])
    if report["generation"]["returncode"] != 0:
        report["status"] = "generation-failed"
        return report

    conf = _load(conf_path)
    if arm in SELECTING_ARMS:
        produced = automapper_output(project_dir)
        if produced is None:
            report["status"] = "automapping-output-missing"
            report["mapping_selection"] = None
            return report
        report["mapping_selection"] = mapping_selection(
            reference_table, produced, roots=profile_roots(project)
        )
        if arm == "llm-automap":
            report["llm_automapping_report"] = llm_audit_summary(project_dir)
    else:
        report["mapping_selection"] = None

    report["baseline"] = measure(conf, engine=engine)

    if arm == "agent":
        report["agent"] = _run_agent(conf, project_dir, engine=engine)
        report["after"] = measure(conf, engine=engine)

    report["status"] = "ok"
    return report


def _run_agent(conf: Dict[str, Any], project_dir: Path, *, engine: bool) -> Dict[str, Any]:
    """`agent fix --all --apply` over the copy, through the project graph again"""

    from argparse import Namespace

    from agent.cli import run_agent_fix
    from agent.reporting import AGENT_OUTPUT_DIR, PROJECT_REPORT_NAME

    started = time.monotonic()
    args = Namespace(
        map=None,
        all_maps=True,
        max_attempts=4,
        max_project_rounds=3,
        resume=None,
        output_dir=str(project_dir),
        offline=not engine,
        use_examples=False,
        apply=engine,
        llm_provider=None,
        llm_model=None,
        llm_base_url=None,
    )
    exit_code = run_agent_fix(args, conf)

    runs = sorted((project_dir / AGENT_OUTPUT_DIR).glob("run-*"))
    report = _load(runs[-1] / PROJECT_REPORT_NAME) if runs else None
    if not isinstance(report, dict):
        return {
            "exit_code": exit_code,
            "elapsed_s": round(time.monotonic() - started, 2),
            "error": "the agent run wrote no project report",
        }

    rows = [
        {
            "map": Path(entry["path"]).name,
            "outcome": entry["outcome"],
            "stop_reason": entry["stop_reason"],
            "round": entry["round"],
            "attempts": entry["attempts"],
            "provider_calls": entry["provider_calls"],
            "cache_hits": entry["cache_hits"],
            "staged": entry["staged"],
            "requeued_for": entry["requeued_for"],
            "report": entry["report"],
        }
        for entry in report.get("maps") or []
    ]
    applied = sum(
        1
        for entry in (report.get("apply") or {}).get("maps") or []
        if entry.get("status") == "replaced"
    )
    return {
        "exit_code": exit_code,
        "outcome": report.get("outcome"),
        "stop_reason": report.get("stop_reason"),
        "stats": report.get("totals"),
        "elapsed_s": round(time.monotonic() - started, 2),
        "provider_calls": sum(row["provider_calls"] for row in rows),
        "cache_hits": sum(row["cache_hits"] for row in rows),
        "attempts": sum(row["attempts"] for row in rows),
        "rounds": len((report.get("graph") or {}).get("rounds") or []),
        "global_findings": [
            finding["finding_id"] for finding in report.get("global_findings") or []
        ],
        "applied": applied,
        "apply_refused": (report.get("apply") or {}).get("refused_because"),
        "run_directory": str(runs[-1]) if runs else None,
        "per_map": rows,
    }


def build_corpus(projects: List[str]) -> Dict[str, Any]:
    """classify the corpus by failure type from the committed maps"""

    from agent.validation import build_target_tree, validate_offline

    cases = []
    for project in projects:
        directory = _REPO / "projects" / project
        profiles = {}
        for path in sorted((directory / "input_profile").glob("*.json")):
            document = _load(path)
            if isinstance(document, dict) and document.get("resourceType") == "StructureDefinition":
                profiles[document.get("url")] = document
        table = _load(directory / "source_data" / "mapping_table.json") or {}
        maps = sorted((directory / "structure_maps").glob("*.json"))
        classes: Dict[str, int] = {}
        graded = 0
        for path in maps:
            document = _load(path)
            if not isinstance(document, dict) or document.get("resourceType") != "StructureMap":
                continue
            url = next(
                (
                    str(entry.get("url") or "").split("|")[0]
                    for entry in document.get("structure") or []
                    if entry.get("mode") == "target"
                ),
                None,
            )
            profile = profiles.get(url)
            if profile is None:
                continue
            graded += 1
            report = validate_offline(
                document,
                target_tree=build_target_tree(profile),
                mapping_table=table,
                profile_url=url,
                profile_id=profile.get("id"),
            )
            for name in classify_findings(report):
                classes[name] = classes.get(name, 0) + 1
        cases.append(
            {
                "project": project,
                "maps": len(maps),
                "graded_maps": graded,
                "mapping_table_entries": len(table),
                "fully_specified": project.endswith("_full_spec"),
                "failure_classes": dict(sorted(classes.items())),
            }
        )
    return {
        "experiment": "E18",
        "kind": "corpus",
        "provenance": _provenance(),
        "classes": list(FAILURE_CLASSES),
        "cases": cases,
    }


def default_projects() -> List[str]:
    """fully specified projects with a complete mapping table"""

    return sorted(
        path.name
        for path in (_REPO / "projects").iterdir()
        if path.is_dir()
        and path.name.endswith("_full_spec")
        and (path / "structure_maps").is_dir()
    )

def _provenance() -> Dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=_REPO, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return ""

    dirty = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(dirty),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _write(name: str, payload: Dict[str, Any]) -> Path:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    path = _REPORTS / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _summarize(report: Dict[str, Any]) -> str:
    if report.get("status") != "ok":
        return f"{report['project']} [{report['arm']}]: {report.get('status')}"
    lines = [f"\n=== {report['project']} [{report['arm']}]"]
    for label, key in (("baseline", "baseline"), ("after agent", "after")):
        block = report.get(key)
        if not block:
            continue
        lines.append(
            f"  {label:12s} maps={block['maps']} clean={block['maps_clean']} "
            f"compiler={block['compiler_acceptance']} "
            f"required-coverage={block['required_path_coverage']} "
            f"transform={block['matchbox_execution']} "
            f"validate={block['profile_validation']}"
        )
    agent = report.get("agent")
    if agent:
        lines.append(
            f"  agent        outcomes={agent['stats']['outcomes']} "
            f"attempts={agent['attempts']} calls={agent['provider_calls']} "
            f"cache={agent['cache_hits']} applied={agent['applied']} "
            f"{agent['elapsed_s']}s"
        )
    selection = report.get("mapping_selection")
    if selection:
        lines.append(
            f"  selection    precision={selection['precision']} "
            f"recall={selection['recall']} "
            f"({selection['agreeing_pairs']}/{selection['reference_pairs']})"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="E18 comparative evaluation of the three mapping-production arms")
    parser.add_argument("project", nargs="?", help="project name")
    parser.add_argument("arm", nargs="?", choices=ARMS)
    parser.add_argument("--engine", action="store_true", help="run Matchbox layers 5-6")
    parser.add_argument("--corpus", action="store_true", help="classify the corpus and exit")
    parser.add_argument("--keep", help="reuse this working directory instead of a temp one")
    args = parser.parse_args(argv)

    if args.corpus:
        projects = [args.project] if args.project else default_projects()
        payload = build_corpus(projects)
        path = _write("e18_corpus.json", payload)
        for case in payload["cases"]:
            print(f"  {case['project']:38s} maps={case['maps']:3d} {case['failure_classes']}")
        print(f"\n{len(payload['cases'])} case(s) -> {path}")
        return 0

    if not args.project or not args.arm:
        parser.error("give a project and an arm, or --corpus")

    report = run_arm(
        args.project, args.arm, engine=args.engine,
        keep=Path(args.keep) if args.keep else None,
    )
    path = _write(f"e18_{args.project}_{args.arm}.json", report)
    print(_summarize(report))
    print(f"\n-> {path}")
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
