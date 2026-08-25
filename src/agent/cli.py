"""Handles the agent fix command"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import logging
from pathlib import Path
import shlex
import sys
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

# define exit codes
EXIT_OK = 0
EXIT_UNRESOLVED = 1
EXIT_SETUP_ERROR = 2


def _command_line() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def run_agent_fix(args, conf: Mapping[str, Any]) -> int:
    """Execute ``agent fix`` and write its run directory.

    :param args: parsed CLI namespace.
    :param conf: the loaded project configuration.
    :return: a process exit code — 0 when every map is clean or accepted and the
        assembled project passes, 1 when anything is unresolved or ``--apply``
        was refused, 2 for a setup failure.
    """

    from agent.loop import LoopLimits
    from agent.reporting import (
        RUN_MANIFEST_NAME,
        apply_project,
        build_manifest,
        manifest_mismatch,
    )
    from agent.service import AgentFixService, AgentSetupError

    try:
        from agent.graph import (
            AgentRuntimeContext,
            baseline_project_findings,
            open_checkpointer,
            run_project_graph,
        )
        from agent.graph.runtime import (
            checkpointer_identity,
            default_project_validators,
        )
        from agent.graph.state import GRAPH_STATE_VERSION, PROJECT_OK, ProjectLimits
    except ImportError as exc:
        logger.error(
            "Agent mode needs the optional 'agent' extra (pip install -e '.[agent]'): "
            "%s",
            exc,
        )
        return EXIT_SETUP_ERROR

    offline = bool(getattr(args, "offline", False))
    resume_id = getattr(args, "resume", None)
    limits = LoopLimits(max_attempts=int(getattr(args, "max_attempts", 4)))
    project_limits = ProjectLimits(
        max_project_rounds=int(getattr(args, "max_project_rounds", 3) or 3)
    )
    effective_conf = _without_llm_section(conf)
    llm_overrides = {
        "provider": getattr(args, "llm_provider", None),
        "model": getattr(args, "llm_model", None),
        "base_url": getattr(args, "llm_base_url", None),
    }

    try:
        service = AgentFixService.create(
            effective_conf,
            limits=limits,
            offline=offline,
            use_examples=bool(getattr(args, "use_examples", False)),
            overrides=llm_overrides,
        )
        targets = service.targets(
            getattr(args, "map", None), all_maps=bool(getattr(args, "all_maps", False))
        )
    except AgentSetupError as exc:
        logger.error("Agent mode cannot run: %s", exc)
        return EXIT_SETUP_ERROR
    except Exception as exc:
        if type(exc).__name__.startswith("LLM"):
            logger.error("Agent mode is unavailable: %s", exc)
            return EXIT_SETUP_ERROR
        raise

    output_root = Path(getattr(args, "output_dir", None) or service.project.project_dir)
    try:
        run_dir = _run_directory(output_root, resume_id)
    except AgentSetupError as exc:
        logger.error("Agent mode cannot run: %s", exc)
        return EXIT_SETUP_ERROR

    manifest = build_manifest(
        run_id=run_dir.root.name,
        command=_command_line(),
        conf=effective_conf,
        targets=targets,
        llm_config=service.llm_config,
        offline=offline,
        checkpointer=checkpointer_identity(),
        graph_state_version=GRAPH_STATE_VERSION,
        project=service.project,
        limits={"map": asdict(limits), "project": asdict(project_limits)},
        use_examples=bool(getattr(args, "use_examples", False)),
    )
    if resume_id:
        recorded = run_dir.read_json(RUN_MANIFEST_NAME)
        if recorded is None:
            logger.error(
                "Run %s has no %s; it was not written by this version and cannot "
                "be resumed.",
                resume_id,
                RUN_MANIFEST_NAME,
            )
            return EXIT_SETUP_ERROR
        mismatch = manifest_mismatch(recorded, manifest)
        if mismatch:
            logger.error(
                "Refusing to resume %s: %s. Start a new run instead.",
                resume_id,
                mismatch,
            )
            return EXIT_SETUP_ERROR
        manifest = recorded
    else:
        run_dir.write_json(RUN_MANIFEST_NAME, manifest)

    logger.info(
        "Agent run directory: %s (%d map(s)%s).",
        run_dir.root,
        len(targets),
        ", resumed" if resume_id else "",
    )

    context = AgentRuntimeContext(
        service=service,
        documents={key: (path, document) for key, (path, document) in targets.items()},
        limits=limits,
        project_limits=project_limits,
        require_engine=service.require_engine,
        project_validators=default_project_validators(
            getattr(service.project, "conf", None)
        ),
    )
    baseline_global_findings = baseline_project_findings(context)

    with open_checkpointer(run_dir.root) as checkpointer:
        run = run_project_graph(
            context=context,
            run_id=run_dir.root.name,
            checkpointer=checkpointer,
            resume=bool(resume_id),
        )
    run.baseline_global_findings = baseline_global_findings

    map_dirs = _map_directories(run_dir, run, multi=len(targets) > 1)
    transaction: Optional[Dict[str, Any]] = None
    if bool(getattr(args, "apply", False)):
        transaction = apply_project(
            run.ordered,
            map_dirs=map_dirs,
            offline=offline,
            globally_admissible=run.globally_admissible,
            global_reason=(
                f"{len(run.global_regressions)} new blocking finding(s) were "
                "introduced on the assembled set."
                if not run.globally_admissible
                else None
            ),
        )

    report_conf = deepcopy(effective_conf)
    if service.llm_config:
        report_conf["llm"] = dict(service.llm_config)
    map_reports = _write_map_reports(
        run, run_dir, map_dirs, service, report_conf, transaction, offline
    )
    _write_project_report(
        run, run_dir, map_reports, report_conf, manifest, transaction, offline
    )
    _summarize(run, run_dir, transaction)

    applied_ok = transaction is None or bool(transaction.get("applied"))
    return EXIT_OK if run.outcome == PROJECT_OK and applied_ok else EXIT_UNRESOLVED


def _without_llm_section(conf: Mapping[str, Any]) -> Dict[str, Any]:
    """check and drop llm section from a project configuratoin (for legacy compatibility)"""

    effective = deepcopy(dict(conf))
    if effective.get("llm"):
        logger.warning(
            "Ignoring the 'llm' section in the project configuration. LLM "
            "settings are read from .env / the environment only."
        )
        effective.pop("llm", None)
    return effective


def _run_directory(output_root: Path, resume_id: Optional[str]):
    from agent.reporting import AGENT_OUTPUT_DIR, RunDirectory  # noqa: PLC0415
    from agent.service import AgentSetupError  # noqa: PLC0415

    if not resume_id:
        return RunDirectory.create(output_root)
    root = Path(output_root) / AGENT_OUTPUT_DIR / resume_id
    if not root.is_dir():
        raise AgentSetupError(f"No agent run directory at {root}.")
    return RunDirectory(root=root)


def _map_directories(run_dir, run, *, multi: bool) -> Dict[str, Any]:
    """get directories (per map) or the root directory of the run itself (if exists)"""

    from agent.reporting import RunDirectory  # noqa: PLC0415

    if not multi:
        return {entry.map_key: run_dir for entry in run.ordered}
    return {
        entry.map_key: RunDirectory.child(run_dir, entry.map_key)
        for entry in run.ordered
    }


def _write_map_reports(
    run, run_dir, map_dirs, service, report_conf, transaction, offline: bool
) -> Dict[str, Path]:
    """Write given mapping report outputs to the map directory"""

    from agent.reporting import build_report, write_run  # noqa: PLC0415

    outcomes = {
        entry.get("map_key"): entry for entry in (transaction or {}).get("maps") or []
    }
    reports: Dict[str, Path] = {}
    for entry in run.ordered:
        map_dir = map_dirs[entry.map_key]
        artifacts = write_run(map_dir, document=entry.baseline, result=entry.result)
        record = outcomes.get(entry.map_key) or {}
        applied = record.get("status") == "replaced"
        if applied:
            artifacts["applied_to"] = entry.map_path
        reports[entry.map_key] = map_dir.write_report(
            build_report(
                command=_command_line(),
                conf=report_conf,
                result=entry.result,
                map_path=Path(entry.map_path),
                run_dir=map_dir,
                coverage_report=service.project.coverage_report,
                applied=applied,
                apply_refused=record.get("reason") if not applied else None,
                offline=offline,
                artifacts=artifacts,
            )
        )
    return reports


def _write_project_report(
    run, run_dir, map_reports, report_conf, manifest, transaction, offline: bool
) -> Path:
    """write given project-level reports to the corresponding report directories"""

    from agent.reporting import PROJECT_REPORT_NAME, build_project_report  # noqa: PLC0415

    return run_dir.write_json(
        PROJECT_REPORT_NAME,
        build_project_report(
            command=_command_line(),
            conf=report_conf,
            run=run,
            run_dir=run_dir,
            map_reports=map_reports,
            manifest=manifest,
            transaction=transaction,
            offline=offline,
        ),
    )


def _summarize(run, run_dir, transaction) -> None:
    """summarize the corresponding agent run for a quick overview"""

    from agent.reporting import PROJECT_REPORT_NAME  # noqa: PLC0415

    stats = run.stats()
    logger.info("Agent project: %s - %s", run.outcome, run.stop_reason)
    logger.info(
        "%d map(s) over %d round(s): %s. %d provider call(s), %d cache hit(s).",
        stats.maps,
        len(run.rounds),
        stats.outcomes,
        stats.provider_calls,
        stats.cache_hits,
    )
    for finding in run.global_findings:
        logger.warning("Project-level: %s", finding.message)
    for key, ids in sorted(run.routing.items()):
        logger.info("Requeued %s for %s.", key, ", ".join(ids))
    if transaction is None:
        staged = [entry.map_key for entry in run.ordered if entry.changed]
        if staged:
            logger.info(
                "%d candidate(s) accepted but not written (%s). Re-run with "
                "--apply to replace the source maps.",
                len(staged),
                ", ".join(staged),
            )
    elif not transaction.get("applied"):
        logger.warning("--apply did not run: %s", transaction.get("refused_because"))
    logger.info("Report: %s", run_dir.root / PROJECT_REPORT_NAME)
