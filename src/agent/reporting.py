"""Agent run output file handling"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from agent.loop import LoopOutcome, LoopResult

logger = logging.getLogger(__name__)

AGENT_OUTPUT_DIR = "agent_output"
REPORT_VERSION = 2
PROJECT_REPORT_NAME = "project_report.json"
RUN_MANIFEST_NAME = "run_manifest.json"

NON_SECRET_CONFIG_KEYS = frozenset(
    {
        "project_path",
        "profile_path",
        "mapping_table_path",
        "fhir_version",
        "package_name",
        "package_version",
    }
)

NON_SECRET_LLM_KEYS = frozenset(
    {
        "provider",
        "model",
        "base_url",
        "api_key_env",
        "temperature",
        "max_output_tokens",
        "timeout_s",
        "seed",
        "structured_output",
        "cache_enabled",
    }
)


def new_run_id(now: Optional[datetime] = None) -> str:
    """A sortable, collision-resistant directory name"""

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}"


def redact_config(conf: Mapping[str, Any]) -> Dict[str, Any]:
    """The recordable part of a configuration, by allow-list."""

    def safe_url(value: Any) -> str:
        """Keep endpoint identity without credentials or query secrets."""

        parsed = urlsplit(str(value))
        hostname = parsed.hostname or ""
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))

    safe = {key: conf[key] for key in sorted(NON_SECRET_CONFIG_KEYS) if key in conf}
    matchbox = conf.get("matchbox_connection")
    if isinstance(matchbox, Mapping) and matchbox.get("url"):
        safe["matchbox_url"] = safe_url(matchbox["url"])
    llm = conf.get("llm")
    if isinstance(llm, Mapping):
        safe["llm"] = {
            key: llm[key] for key in sorted(NON_SECRET_LLM_KEYS) if key in llm
        }
        if safe["llm"].get("base_url"):
            safe["llm"]["base_url"] = safe_url(safe["llm"]["base_url"])
    return safe


@dataclass
class RunDirectory:
    """handle run artifacts on disk"""

    root: Path

    @classmethod
    def create(cls, project_dir: Path, run_id: Optional[str] = None) -> "RunDirectory":
        root = Path(project_dir) / AGENT_OUTPUT_DIR / (run_id or new_run_id())
        suffix = 1
        while root.exists():
            root = root.with_name(f"{root.name}-{suffix}")
            suffix += 1
        (root / "patches").mkdir(parents=True)
        (root / "validation").mkdir(parents=True)
        return cls(root=root)

    @classmethod
    def child(cls, parent: "RunDirectory", name: str) -> "RunDirectory":
        """build per-map directory inside one multi-map run"""

        root = parent.root / "maps" / name
        (root / "patches").mkdir(parents=True, exist_ok=True)
        (root / "validation").mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    def write_json(self, relative: str, payload: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path

    def read_json(self, relative: str) -> Optional[Any]:
        """Read one artifact back, or ``None`` when it is absent or unreadable."""

        path = self.root / relative
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def write_baseline(self, document: Mapping[str, Any]) -> Path:
        return self.write_json("baseline_structure_map.json", dict(document))

    def write_candidate(self, document: Mapping[str, Any]) -> Path:
        return self.write_json("accepted_candidate.json", dict(document))

    def write_attempt(self, index: int, patch: Any, validation: Any) -> None:
        self.write_json(f"patches/attempt-{index:03d}.json", patch)
        self.write_json(f"validation/attempt-{index:03d}.json", validation)

    def write_report(self, report: Mapping[str, Any]) -> Path:
        return self.write_json("agent_report.json", dict(report))


def _finding_payload(findings: Sequence[Any]) -> List[Dict[str, Any]]:
    return [finding.model_dump(mode="json", exclude_none=True) for finding in findings]


def _partition_findings(report) -> Dict[str, List[Dict[str, Any]]]:
    """group findings from a given report"""

    if report is None:
        return {}
    from agent.validation import Producer, Stage  # noqa: PLC0415

    buckets: Dict[str, List[Any]] = {
        "offline": [],
        "engine_transform": [],
        "engine_validate": [],
        "generator": [],
    }
    for finding in report.findings:
        if finding.producer is Producer.GENERATOR:
            buckets["generator"].append(finding)
        elif finding.producer is Producer.ENGINE:
            if finding.stage is Stage.VALIDATE:
                buckets["engine_validate"].append(finding)
            else:
                buckets["engine_transform"].append(finding)
        else:
            buckets["offline"].append(finding)
    return {name: _finding_payload(items) for name, items in buckets.items() if items}


def build_report(
    *,
    command: str,
    conf: Mapping[str, Any],
    result: LoopResult,
    map_path: Path,
    run_dir: RunDirectory,
    coverage_report=None,
    applied: bool = False,
    apply_refused: Optional[str] = None,
    offline: bool = False,
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble agent_report.json

    :param command: the invocation, for reproducibility.
    :param applied: whether the accepted candidate replaced the source map.
    :param apply_refused: why --apply did not run, when it did not.
    :param offline: whether engine evidence was deliberately skipped
    """

    baseline = result.baseline_report
    accepted = result.accepted_report

    return {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "configuration": redact_config(conf),
        "offline": offline,
        "map": {
            "path": str(map_path),
            "url": result.map_url,
            "id": result.map_id,
            "baseline_sha256": result.baseline_sha256,
            "candidate_sha256": result.accepted_sha256,
        },
        "coverage_report": {
            "report_version": getattr(coverage_report, "report_version", None),
            "diagnostic_ids": [
                diagnostic.diagnostic_id
                for diagnostic in getattr(coverage_report, "mapping_diagnostics", [])
            ],
        }
        if coverage_report is not None
        else None,
        "outcome": result.outcome.value,
        "stop_reason": result.stop_reason,
        "provider_calls": result.provider_calls,
        "cache_hits": result.cache_hits,
        "elapsed_s": round(result.elapsed_s, 3),
        "attempts": [
            {
                **attempt.model_dump(mode="json", exclude_none=True),
               "cache_hit": attempt.cache_hit,
            }
            for attempt in result.attempts
        ],
        "baseline_findings": _partition_findings(baseline),
        "candidate_findings": _partition_findings(accepted),
        "acceptance": {
            "accepted": result.outcome is LoopOutcome.ACCEPTED,
            "invariants": (
                result.attempts[-1].decision.get("invariants")
                if result.attempts and result.attempts[-1].decision
                else None
            ),
        },
        "apply": {
            "applied": applied,
            "refused_because": apply_refused,
        },
        "unresolved": {
            owner: ids
            for owner, ids in _unresolved_by_owner(result).items()
        },
        "artifacts": {
            "run_directory": str(run_dir.root),
            **(dict(artifacts) if artifacts else {}),
        },
    }


def _unresolved_by_owner(result: LoopResult) -> Dict[str, List[str]]:
    from agent.loop import unresolved_by_owner  # noqa: PLC0415

    return unresolved_by_owner(result)


def write_run(
    run_dir: RunDirectory,
    *,
    document: Mapping[str, Any],
    result: LoopResult,
) -> Dict[str, str]:
    """write per-attempt and per-revision artifacts

    :return: {name: path} for the report's artifact index
    """

    artifacts: Dict[str, str] = {
        "baseline": str(run_dir.write_baseline(document))
    }
    for attempt in result.attempts:
        run_dir.write_attempt(
            attempt.index,
            {
                "operations": attempt.operations,
                "rationale": attempt.rationale,
                "worklist": attempt.worklist,
                "application": attempt.application,
            },
            {
                "candidate_sha256": attempt.candidate_sha256,
                "decision": attempt.decision,
                "report": (
                    attempt.validation_report.model_dump(
                        mode="json", exclude_none=True
                    )
                    if attempt.validation_report is not None
                    else None
                ),
                "note": attempt.note,
            },
        )
    if result.accepted_candidate is not None:
        artifacts["accepted_candidate"] = str(
            run_dir.write_candidate(result.accepted_candidate)
        )
    return artifacts


_EXECUTION_PROVIDER_KEYS = (
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "structured_output",
    "temperature",
    "seed",
    "max_output_tokens",
    "timeout_s",
    "cache_enabled",
)

_EXECUTION_PROJECT_INPUTS = (
    "profiles",
    "mapping_table",
    "concept_maps",
    "source_model",
    "examples",
    "coverage_report",
)


def _digestible(value: Any) -> Any:
    """JSON view of a project input, whatever shape it arrives in"""

    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    return value


def _project_input_digests(project) -> Dict[str, Optional[str]]:
    """A digest per project input the graph decisions depend on."""

    from agent.patch import canonical_sha256  # noqa: PLC0415

    digests: Dict[str, Optional[str]] = {}
    for name in _EXECUTION_PROJECT_INPUTS:
        value = getattr(project, name, None) if project is not None else None
        digests[name] = canonical_sha256(_digestible(value)) if value else None
    return digests


def build_manifest(
    *,
    run_id: str,
    command: str,
    conf: Mapping[str, Any],
    targets: Mapping[str, Tuple[Path, Mapping[str, Any]]],
    llm_config: Mapping[str, Any],
    offline: bool,
    checkpointer: Mapping[str, Any],
    graph_state_version: int,
    project=None,
    limits: Optional[Mapping[str, Any]] = None,
    use_examples: bool = False,
) -> Dict[str, Any]:
    """build identity a resume has to match before it continues a run

    :param project: the resolved ProjectContext (for inputs run)
    :param limits: the per-map and project bounds in force, as a plain mapping.
    """

    from agent.patch import canonical_sha256  # noqa: PLC0415

    identity = {
        "graph_state_version": graph_state_version,
        "report_version": REPORT_VERSION,
        "patch_schema_version": _patch_schema_version(),
        "checkpointer": dict(checkpointer),
        "offline": offline,
        "use_examples": bool(use_examples),
        "configuration_sha256": canonical_sha256(redact_config(conf)),
        "project_inputs": _project_input_digests(project),
        "provider": {key: llm_config.get(key) for key in _EXECUTION_PROVIDER_KEYS},
        "limits": dict(limits or {}),
        "maps": [
            {
                "map_key": key,
                "path": str(path),
                "baseline_sha256": canonical_sha256(dict(document)),
            }
            for key, (path, document) in sorted(targets.items())
        ],
    }
    return {
        "run_id": run_id,
        "command": command,
        "execution_sha256": canonical_sha256(identity),
        **identity,
    }


def _patch_schema_version() -> int:
    from agent.models import PATCH_SCHEMA_VERSION  # noqa: PLC0415

    return PATCH_SCHEMA_VERSION


_MANIFEST_FIELDS = (
    ("graph_state_version", "graph state version"),
    ("patch_schema_version", "patch schema version"),
    ("offline", "offline mode"),
    ("use_examples", "example-fixture mode"),
    ("configuration_sha256", "project configuration"),
    ("limits", "attempt or project bounds"),
)

def manifest_mismatch(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> Optional[str]:
    """first field that makes the continuation unsafe, named, or None"""

    if recorded.get("execution_sha256") == current.get("execution_sha256"):
        return None

    for field_name, label in _MANIFEST_FIELDS:
        if recorded.get(field_name) != current.get(field_name):
            return (
                f"the {label} changed since the run started "
                f"({recorded.get(field_name)!r} → {current.get(field_name)!r})"
            )

    recorded_provider = recorded.get("provider") or {}
    current_provider = current.get("provider") or {}
    for key in _EXECUTION_PROVIDER_KEYS:
        if recorded_provider.get(key) != current_provider.get(key):
            return (
                f"the provider setting {key!r} changed since the run started "
                f"({recorded_provider.get(key)!r} → {current_provider.get(key)!r})"
            )

    recorded_inputs = recorded.get("project_inputs") or {}
    current_inputs = current.get("project_inputs") or {}
    for name in _EXECUTION_PROJECT_INPUTS:
        if recorded_inputs.get(name) != current_inputs.get(name):
            return f"the project's {name.replace('_', ' ')} changed since the run started"

    recorded_maps = {entry["map_key"]: entry for entry in recorded.get("maps") or []}
    current_maps = {entry["map_key"]: entry for entry in current.get("maps") or []}
    if set(recorded_maps) != set(current_maps):
        added = sorted(set(current_maps) - set(recorded_maps))
        removed = sorted(set(recorded_maps) - set(current_maps))
        return f"the selected maps changed (added {added}, removed {removed})"
    for key, entry in recorded_maps.items():
        if entry.get("baseline_sha256") != current_maps[key].get("baseline_sha256"):
            return f"{key} changed on disk since the run started"
        if entry.get("path") != current_maps[key].get("path"):
            return (
                f"{key} moved since the run started "
                f"({entry.get('path')} → {current_maps[key].get('path')})"
            )

    recorded_saver = recorded.get("checkpointer") or {}
    current_saver = current.get("checkpointer") or {}
    for key, label in (
        ("package_version", "checkpointer package version"),
        ("langgraph_version", "LangGraph version"),
        ("kind", "checkpointer"),
    ):
        if recorded_saver.get(key) != current_saver.get(key):
            return (
                f"the {label} changed since the run started "
                f"({recorded_saver.get(key)!r} → {current_saver.get(key)!r})"
            )

    if recorded.get("report_version") != current.get("report_version"):
        return "the report version changed since the run started"
    return "the run's execution identity changed in an unrecognised way"


def build_project_report(
    *,
    command: str,
    conf: Mapping[str, Any],
    run,
    run_dir: RunDirectory,
    map_reports: Mapping[str, Path],
    manifest: Mapping[str, Any],
    transaction: Optional[Mapping[str, Any]] = None,
    offline: bool = False,
) -> Dict[str, Any]:
    """create the current runs project_report.json result"""

    stats = run.stats()
    return {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "configuration": redact_config(conf),
        "offline": offline,
        "graph": {
            "state_version": manifest.get("graph_state_version"),
            "checkpointer": manifest.get("checkpointer"),
            "resumed": bool(getattr(run, "resumed", False)),
            "rounds": run.rounds,
            "assembled_digests": run.assembled_digests,
            "routing": run.routing,
        },
        "outcome": run.outcome,
        "stop_reason": run.stop_reason,
        "totals": stats.as_dict(),
        "maps": [
            {
                "map_key": entry.map_key,
                "path": entry.map_path,
                "outcome": entry.result.outcome.value,
                "stop_reason": entry.result.stop_reason,
                "round": entry.round,
                "attempts": entry.attempts_used,
                "provider_calls": entry.provider_calls_used,
                "cache_hits": entry.cache_hits,
                "origin_sha256": entry.origin_sha256,
                "staged_sha256": entry.staged_sha256,
                "staged": entry.changed,
                "requeued_for": entry.requeued_for,
                "report": str(map_reports.get(entry.map_key, "")),
            }
            for entry in run.ordered
        ],
        "global_findings": _finding_payload(run.global_findings),
        "baseline_global_findings": _finding_payload(
            getattr(run, "baseline_global_findings", [])
        ),
        "global_regressions": _finding_payload(
            getattr(run, "global_regressions", [])
        ),
        "apply": transaction
        or {"attempted": False, "applied": False, "refused_because": None, "maps": []},
        "artifacts": {"run_directory": str(run_dir.root)},
    }


def candidate_refusal(entry, *, offline: bool = False) -> Optional[str]:
    """check if a candidate of a map should be written, (or None):

    - there is no accepted candidate
    - the candidate object no longer hashes to the accepted digest
    - the run was offline
    - the file on disk no longer hashes to what the run first read (file changed in between)
    - the accepted report does not prove engine validation was both requested
       and available
    """

    from agent.patch import canonical_sha256  # noqa: PLC0415

    result = entry.result
    if result.outcome is not LoopOutcome.ACCEPTED or result.accepted_candidate is None:
        return f"No accepted candidate to apply (outcome: {result.outcome.value})."
    accepted_sha = canonical_sha256(result.accepted_candidate)
    if not result.accepted_sha256 or accepted_sha != result.accepted_sha256:
        return (
            "The accepted candidate content does not match the digest recorded by "
            "the repair run."
        )
    if offline:
        return (
            "This was an offline run, so no engine executed the candidate. An "
            "offline diagnostic candidate is never applicable."
        )

    accepted_attempt = next(
        (record for record in reversed(result.attempts) if record.accepted), None
    )
    decision = (accepted_attempt.decision if accepted_attempt else None) or {}
    if decision.get("provisional"):
        excused = ", ".join(decision.get("excused_evidence") or []) or "engine evidence"
        return (
            f"The candidate was accepted only because the environment was excused "
            f"({excused}), so no engine evidence judged it. Resolve the "
            "environment finding and re-run before applying."
        )

    try:
        current = json.loads(Path(entry.map_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"The source map could not be re-read: {exc}"
    if canonical_sha256(current) != entry.origin_sha256:
        return (
            "The source map changed since validation; the candidate was written "
            "against a revision that is no longer on disk."
        )

    report = result.accepted_report
    if (
        report is None
        or report.map_sha256 != accepted_sha
        or not report.engine_requested
        or not report.engine_available
    ):
        return (
            "The accepted candidate has no matching successful engine-validation "
            "report and is therefore not applicable."
        )
    return None


def apply_project(
    entries: Sequence[Any],
    *,
    map_dirs: Mapping[str, RunDirectory],
    offline: bool = False,
    globally_admissible: bool = True,
    global_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """final write whole projects accepted candidates, or NONE of them:
    1. refuse the batch on an offline run or a new blocking global finding
    2. evaluate every precondition before the first write, so a refusal
       leaves the working tree untouched
    3. preserve every baseline into the run directory, including for maps that
       will not change
    4. write all replacements to temporary files in their target directories
    5. only then rename them into place
    6. restore from the preserved bytes if a rename fails partway

    :return: the transaction record for the project report
    """

    entries = list(entries)
    maps: List[Dict[str, Any]] = []
    record: Dict[str, Any] = {
        "attempted": True,
        "applied": False,
        "atomic": True,
        "refused_because": None,
        "rolled_back": False,
        "maps": maps,
    }

    if offline:
        record["refused_because"] = (
            "This was an offline run, so no engine executed any candidate. Offline "
            "diagnostic candidates are never applicable."
        )
    elif not globally_admissible:
        record["refused_because"] = global_reason or (
            "The assembled set introduces blocking project-level findings; no "
            "map is written when the staged project regresses."
        )

    if record["refused_because"]:
        for entry in entries:
            maps.append(
                {
                    "map_key": entry.map_key,
                    "path": entry.map_path,
                    "status": "refused",
                    "reason": record["refused_because"],
                }
            )
        logger.error("--apply refused for the project: %s", record["refused_because"])
        return record

    writable = []
    for entry in entries:
        if not entry.changed:
            maps.append(
                {
                    "map_key": entry.map_key,
                    "path": entry.map_path,
                    "status": "unchanged",
                    "reason": f"Nothing to write (outcome: {entry.result.outcome.value}).",
                }
            )
            continue
        reason = candidate_refusal(entry, offline=offline)
        if reason:
            record["refused_because"] = (
                f"{entry.map_key}: {reason} No map was written."
            )
            break
        writable.append(entry)

    if record["refused_because"]:
        maps.clear()
        for entry in entries:
            maps.append(
                {
                    "map_key": entry.map_key,
                    "path": entry.map_path,
                    "status": "refused",
                    "reason": record["refused_because"],
                }
            )
        logger.error("--apply refused for the project: %s", record["refused_because"])
        return record

    preserved: Dict[str, bytes] = {}
    for entry in entries:
        run_dir = map_dirs[entry.map_key]
        try:
            preserved[entry.map_key] = Path(entry.map_path).read_bytes()
        except OSError:
            preserved[entry.map_key] = b""
        run_dir.write_baseline(entry.baseline)

    staged_files: List[Tuple[Any, str]] = []
    replaced: List[Any] = []
    try:
        for entry in writable:
            staged_files.append((entry, _stage_replacement(entry)))
        for entry, temporary in staged_files:
            os.replace(temporary, entry.map_path)
            replaced.append(entry)
        staged_files = []
    except OSError as exc:
        for _entry, temporary in staged_files:
            Path(temporary).unlink(missing_ok=True)
        rolled = _rollback(replaced, preserved)
        record["refused_because"] = (
            f"Writing the project failed after {len(replaced)} replacement(s): {exc}. "
            f"{rolled} file(s) were restored from the run directory."
        )
        record["rolled_back"] = True
        maps.clear()
        for entry in entries:
            maps.append(
                {
                    "map_key": entry.map_key,
                    "path": entry.map_path,
                    "status": "rolled-back" if entry in replaced else "refused",
                    "reason": record["refused_because"],
                }
            )
        logger.error("Project apply rolled back: %s", record["refused_because"])
        return record

    for entry in writable:
        maps.append(
            {
                "map_key": entry.map_key,
                "path": entry.map_path,
                "status": "replaced",
                "baseline_sha256": entry.origin_sha256,
                "replaced_with_sha256": entry.result.accepted_sha256,
            }
        )
    record["applied"] = True
    logger.info(
        "Applied %d accepted candidate(s); %d map(s) left unchanged.",
        len(writable),
        len(entries) - len(writable),
    )
    return record


def _stage_replacement(entry) -> str:
    """Write one candidate to a flushed temporary file beside its target."""

    target = Path(entry.map_path)
    payload = json.dumps(entry.staged, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return handle.name


def _rollback(replaced: Sequence[Any], preserved: Mapping[str, bytes]) -> int:
    """Put back every file this transaction already overwrote."""

    restored = 0
    for entry in replaced:
        original = preserved.get(entry.map_key)
        if not original:
            continue
        try:
            Path(entry.map_path).write_bytes(original)
            restored += 1
        except OSError as exc:  # pragma: no cover - the filesystem is failing
            logger.error("Could not restore %s: %s", entry.map_path, exc)
    return restored
