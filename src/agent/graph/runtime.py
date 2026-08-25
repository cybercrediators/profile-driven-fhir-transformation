"""helper functions for graph nodes"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as package_version
import logging
from pathlib import Path
import sqlite3
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from agent.loop import LoopLimits
from agent.graph.state import ProjectLimits

logger = logging.getLogger(__name__)

CHECKPOINT_DB_NAME = "checkpoints.sqlite"

# define checkpoint types for reporting and validation
CHECKPOINT_TYPES: Tuple[Tuple[str, str], ...] = (
    ("agent.context", "AgentContext"),
    ("agent.validation", "ActionOwner"),
    ("agent.validation", "GateStatus"),
    ("agent.validation", "Producer"),
    ("agent.validation", "Stage"),
    ("agent.loop", "LoopOutcome"),
    ("agent.models", "PatchOp"),
    ("agent.models", "RejectionCode"),
    ("agent.context", "ContextListStatus"),
    ("agent.context", "FindingBrief"),
    ("agent.context", "InsertionPoint"),
    ("agent.context", "RejectedAttempt"),
    ("agent.context", "RulePointer"),
    ("agent.context", "TargetElementBrief"),
    ("agent.loop", "AttemptRecord"),
    ("agent.loop", "LoopResult"),
    ("agent.loop", "MapRunResult"),
    ("agent.models", "AddOperation"),
    ("agent.models", "AgentPatch"),
    ("agent.models", "PatchApplication"),
    ("agent.models", "PatchRejection"),
    ("agent.models", "RemoveOperation"),
    ("agent.models", "ReplaceOperation"),
    ("agent.models", "TestOperation"),
    ("agent.validation", "AcceptanceDecision"),
    ("agent.validation", "InvariantResult"),
    ("agent.validation", "ValidationFinding"),
    ("agent.validation", "ValidationReport"),
    ("agent.validation.models", "ActionOwner"),
    ("agent.validation.models", "GateStatus"),
    ("agent.validation.models", "Producer"),
    ("agent.validation.models", "Stage"),
    ("agent.validation.models", "ValidationFinding"),
    ("agent.validation.models", "ValidationReport"),
    ("agent.validation.acceptance", "AcceptanceDecision"),
    ("agent.validation.acceptance", "InvariantResult"),
)


def checkpointer_identity() -> Dict[str, Any]:
    """record run information for report and resume functionality"""

    try:
        saver_version = package_version("langgraph-checkpoint-sqlite")
    except PackageNotFoundError:  # pragma: no cover - packaging accident
        saver_version = "unknown"
    try:
        graph_version = package_version("langgraph")
    except PackageNotFoundError:  # pragma: no cover - packaging accident
        graph_version = "unknown"
    return {
        "kind": "sqlite",
        "package": "langgraph-checkpoint-sqlite",
        "package_version": saver_version,
        "langgraph_version": graph_version,
        "database": CHECKPOINT_DB_NAME,
    }


@contextmanager
def open_checkpointer(run_root: Path) -> Iterator[Any]:
    """checkpointer inside the run directory"""

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: PLC0415
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415

    serde: Any = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)

    Path(run_root).mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(Path(run_root) / CHECKPOINT_DB_NAME), check_same_thread=False
    )
    try:
        yield SqliteSaver(connection, serde=serde)
    finally:
        connection.close()


@dataclass
class AgentRuntimeContext:
    """everything the nodes call, and nothing they store

    :param service: the :class:`~agent.service.AgentFixService` for this project.
    :param documents: ``{map key: (path, baseline document)}`` resolved once, so
        every round reads the same revision and assembly does not re-read files
        that a concurrent edit may have changed underneath the run.
    :param project_validators: deterministic checks over the assembled set. The
        shipped one is ``recheck_cross_map_references``; the list exists so a
        new project-level check becomes a routing input without touching the
        graph.
    :param evaluators: cache of per-map evaluators. A ``MapEvaluator`` bootstraps
        a Matchbox session, which must not happen twice for one map in one run.
    """

    service: Any
    documents: Dict[str, Tuple[Path, Dict[str, Any]]] = field(default_factory=dict)
    limits: LoopLimits = field(default_factory=LoopLimits)
    project_limits: ProjectLimits = field(default_factory=ProjectLimits)
    require_engine: bool = True
    project_validators: Sequence[Callable[[Sequence[Any]], List[Any]]] = ()
    started_at: float = field(default_factory=time.monotonic)
    evaluators: Dict[str, Any] = field(default_factory=dict)

    def document(self, map_key: str) -> Dict[str, Any]:
        return self.documents[map_key][1]

    def path(self, map_key: str) -> Path:
        return self.documents[map_key][0]

    def evaluator(self, map_key: str):
        """evaluation for one map, built once per run."""

        evaluator = self.evaluators.get(map_key)
        if evaluator is None:
            evaluator = self.service.evaluator_for(self.document(map_key))
            self.evaluators[map_key] = evaluator
        return evaluator

    def profile_for(self, document: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        return self.service.project.profile_for(document)

    def out_of_time(self) -> bool:
        return (time.monotonic() - self.started_at) > self.project_limits.max_seconds

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


def default_project_validators(
    conf: Optional[Mapping[str, Any]] = None,
) -> List[Callable[[Sequence[Any]], List[Any]]]:
    """The deterministic project-level checks that ship with the tool

    :param conf: the project configuration, so the check sees the same
        ``external_reference_defaults`` the bundle assembler applies. Omitted,
        the check runs without them and reports a configured reference as an
        unmet dependency.
    """

    from functools import partial  # noqa: PLC0415

    from agent.validation import recheck_cross_map_references  # noqa: PLC0415

    external = list((conf or {}).get("external_reference_defaults") or [])
    return [partial(recheck_cross_map_references, external_defaults=external)]
