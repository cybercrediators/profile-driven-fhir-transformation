"""graph state definitions for langgraph workflow"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional

from typing_extensions import TypedDict

from agent.context import AgentContext, RejectedAttempt
from agent.loop import AttemptRecord, MapRunResult
from agent.models import AgentPatch, PatchApplication
from agent.validation import ValidationFinding, ValidationReport

GRAPH_STATE_VERSION = 1

# define project outcomes
PROJECT_OK = "ok"
PROJECT_UNRESOLVED = "unresolved"
PROJECT_NO_PROGRESS = "no-progress"
PROJECT_LIMIT_REACHED = "limit-reached"

def merge_results(
    old: Optional[Dict[str, "MapRunResult"]], new: Optional[Dict[str, "MapRunResult"]]
) -> Dict[str, "MapRunResult"]:
    """compile per-map results, keyed by stable map key"""

    merged = dict(old or {})
    merged.update(new or {})
    return merged


@dataclass(frozen=True)
class ProjectLimits:
    """bounds the project graph enforces on top of the per-map ones"""

    max_project_rounds: int = 3
    max_provider_calls: int = 40
    max_seconds: float = 3600.0


class MapRepairState(TypedDict, total=False):
    """track state of single map repair subgraphs"""

    map_key: str
    map_path: str
    map_url: Optional[str]
    map_id: Optional[str]
    round: int

    staged_before: Optional[Dict[str, Any]]
    baseline: Dict[str, Any]
    baseline_sha256: Optional[str]
    baseline_report: Optional[ValidationReport]
    generator_findings: List[ValidationFinding]
    global_feedback: List[ValidationFinding]
    worklist: List[ValidationFinding]

    attempt: Optional[AgentContext]
    authorized_pointers: List[str]
    patch: Optional[AgentPatch]
    proposal_cache_hit: bool
    proposal_prompt_chars: int
    proposal_error: Optional[str]
    proposal_fatal: bool

    application: Optional[PatchApplication]
    candidate: Optional[Dict[str, Any]]
    candidate_sha256: Optional[str]
    candidate_report: Optional[ValidationReport]
    decision: Optional[Dict[str, Any]]
    record: Optional[AttemptRecord]


    retry: bool
    
    accepted_so_far: Optional[Dict[str, Any]]
    accepted_sha256_so_far: Optional[str]
    accepted_report_so_far: Optional[ValidationReport]
    continuations: int

    attempts: List[AttemptRecord]
    rejected: List[RejectedAttempt]
    seen_candidates: List[str]
    
    seen_signatures: List[List[str]]
    attempts_used: int
    provider_calls: int
    cache_hits: int
    
    project_provider_quota: int
    elapsed_s: float
    requeued_for: List[List[str]]

    outcome: Optional[str]
    stop_reason: str

    results: Annotated[Dict[str, MapRunResult], merge_results]


class MapRepairOutput(TypedDict, total=False):
    """single map repair subgraph outcomes"""

    results: Annotated[Dict[str, MapRunResult], merge_results]


class ProjectAgentState(TypedDict, total=False):
    """track the state of project-level graph runs (no single-map)"""

    graph_state_version: int
    run_id: str
    map_order: List[str]
    pending: List[str]
    queue: List[str]
    dispatched: List[str]
    
    dispatch: List[Dict[str, Any]]
    feedback: Dict[str, List[ValidationFinding]]
    round: int
    results: Annotated[Dict[str, MapRunResult], merge_results]

    assembled_digests: List[str]
    global_findings: List[ValidationFinding]
    routing: Dict[str, List[str]]
    rounds: List[Dict[str, Any]]
    project_signatures: List[str]

    outcome: Optional[str]
    stop_reason: str
