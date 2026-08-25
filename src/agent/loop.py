"""Define rules and bounds for the agent loop"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from agent.context import AgentContext, RejectedAttempt
from agent.models import AgentPatch
from agent.validation import (
    AcceptanceDecision,
    InvariantResult,
    ValidationFinding,
    ValidationReport,
)

logger = logging.getLogger(__name__)

FEATURE = "agent-fix"


class LoopOutcome(str, Enum):
    """define reasons, why the loop stopped. Only ACCEPTED produces a candidate."""

    CLEAN = "clean"
    BLOCKED = "blocked"
    ACCEPTED = "accepted"
    EXHAUSTED = "exhausted"
    NO_PROGRESS = "no-progress"
    LIMIT_REACHED = "limit-reached"
    PROVIDER_ERROR = "provider-error"


@dataclass(frozen=True)
class LoopLimits:
    """Every bound the loop enforces"""

    max_attempts: int = 4

    max_operations: int = 40
    
    max_prompt_chars: int = 120000
    
    max_findings_per_attempt: int = 6

    max_provider_calls: int = 8
    max_seconds: float = 600.0


@dataclass
class Proposal:
    """One model answer, as the loop needs it."""

    patch: Optional[AgentPatch]
    cache_hit: bool = False
    prompt_chars: int = 0
    error: Optional[str] = None
    fatal: bool = False


class Proposer(Protocol):
    """Whatever turns a context into a patch. An LLM in production, a stub in tests."""

    def propose(self, context: AgentContext) -> Proposal: ...


class CandidateEvaluator(Protocol):
    """validate given candidates"""

    def evaluate(self, document: Mapping[str, Any]) -> ValidationReport:
        """Validate one document on its own"""
        ...

    def evaluate_pair(
        self, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> Tuple[ValidationReport, ValidationReport]:
        """validate both on identical fixtures and services"""
        ...


class AttemptRecord(BaseModel):
    """One (full) pass through the loop for the run report"""

    model_config = ConfigDict(extra="forbid")

    index: int
    prompt_chars: int = 0
    cache_hit: bool = False

    worklist: List[str] = Field(default_factory=list)

    operations: List[Dict[str, Any]] = Field(default_factory=list)
    rationale: Optional[str] = None

    application: Optional[Dict[str, Any]] = None
    candidate_sha256: Optional[str] = None
    decision: Optional[Dict[str, Any]] = None

    validation_report: Optional[ValidationReport] = None
    accepted: bool = False
    note: Optional[str] = None


class LoopResult(BaseModel):
    """Everything one maps repair run produced."""

    model_config = ConfigDict(extra="forbid")

    outcome: LoopOutcome
    map_url: Optional[str] = None
    map_id: Optional[str] = None
    baseline_sha256: Optional[str] = None
    baseline_report: Optional[ValidationReport] = None

    accepted_candidate: Optional[Dict[str, Any]] = None
    accepted_sha256: Optional[str] = None
    accepted_report: Optional[ValidationReport] = None

    attempts: List[AttemptRecord] = Field(default_factory=list)

    unresolved: List[ValidationFinding] = Field(default_factory=list)

    provider_calls: int = 0
    cache_hits: int = 0
    elapsed_s: float = 0.0
    stop_reason: str = ""

    @property
    def applied_candidate(self) -> Optional[Dict[str, Any]]:
        return self.accepted_candidate if self.outcome is LoopOutcome.ACCEPTED else None


class MapRunResult(BaseModel):
    """durable outcome of a map as the project graph keeps it"""

    model_config = ConfigDict(extra="forbid")

    map_key: str
    map_path: str
    round: int = 1
    result: LoopResult
    
    baseline: Dict[str, Any] = Field(default_factory=dict)
    origin_sha256: Optional[str] = None
    
    staged_candidate: Optional[Dict[str, Any]] = None
    staged_sha256: Optional[str] = None

    # track attempts/calls
    attempts_used: int = 0
    provider_calls_used: int = 0
    cache_hits: int = 0

    # which findings caused a re-queue
    requeued_for: List[List[str]] = Field(default_factory=list)

    
    # list rejected attempts, candidates and signatures
    rejected: List[RejectedAttempt] = Field(default_factory=list)
    seen_candidates: List[str] = Field(default_factory=list)
    seen_signatures: List[List[str]] = Field(default_factory=list)

    @property
    def staged(self) -> Dict[str, Any]:
        """What this map would be if the run were applied."""

        return self.staged_candidate or self.baseline

    @property
    def changed(self) -> bool:
        """Whether applying this run would rewrite the file."""

        return self.staged_candidate is not None


def finding_signature(report: ValidationReport) -> frozenset:
    """Normalized identity of a report's blocking findings"""

    return frozenset(report.blocking_ids())


def remaining_generator_findings(
    findings: Sequence[ValidationFinding],
    candidate: ValidationReport,
    proposal: AgentPatch,
    baseline_document: Mapping[str, Any],
    candidate_document: Mapping[str, Any],
) -> List[ValidationFinding]:
    """generator diagnostics a patch has not deterministically displaced"""

    from agent.validation import addressed_paths  # noqa: PLC0415

    addressed = set(proposal.diagnostic_ids)
    blocking_paths = {
        finding.path for finding in candidate.blocking_findings if finding.path
    }
    baseline_targets = {entry.path for entry in addressed_paths(baseline_document)[1]}
    candidate_targets = {entry.path for entry in addressed_paths(candidate_document)[1]}

    def demonstrably_displaced(finding: ValidationFinding) -> bool:
        return bool(
            finding.finding_id in addressed
            and finding.path
            and finding.path not in baseline_targets
            and finding.path in candidate_targets
            and finding.path not in blocking_paths
        )

    return [finding for finding in findings if not demonstrably_displaced(finding)]


def require_targeted_progress(
    decision: AcceptanceDecision,
    worklist: Sequence[ValidationFinding],
    candidate: ValidationReport,
) -> None:
    """turn a merely non-regressing candidate into a rejection"""

    targeted = {finding.finding_id for finding in worklist}
    remaining = candidate.finding_ids()
    resolved = sorted(targeted - remaining)
    invariant = InvariantResult(
        name="targeted-repair-progress",
        passed=bool(resolved),
        comparative=True,
        detail=(
            f"The candidate resolved {len(resolved)} targeted finding(s)."
            if resolved
            else "The candidate resolved none of the findings it was asked to repair."
        ),
        evidence={"targeted": sorted(targeted), "resolved": resolved},
    )
    decision.invariants.append(invariant)
    decision.accepted = decision.accepted and invariant.passed


def introduced_briefs(report: ValidationReport, decision: AcceptanceDecision):
    """The blocking findings this candidate added, phrased for the retry prompt."""

    from agent.context import _brief  # noqa: PLC0415

    introduced = set(decision.new_blocking_ids)
    return [
        _brief(finding)
        for finding in report.findings
        if finding.finding_id in introduced
    ][:10]


@dataclass
class LoopStats:
    """Aggregate counters across several maps, for a multi-map run summary."""

    maps: int = 0
    accepted: int = 0
    clean: int = 0
    blocked: int = 0
    unresolved: int = 0
    provider_calls: int = 0
    cache_hits: int = 0
    outcomes: Dict[str, int] = field(default_factory=dict)

    def add(self, result: LoopResult) -> None:
        self.maps += 1
        self.provider_calls += result.provider_calls
        self.cache_hits += result.cache_hits
        self.outcomes[result.outcome.value] = (
            self.outcomes.get(result.outcome.value, 0) + 1
        )
        if result.outcome is LoopOutcome.ACCEPTED:
            self.accepted += 1
        elif result.outcome is LoopOutcome.CLEAN:
            self.clean += 1
        elif result.outcome is LoopOutcome.BLOCKED:
            self.blocked += 1
        else:
            self.unresolved += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "maps": self.maps,
            "accepted": self.accepted,
            "clean": self.clean,
            "blocked": self.blocked,
            "unresolved": self.unresolved,
            "provider_calls": self.provider_calls,
            "cache_hits": self.cache_hits,
            "outcomes": dict(self.outcomes),
        }


def unresolved_by_owner(result: LoopResult) -> Dict[str, List[str]]:
    """Unresolved findings grouped by who has to act, for the run report."""

    grouped: Dict[str, List[str]] = {}
    for finding in result.unresolved:
        grouped.setdefault(finding.action_owner.value, []).append(finding.finding_id)
    return grouped

