"""Build langgraph graph for fixing structure maps

START -> load_project -> begin_round -> dispatch_next
                                         |-- queue empty ------> assemble
                                         `-> Send(repair_map) -> dispatch_next
         assemble -> validate_project -> route_global
                                          |-- actionable, named ---> begin_round
                                          `-> finalize -> END
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from agent.graph.repair_graph import build_repair_graph, initial_repair_state
from agent.graph.runtime import AgentRuntimeContext
from agent.graph.state import (
    GRAPH_STATE_VERSION,
    PROJECT_LIMIT_REACHED,
    PROJECT_NO_PROGRESS,
    PROJECT_OK,
    PROJECT_UNRESOLVED,
    ProjectAgentState,
)
from agent.loop import LoopOutcome, LoopStats, MapRunResult
from agent.patch import canonical_sha256
from agent.validation import ActionOwner, ValidationFinding

logger = logging.getLogger(__name__)

PROJECT_RECURSION_LIMIT = 200


def project_recursion_limit(maps: int, rounds: int) -> int:
    """Supersteps to allow: two per map dispatched, plus the round's own nodes."""

    return max(PROJECT_RECURSION_LIMIT, rounds * (2 * max(1, maps) + 8) + 20)

_SETTLED = frozenset({LoopOutcome.ACCEPTED.value, LoopOutcome.CLEAN.value})


def load_project(
    state: ProjectAgentState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """create fixed set of given maps for graph and order for the complete run"""

    context = runtime.context
    assert context is not None
    order = list(context.documents)
    logger.info("Agent project graph: %d map(s) selected.", len(order))
    return {
        "graph_state_version": GRAPH_STATE_VERSION,
        "map_order": order,
        "pending": order,
        "round": 0,
        "results": {},
        "assembled_digests": [],
        "global_findings": [],
        "routing": {},
        "rounds": [],
        "project_signatures": [],
        "outcome": None,
        "stop_reason": "",
    }


def begin_round(
    state: ProjectAgentState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """Open a round: queue the pending maps, in the run's fixed order."""

    return {
        "round": int(state.get("round", 0)) + 1,
        "queue": list(state.get("pending") or []),
        "dispatched": [],
        "dispatch": [],
    }


def dispatch_next(
    state: ProjectAgentState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """create payload for one maps fix request"""

    context = runtime.context
    assert context is not None
    queue = list(state.get("queue") or [])
    if not queue:
        return {"dispatch": [], "queue": []}

    key, rest = queue[0], queue[1:]
    results: Dict[str, MapRunResult] = dict(state.get("results") or {})
    feedback: Dict[str, List[ValidationFinding]] = dict(state.get("feedback") or {})
    previous = results.get(key)
    routed = feedback.get(key) or []
    used = previous.provider_calls_used if previous else 0
    spent = sum(entry.provider_calls_used for entry in results.values())
    remaining = max(0, context.project_limits.max_provider_calls - spent + used)

    payload = initial_repair_state(
        key,
        round_index=int(state.get("round", 1)),
        staged_before=previous.staged_candidate if previous else None,
        generator_findings=context.service.generator_findings(context.document(key)),
        global_feedback=routed,
        attempts_used=previous.attempts_used if previous else 0,
        provider_calls=used,
        cache_hits=previous.cache_hits if previous else 0,
        project_provider_quota=min(context.limits.max_provider_calls, remaining),
        elapsed_s=previous.result.elapsed_s if previous else 0.0,
        attempts=previous.result.attempts if previous else (),
        rejected=previous.rejected if previous else (),
        seen_candidates=previous.seen_candidates if previous else (),
        seen_signatures=previous.seen_signatures if previous else (),
        requeued_for=(
            [*previous.requeued_for, [f.finding_id for f in routed]]
            if previous and routed
            else (previous.requeued_for if previous else [])
        ),
    )
    return {
        "dispatch": [payload],
        "queue": rest,
        "dispatched": [*(state.get("dispatched") or []), key],
    }


def dispatch(state: ProjectAgentState):
    """Send the map this step prepared, or move on to assembly."""

    payloads = state.get("dispatch") or []
    if not payloads:
        return "assemble"
    return [Send("repair_map", payload) for payload in payloads]


def assemble(state: ProjectAgentState, runtime: Runtime[AgentRuntimeContext]) -> dict:
    """The set as it would exist on disk: accepted candidates, baselines elsewhere."""

    context = runtime.context
    assert context is not None
    results: Dict[str, MapRunResult] = dict(state.get("results") or {})
    order = [key for key in state.get("map_order") or [] if key in results]
    digest = canonical_sha256(
        {"maps": [[key, results[key].staged_sha256] for key in order]}
    )
    rounds = list(state.get("rounds") or [])
    rounds.append(
        {
            "round": int(state.get("round", 1)),
            "dispatched": list(state.get("dispatched") or []),
            "assembled_sha256": digest,
            "outcomes": {
                key: results[key].result.outcome.value for key in order
            },
        }
    )
    return {
        "assembled_digests": [*(state.get("assembled_digests") or []), digest],
        "rounds": rounds,
        "dispatch": [],
        "queue": [],
    }


def validate_project(
    state: ProjectAgentState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """validate maps on a project-level based on deterministic validation checks"""

    context = runtime.context
    assert context is not None
    assembled = assembled_set(state, context)
    findings: List[ValidationFinding] = []
    for validator in context.project_validators:
        findings.extend(validator(assembled))
    for finding in findings:
        logger.warning("Project-level: %s", finding.message)
    return {"global_findings": findings}


def assembled_set(
    state: ProjectAgentState, context: AgentRuntimeContext
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """(staged map, its target profile) pairs, in the run map order"""

    results: Dict[str, MapRunResult] = dict(state.get("results") or {})
    assembled = []
    for key in state.get("map_order") or []:
        entry = results.get(key)
        if entry is None:
            continue
        staged = entry.staged
        assembled.append((staged, context.profile_for(staged)))
    return assembled


def baseline_project_findings(
    context: AgentRuntimeContext,
) -> List[ValidationFinding]:
    """retrieve project validator results over unmodified set of maps"""

    assembled = [
        (context.document(key), context.profile_for(context.document(key)))
        for key in context.documents
    ]
    findings: List[ValidationFinding] = []
    for validator in context.project_validators:
        findings.extend(validator(assembled))
    return findings


def route_global(
    state: ProjectAgentState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """Decide whether any map can act on a global finding."""

    context = runtime.context
    assert context is not None
    results: Dict[str, MapRunResult] = dict(state.get("results") or {})
    findings = list(state.get("global_findings") or [])
    blocking = [finding for finding in findings if finding.blocking]

    routing: Dict[str, List[str]] = {}
    feedback: Dict[str, List[ValidationFinding]] = {}
    for finding in findings:
        if finding.action_owner is not ActionOwner.MAP_FIXABLE:
            continue
        key = _owning_map(finding, results)
        if key is None:
            logger.warning(
                "Global finding %s is map-fixable but names no map in this run; "
                "it is reported rather than routed.",
                finding.finding_id,
            )
            continue
        routing.setdefault(key, []).append(finding.finding_id)
        feedback.setdefault(key, []).append(finding)

    signature = canonical_sha256(
        {
            "assembled": (state.get("assembled_digests") or [None])[-1],
            "blocking": sorted(finding.finding_id for finding in blocking),
        }
    )
    seen = list(state.get("project_signatures") or [])
    rounds = list(state.get("rounds") or [])
    if rounds:
        rounds[-1] = {
            **rounds[-1],
            "global_findings": sorted(finding.finding_id for finding in findings),
            "routed": {key: list(ids) for key, ids in routing.items()},
        }
    update: Dict[str, Any] = {
        "routing": routing,
        "rounds": rounds,
        "project_signatures": [*seen, signature],
    }

    if not routing:
        return {**update, "pending": [], "feedback": {}}

    limits = context.project_limits
    if signature in seen:
        return {
            **update,
            "pending": [],
            "feedback": {},
            "outcome": PROJECT_NO_PROGRESS,
            "stop_reason": (
                "The assembled set and its blocking global findings repeated; "
                "the project loop is not converging."
            ),
        }
    if int(state.get("round", 1)) >= limits.max_project_rounds:
        return {
            **update,
            "pending": [],
            "feedback": {},
            "outcome": PROJECT_LIMIT_REACHED,
            "stop_reason": (
                f"The project round budget of {limits.max_project_rounds} is "
                "exhausted with global findings still routable."
            ),
        }
    if context.out_of_time():
        return {
            **update,
            "pending": [],
            "feedback": {},
            "outcome": PROJECT_LIMIT_REACHED,
            "stop_reason": f"The project time budget of {limits.max_seconds}s ran out.",
        }
    spent = sum(entry.provider_calls_used for entry in results.values())
    if spent >= limits.max_provider_calls:
        return {
            **update,
            "pending": [],
            "feedback": {},
            "outcome": PROJECT_LIMIT_REACHED,
            "stop_reason": (
                f"The project provider-call budget of {limits.max_provider_calls} "
                "is exhausted."
            ),
        }

    logger.info(
        "Round %d routes %d global finding(s) back to %d map(s).",
        int(state.get("round", 1)),
        sum(len(ids) for ids in routing.values()),
        len(routing),
    )
    return {**update, "pending": sorted(routing), "feedback": feedback}


def _owning_map(
    finding: ValidationFinding, results: Dict[str, MapRunResult]
) -> Optional[str]:
    """Which map a global finding names, by identity rather than by wording."""

    for key, entry in results.items():
        staged = entry.staged
        if finding.map_url and finding.map_url == staged.get("url"):
            return key
        if finding.map_id and finding.map_id == staged.get("id"):
            return key
    return None


def after_route(state: ProjectAgentState) -> str:
    if state.get("outcome"):
        return "finalize"
    return "begin_round" if state.get("pending") else "finalize"


def finalize(state: ProjectAgentState, runtime: Runtime[AgentRuntimeContext]) -> dict:
    """Finalize project node (for a complete project-level run, not map-level verdicts)"""

    results: Dict[str, MapRunResult] = dict(state.get("results") or {})
    blocking = [
        finding for finding in state.get("global_findings") or [] if finding.blocking
    ]
    unsettled = sorted(
        key
        for key, entry in results.items()
        if entry.result.outcome.value not in _SETTLED
    )

    if state.get("outcome"):
        return {}
    if blocking:
        return {
            "outcome": PROJECT_UNRESOLVED,
            "stop_reason": (
                f"{len(blocking)} blocking finding(s) remain on the assembled set."
            ),
        }
    if unsettled:
        return {
            "outcome": PROJECT_UNRESOLVED,
            "stop_reason": (
                f"{len(unsettled)} map(s) did not reach an accepted or clean "
                "outcome: " + ", ".join(unsettled)
            ),
        }
    return {
        "outcome": PROJECT_OK,
        "stop_reason": "Every map is clean or accepted and the assembled set passes.",
    }


def build_project_graph(checkpointer=None, *, repair_graph=None):
    """compile the project graph, with the repair subgraph as a node"""

    graph = StateGraph(ProjectAgentState, context_schema=AgentRuntimeContext)
    graph.add_node("load_project", load_project)
    graph.add_node("begin_round", begin_round)
    graph.add_node("dispatch_next", dispatch_next)
    graph.add_node("repair_map", repair_graph or build_repair_graph())
    graph.add_node("assemble", assemble)
    graph.add_node("validate_project", validate_project)
    graph.add_node("route_global", route_global)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "load_project")
    graph.add_edge("load_project", "begin_round")
    graph.add_edge("begin_round", "dispatch_next")
    graph.add_conditional_edges("dispatch_next", dispatch, ["repair_map", "assemble"])
    graph.add_edge("repair_map", "dispatch_next")
    graph.add_edge("assemble", "validate_project")
    graph.add_edge("validate_project", "route_global")
    graph.add_conditional_edges("route_global", after_route, ["begin_round", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


@dataclass
class ProjectRun:
    """What one invocation of the project graph produced"""

    run_id: str
    outcome: str
    stop_reason: str
    results: Dict[str, MapRunResult] = field(default_factory=dict)
    map_order: List[str] = field(default_factory=list)
    global_findings: List[ValidationFinding] = field(default_factory=list)
    baseline_global_findings: List[ValidationFinding] = field(default_factory=list)
    routing: Dict[str, List[str]] = field(default_factory=dict)
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    assembled_digests: List[str] = field(default_factory=list)
    resumed: bool = False

    @property
    def ordered(self) -> List[MapRunResult]:
        return [self.results[key] for key in self.map_order if key in self.results]

    @property
    def blocking_global(self) -> List[ValidationFinding]:
        return [finding for finding in self.global_findings if finding.blocking]

    @property
    def global_regressions(self) -> List[ValidationFinding]:
        """Blocking project findings introduced by the staged map set."""

        baseline = {
            finding.finding_id
            for finding in self.baseline_global_findings
            if finding.blocking
        }
        return [
            finding
            for finding in self.blocking_global
            if finding.finding_id not in baseline
        ]

    @property
    def globally_admissible(self) -> bool:
        """whether the assembled set introduced no project-level regression"""

        return not self.global_regressions

    def stats(self) -> LoopStats:
        stats = LoopStats()
        for entry in self.ordered:
            stats.add(entry.result)
        return stats


def run_project_graph(
    *,
    context: AgentRuntimeContext,
    run_id: str,
    checkpointer=None,
    resume: bool = False,
) -> ProjectRun:
    """Invoke, or resume, the project graph for one run.

    :param resume: continue the checkpointed thread instead of starting it.
    """

    compiled = build_project_graph(checkpointer)
    config = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": project_recursion_limit(
            len(context.documents), context.project_limits.max_project_rounds
        ),
        "max_concurrency": 1,
    }
    state = compiled.invoke(None if resume else {}, config=config, context=context)
    return ProjectRun(
        run_id=run_id,
        outcome=str(state.get("outcome") or PROJECT_UNRESOLVED),
        stop_reason=str(state.get("stop_reason") or ""),
        results=dict(state.get("results") or {}),
        map_order=list(state.get("map_order") or []),
        global_findings=list(state.get("global_findings") or []),
        routing=dict(state.get("routing") or {}),
        rounds=list(state.get("rounds") or []),
        assembled_digests=list(state.get("assembled_digests") or []),
        resumed=resume,
    )


__all__: Sequence[str] = (
    "PROJECT_RECURSION_LIMIT",
    "project_recursion_limit",
    "ProjectRun",
    "assembled_set",
    "baseline_project_findings",
    "build_project_graph",
    "run_project_graph",
)
