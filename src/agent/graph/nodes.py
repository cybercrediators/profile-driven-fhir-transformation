"""Defined langgraph nodes"""

from __future__ import annotations

from dataclasses import replace
import logging
import time
from typing import Any, Dict, List

from langgraph.runtime import Runtime

from agent.context import RejectedAttempt, build_context
from agent.graph.runtime import AgentRuntimeContext
from agent.graph.state import MapRepairState
from agent.loop import (
    AttemptRecord,
    LoopOutcome,
    LoopResult,
    MapRunResult,
    finding_signature,
    introduced_briefs,
    remaining_generator_findings,
    require_targeted_progress,
)
from agent.patch import apply_patch, canonical_sha256
from agent.policy import PatchPolicy
from agent.validation import ValidationFinding, decide_acceptance

logger = logging.getLogger(__name__)


def _tick(state: MapRepairState, started: float) -> float:
    """track elapsed time for reporting purposes"""

    return float(state.get("elapsed_s", 0.0)) + (time.monotonic() - started)


def _stop(outcome: LoopOutcome, reason: str, **extra) -> dict:
    return {"outcome": outcome.value, "stop_reason": reason, **extra}


def _close_attempt(state: MapRepairState, record: AttemptRecord, **extra) -> dict:
    """File the attempt record and clear the slot for the next one."""

    return {
        "attempts": [*(state.get("attempts") or []), record],
        "record": None,
        **extra,
    }


def _rejected(
    state: MapRepairState, attempt: RejectedAttempt
) -> List[RejectedAttempt]:
    return [*(state.get("rejected") or []), attempt]


def _extend(existing, additions) -> List[Any]:
    """Append without duplicating, the carried history plus this round"""

    merged = list(existing or [])
    for item in additions:
        if item not in merged:
            merged.append(item)
    return merged


def _inherited_findings(state: MapRepairState) -> List[ValidationFinding]:
    """Findings this map was handed rather than derived: WP1's and the project's"""

    return [
        *(state.get("generator_findings") or []),
        *(state.get("global_feedback") or []),
    ]

def validate_baseline(
    state: MapRepairState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """Baseline validation node"""

    started = time.monotonic()
    context = runtime.context
    assert context is not None
    key = state["map_key"]
    document = dict(state.get("staged_before") or context.document(key))

    report = context.evaluator(key).evaluate(document)
    report.findings.extend(_inherited_findings(state))

    complete_worklist = report.worklist()
    worklist = complete_worklist[: context.limits.max_findings_per_attempt]
    if len(worklist) < len(complete_worklist):
        logger.info(
            "Repairing a bounded batch of %d/%d map-fixable findings.",
            len(worklist),
            len(complete_worklist),
        )
    update: Dict[str, Any] = {
        "baseline": document,
        "baseline_sha256": report.map_sha256,
        "baseline_report": report,
        "map_url": document.get("url"),
        "map_id": document.get("id"),
        "map_path": str(context.path(key)),
        "worklist": worklist,
        "seen_candidates": _extend(
            state.get("seen_candidates"), [report.map_sha256] if report.map_sha256 else []
        ),
        "seen_signatures": _extend(
            state.get("seen_signatures"), [sorted(report.blocking_ids())]
        ),
        "retry": False,
        "elapsed_s": _tick(state, started),
    }

    if worklist:
        return update

    blocking = report.blocking_findings
    if blocking:
        reason = (
            f"{len(blocking)} blocking finding(s) remain, none of them "
            "map-fixable; no provider call was made."
        )
        logger.info("Agent mode stopped without calling a provider: %s", reason)
        update.update(_stop(LoopOutcome.BLOCKED, reason))
    else:
        update.update(_stop(LoopOutcome.CLEAN, "The baseline has no findings."))
    return update


def open_attempt(state: MapRepairState, runtime: Runtime[AgentRuntimeContext]) -> dict:
    """Node to check the remaining fix attempts"""

    context = runtime.context
    assert context is not None
    limits = context.limits
    used = int(state.get("attempts_used", 0))
    provider_calls = int(state.get("provider_calls", 0))

    if used >= limits.max_attempts:
        return _stop(
            LoopOutcome.EXHAUSTED,
            f"No candidate passed within {limits.max_attempts} attempt(s).",
        )
    if float(state.get("elapsed_s", 0.0)) > limits.max_seconds:
        return _stop(
            LoopOutcome.LIMIT_REACHED,
            f"Time budget of {limits.max_seconds}s exhausted.",
        )
    if provider_calls >= limits.max_provider_calls:
        return _stop(
            LoopOutcome.LIMIT_REACHED,
            f"Provider-call budget of {limits.max_provider_calls} exhausted.",
        )
    quota = int(
        state.get("project_provider_quota", context.project_limits.max_provider_calls)
    )
    if provider_calls >= quota:
        return _stop(
            LoopOutcome.LIMIT_REACHED,
            f"This map's reserved share of the project provider budget ({quota} "
            f"call(s) of {context.project_limits.max_provider_calls}) is exhausted.",
        )
    if context.out_of_time():
        return _stop(
            LoopOutcome.LIMIT_REACHED,
            f"The project time budget of {context.project_limits.max_seconds}s "
            "was exceeded.",
        )

    index = used + 1
    return {
        "attempts_used": index,
        "retry": False,
        "record": AttemptRecord(
            index=index,
            worklist=[finding.finding_id for finding in state.get("worklist") or []],
        ),
    }


def build_prompt_context(
    state: MapRepairState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """Node to create the context for the prompt"""

    started = time.monotonic()
    context = runtime.context
    assert context is not None
    evaluator = context.evaluator(state["map_key"])
    report = state.get("baseline_report")
    record = state.get("record") or AttemptRecord(index=int(state.get("attempts_used", 1)))

    agent_context = build_context(
        state["baseline"],
        state.get("worklist") or [],
        all_findings=report.findings if report is not None else [],
        target_tree=getattr(evaluator, "target_tree", None),
        mapping_table=context.service.project.mapping_table,
        source_fields=getattr(evaluator, "source_field_specs", []),
        profile_url=getattr(evaluator, "profile_url", None),
        attempts=state.get("rejected") or [],
    )
    authorized = [
        pointer.pointer for pointer in agent_context.pointers if pointer.focused
    ] + [point.pointer for point in agent_context.insertion_points]

    if not authorized:
        from agent.context import finding_path  # noqa: PLC0415

        worklist = list(state.get("worklist") or [])
        unlocated = [
            finding
            for finding in worklist
            if not finding_path(finding)
            and not finding.pointer
            and not finding.evidence.get("rule_hint")
        ]
        codes = sorted({finding.code for finding in unlocated or worklist})
        if unlocated:
            reason = (
                f"{len(unlocated)} of {len(worklist)} map-fixable finding(s) name "
                "no element, rule or pointer in the map, so there is nowhere a "
                f"patch could be authorized to write ({', '.join(codes)}); no "
                "provider call was made."
            )
        else:
            located = sorted({finding_path(finding) for finding in worklist} - {""})
            reason = (
                f"{len(worklist)} map-fixable finding(s) name an element no rule "
                f"in this map addresses ({', '.join(located) or ', '.join(codes)}), "
                "so there is nowhere a patch could be authorized to write; no "
                "provider call was made."
            )
        record.note = reason
        return _close_attempt(
            state,
            record,
            elapsed_s=_tick(state, started),
            **_stop(LoopOutcome.BLOCKED, reason),
        )

    preflight = getattr(context.service.proposer, "prompt_chars", None)
    if callable(preflight):
        estimated = int(preflight(agent_context))
        if estimated > context.limits.max_prompt_chars:
            reason = (
                f"Prompt of {estimated} characters exceeds the "
                f"{context.limits.max_prompt_chars} budget."
            )
            record.prompt_chars = estimated
            record.note = reason
            return _close_attempt(
                state,
                record,
                elapsed_s=_tick(state, started),
                **_stop(LoopOutcome.LIMIT_REACHED, reason),
            )

    return {
        "attempt": agent_context,
        "authorized_pointers": authorized,
        "record": record,
        "elapsed_s": _tick(state, started),
    }


def request_patch(state: MapRepairState, runtime: Runtime[AgentRuntimeContext]) -> dict:
    """The one node that may talk to a provider."""

    started = time.monotonic()
    context = runtime.context
    assert context is not None
    record = state["record"]
    assert record is not None

    proposal = context.service.proposer.propose(state["attempt"])
    record.prompt_chars = proposal.prompt_chars
    record.cache_hit = proposal.cache_hit

    update: Dict[str, Any] = {
        "patch": proposal.patch,
        "proposal_cache_hit": proposal.cache_hit,
        "proposal_prompt_chars": proposal.prompt_chars,
        "proposal_error": proposal.error,
        "proposal_fatal": proposal.fatal,
        "record": record,
        "cache_hits": int(state.get("cache_hits", 0)) + (1 if proposal.cache_hit else 0),
        "provider_calls": int(state.get("provider_calls", 0))
        + (0 if proposal.cache_hit else 1),
        "elapsed_s": _tick(state, started),
    }

    if proposal.prompt_chars > context.limits.max_prompt_chars:
        reason = (
            f"Prompt of {proposal.prompt_chars} characters exceeds the "
            f"{context.limits.max_prompt_chars} budget."
        )
        record.note = reason
        update.update(_close_attempt(state, record))
        update.update(_stop(LoopOutcome.LIMIT_REACHED, reason))
        return update

    if update["elapsed_s"] > context.limits.max_seconds:
        reason = (
            f"Time budget of {context.limits.max_seconds}s was exceeded during "
            "the provider call; its answer was not evaluated."
        )
        record.note = reason
        update.update(_close_attempt(state, record))
        update.update(_stop(LoopOutcome.LIMIT_REACHED, reason))
    return update


def screen_patch(state: MapRepairState, runtime: Runtime[AgentRuntimeContext]) -> dict:
    """Check returned patches for the given maps"""

    record = state["record"]
    assert record is not None
    patch = state.get("patch")

    if patch is None:
        note = state.get("proposal_error") or "The provider returned no usable patch."
        record.note = note
        if state.get("proposal_fatal"):
            return _close_attempt(
                state, record, **_stop(LoopOutcome.PROVIDER_ERROR, note)
            )
        return _close_attempt(
            state,
            record,
            retry=True,
            rejected=_rejected(
                state,
                RejectedAttempt(
                    attempt=record.index,
                    rejections=[{"code": "no-patch", "message": note}],
                ),
            ),
        )

    record.operations = patch.as_rfc6902()
    record.rationale = patch.rationale

    offered = {finding.finding_id for finding in state.get("worklist") or []}
    claimed = set(patch.diagnostic_ids)
    if not claimed or not claimed.issubset(offered):
        note = (
            "The patch diagnostic_ids must name at least one finding from this "
            "attempt's worklist and no finding outside it."
        )
        record.note = note
        record.application = {
            "applied": False,
            "rejections": [{"code": "diagnostic-scope-mismatch", "message": note}],
        }
        return _close_attempt(
            state,
            record,
            retry=True,
            rejected=_rejected(
                state,
                RejectedAttempt(
                    attempt=record.index,
                    rejections=record.application["rejections"],
                    operations=record.operations,
                ),
            ),
        )

    if not patch.patch:
        note = patch.rationale or (
            "The model reported that no valid map patch can resolve the worklist."
        )
        record.note = note
        record.application = {"applied": False, "abstained": True, "rejections": []}
        return _close_attempt(
            state,
            record,
            **_stop(
                LoopOutcome.NO_PROGRESS,
                "The model abstained; the baseline was left unchanged and its "
                "findings remain unresolved.",
            ),
        )

    return {"record": record}


def apply_patch_in_memory(
    state: MapRepairState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """Check if created patch is valid and apply to document in-memory (without writing)"""

    started = time.monotonic()
    context = runtime.context
    assert context is not None
    record = state["record"]
    patch = state.get("patch")
    assert record is not None and patch is not None

    policy = PatchPolicy(max_operations=context.limits.max_operations)
    policy = replace(
        policy,
        max_operations=min(policy.max_operations, context.limits.max_operations),
        allowed_pointer_prefixes=tuple(state.get("authorized_pointers") or []),
    )
    application = apply_patch(dict(state["baseline"]), patch, policy=policy)
    record.application = application.as_report_dict()

    if not application.applied or application.candidate is None:
        record.note = "The patch was rejected by policy."
        return _close_attempt(
            state,
            record,
            retry=True,
            elapsed_s=_tick(state, started),
            rejected=_rejected(
                state,
                RejectedAttempt(
                    attempt=record.index,
                    rejections=[
                        rejection.as_dict() for rejection in application.rejections
                    ],
                    operations=record.operations,
                ),
            ),
        )

    record.candidate_sha256 = application.candidate_sha256
    if application.candidate_sha256 in set(state.get("seen_candidates") or []):
        record.note = "This candidate was produced before."
        return _close_attempt(
            state,
            record,
            elapsed_s=_tick(state, started),
            **_stop(
                LoopOutcome.NO_PROGRESS,
                "The model produced a candidate identical to one already "
                "evaluated; further attempts cannot differ.",
            ),
        )

    return {
        "candidate": application.candidate,
        "candidate_sha256": application.candidate_sha256,
        "application": application,
        "seen_candidates": [
            *(state.get("seen_candidates") or []),
            application.candidate_sha256,
        ],
        "record": record,
        "elapsed_s": _tick(state, started),
    }


def validate_candidate(
    state: MapRepairState, runtime: Runtime[AgentRuntimeContext]
) -> dict:
    """node for the prelim. validation of a possible given map fix"""

    started = time.monotonic()
    context = runtime.context
    assert context is not None
    record = state["record"]
    patch = state.get("patch")
    candidate = state.get("candidate")
    assert record is not None and patch is not None and candidate is not None

    evaluator = context.evaluator(state["map_key"])
    baseline_report, candidate_report = evaluator.evaluate_pair(
        state["baseline"], candidate
    )
    baseline_report.findings.extend(_inherited_findings(state))
    candidate_report.findings.extend(
        remaining_generator_findings(
            state.get("generator_findings") or [],
            candidate_report,
            patch,
            state["baseline"],
            candidate,
        )
    )

    decision = decide_acceptance(
        baseline_report,
        candidate_report,
        application=state.get("application"),
        require_engine=context.require_engine,
        baseline_document=state["baseline"],
        candidate_document=candidate,
        target_tree=getattr(context.evaluator(state["map_key"]), "target_tree", None),
    )
    claimed = set(patch.diagnostic_ids)
    require_targeted_progress(
        decision,
        [
            finding
            for finding in state.get("worklist") or []
            if finding.finding_id in claimed
        ],
        candidate_report,
    )
    record.validation_report = candidate_report
    record.decision = decision.as_report_dict()

    update: Dict[str, Any] = {
        "baseline_report": baseline_report,
        "candidate_report": candidate_report,
        "decision": record.decision,
        "elapsed_s": _tick(state, started),
    }

    if update["elapsed_s"] > context.limits.max_seconds:
        reason = (
            f"Time budget of {context.limits.max_seconds}s was exceeded during "
            "validation; the candidate was not accepted."
        )
        record.note = reason
        update.update(_close_attempt(state, record))
        update.update(_stop(LoopOutcome.LIMIT_REACHED, reason))
        return update

    if decision.accepted:
        record.accepted = True
        update.update(_close_attempt(state, record))
        update.update(
            {
                "accepted_so_far": candidate,
                "accepted_sha256_so_far": state.get("candidate_sha256"),
                "accepted_report_so_far": candidate_report,
            }
        )

        remaining = candidate_report.worklist()
        used = int(state.get("attempts_used", 0))
        budget_left = (
            used < context.limits.max_attempts
            and int(state.get("provider_calls", 0)) < context.limits.max_provider_calls
            and not context.out_of_time()
        )
        if remaining and budget_left:
            logger.info(
                "Candidate accepted on attempt %d with %d map-fixable finding(s) "
                "still open; continuing from the repaired revision.",
                record.index,
                len(remaining),
            )
            update.update(
                {
                    "staged_before": candidate,
                    "continuations": int(state.get("continuations", 0)) + 1,
                    "retry": False,
                    "patch": None,
                    "application": None,
                    "attempt": None,
                    "authorized_pointers": [],
                    "record": None,
                }
            )
            return update

        update.update(
            _stop(
                LoopOutcome.ACCEPTED,
                f"Candidate accepted on attempt {record.index}; every hard "
                "invariant passed."
                + (
                    ""
                    if not remaining
                    else f" {len(remaining)} map-fixable finding(s) remain, but the "
                    "budget for this map is spent."
                ),
            )
        )
        return update

    record.note = "; ".join(item.name for item in decision.failures)
    signature = sorted(finding_signature(candidate_report))
    update.update(
        _close_attempt(
            state,
            record,
            rejected=_rejected(
                state,
                RejectedAttempt(
                    attempt=record.index,
                    failed_invariants=[
                        {"name": item.name, "detail": item.detail}
                        for item in decision.failures
                    ],
                    introduced=introduced_briefs(candidate_report, decision),
                    operations=record.operations,
                ),
            ),
        )
    )

    if signature in (state.get("seen_signatures") or []):
        update.update(
            _stop(
                LoopOutcome.NO_PROGRESS,
                "A different candidate produced the same blocking findings; the "
                "loop is oscillating rather than converging.",
            )
        )
        return update

    update["seen_signatures"] = [*(state.get("seen_signatures") or []), signature]
    update["retry"] = True
    return update


def finish(state: MapRepairState, runtime: Runtime[AgentRuntimeContext]) -> dict:
    """Final graph state before providing results back to caller"""

    context = runtime.context
    assert context is not None
    key = state["map_key"]
    outcome = LoopOutcome(state.get("outcome") or LoopOutcome.EXHAUSTED.value)
    baseline_report = state.get("baseline_report")

    banked = state.get("accepted_so_far")
    if banked is not None and outcome is not LoopOutcome.ACCEPTED:
        logger.info(
            "%s ended %s, but an earlier candidate passed every invariant; "
            "reporting that repair.",
            key,
            outcome.value,
        )
        state = dict(state)  # type: ignore[assignment]
        state["stop_reason"] = (
            f"{state.get('stop_reason', '')} An earlier candidate passed every "
            "hard invariant and is what this map ends with."
        ).strip()
        outcome = LoopOutcome.ACCEPTED

    accepted = outcome is LoopOutcome.ACCEPTED
    accepted_candidate = banked if banked is not None else state.get("candidate")
    accepted_sha = (
        state.get("accepted_sha256_so_far")
        if banked is not None
        else state.get("candidate_sha256")
    )
    accepted_report = (
        state.get("accepted_report_so_far")
        if banked is not None
        else state.get("candidate_report")
    )

    result = LoopResult(
        outcome=outcome,
        map_url=state.get("map_url"),
        map_id=state.get("map_id"),
        baseline_sha256=state.get("baseline_sha256"),
        baseline_report=baseline_report,
        accepted_candidate=accepted_candidate if accepted else None,
        accepted_sha256=accepted_sha if accepted else None,
        accepted_report=accepted_report if accepted else None,
        attempts=list(state.get("attempts") or []),
        provider_calls=int(state.get("provider_calls", 0)),
        cache_hits=int(state.get("cache_hits", 0)),
        elapsed_s=float(state.get("elapsed_s", 0.0)),
        stop_reason=state.get("stop_reason", ""),
    )
    if accepted and accepted_report is not None:
        result.unresolved = list(accepted_report.findings)
    elif baseline_report is not None:
        result.unresolved = list(baseline_report.blocking_findings)

    logger.info(
        "Agent mode finished %s: %s (%s) after %d attempt(s), %d provider call(s).",
        key,
        result.outcome.value,
        result.stop_reason,
        len(result.attempts),
        result.provider_calls,
    )

    origin = context.document(key)
    staged_candidate = result.accepted_candidate or state.get("staged_before")
    run_result = MapRunResult(
        map_key=key,
        map_path=str(context.path(key)),
        round=int(state.get("round", 1)),
        result=result,
        baseline=dict(origin),
        origin_sha256=canonical_sha256(origin),
        staged_candidate=dict(staged_candidate) if staged_candidate else None,
        staged_sha256=(
            canonical_sha256(staged_candidate)
            if staged_candidate
            else canonical_sha256(origin)
        ),
        attempts_used=int(state.get("attempts_used", 0)),
        provider_calls_used=int(state.get("provider_calls", 0)),
        cache_hits=int(state.get("cache_hits", 0)),
        requeued_for=list(state.get("requeued_for") or []),
        rejected=list(state.get("rejected") or []),
        seen_candidates=list(state.get("seen_candidates") or []),
        seen_signatures=[list(entry) for entry in state.get("seen_signatures") or []],
    )
    return {"results": {key: run_result}}


def after_baseline(state: MapRepairState) -> str:
    """edge definition after baseline"""
    return "finish" if state.get("outcome") else "open_attempt"


def after_gate(state: MapRepairState) -> str:
    """edge definition after validation gate"""
    return "finish" if state.get("outcome") else "build_prompt_context"


def after_context(state: MapRepairState) -> str:
    """edge definition after context build"""
    return "finish" if state.get("outcome") else "request_patch"


def after_request(state: MapRepairState) -> str:
    """edge definition after request definition"""
    return "finish" if state.get("outcome") else "screen_patch"


def after_screen(state: MapRepairState) -> str:
    """edge definition after prelim. validation"""
    if state.get("outcome"):
        return "finish"
    return "open_attempt" if state.get("retry") else "apply_patch"


def after_apply(state: MapRepairState) -> str:
    if state.get("outcome"):
        return "finish"
    return "open_attempt" if state.get("retry") else "validate_candidate"


def after_validate(state: MapRepairState) -> str:
    if state.get("outcome"):
        return "finish"
    # An accepted candidate that left map-fixable work open is staged as the new
    # baseline; re-validating it is what produces the worklist for the next
    # attempt. Routing to `open_attempt` instead would prompt against the
    # revision that has already been repaired.
    if state.get("staged_before") is not None and state.get("record") is None:
        return "validate_baseline"
    return "open_attempt"
