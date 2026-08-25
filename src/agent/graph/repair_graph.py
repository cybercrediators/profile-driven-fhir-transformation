"""define the map repair subgraph"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from langgraph.graph import END, START, StateGraph

from agent.graph import nodes
from agent.graph.runtime import AgentRuntimeContext
from agent.graph.state import MapRepairOutput, MapRepairState
from agent.context import RejectedAttempt
from agent.loop import AttemptRecord, LoopLimits, LoopResult
from agent.validation import ValidationFinding


def build_repair_graph():
    """compile the repair subgraph"""

    graph = StateGraph(
        MapRepairState,
        context_schema=AgentRuntimeContext,
        output_schema=MapRepairOutput,
    )
    graph.add_node("validate_baseline", nodes.validate_baseline)
    graph.add_node("open_attempt", nodes.open_attempt)
    graph.add_node("build_prompt_context", nodes.build_prompt_context)
    graph.add_node("request_patch", nodes.request_patch)
    graph.add_node("screen_patch", nodes.screen_patch)
    graph.add_node("apply_patch", nodes.apply_patch_in_memory)
    graph.add_node("validate_candidate", nodes.validate_candidate)
    graph.add_node("finish", nodes.finish)

    graph.add_edge(START, "validate_baseline")
    graph.add_conditional_edges(
        "validate_baseline", nodes.after_baseline, ["open_attempt", "finish"]
    )
    graph.add_conditional_edges(
        "open_attempt", nodes.after_gate, ["build_prompt_context", "finish"]
    )
    graph.add_conditional_edges(
        "build_prompt_context", nodes.after_context, ["request_patch", "finish"]
    )
    graph.add_conditional_edges(
        "request_patch", nodes.after_request, ["screen_patch", "finish"]
    )
    graph.add_conditional_edges(
        "screen_patch", nodes.after_screen, ["apply_patch", "open_attempt", "finish"]
    )
    graph.add_conditional_edges(
        "apply_patch",
        nodes.after_apply,
        ["validate_candidate", "open_attempt", "finish"],
    )
    graph.add_conditional_edges(
        "validate_candidate",
        nodes.after_validate,
        ["validate_baseline", "open_attempt", "finish"],
    )
    graph.add_edge("finish", END)
    return graph.compile()


REPAIR_RECURSION_LIMIT = 120


def initial_repair_state(
    map_key: str,
    *,
    round_index: int = 1,
    staged_before: Optional[Mapping[str, Any]] = None,
    generator_findings: Sequence[ValidationFinding] = (),
    global_feedback: Sequence[ValidationFinding] = (),
    attempts_used: int = 0,
    provider_calls: int = 0,
    cache_hits: int = 0,
    project_provider_quota: int = LoopLimits().max_provider_calls,
    elapsed_s: float = 0.0,
    attempts: Sequence[AttemptRecord] = (),
    rejected: Sequence[RejectedAttempt] = (),
    seen_candidates: Sequence[str] = (),
    seen_signatures: Sequence[Sequence[str]] = (),
    requeued_for: Sequence[Sequence[str]] = (),
) -> Dict[str, Any]:
    """The payload one dispatch of one map starts from"""

    return {
        "map_key": map_key,
        "round": round_index,
        "staged_before": dict(staged_before) if staged_before else None,
        "generator_findings": list(generator_findings),
        "global_feedback": list(global_feedback),
        "attempts_used": attempts_used,
        "provider_calls": provider_calls,
        "cache_hits": cache_hits,
        "project_provider_quota": project_provider_quota,
        "requeued_for": [list(entry) for entry in requeued_for],
        "attempts": list(attempts),
        "rejected": list(rejected),
        "seen_candidates": list(seen_candidates),
        "seen_signatures": [list(entry) for entry in seen_signatures],
        "elapsed_s": elapsed_s,
        "retry": False,
        "outcome": None,
        "stop_reason": "",
    }


def run_repair_graph(
    map_key: str,
    *,
    context: AgentRuntimeContext,
    round_index: int = 1,
    staged_before: Optional[Mapping[str, Any]] = None,
    generator_findings: Sequence[ValidationFinding] = (),
    global_feedback: Sequence[ValidationFinding] = (),
    graph=None,
) -> LoopResult:
    """Run one map's repair subgraph on its own and return the coreresponding result"""

    compiled = graph or build_repair_graph()
    payload = initial_repair_state(
        map_key,
        round_index=round_index,
        staged_before=staged_before,
        generator_findings=generator_findings,
        global_feedback=global_feedback,
    )
    output = compiled.invoke(
        payload,
        context=context,
        config={"recursion_limit": REPAIR_RECURSION_LIMIT},
    )
    results: Dict[str, Any] = output.get("results") or {}
    return results[map_key].result


__all__: List[str] = [
    "REPAIR_RECURSION_LIMIT",
    "build_repair_graph",
    "initial_repair_state",
    "run_repair_graph",
]
