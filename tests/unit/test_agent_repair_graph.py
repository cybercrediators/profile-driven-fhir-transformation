"""The repair subgraph's shape and its dependency boundary (WP10.2).

The *behaviour* of the subgraph is covered by the parity suite in
``test_agent_loop.py`` — those tests are unchanged from when a plain ``for`` loop
ran the same stages. What is asserted here is what the loop could not have:

- proposal, application, validation and routing are **separate** nodes, because
  a node boundary is a checkpoint boundary and "wrap the old loop in one node"
  would give a graph with none of the resumability that motivated it;
- routing is a pure function of ``outcome`` and ``retry``, so the path a run
  took is inspectable rather than inferred;
- nothing outside agent mode imports a graph runtime.
"""

import ast
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from agent.graph import nodes
from agent.graph.repair_graph import build_repair_graph, initial_repair_state

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every stage WP10.2 requires to be independently checkpointable.
REQUIRED_NODES = {
    "validate_baseline",
    "open_attempt",
    "build_prompt_context",
    "request_patch",
    "screen_patch",
    "apply_patch",
    "validate_candidate",
    "finish",
}


def test_every_stage_is_its_own_node():
    graph = build_repair_graph().get_graph()
    assert REQUIRED_NODES <= {node for node in graph.nodes}


def test_only_one_node_may_reach_a_provider():
    """A second calling site is a second place a budget can be bypassed."""

    source = (REPO_ROOT / "src" / "agent" / "graph" / "nodes.py").read_text(
        encoding="utf-8"
    )
    calling = set()
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, ast.FunctionDef):
            continue
        for call in ast.walk(function):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "propose"
            ):
                calling.add(function.name)
    assert calling == {"request_patch"}


# --- routing is data ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "state", "expected"),
    [
        (nodes.after_baseline, {}, "open_attempt"),
        (nodes.after_baseline, {"outcome": "clean"}, "finish"),
        (nodes.after_gate, {}, "build_prompt_context"),
        (nodes.after_gate, {"outcome": "exhausted"}, "finish"),
        (nodes.after_context, {}, "request_patch"),
        (nodes.after_context, {"outcome": "blocked"}, "finish"),
        (nodes.after_request, {}, "screen_patch"),
        (nodes.after_request, {"outcome": "limit-reached"}, "finish"),
        (nodes.after_screen, {}, "apply_patch"),
        (nodes.after_screen, {"retry": True}, "open_attempt"),
        (nodes.after_screen, {"outcome": "provider-error"}, "finish"),
        (nodes.after_apply, {}, "validate_candidate"),
        (nodes.after_apply, {"retry": True}, "open_attempt"),
        (nodes.after_apply, {"outcome": "no-progress"}, "finish"),
        (nodes.after_validate, {}, "open_attempt"),
        (nodes.after_validate, {"outcome": "accepted"}, "finish"),
    ],
)
def test_routes_read_state_rather_than_recomputing_a_decision(route, state, expected):
    assert route(state) == expected


def test_a_terminal_outcome_always_wins_over_a_retry_request():
    """Both set at once must never mean "keep going": the outcome is terminal."""

    for route in (nodes.after_screen, nodes.after_apply):
        assert route({"outcome": "no-progress", "retry": True}) == "finish"


# --- the dispatch payload ------------------------------------------------------------


def test_a_fresh_dispatch_starts_every_budget_at_zero():
    payload = initial_repair_state("a")
    assert payload["attempts_used"] == 0
    assert payload["provider_calls"] == 0
    assert payload["elapsed_s"] == 0.0
    assert payload["staged_before"] is None
    assert payload["round"] == 1


def test_a_requeue_payload_continues_the_map_rather_than_restarting_it():
    """Budgets, history and no-progress evidence all cross the round boundary.
    Resetting any of them lets one map be asked the same failed question once
    per round while every individual round still looks bounded."""

    from agent.context import RejectedAttempt
    from agent.loop import AttemptRecord

    payload = initial_repair_state(
        "a",
        round_index=2,
        staged_before={"id": "a", "fixed": True},
        attempts_used=3,
        provider_calls=3,
        project_provider_quota=5,
        elapsed_s=12.5,
        attempts=[AttemptRecord(index=1), AttemptRecord(index=2)],
        rejected=[RejectedAttempt(attempt=1)],
        seen_candidates=["sha-1", "sha-2"],
        seen_signatures=[["vf-1"]],
        requeued_for=[["vf-1"]],
    )

    assert payload["attempts_used"] == 3
    assert payload["provider_calls"] == 3
    assert payload["project_provider_quota"] == 5
    assert payload["elapsed_s"] == 12.5
    assert [record.index for record in payload["attempts"]] == [1, 2]
    assert len(payload["rejected"]) == 1
    assert payload["seen_candidates"] == ["sha-1", "sha-2"]
    assert payload["seen_signatures"] == [["vf-1"]]
    assert payload["staged_before"] == {"id": "a", "fixed": True}
    assert payload["requeued_for"] == [["vf-1"]]


# --- the dependency boundary ------------------------------------------------------


def _probe(script: str) -> str:
    """Run *script* in a fresh interpreter and report the graph packages loaded."""

    program = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        sys.path.insert(0, {str(REPO_ROOT)!r})
        {textwrap.indent(textwrap.dedent(script), "        ").strip()}
        loaded = sorted(
            name
            for name in sys.modules
            if name.split(".", 1)[0]
            in {{"langgraph", "langchain_core", "agent", "openai", "sqlite3"}}
        )
        print(",".join(loaded))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_llm_automapping_needs_no_graph_runtime():
    """The `llm` extra is a complete feature on its own; only agent mode adds
    LangGraph."""

    loaded = _probe(
        """
        import llm
        from llm.models import LLMSettings
        """
    )
    roots = {name.split(".", 1)[0] for name in loaded.split(",") if name}
    assert "langgraph" not in roots


def test_the_agent_extra_declares_the_graph_runtime_and_its_checkpointer():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    agent_extra = pyproject.split("agent = [", 1)[1].split("]", 1)[0]
    llm_extra = pyproject.split("llm = [", 1)[1].split("]", 1)[0]

    assert "langgraph" in agent_extra
    assert "langgraph-checkpoint-sqlite" in agent_extra
    # Everything the patch envelope and the provider client need is still there,
    # so `agent` is usable on its own rather than only alongside `llm`.
    assert "openai" in agent_extra and "jsonpatch" in agent_extra
    assert "langgraph" not in llm_extra


def test_agent_mode_without_the_extra_fails_with_a_named_setup_error(monkeypatch):
    """A missing optional package must be an exit code and a sentence, not a
    traceback about a module name."""

    import builtins

    from agent.cli import EXIT_SETUP_ERROR, run_agent_fix

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("agent.graph") or name == "langgraph":
            raise ImportError("No module named 'langgraph'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    from argparse import Namespace

    args = Namespace(
        map=None,
        all_maps=False,
        max_attempts=1,
        max_project_rounds=1,
        resume=None,
        output_dir="",
        offline=False,
        use_examples=False,
        apply=False,
        llm_provider=None,
        llm_model=None,
        llm_base_url=None,
    )
    assert run_agent_fix(args, {"project_path": "."}) == EXIT_SETUP_ERROR
