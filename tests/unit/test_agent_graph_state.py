"""The graph's state contract (WP10.1).

State is written to a durable checkpoint after every node and read back by a
process that may be a different invocation entirely. Three properties follow,
and all three are asserted here rather than left to review:

- fan-in has an explicit reducer, so two maps finishing in one superstep merge
  instead of clobbering each other;
- nothing that holds a socket, a credential or a file handle is in state;
- there is no unbounded transcript, because a checkpoint that grows with every
  provider message is a checkpoint nobody can resume from.
"""

from pathlib import Path
import typing

import pytest
from pydantic import BaseModel

from agent.graph.state import (
    GRAPH_STATE_VERSION,
    MapRepairOutput,
    MapRepairState,
    ProjectAgentState,
    ProjectLimits,
    merge_results,
)
from agent.loop import LoopOutcome, LoopResult, MapRunResult

pytestmark = pytest.mark.unit


def result(key, outcome=LoopOutcome.CLEAN, candidate=None):
    return MapRunResult(
        map_key=key,
        map_path=f"{key}.json",
        result=LoopResult(outcome=outcome, stop_reason="scripted"),
        baseline={"resourceType": "StructureMap", "id": key},
        origin_sha256=f"origin-{key}",
        staged_candidate=candidate,
    )


# --- reducers -------------------------------------------------------------------


def test_results_merge_by_map_key_rather_than_replacing_the_dictionary():
    merged = merge_results({"a": result("a")}, {"b": result("b")})
    assert sorted(merged) == ["a", "b"]


def test_a_requeued_map_replaces_its_earlier_result_rather_than_duplicating_it():
    first = result("a", LoopOutcome.EXHAUSTED)
    second = result("a", LoopOutcome.ACCEPTED, candidate={"id": "a", "fixed": True})

    merged = merge_results({"a": first}, {"a": second})

    assert list(merged) == ["a"]
    assert merged["a"].result.outcome is LoopOutcome.ACCEPTED


def test_the_reducer_tolerates_an_empty_side():
    assert merge_results(None, {"a": result("a")})["a"].map_key == "a"
    assert merge_results({"a": result("a")}, None)["a"].map_key == "a"
    assert merge_results(None, None) == {}


def test_the_reducer_does_not_mutate_either_input():
    left = {"a": result("a")}
    right = {"b": result("b")}
    merge_results(left, right)
    assert list(left) == ["a"] and list(right) == ["b"]


# --- what may and may not live in state -------------------------------------------


def _annotations(state) -> dict:
    return typing.get_type_hints(state, include_extras=True)


def test_the_subgraph_publishes_only_its_results_channel():
    """The output schema is what lets the subgraph be a node of the project
    graph: without it, every per-map key would be an update against a schema
    that does not declare it."""

    assert set(_annotations(MapRepairOutput)) == {"results"}


def test_neither_state_carries_a_service_a_client_or_a_transcript():
    forbidden = {
        "service",
        "client",
        "proposer",
        "evaluator",
        "engine",
        "matchbox",
        "session",
        "logger",
        "credentials",
        "api_key",
        "messages",
    }
    for state in (MapRepairState, ProjectAgentState):
        assert not forbidden & set(_annotations(state)), state.__name__


def test_the_repair_state_records_the_budgets_a_requeue_must_continue():
    fields = set(_annotations(MapRepairState))
    assert {"attempts_used", "provider_calls", "cache_hits", "elapsed_s"} <= fields
    # And the evidence a no-progress stop is decided from.
    assert {"seen_candidates", "seen_signatures"} <= fields


def test_the_project_state_records_what_a_round_did():
    fields = set(_annotations(ProjectAgentState))
    assert {
        "map_order",
        "pending",
        "results",
        "round",
        "rounds",
        "assembled_digests",
        "global_findings",
        "routing",
        "project_signatures",
    } <= fields


def test_the_state_version_is_independent_of_the_patch_and_report_versions():
    from agent.models import PATCH_SCHEMA_VERSION
    from agent.reporting import REPORT_VERSION

    # Not an assertion that they differ — an assertion that all three exist and
    # are recorded separately, so an incompatible resume can name which one.
    assert isinstance(GRAPH_STATE_VERSION, int)
    assert isinstance(PATCH_SCHEMA_VERSION, int)
    assert isinstance(REPORT_VERSION, int)


# --- durable per-map result -------------------------------------------------------


def test_staging_falls_back_to_the_baseline_when_nothing_was_accepted():
    entry = result("a")
    assert entry.changed is False
    assert entry.staged == entry.baseline


def test_a_staged_candidate_is_what_would_be_written():
    entry = result("a", LoopOutcome.ACCEPTED, candidate={"id": "a", "fixed": True})
    assert entry.changed is True
    assert entry.staged == {"id": "a", "fixed": True}


# --- project bounds ----------------------------------------------------------------


def test_project_limits_are_separate_from_the_per_map_ones():
    """A cycle can be bounded per map and still not terminate: four maps
    requeueing each other four times is sixteen repairs, each within budget."""

    limits = ProjectLimits()
    assert limits.max_project_rounds >= 1
    assert limits.max_provider_calls >= 1
    assert limits.max_seconds > 0


# --- checkpoint allow-list ----------------------------------------------------------


def test_every_checkpointed_type_is_importable_and_named_explicitly():
    """The checkpoint deserializer is an allow-list, not "construct whatever
    class the file names". Each entry has to resolve, or a resume fails on the
    first state that contains it."""

    import importlib

    from agent.graph.runtime import CHECKPOINT_TYPES

    for module_name, attribute in CHECKPOINT_TYPES:
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), f"{module_name}.{attribute}"


def test_the_checkpointer_identity_names_what_wrote_the_run():
    from agent.graph.runtime import checkpointer_identity

    identity = checkpointer_identity()
    assert identity["kind"] == "sqlite"
    assert identity["package_version"]
    assert identity["langgraph_version"]


def test_the_checkpoint_database_lives_in_the_run_directory(tmp_path):
    from agent.graph.runtime import CHECKPOINT_DB_NAME, open_checkpointer

    root = tmp_path / "agent_output" / "run-1"
    with open_checkpointer(root) as saver:
        assert saver is not None
    assert (root / CHECKPOINT_DB_NAME).is_file()
    assert Path(root).is_dir()


def test_the_saver_is_actually_restricted_to_the_allow_list(tmp_path):
    """Inspect the configured serializer, not the call that configured it.

    The serializer's default is the permissive ``True``, and relaxing *into* an
    allow-list afterwards is a no-op against it — the merge has nothing to merge
    into and returns the same object. Only the constructor argument restricts
    anything, so this asserts the state of the saver a run would use.
    """

    from agent.graph.runtime import CHECKPOINT_TYPES, open_checkpointer

    with open_checkpointer(tmp_path / "run-1") as saver:
        allowed = saver.serde._allowed_msgpack_modules

    assert allowed is not True, "the checkpoint deserializer is unrestricted"
    assert set(CHECKPOINT_TYPES) <= set(allowed)


def test_a_type_outside_the_allow_list_is_not_reconstructed(tmp_path):
    """An allow-list that lets anything through is not an allow-list."""

    from agent.graph.runtime import open_checkpointer

    class NotOnTheList(BaseModel):
        value: str

    with open_checkpointer(tmp_path / "run-1") as saver:
        serde = saver.serde
        restored = serde.loads_typed(serde.dumps_typed(NotOnTheList(value="x")))

    assert not isinstance(restored, NotOnTheList)


def test_a_type_on_the_allow_list_still_round_trips(tmp_path):
    from agent.graph.runtime import open_checkpointer

    entry = result("a", LoopOutcome.ACCEPTED, candidate={"id": "a"})
    with open_checkpointer(tmp_path / "run-1") as saver:
        serde = saver.serde
        restored = serde.loads_typed(serde.dumps_typed({"results": {"a": entry}}))

    assert restored["results"]["a"].map_key == "a"
    assert restored["results"]["a"].result.outcome is LoopOutcome.ACCEPTED
