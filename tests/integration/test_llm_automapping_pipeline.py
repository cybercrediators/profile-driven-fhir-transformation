"""End-to-end LLM automapping against a fake backend (WP4).

Runs the real pipeline — profile parsing, candidate generation, model selection,
mapping-table compilation, StructureMap emission — with a scripted client
standing in for a provider. Offline: everything resolves from the repo, no
matchbox, no valkey, no network.

The point is the acceptance criterion that unit tests cannot show: a model
selection actually survives compilation and reaches a generated StructureMap,
and the deterministic route is untouched by any of it.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

from controller.pipeline_controller.pipeline_controller import PipelineController
from llm.models import LLMResponse, LLMUsage

REPO = Path(__file__).resolve().parents[2]
PROJECT = "example_project1"

pytestmark = pytest.mark.integration


class ScriptedClient:
    """Always picks the top-ranked offer, so every mappable field gets mapped."""

    provider = "openai-compatible"

    def __init__(self):
        self.calls = []

    def complete(
        self, *, messages, response_model, settings, feature="llm", context=None
    ):
        self.calls.append(
            {
                "source_id": (context or {}).get("source_id"),
                "prompt": messages[-1].content,
            }
        )
        value = response_model(
            candidate_id="c1", confidence=0.8, reason="highest ranked candidate"
        )
        return LLMResponse(
            value=value,
            raw_text=value.model_dump_json(),
            provider=self.provider,
            model=settings.model,
            usage=LLMUsage(total_tokens=42),
            cache_hit=False,
        )


def _conf(tmp_path: Path, work: Path) -> dict:
    conf = json.loads((REPO / "conf" / f"{PROJECT}.json").read_text())
    conf["project_path"] = str(work)
    conf["resource_cache_path"] = str(REPO / "data" / "resource_cache") + "/"
    conf["external_cache_service"] = "DISK"
    conf["cache_args"] = {"cache_dir": str(tmp_path / "url_cache")}
    # No mapping_table_path: a user-authored table would (correctly) skip automapping.
    conf.pop("mapping_table_path", None)
    return conf


def _work_copy(tmp_path: Path) -> Path:
    work = tmp_path / PROJECT
    shutil.copytree(REPO / "projects" / PROJECT, work)
    for existing in (work / "structure_maps").glob("*.json"):
        existing.unlink()
    return work


def _run(conf, **controller_kwargs):
    pc = PipelineController(
        conf=conf,
        force_overwrite=True,
        reprocess_profile=True,
        reprocess_helper_definition=True,
        reprocess_structure_map=True,
        create_references=True,
        minimal_mode=True,
        **controller_kwargs,
    )
    assert pc.run_source_def(), "source-def generation failed"
    assert pc.run_static_gen_sm(), "StructureMap generation failed"
    return pc


@pytest.fixture
def scripted(monkeypatch):
    # LLM settings come from the environment only, so they are set here rather
    # than in the project config — and pinned, so a developer's own .env cannot
    # change what this test asserts.
    monkeypatch.setenv("LLM_MODEL_NAME", "fake-model")
    monkeypatch.setenv("LLM_CACHE_ENABLED", "false")
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_EXTRA_BODY", raising=False)
    client = ScriptedClient()
    monkeypatch.setattr("llm.backend.build_client", lambda settings, **kw: client)
    return client


def test_llm_selection_reaches_a_generated_structure_map(tmp_path, scripted):
    work = _work_copy(tmp_path)

    _run(_conf(tmp_path, work), automapping=True, auto_mapping_mode="llm")

    assert scripted.calls, "the strategy never consulted the model"

    maps = list((work / "structure_maps").glob("*.json"))
    assert maps, "no StructureMap was generated"

    table = json.loads(
        (
            work / "source_data" / "structure_map_example_project1_automapping.json"
        ).read_text()
    )
    assert table, "the model selected nothing at all"
    assert len(set(table.values())) == len(table), (
        "a target was assigned more than once"
    )

    # every emitted target must be one the model was actually offered
    offered = set()
    report = json.loads(
        (
            work
            / "source_data"
            / "structure_map_example_project1_llm_automapping_report.json"
        ).read_text()
    )
    for field in report["fields"]:
        for attempt in field["attempts"]:
            offered.update(offer["target"] for offer in attempt["offers"])
    assert set(table.values()) <= offered
    assert report["compiled_mapping_table"] == table
    assert report["mapping_table"] == table


def test_report_is_written_with_full_provenance(tmp_path, scripted):
    work = _work_copy(tmp_path)

    _run(_conf(tmp_path, work), automapping=True, auto_mapping_mode="llm")

    report = json.loads(
        (
            work
            / "source_data"
            / "structure_map_example_project1_llm_automapping_report.json"
        ).read_text()
    )

    assert report["mode"] == "llm"
    assert report["model"] == "fake-model"
    assert report["summary"]["provider_calls"] == len(scripted.calls)
    assert report["summary"]["mapped"] >= 1
    assert report["mapping_table"]
    assert report["compiled_mapping_table"] == report["mapping_table"]
    assert report["summary"]["compiler_rejections"] == len(
        report["compiler_rejections"]
    )
    first = report["fields"][0]
    assert first["attempts"][0]["offers"]


def test_prompts_only_ever_offer_resolved_targets(tmp_path, scripted):
    """Nothing unresolved may be shown to the model, even in a real project."""

    work = _work_copy(tmp_path)

    _run(_conf(tmp_path, work), automapping=True, auto_mapping_mode="llm")

    report = json.loads(
        (
            work
            / "source_data"
            / "structure_map_example_project1_llm_automapping_report.json"
        ).read_text()
    )
    for field in report["fields"]:
        for attempt in field["attempts"]:
            for offer in attempt["offers"]:
                assert offer["element_id"], f"unresolved candidate offered: {offer}"


def test_deterministic_automapping_makes_no_llm_call(tmp_path, monkeypatch):
    """`--auto-mapping` without a mode must not touch the LLM layer at all."""

    def explode(*_, **__):
        raise AssertionError("deterministic automapping built an LLM client")

    monkeypatch.setattr("llm.backend.build_client", explode)
    work = _work_copy(tmp_path)

    pc = _run(_conf(tmp_path, work), automapping=True)

    assert pc.llm_automapping is None
    assert (work / "structure_maps").glob("*.json")
    assert not (
        work
        / "source_data"
        / "structure_map_example_project1_llm_automapping_report.json"
    ).exists()


def test_ordinary_generation_loads_no_provider_sdk(tmp_path, monkeypatch):
    """The WP2 isolation guarantee, checked on a real deterministic run."""

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    work = _work_copy(tmp_path)

    _run(_conf(tmp_path, work), automapping=True)

    assert "openai" not in sys.modules


def test_provider_failure_fails_the_run_rather_than_falling_back(tmp_path, monkeypatch):
    from llm.errors import LLMProviderError

    class Failing:
        provider = "openai-compatible"

        def complete(self, **_):
            raise LLMProviderError("upstream refused the request", 503)

    monkeypatch.setattr("llm.backend.build_client", lambda settings, **kw: Failing())
    work = _work_copy(tmp_path)

    with pytest.raises(LLMProviderError):
        _run(_conf(tmp_path, work), automapping=True, auto_mapping_mode="llm")
