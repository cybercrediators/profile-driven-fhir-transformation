from types import SimpleNamespace

import pytest

import controller.pipeline_controller.pipeline_build_service as bs_mod
from controller.pipeline_controller.pipeline_build_service import (
    BuildOptions,
    PipelineBuildService,
)


pytestmark = pytest.mark.unit


class FakeHelperMap:
    """Minimal stand-in for a fhir.resources StructureDefinition model."""

    def __init__(self, url="http://x/h", snapshot=None):
        self._url = url
        self.snapshot = snapshot

    def model_dump(self):
        return {"resourceType": "StructureDefinition", "url": self._url}

    def model_dump_json(self, indent=None):
        return '{"resourceType": "StructureDefinition"}'


def make_service(make_app_state, state, *, options=None, plugins=None, automapper=None,
                 dataIO=None, cache=None, conf=None):
    app_state = make_app_state(dataIO=dataIO, cache=cache, conf=conf or {"profile_path": "p"})
    return PipelineBuildService(
        app_state, state, options or BuildOptions(project_name="proj"),
        plugins=plugins, automapper=automapper,
    )


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


def test_clean_name(make_app_state, state):
    svc = make_service(make_app_state, state)
    assert svc._clean_name("My Project_Name/x") == "my-project-name-x"


def test_naming_scheme(make_app_state, state):
    svc = make_service(
        make_app_state,
        state,
        options=BuildOptions(project_name="My-Proj"),
        conf={"base_profile_url": "http://base.org/", "profile_path": "p"},
    )
    sd_name, sd_id, sd_url, sm_name, sm_url, sm_title = svc._naming_scheme()
    assert sd_name == "SourceDefinition_My-Proj"
    assert sd_id == "source-definition-my-proj"
    assert sd_url == "http://base.org/StructureDefinition/source-definition-my-proj"
    assert sm_name == "structure_map_my_proj"
    assert sm_title == "Structure Map for My-Proj"


# --------------------------------------------------------------------------- #
# process steps
# --------------------------------------------------------------------------- #


def test_run_process_propagates_force_overwrite(make_app_state, state, fake_dataio, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        bs_mod.resource_processing, "process_files",
        lambda files, app_state, overwrite: calls.update(files=files, overwrite=overwrite),
    )
    dataIO = fake_dataio(profile_files=["a.json"])
    svc = make_service(
        make_app_state, state, dataIO=dataIO,
        options=BuildOptions(project_name="p", force_overwrite=True),
    )
    assert svc.run_process() is True
    assert calls == {"files": ["a.json"], "overwrite": True}


def test_ensure_processed_uses_reprocess_flag_only(make_app_state, state, fake_dataio, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        bs_mod.resource_processing, "process_files",
        lambda files, app_state, overwrite: calls.update(overwrite=overwrite),
    )
    # force_overwrite True but reprocess_profile False -> dependency run must NOT force
    svc = make_service(
        make_app_state, state, dataIO=fake_dataio(profile_files=["a.json"]),
        options=BuildOptions(project_name="p", force_overwrite=True, reprocess_profile=False),
    )
    assert svc._ensure_processed() is True
    assert calls == {"overwrite": False}


# --------------------------------------------------------------------------- #
# create_source_map
# --------------------------------------------------------------------------- #


def test_create_source_map_success_caches(make_app_state, state, fake_cache, monkeypatch):
    monkeypatch.setattr(
        bs_mod.fml_structure, "generate_helper_map",
        lambda *a, **k: FakeHelperMap(url="http://x/h"),
    )
    svc = make_service(make_app_state, state, cache=fake_cache)
    out = svc.create_source_map("path", "id", "http://x/h", "name")
    assert isinstance(out, FakeHelperMap)
    assert fake_cache.store["http://x/h"]["url"] == "http://x/h"


def test_create_source_map_handles_generation_error(make_app_state, state, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(bs_mod.fml_structure, "generate_helper_map", _boom)
    svc = make_service(make_app_state, state)
    assert svc.create_source_map("path", "id", "url", "name") is None


def test_create_source_map_runs_plugins(make_app_state, state, fake_cache, monkeypatch):
    monkeypatch.setattr(
        bs_mod.fml_structure, "generate_helper_map", lambda *a, **k: FakeHelperMap()
    )
    # Fake StructureDefinition.model_validate so plugin rebuild doesn't need real FHIR models.
    enriched = FakeHelperMap(url="http://x/h-enriched")
    monkeypatch.setattr(
        bs_mod, "StructureDefinition",
        SimpleNamespace(model_validate=lambda d: enriched),
    )
    seen = {}

    class Plugin:
        plugin_id = "p1"

        def post_source_def(self, sd_dict, source_fields):
            seen["called"] = True
            sd_dict["enriched"] = True
            return sd_dict

    svc = make_service(make_app_state, state, cache=fake_cache, plugins=[Plugin()])
    out = svc.create_source_map("path", "model-id", "http://x/h", "name")
    assert seen.get("called") is True
    assert out is enriched
    # enriched SD persisted to disk via dataIO.store_project_file
    assert svc.app_state.dataIO.stored, "enriched SD should be written to disk"


# --------------------------------------------------------------------------- #
# run_static_gen_sm
# --------------------------------------------------------------------------- #


def test_static_gen_sm_errors_without_mapping_source(make_app_state, state):
    # neither automapping nor a custom mapping table -> cannot generate
    svc = make_service(make_app_state, state, options=BuildOptions(project_name="p", automapping=False))
    assert svc.run_static_gen_sm() is False


def test_static_gen_sm_happy_path_records_urls(
    make_app_state, state, fake_cache, monkeypatch
):
    state.custom_mapping_table = {"src": "tgt"}
    monkeypatch.setattr(bs_mod.resource_processing, "process_files", lambda *a, **k: None)

    svc = make_service(
        make_app_state, state, cache=fake_cache,
        options=BuildOptions(project_name="p"),
    )
    # stub the two heavy collaborators
    monkeypatch.setattr(svc, "_load_helper_map", lambda sd_url: FakeHelperMap())
    sms = [SimpleNamespace(url="http://x/sm1"), SimpleNamespace(url="http://x/sm2")]
    monkeypatch.setattr(svc, "generate_structure_maps", lambda *a, **k: sms)

    assert svc.run_static_gen_sm() is True
    assert state.structure_map_urls == ["http://x/sm1", "http://x/sm2"]


def test_static_gen_sm_fails_when_helper_missing(make_app_state, state, monkeypatch):
    state.custom_mapping_table = {"src": "tgt"}
    monkeypatch.setattr(bs_mod.resource_processing, "process_files", lambda *a, **k: None)
    svc = make_service(make_app_state, state, options=BuildOptions(project_name="p"))
    monkeypatch.setattr(svc, "_load_helper_map", lambda sd_url: None)
    assert svc.run_static_gen_sm() is False


# --------------------------------------------------------------------------- #
# run_source_def
# --------------------------------------------------------------------------- #


def test_run_source_def_records_helper_url(make_app_state, state, fake_cache, monkeypatch):
    monkeypatch.setattr(bs_mod.resource_processing, "process_files", lambda *a, **k: None)
    monkeypatch.setattr(
        bs_mod.fml_structure, "generate_helper_map",
        lambda source, app_state, model_id, model_url, model_name, overwrite=False: FakeHelperMap(
            url=model_url
        ),
    )
    svc = make_service(
        make_app_state, state, cache=fake_cache,
        options=BuildOptions(project_name="proj", input_source_example="ex.json"),
        conf={"base_profile_url": "http://base.org", "profile_path": "p"},
    )
    assert svc.run_source_def() is True
    expected_url = "http://base.org/StructureDefinition/source-definition-proj"
    assert state.source_helper_urls == [expected_url]
