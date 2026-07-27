"""Unit tests for MatchboxTransformService (controller/external_services/matchbox_transform_service.py).

Uses the FakeDataIO / FakeMatchbox fakes from conftest.py so no real project IO or
matchbox HTTP traffic is involved. MatchboxTransformService constructs its own
DataIO(project_path) internally when a project_path is given, so we monkeypatch the
module's DataIO reference to conftest's FakeDataIO for those cases.
"""

import json
from types import SimpleNamespace

import pytest

import controller.external_services.matchbox_transform_service as mts_module
from controller.external_services.matchbox_transform_service import (
    MatchboxTransformService,
)

pytestmark = pytest.mark.unit


class RecordingMatchbox:
    """Minimal matchbox_controller stand-in recording transform_data calls."""

    def __init__(self, results=None):
        # results: dict keyed by (item repr / index) not needed -- simple queue/callable
        self._results = results
        self.calls = []

    def transform_data(self, source_obj, structure_map_url):
        self.calls.append((source_obj, structure_map_url))
        if callable(self._results):
            return self._results(source_obj, structure_map_url)
        return self._results


class FakeProfileData:
    """Module-level (so jsonpickle.decode can re-resolve the class by name)
    stand-in for a parsed profile carried by a RegistryObject."""

    def __init__(self):
        self.url = "http://example.org/StructureDefinition/x"
        self.id = "x"
        self.type = "Patient"


def sm_file(tmp_path, name, url, ftype="Patient"):
    path = tmp_path / name
    path.write_text(
        json.dumps({"resourceType": "StructureMap", "url": url, "group": []})
    )
    return path


def use_fake_dataio(monkeypatch, fake_dataio_factory, **kwargs):
    fake = fake_dataio_factory(**kwargs)
    monkeypatch.setattr(mts_module, "DataIO", lambda project_path: fake)
    return fake


# --------------------------------------------------------------------------- #
# _resolve_sm_urls
# --------------------------------------------------------------------------- #


def test_resolve_sm_urls_explicit_url_wins():
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=None)
    assert svc._resolve_sm_urls("http://explicit/sm") == ["http://explicit/sm"]


def test_resolve_sm_urls_no_project_path_and_no_explicit_returns_empty(caplog):
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=None)
    with caplog.at_level("ERROR"):
        result = svc._resolve_sm_urls(None)
    assert result == []
    assert any("No --structure-map-url" in r.message for r in caplog.records)


def test_resolve_sm_urls_from_project_structure_maps(tmp_path, monkeypatch, fake_dataio):
    f = sm_file(tmp_path, "sm1.json", "http://x/sm1")
    use_fake_dataio(monkeypatch, fake_dataio, sm_files=[f])
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    assert svc._resolve_sm_urls(None) == ["http://x/sm1"]


def test_resolve_sm_urls_project_path_but_no_maps_found(tmp_path, monkeypatch, fake_dataio, caplog):
    use_fake_dataio(monkeypatch, fake_dataio, sm_files=[])
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    with caplog.at_level("ERROR"):
        result = svc._resolve_sm_urls(None)
    assert result == []
    assert any("No StructureMap URLs found" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# _load_structure_maps / _load_structure_map_urls
# --------------------------------------------------------------------------- #


def test_load_structure_maps_reads_and_filters_json(tmp_path, monkeypatch, fake_dataio):
    sm = sm_file(tmp_path, "sm1.json", "http://x/sm1")
    other = tmp_path / "not_a_sm.json"
    other.write_text(json.dumps({"resourceType": "ConceptMap"}))
    use_fake_dataio(monkeypatch, fake_dataio, sm_files=[sm, other])
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    maps = svc._load_structure_maps()
    assert len(maps) == 1
    assert maps[0]["url"] == "http://x/sm1"


def test_load_structure_maps_without_dataio_returns_empty():
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=None)
    assert svc._load_structure_maps() == []


# --------------------------------------------------------------------------- #
# _inject_resource_type / _infer_source_resource_type
# --------------------------------------------------------------------------- #


def test_inject_resource_type_no_project_path_leaves_unchanged():
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=None)
    data = {"foo": "bar"}
    assert svc._inject_resource_type(data) is data


def test_inject_resource_type_already_present_unchanged(tmp_path, monkeypatch, fake_dataio):
    use_fake_dataio(monkeypatch, fake_dataio)
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    data = {"resourceType": "Patient"}
    assert svc._inject_resource_type(data) == {"resourceType": "Patient"}


def test_inject_resource_type_infers_from_source_helper_files(tmp_path, monkeypatch, fake_dataio):
    helper = tmp_path / "source.json"
    helper.write_text(json.dumps({"resourceType": "StructureDefinition", "type": "Encounter"}))
    use_fake_dataio(monkeypatch, fake_dataio, helper_files=[helper])
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    result = svc._inject_resource_type({"foo": "bar"})
    assert result == {"foo": "bar", "resourceType": "Encounter"}


def test_inject_resource_type_infers_for_list_input(tmp_path, monkeypatch, fake_dataio):
    helper = tmp_path / "source.json"
    helper.write_text(json.dumps({"resourceType": "StructureDefinition", "type": "Encounter"}))
    use_fake_dataio(monkeypatch, fake_dataio, helper_files=[helper])
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    result = svc._inject_resource_type([{"foo": "bar"}, {"resourceType": "Patient"}])
    assert result == [{"foo": "bar", "resourceType": "Encounter"}, {"resourceType": "Patient"}]


def test_inject_resource_type_no_inference_available_unchanged(tmp_path, monkeypatch, fake_dataio):
    use_fake_dataio(monkeypatch, fake_dataio, helper_files=[])
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    data = {"foo": "bar"}
    assert svc._inject_resource_type(data) == {"foo": "bar"}


def test_infer_source_resource_type_no_dataio_returns_none():
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=None)
    assert svc._infer_source_resource_type() is None


def test_infer_source_resource_type_no_type_field_returns_none(tmp_path, monkeypatch, fake_dataio):
    helper = tmp_path / "source.json"
    helper.write_text(json.dumps({"resourceType": "StructureDefinition"}))
    use_fake_dataio(monkeypatch, fake_dataio, helper_files=[helper])
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    assert svc._infer_source_resource_type() is None


# --------------------------------------------------------------------------- #
# _transform_one
# --------------------------------------------------------------------------- #


def test_transform_one_extracts_bundle_entries():
    mb = RecordingMatchbox(
        results={
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient"}},
                {"resource": {"resourceType": "Observation"}},
                {"no_resource_key": True},
            ],
        }
    )
    svc = MatchboxTransformService(mb, project_path=None)
    resources = svc._transform_one({"resourceType": "Foo"}, ["http://sm1"])
    assert resources == [
        {"resourceType": "Patient"},
        {"resourceType": "Observation"},
    ]


def test_transform_one_appends_direct_resource():
    mb = RecordingMatchbox(results={"resourceType": "Patient"})
    svc = MatchboxTransformService(mb, project_path=None)
    resources = svc._transform_one({"resourceType": "Foo"}, ["http://sm1"])
    assert resources == [{"resourceType": "Patient"}]


def test_transform_one_skips_falsy_results_with_warning(caplog):
    mb = RecordingMatchbox(results=None)
    svc = MatchboxTransformService(mb, project_path=None)
    with caplog.at_level("WARNING"):
        resources = svc._transform_one({"resourceType": "Foo"}, ["http://sm1", "http://sm2"])
    assert resources == []
    assert len(mb.calls) == 2


# --------------------------------------------------------------------------- #
# transform(): end-to-end, batch, bundle
# --------------------------------------------------------------------------- #


def test_transform_returns_none_when_no_sm_urls_resolvable():
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=None)
    assert svc.transform({"resourceType": "Foo"}, structure_map_url=None) is None


def test_transform_non_bundle_returns_normalized_list():
    mb = RecordingMatchbox(results={"resourceType": "Patient", "identifier": {"system": "s", "value": "v"}})
    svc = MatchboxTransformService(mb, project_path=None)
    result = svc.transform({"resourceType": "Foo"}, structure_map_url="http://sm1")
    assert result == [
        {"resourceType": "Patient", "identifier": [{"system": "s", "value": "v"}]}
    ]


def test_transform_no_resources_produced_returns_none(caplog):
    mb = RecordingMatchbox(results=None)
    svc = MatchboxTransformService(mb, project_path=None)
    with caplog.at_level("ERROR"):
        result = svc.transform({"resourceType": "Foo"}, structure_map_url="http://sm1")
    assert result is None
    assert any("Data transformation produced no resources" in r.message for r in caplog.records)


def test_transform_bundle_true_wraps_result():
    mb = RecordingMatchbox(results={"resourceType": "Patient"})
    svc = MatchboxTransformService(mb, project_path=None)
    result = svc.transform({"resourceType": "Foo"}, structure_map_url="http://sm1", bundle=True)
    assert result["resourceType"] == "Bundle"
    assert result["entry"][0]["resource"] == {"resourceType": "Patient"}


def test_transform_bundle_applies_explicit_external_reference_default():
    mb = RecordingMatchbox(results={"resourceType": "Observation"})
    defaults = [
        {
            "path": "Observation.subject",
            "reference": "Patient/eval-patient-1",
        }
    ]
    svc = MatchboxTransformService(
        mb,
        project_path=None,
        external_reference_defaults=defaults,
    )

    result = svc.transform(
        {"resourceType": "Foo"},
        structure_map_url="http://sm1",
        bundle=True,
    )

    observation = result["entry"][0]["resource"]
    assert observation["subject"] == {
        "reference": "Patient/eval-patient-1",
        "type": "Patient",
    }


def test_transform_batch_true_aggregates_per_item():
    mb = RecordingMatchbox(results={"resourceType": "Patient"})
    svc = MatchboxTransformService(mb, project_path=None)
    result = svc.transform(
        [{"resourceType": "Foo"}, {"resourceType": "Bar"}],
        structure_map_url="http://sm1",
        batch=True,
    )
    assert len(result) == 2
    assert result[0] == [{"resourceType": "Patient"}]


def test_transform_batch_true_wraps_non_list_single_item():
    mb = RecordingMatchbox(results={"resourceType": "Patient"})
    svc = MatchboxTransformService(mb, project_path=None)
    result = svc.transform({"resourceType": "Foo"}, structure_map_url="http://sm1", batch=True)
    assert len(result) == 1


def test_transform_batch_all_empty_returns_none(caplog):
    mb = RecordingMatchbox(results=None)
    svc = MatchboxTransformService(mb, project_path=None)
    with caplog.at_level("ERROR"):
        result = svc.transform(
            [{"resourceType": "Foo"}], structure_map_url="http://sm1", batch=True
        )
    assert result is None
    assert any("Batch transformation produced no resources" in r.message for r in caplog.records)


def test_transform_batch_bundle_true_wraps_each_result():
    mb = RecordingMatchbox(results={"resourceType": "Patient"})
    svc = MatchboxTransformService(mb, project_path=None)
    result = svc.transform(
        [{"resourceType": "Foo"}],
        structure_map_url="http://sm1",
        batch=True,
        bundle=True,
    )
    assert len(result) == 1
    assert result[0]["resourceType"] == "Bundle"


# --------------------------------------------------------------------------- #
# _load_registry
# --------------------------------------------------------------------------- #


def test_load_registry_without_dataio_returns_none():
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=None)
    assert svc._load_registry() is None


def test_load_registry_skips_non_file_entries(tmp_path, monkeypatch, fake_dataio):
    # a directory passed as a "processed resource" path must be skipped, not crash
    subdir = tmp_path / "adir"
    subdir.mkdir()
    fake = fake_dataio()
    fake.get_processed_resources = lambda: [subdir]
    monkeypatch.setattr(mts_module, "DataIO", lambda project_path: fake)
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    assert svc._load_registry() is None


def test_load_registry_round_trips_writer_encoded_registry_object(tmp_path, monkeypatch, fake_dataio):
    # Positive round-trip using the REAL writer encoding path. The writer
    # (parser/resource_processing.py store_results) does
    #   utils.store_json(jsonpickle.encode(obj), path)
    # i.e. the jsonpickle payload is stored as a JSON *string literal*
    # (double-encoded). The reader in matchbox_transform_service.py:172,
    #   jsonpickle.decode(json.loads(f.read_text())),
    # exactly mirrors that: json.loads() unwraps the string literal, then
    # jsonpickle.decode() rebuilds the object. This test proves the pairing is
    # correct end-to-end (my earlier "double-decode bug" report was a false
    # alarm based on writing raw jsonpickle to disk, which the writer never does).
    import jsonpickle

    from data_handling.registry.registry_object import RegistryObject
    from helpers import utils

    obj = RegistryObject(data=FakeProfileData(), res_type="Patient")
    obj.set_root()

    f = tmp_path / "processed" / "x"
    f.parent.mkdir(parents=True, exist_ok=True)
    utils.store_json(jsonpickle.encode(obj), f)  # same call chain as store_results

    fake = fake_dataio()
    fake.get_processed_resources = lambda: [f]
    monkeypatch.setattr(mts_module, "DataIO", lambda project_path: fake)
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    registry = svc._load_registry()
    assert registry is not None
    rebuilt = registry.registry_objects["http://example.org/StructureDefinition/x"]
    assert isinstance(rebuilt, RegistryObject)
    assert rebuilt.is_root is True
    assert rebuilt.data.type == "Patient"


def test_load_registry_unreadable_file_is_skipped_and_logged(tmp_path, monkeypatch, fake_dataio, caplog):
    # A file in a wrong/corrupt format -- here RAW jsonpickle JSON, i.e. missing
    # the writer's outer JSON-string wrapping (see the round-trip test above) --
    # must be skipped with a warning, not crash _load_registry. json.loads()
    # yields a dict for such content and jsonpickle.decode(dict) raises
    # TypeError, which lands in the skip-and-log except branch.
    import jsonpickle

    class Dummy:
        def __init__(self, url):
            self.data = SimpleNamespace(url=url)

    f = tmp_path / "processed" / "obj1.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(jsonpickle.encode(Dummy("http://example.org/obj1")))

    fake = fake_dataio()
    fake.get_processed_resources = lambda: [f]
    monkeypatch.setattr(mts_module, "DataIO", lambda project_path: fake)
    svc = MatchboxTransformService(RecordingMatchbox(), project_path=str(tmp_path))
    with caplog.at_level("WARNING"):
        result = svc._load_registry()
    assert result is None
    assert any("Skipping unreadable processed resource" in r.message for r in caplog.records)
