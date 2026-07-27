import json

import pytest

from controller.pipeline_controller.pipeline_transform_service import (
    PipelineTransformService,
)


pytestmark = pytest.mark.unit


def make_service(make_app_state, state, *, dataIO=None, cache=None, matchbox=None, conf=None):
    app_state = make_app_state(dataIO=dataIO, cache=cache, conf=conf or {})
    return PipelineTransformService(app_state, state, matchbox)


# --------------------------------------------------------------------------- #
# _single_transform
# --------------------------------------------------------------------------- #


def test_single_transform_requires_resource_type(make_app_state, state, fake_matchbox):
    svc = make_service(make_app_state, state, matchbox=fake_matchbox())
    assert svc._single_transform({}, "http://x/sm") is None


def test_single_transform_delegates_to_matchbox(make_app_state, state, fake_matchbox):
    mb = fake_matchbox(transform_result={"resourceType": "Patient"})
    svc = make_service(make_app_state, state, matchbox=mb)
    out = svc._single_transform({"resourceType": "X"}, "http://x/sm")
    assert out == {"resourceType": "Patient"}
    assert mb.transform_calls == [({"resourceType": "X"}, "http://x/sm")]


# --------------------------------------------------------------------------- #
# _infer_source_resource_type
# --------------------------------------------------------------------------- #


def test_infer_from_cache(make_app_state, state, fake_cache, fake_matchbox):
    state.source_helper_urls = ["http://x/h"]
    fake_cache.store["http://x/h"] = {"url": "http://x/h", "type": "Patient"}
    svc = make_service(make_app_state, state, cache=fake_cache, matchbox=fake_matchbox())
    assert svc._infer_source_resource_type() == "Patient"


def test_infer_from_disk_when_not_cached(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    helper = write_json("h.json", {"url": "http://x/h", "type": "Observation"})
    dataIO = fake_dataio(helper_files=[helper])
    svc = make_service(make_app_state, state, dataIO=dataIO, matchbox=fake_matchbox())
    assert svc._infer_source_resource_type() == "Observation"


def test_infer_returns_none_when_nothing_available(make_app_state, state, fake_matchbox):
    svc = make_service(make_app_state, state, matchbox=fake_matchbox())
    assert svc._infer_source_resource_type() is None


# --------------------------------------------------------------------------- #
# transform_data
# --------------------------------------------------------------------------- #


def test_transform_data_single_map_returns_raw_result(make_app_state, state, fake_matchbox):
    mb = fake_matchbox(transform_result={"resourceType": "Patient", "id": "1"})
    svc = make_service(make_app_state, state, matchbox=mb)
    out = svc.transform_data({"resourceType": "Src"}, structure_map_url="http://x/sm")
    assert out == {"resourceType": "Patient", "id": "1"}


def test_transform_data_infers_missing_resource_type(make_app_state, state, fake_cache, fake_matchbox):
    state.source_helper_urls = ["http://x/h"]
    fake_cache.store["http://x/h"] = {"url": "http://x/h", "type": "Bundle"}
    captured = {}

    def _transform(src, url):
        captured["src"] = src
        return {"resourceType": "Patient"}

    mb = fake_matchbox(transform_result=_transform)
    svc = make_service(make_app_state, state, cache=fake_cache, matchbox=mb)
    svc.transform_data({"field": 1}, structure_map_url="http://x/sm")
    # resourceType inferred and injected before transform (without mutating caller dict)
    assert captured["src"]["resourceType"] == "Bundle"


def test_transform_data_no_resource_type_and_no_inference_returns_none(
    make_app_state, state, fake_matchbox
):
    svc = make_service(make_app_state, state, matchbox=fake_matchbox())
    assert svc.transform_data({"field": 1}) is None


def test_transform_data_all_maps_flattens_bundles(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    sm = write_json("sm.json", {"resourceType": "StructureMap", "url": "http://x/sm"})
    dataIO = fake_dataio(sm_files=[sm])
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Patient"}},
            {"resource": {"resourceType": "Observation"}},
        ],
    }
    mb = fake_matchbox(transform_result=bundle)
    svc = make_service(make_app_state, state, dataIO=dataIO, matchbox=mb)

    out = svc.transform_data({"resourceType": "Src"})
    assert out == [{"resourceType": "Patient"}, {"resourceType": "Observation"}]
    # url loaded from disk into shared state
    assert state.structure_map_urls == ["http://x/sm"]


def test_transform_data_no_maps_returns_none(make_app_state, state, fake_dataio, fake_matchbox):
    svc = make_service(make_app_state, state, dataIO=fake_dataio(), matchbox=fake_matchbox())
    assert svc.transform_data({"resourceType": "Src"}) is None


def test_transform_data_aggregates_non_bundle_results(make_app_state, state, fake_matchbox):
    # two maps preloaded in state, single (non-bundle) resource each
    state.structure_map_urls = ["http://x/sm1", "http://x/sm2"]
    mb = fake_matchbox(transform_result={"resourceType": "Patient"})
    svc = make_service(make_app_state, state, matchbox=mb)
    out = svc.transform_data({"resourceType": "Src"})
    assert out == [{"resourceType": "Patient"}, {"resourceType": "Patient"}]


# --------------------------------------------------------------------------- #
# transform_data_batch / transform_data_from_disk
# --------------------------------------------------------------------------- #


def test_transform_data_batch_skips_none_results(make_app_state, state, fake_matchbox):
    state.structure_map_urls = ["http://x/sm"]
    # first input yields a resource, second yields nothing.
    # No explicit url -> multi-map path, which returns a flattened list per input.
    results_iter = iter([{"resourceType": "Patient"}, None])
    mb = fake_matchbox(transform_result=lambda *_: next(results_iter))
    svc = make_service(make_app_state, state, matchbox=mb)
    out = svc.transform_data_batch([{"resourceType": "A"}, {"resourceType": "B"}])
    assert out == [[{"resourceType": "Patient"}]]


def test_transform_data_from_disk_writes_output(
    make_app_state, state, fake_matchbox, write_json, tmp_path
):
    src = write_json("in.json", {"resourceType": "Src"})
    out_path = tmp_path / "out.json"
    mb = fake_matchbox(transform_result={"resourceType": "Patient"})
    svc = make_service(make_app_state, state, matchbox=mb)

    assert svc.transform_data_from_disk(src, out_path, structure_map_url="http://x/sm") is True
    assert json.loads(out_path.read_text()) == {"resourceType": "Patient"}


def test_transform_data_from_disk_returns_false_on_failure(
    make_app_state, state, fake_dataio, fake_matchbox, write_json, tmp_path
):
    src = write_json("in.json", {"resourceType": "Src"})
    out_path = tmp_path / "out.json"
    # no maps available -> transform_data returns None
    svc = make_service(make_app_state, state, dataIO=fake_dataio(), matchbox=fake_matchbox())
    assert svc.transform_data_from_disk(src, out_path) is False
    assert not out_path.exists()


# ── _prune_empty_values: REDCap ""-as-absent normalization ────────────────────
def test_prune_empty_values_drops_empty_leaves():
    rec = {
        "resourceType": "Src",
        "record_id": "2",
        "answered": "yes",
        "unanswered": "",          # REDCap serializes absent answers as ""
        "missing": None,
        "empty_list": [],
        "nested": {"kept": "1", "dropped": ""},
        "all_empty_nested": {"a": "", "b": None},
        "list_mixed": ["x", "", None],
        "zero": 0,                  # falsy but real values must survive
        "false": False,
    }
    pruned = PipelineTransformService._prune_empty_values(rec)
    assert pruned == {
        "resourceType": "Src",
        "record_id": "2",
        "answered": "yes",
        "nested": {"kept": "1"},
        "list_mixed": ["x"],
        "zero": 0,
        "false": False,
    }


def test_transform_data_prunes_empty_values_before_matchbox(
    make_app_state, state, fake_matchbox
):
    mb = fake_matchbox(transform_result={"resourceType": "Patient"})
    svc = make_service(make_app_state, state, matchbox=mb)
    svc.transform_data(
        {"resourceType": "Src", "kept": "v", "gone": ""},
        structure_map_url="http://x/sm",
    )
    sent = mb.transform_calls[0][0]
    assert sent == {"resourceType": "Src", "kept": "v"}


# --------------------------------------------------------------------------- #
# map_errors: engine failures must be reportable, never silent
# --------------------------------------------------------------------------- #


def test_single_map_failure_is_recorded_in_map_errors(make_app_state, state, fake_matchbox):
    mb = fake_matchbox(transform_result=None)  # engine returns nothing
    svc = make_service(make_app_state, state, matchbox=mb)
    errors = {}
    out = svc.transform_data(
        {"resourceType": "Src"}, structure_map_url="http://x/sm", map_errors=errors
    )
    assert out is None
    assert "http://x/sm" in errors


def test_multi_map_failure_is_recorded_but_gated_skip_is_not(
    make_app_state, state, fake_matchbox
):
    # One loaded map whose (non-gated) transform fails -> map_errors entry;
    # transform_data returns None because nothing was produced.
    mb = fake_matchbox(transform_result=None)
    svc = make_service(make_app_state, state, matchbox=mb)
    state.structure_map_urls = ["http://x/sm-a"]
    errors = {}
    out = svc.transform_data({"resourceType": "Src", "f": "1"}, map_errors=errors)
    assert out is None
    assert errors == {
        "http://x/sm-a": "transform returned no output (engine error)"
    }
