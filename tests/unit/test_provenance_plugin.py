"""Unit tests for the provenance plugin and the post_bundle runtime hook.

The provenance plugin augments assembled transform bundles with Provenance
(+ Device) entries via the plugin system's only runtime hook (post_bundle,
augmentation-only). These tests pin: hook dispatch in BundleService
(default no-op, plugin isolation), plugin registration, and the emitted
Provenance structure (2-step chain, agents, map entities, reference modes).
"""

import pytest

from controller.bundle_service import BundleService
from plugins.base import PipelinePlugin
from plugins.provenance.plugin import ProvenancePlugin
from plugins.registry import load_plugins

pytestmark = pytest.mark.unit


# ── fixtures ─────────────────────────────────────────────────────────────────

STRUCTURE_MAPS = [
    {
        "resourceType": "StructureMap",
        "id": "sm-patient-abcd1234",
        "url": "http://example.org/StructureMap/patient",
        "name": "PatientMap",
    },
    {
        "resourceType": "StructureMap",
        "id": "sm-condition-ef567890",
        "url": "http://example.org/StructureMap/condition",
        "name": "ConditionMap",
    },
]

SOURCE_RECORD = {"record_id": "rec-7", "family": "Smith", "status": "2"}


def _resources():
    """fresh transformed resources (create_bundle mutates its inputs)"""
    return [
        {"resourceType": "Patient", "name": [{"family": "Smith"}]},
        {"resourceType": "Condition", "code": {"coding": [{"code": "x"}]}},
        {"resourceType": "QuestionnaireResponse", "status": "completed"},
    ]


def _plugin(**overrides) -> ProvenancePlugin:
    config = {
        "type": "provenance",
        "source_system": "http://example.org/redcap/record-id",
        "source_resource_type": "QuestionnaireResponse",
        "recorded": "2026-07-07T12:00:00Z",  # fixed for determinism
        "tool_version": "0.1.0+test",
    }
    config.update(overrides)
    return ProvenancePlugin(config)


def _entries_of_type(bundle, resource_type):
    return [
        e for e in bundle["entry"]
        if e["resource"].get("resourceType") == resource_type
    ]


# ── hook dispatch in BundleService ───────────────────────────────────────────

def test_create_bundle_without_plugins_is_unchanged():
    bundle = BundleService.create_bundle(_resources())
    types = sorted(e["resource"]["resourceType"] for e in bundle["entry"])
    assert types == ["Condition", "Patient", "QuestionnaireResponse"]


def test_create_bundle_passes_context_to_post_bundle():
    seen = {}

    class Probe(PipelinePlugin):
        @property
        def plugin_id(self):
            return "probe"

        def post_bundle(self, bundle, context):
            seen.update(context)
            return bundle

    BundleService.create_bundle(
        _resources(),
        structure_maps=STRUCTURE_MAPS,
        plugins=[Probe({})],
        source_record=SOURCE_RECORD,
    )
    assert seen["source_record"] == SOURCE_RECORD
    assert seen["structure_maps"] == STRUCTURE_MAPS


def test_create_bundle_isolates_failing_plugin():
    class Broken(PipelinePlugin):
        @property
        def plugin_id(self):
            return "broken"

        def post_bundle(self, bundle, context):
            raise RuntimeError("boom")

    bundle = BundleService.create_bundle(_resources(), plugins=[Broken({})])
    assert len(bundle["entry"]) == 3  # transform output unaffected


def test_create_bundle_ignores_non_bundle_hook_return():
    class BadReturn(PipelinePlugin):
        @property
        def plugin_id(self):
            return "bad"

        def post_bundle(self, bundle, context):
            return "not a bundle"

    bundle = BundleService.create_bundle(_resources(), plugins=[BadReturn({})])
    assert bundle["resourceType"] == "Bundle"


# ── plugin registration ──────────────────────────────────────────────────────

def test_registry_loads_provenance_plugin():
    plugins = load_plugins([{"type": "provenance"}])
    assert len(plugins) == 1
    assert plugins[0].plugin_id == "provenance"


# ── emitted provenance structure ─────────────────────────────────────────────

def test_two_step_chain_with_source_representation():
    bundle = BundleService.create_bundle(
        _resources(),
        structure_maps=STRUCTURE_MAPS,
        plugins=[_plugin()],
        source_record=SOURCE_RECORD,
    )
    provenances = _entries_of_type(bundle, "Provenance")
    assert len(provenances) == 2
    ingest, transform = provenances[0]["resource"], provenances[1]["resource"]

    # step 1: raw record (identifier entity) -> QuestionnaireResponse
    assert len(ingest["target"]) == 1
    qr_entry = _entries_of_type(bundle, "QuestionnaireResponse")[0]
    assert ingest["target"][0]["reference"] == qr_entry["fullUrl"]
    ident = ingest["entity"][0]["what"]["identifier"]
    assert ident == {"system": "http://example.org/redcap/record-id", "value": "rec-7"}

    # step 2: QR (reference entity) + maps -> Patient/Condition
    assert len(transform["target"]) == 2
    assert transform["entity"][0]["role"] == "source"
    assert transform["entity"][0]["what"]["reference"] == qr_entry["fullUrl"]
    map_entities = [e for e in transform["entity"] if e["role"] == "derivation"]
    assert len(map_entities) == 2
    assert transform["policy"] == [sm["url"] for sm in STRUCTURE_MAPS]
    assert transform["recorded"] == "2026-07-07T12:00:00Z"
    assert transform["activity"]["coding"][0]["code"] == "transform"


def test_single_step_without_source_representation():
    plugin = _plugin(source_resource_type=None)
    bundle = BundleService.create_bundle(
        _resources(),
        structure_maps=STRUCTURE_MAPS,
        plugins=[plugin],
        source_record=SOURCE_RECORD,
    )
    provenances = _entries_of_type(bundle, "Provenance")
    assert len(provenances) == 1
    prov = provenances[0]["resource"]
    assert len(prov["target"]) == 3  # QR is just another derived resource now
    assert prov["entity"][0]["what"]["identifier"]["value"] == "rec-7"


def test_device_agent_and_version():
    bundle = BundleService.create_bundle(
        _resources(), plugins=[_plugin()], source_record=SOURCE_RECORD
    )
    devices = _entries_of_type(bundle, "Device")
    assert len(devices) == 1
    device = devices[0]
    assert device["request"] == {"method": "PUT", "url": "Device/fsh-nifi-bridge"}
    assert device["resource"]["version"] == [{"value": "0.1.0+test"}]
    prov = _entries_of_type(bundle, "Provenance")[0]["resource"]
    assert prov["agent"][0]["who"]["reference"] == device["fullUrl"]


def test_include_device_false_uses_plain_reference():
    bundle = BundleService.create_bundle(
        _resources(),
        plugins=[_plugin(include_device=False)],
        source_record=SOURCE_RECORD,
    )
    assert not _entries_of_type(bundle, "Device")
    prov = _entries_of_type(bundle, "Provenance")[0]["resource"]
    assert prov["agent"][0]["who"]["reference"] == "Device/fsh-nifi-bridge"


def test_map_reference_modes():
    # default: identifier-only (referential-integrity-safe)
    bundle = BundleService.create_bundle(
        _resources(),
        structure_maps=STRUCTURE_MAPS,
        plugins=[_plugin()],
        source_record=SOURCE_RECORD,
    )
    transform = _entries_of_type(bundle, "Provenance")[1]["resource"]
    map_entity = [e for e in transform["entity"] if e["role"] == "derivation"][0]
    assert "reference" not in map_entity["what"]
    assert map_entity["what"]["identifier"]["value"] == STRUCTURE_MAPS[0]["url"]

    # opt-in: resolvable reference
    bundle = BundleService.create_bundle(
        _resources(),
        structure_maps=STRUCTURE_MAPS,
        plugins=[_plugin(map_reference_mode="reference")],
        source_record=SOURCE_RECORD,
    )
    transform = _entries_of_type(bundle, "Provenance")[1]["resource"]
    map_entity = [e for e in transform["entity"] if e["role"] == "derivation"][0]
    assert map_entity["what"]["reference"] == "StructureMap/sm-patient-abcd1234"


def test_no_source_record_still_emits_transform_provenance():
    bundle = BundleService.create_bundle(
        _resources(), structure_maps=STRUCTURE_MAPS, plugins=[_plugin()]
    )
    provenances = _entries_of_type(bundle, "Provenance")
    # no record identifier -> no ingest step, but lineage to maps/tool remains
    assert len(provenances) == 1
    prov = provenances[0]["resource"]
    roles = {e["role"] for e in prov["entity"]}
    assert roles == {"source", "derivation"}  # QR as source rep + maps


# ── per-map attribution (map_outputs) ────────────────────────────────────────

def _bundle_with_map_outputs(plugin, resources=None):
    """create_bundle with per-map attribution: patient map -> Patient,
    condition map -> Condition, and a QR-producing map"""
    resources = resources if resources is not None else _resources()
    patient, condition, qr = resources
    qr_map = {
        "resourceType": "StructureMap",
        "id": "sm-qr-11112222",
        "url": "http://example.org/StructureMap/qr",
        "name": "QRMap",
    }
    return BundleService.create_bundle(
        resources,
        structure_maps=STRUCTURE_MAPS + [qr_map],
        plugins=[plugin],
        source_record=SOURCE_RECORD,
        map_outputs={
            STRUCTURE_MAPS[0]["url"]: [patient],
            STRUCTURE_MAPS[1]["url"]: [condition],
            qr_map["url"]: [qr],
        },
    )


def test_per_map_attribution_one_provenance_per_map():
    bundle = _bundle_with_map_outputs(_plugin())
    provenances = [e["resource"] for e in _entries_of_type(bundle, "Provenance")]
    # ingest (QR) + one per derived-resource-producing map
    assert len(provenances) == 3
    ingest, prov_patient, prov_condition = provenances

    # the QR-producing map is attributed to the ingest step
    ingest_maps = [e for e in ingest["entity"] if e["role"] == "derivation"]
    assert len(ingest_maps) == 1
    assert ingest_maps[0]["what"]["identifier"]["value"] == "http://example.org/StructureMap/qr"

    # each transform Provenance targets exactly its map's output
    by_policy = {p["policy"][0]: p for p in (prov_patient, prov_condition)}
    patient_entry = _entries_of_type(bundle, "Patient")[0]
    condition_entry = _entries_of_type(bundle, "Condition")[0]
    assert by_policy[STRUCTURE_MAPS[0]["url"]]["target"] == [
        {"reference": patient_entry["fullUrl"]}
    ]
    assert by_policy[STRUCTURE_MAPS[1]["url"]]["target"] == [
        {"reference": condition_entry["fullUrl"]}
    ]
    # exactly ONE map entity per transform Provenance
    for prov in (prov_patient, prov_condition):
        assert len([e for e in prov["entity"] if e["role"] == "derivation"]) == 1
        assert prov["entity"][0]["role"] == "source"  # QR remains the source


def test_partial_attribution_leftover_gets_aggregate_provenance():
    resources = _resources()
    patient = resources[0]
    plugin = _plugin(source_resource_type=None)
    bundle = BundleService.create_bundle(
        resources,
        structure_maps=STRUCTURE_MAPS,
        plugins=[plugin],
        source_record=SOURCE_RECORD,
        map_outputs={STRUCTURE_MAPS[0]["url"]: [patient]},  # only Patient attributed
    )
    provenances = [e["resource"] for e in _entries_of_type(bundle, "Provenance")]
    assert len(provenances) == 2
    exact = [p for p in provenances if len(p["target"]) == 1][0]
    aggregate = [p for p in provenances if len(p["target"]) == 2][0]
    assert exact["policy"] == [STRUCTURE_MAPS[0]["url"]]
    # unattributed resources fall back to naming ALL maps
    assert len([e for e in aggregate["entity"] if e["role"] == "derivation"]) == 2


def test_no_map_outputs_falls_back_to_aggregate():
    """callers that don't thread map_outputs keep the per-record behavior"""
    bundle = BundleService.create_bundle(
        _resources(),
        structure_maps=STRUCTURE_MAPS,
        plugins=[_plugin(source_resource_type=None)],
        source_record=SOURCE_RECORD,
    )
    provenances = [e["resource"] for e in _entries_of_type(bundle, "Provenance")]
    assert len(provenances) == 1
    assert len(provenances[0]["target"]) == 3
    assert len([e for e in provenances[0]["entity"] if e["role"] == "derivation"]) == 2


def test_map_outputs_ignores_resources_dropped_at_assembly():
    """a resource in map_outputs but not in the final bundle must not be targeted"""
    resources = _resources()
    dropped = {"resourceType": "Observation", "status": "final"}
    plugin = _plugin(source_resource_type=None)
    bundle = BundleService.create_bundle(
        resources,
        structure_maps=STRUCTURE_MAPS,
        plugins=[plugin],
        source_record=SOURCE_RECORD,
        map_outputs={STRUCTURE_MAPS[0]["url"]: [resources[0], dropped]},
    )
    exact = [
        e["resource"] for e in _entries_of_type(bundle, "Provenance")
        if e["resource"].get("policy") == [STRUCTURE_MAPS[0]["url"]]
    ][0]
    assert len(exact["target"]) == 1  # only the Patient that made it into the bundle


def test_empty_bundle_is_left_alone():
    plugin = _plugin()
    bundle = {"resourceType": "Bundle", "type": "transaction", "entry": []}
    assert plugin.post_bundle(bundle, {"source_record": None, "structure_maps": []}) == bundle


def test_provenance_targets_use_type_id_when_no_fullurl():
    plugin = _plugin(source_resource_type=None)
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "resource": {"resourceType": "Patient", "id": "p1"},
                "request": {"method": "PUT", "url": "Patient/p1"},
            }
        ],
    }
    out = plugin.post_bundle(
        bundle, {"source_record": SOURCE_RECORD, "structure_maps": []}
    )
    prov = [
        e["resource"] for e in out["entry"]
        if e["resource"]["resourceType"] == "Provenance"
    ][0]
    assert prov["target"] == [{"reference": "Patient/p1"}]
