"""
WIP WIP WIP

Provenance plugin — runtime lineage metadata via the ``post_bundle`` hook.

Augments every assembled transform bundle with FHIR Provenance resources so
the audit trail *source record → (source representation →) derived resources*
becomes queryable with plain FHIR search (``Provenance?target=``, ``?entity=``,
``?agent=``). Strictly additive: removing the plugin (or deleting the
Provenance/Device resources server-side) restores the plain pipeline output.

Emitted per bundle (mirroring the pipeline lifecycle):
  1. optional *ingest* Provenance — the raw source record (identifier-only
     ``entity.what``, no FHIR resource needed) → the source-representation
     resource (e.g. the QuestionnaireResponse of a REDCap record), when
     ``source_resource_type`` is configured and present in the bundle; the
     map that produced the source representation is attributed here
  2. *transform* Provenance(s) — when the transform layer supplies per-map
     attribution (``context["map_outputs"]``, threaded through
     ``create_bundle``), ONE Provenance per map execution: targets are only
     that map's outputs, so impact queries ("map X defective → what needs
     regeneration?") are exact. Resources without attribution (or callers
     that supply no map_outputs) get one aggregate Provenance naming all
     maps — per-record attribution, the previous behavior.

Modeling notes (R4):
  - activity = ISO 21089 lifecycle ``transform``
  - the tool is an ``assembler`` agent (Device, versioned); the Device entry is
    included with PUT semantics (idempotent) unless ``include_device`` is false
  - StructureMaps enter as entities with role ``derivation`` (R5 has the exact
    role ``instantiates``); by default their ``what`` is an identifier-only
    reference (canonical URL) so servers enforcing referential integrity accept
    the bundle without the maps being uploaded — set ``map_reference_mode`` to
    "reference" for resolvable ``StructureMap/<id>`` references instead
    (queryable via ``Provenance?entity=``; requires the maps on the server)

Config (all optional):
  {
    "type": "provenance",
    "record_id_field": "record_id",            # source field carrying the record id
    "source_system": "urn:example:records",    # identifier system of the origin record
    "source_resource_type": "QuestionnaireResponse",  # enables the 2-step chain
    "device_id": "fsh-nifi-bridge",
    "include_device": true,
    "map_reference_mode": "identifier",         # or "reference"
    "recorded": "2026-07-07T12:00:00Z"          # fixed timestamp (tests); default now-UTC
  }
"""

import datetime
import uuid

import logging
logger = logging.getLogger(__name__)

from plugins.base import PipelinePlugin

LIFECYCLE_TRANSFORM = {
    "coding": [
        {
            "system": "http://terminology.hl7.org/CodeSystem/iso-21089-lifecycle",
            "code": "transform",
            "display": "Transform/Translate Record Lifecycle Event",
        }
    ]
}

PARTICIPANT_ASSEMBLER = {
    "coding": [
        {
            "system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
            "code": "assembler",
        }
    ]
}


def _tool_version() -> str:
    """installed package version (best effort — plugins must never raise here)"""
    try:
        from importlib.metadata import version

        return version("fsh-nifi-bridge")
    except Exception:
        return "unknown"


class ProvenancePlugin(PipelinePlugin):
    """Adds Provenance (+ Device) entries to assembled transform bundles."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.record_id_field = config.get("record_id_field", "record_id")
        self.source_system = config.get("source_system", "urn:example:source-record")
        self.source_resource_type = config.get("source_resource_type")
        self.device_id = config.get("device_id", "fsh-nifi-bridge")
        self.include_device = config.get("include_device", True)
        self.map_reference_mode = config.get("map_reference_mode", "identifier")
        self.fixed_recorded = config.get("recorded")
        self.tool_version = config.get("tool_version") or _tool_version()

    @property
    def plugin_id(self) -> str:
        return "provenance"

    # ── hook ──────────────────────────────────────────────────────────────────

    def post_bundle(self, bundle: dict, context: dict) -> dict:
        entries = bundle.get("entry") or []
        # (entry_ref, resource) for every transformed resource in the bundle
        resource_refs = []
        for entry in entries:
            resource = entry.get("resource")
            if not isinstance(resource, dict) or "resourceType" not in resource:
                continue
            ref = self._entry_ref(entry, resource)
            if ref:
                resource_refs.append((ref, resource))
        if not resource_refs:
            return bundle

        recorded = self.fixed_recorded or (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        source_record = context.get("source_record") or {}
        structure_maps = context.get("structure_maps") or []
        # {map_url: [entry refs]} — which map produced which resource; supplied
        # by the transform layer via create_bundle. Empty/absent -> aggregate
        # (per-record) attribution fallback.
        map_outputs = context.get("map_outputs") or {}
        sm_by_url = {sm.get("url"): sm for sm in structure_maps if sm.get("url")}

        new_entries = []
        agent_ref = f"Device/{self.device_id}"
        if self.include_device:
            device_urn = f"urn:uuid:{uuid.uuid4()}"
            new_entries.append(
                {
                    "fullUrl": device_urn,
                    "resource": self._device(),
                    # PUT: idempotent across bundles — one Device per tool id
                    "request": {"method": "PUT", "url": f"Device/{self.device_id}"},
                }
            )
            agent_ref = device_urn
        agent = [{"type": PARTICIPANT_ASSEMBLER, "who": {"reference": agent_ref}}]

        record_entity = self._record_entity(source_record)
        source_reps = [
            (ref, res)
            for ref, res in resource_refs
            if self.source_resource_type
            and res["resourceType"] == self.source_resource_type
        ]
        derived = [
            (ref, res) for ref, res in resource_refs if (ref, res) not in source_reps
        ]

        source_rep_refs = {ref for ref, _ in source_reps}
        derived_refs = [ref for ref, _ in derived]

        # step 1: raw source record -> source representation (e.g. the QR).
        # A map whose outputs are all source-rep resources belongs to this
        # ingest activity (it *produced* the source representation).
        if source_reps and record_entity:
            ingest_entities = [record_entity]
            for url, refs in map_outputs.items():
                if refs and all(r in source_rep_refs for r in refs):
                    ingest_entities.append(
                        self._map_entity(sm_by_url.get(url, {"url": url}))
                    )
            new_entries.append(
                self._provenance_entry(
                    targets=[ref for ref, _ in source_reps],
                    entities=ingest_entities,
                    agent=agent,
                    recorded=recorded,
                )
            )

        # step 2: source representation (or raw record) + map(s) -> derived
        # resources. With map_outputs: one Provenance per map execution
        # (exact lineage — targets are only that map's outputs). Resources
        # without attribution (or when the transform layer supplied no
        # map_outputs at all) fall back to one aggregate Provenance naming
        # all maps (per-record attribution).
        if derived:
            if source_reps:
                source_entity = {
                    "role": "source",
                    "what": {"reference": source_reps[0][0]},
                }
            else:
                source_entity = record_entity
            base_entities = [source_entity] if source_entity else []

            attributed = set()
            for url, refs in map_outputs.items():
                targets = [r for r in refs if r in derived_refs]
                if not targets:
                    continue
                attributed.update(targets)
                provenance = self._provenance_entry(
                    targets=targets,
                    entities=base_entities
                    + [self._map_entity(sm_by_url.get(url, {"url": url}))],
                    agent=agent,
                    recorded=recorded,
                )
                provenance["resource"]["policy"] = [url]
                new_entries.append(provenance)

            leftover = [r for r in derived_refs if r not in attributed]
            if leftover:
                entities = base_entities + [
                    self._map_entity(sm) for sm in structure_maps
                ]
                provenance = self._provenance_entry(
                    targets=leftover,
                    entities=entities,
                    agent=agent,
                    recorded=recorded,
                )
                policies = [sm.get("url") for sm in structure_maps if sm.get("url")]
                if policies:
                    provenance["resource"]["policy"] = policies
                new_entries.append(provenance)

        if new_entries:
            bundle["entry"] = entries + new_entries
        return bundle

    # ── builders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _entry_ref(entry: dict, resource: dict):
        """intra-bundle reference for an entry: fullUrl (urn:uuid) or type/id"""
        if entry.get("fullUrl"):
            return entry["fullUrl"]
        if resource.get("id"):
            return f"{resource['resourceType']}/{resource['id']}"
        return None

    def _device(self) -> dict:
        return {
            "resourceType": "Device",
            "id": self.device_id,
            "deviceName": [{"name": "fsh-nifi-bridge", "type": "manufacturer-name"}],
            "version": [{"value": self.tool_version}],
        }

    def _record_entity(self, source_record: dict):
        """entity for the raw (non-FHIR) origin record, or None without an id"""
        record_id = source_record.get(self.record_id_field)
        if record_id in (None, ""):
            return None
        return {
            "role": "source",
            "what": {
                "identifier": {"system": self.source_system, "value": str(record_id)},
                "display": f"source record {record_id}",
            },
        }

    def _map_entity(self, structure_map: dict) -> dict:
        """the transform recipe as an entity (R4 'derivation'; R5 would be 'instantiates')"""
        url = structure_map.get("url", "")
        if self.map_reference_mode == "reference" and structure_map.get("id"):
            what = {
                "reference": f"StructureMap/{structure_map['id']}",
                "display": url,
            }
        else:
            # identifier-only reference: queryable, and servers that enforce
            # referential integrity accept it without the map being uploaded
            what = {
                "identifier": {"system": "urn:ietf:rfc:3986", "value": url},
                "display": structure_map.get("name") or url,
            }
        return {"role": "derivation", "what": what}

    @staticmethod
    def _provenance_entry(targets, entities, agent, recorded) -> dict:
        return {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Provenance",
                "target": [{"reference": t} for t in targets],
                "recorded": recorded,
                "activity": LIFECYCLE_TRANSFORM,
                "agent": agent,
                "entity": entities,
            },
            "request": {"method": "POST", "url": "Provenance"},
        }
