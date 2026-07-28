"""Pure helpers for reverse mapping (WIP WIP WIP).
"""

import logging
from pathlib import Path

from helpers import utils
import data_handling.instance_validation as iv
from mapping.rule_ir import mapping_target_path

logger = logging.getLogger(__name__)

# generated ConceptMaps are named after the target element leaf by the static
# generator and after the source field by the REDCap plugin
_CM_NAME_PREFIXES = ("ConceptMap_", "REDCap_")


def load_inverse_concept_maps(cm_dir) -> dict:
    """name-key -> {output_code: [source_codes]} from a project's generated ConceptMaps"""
    inverse = {}
    cm_dir = Path(cm_dir) if cm_dir else None
    if cm_dir is None or not cm_dir.is_dir():
        return inverse
    for path in sorted(cm_dir.glob("*.json")):
        cm = utils.get_json(path)
        if not isinstance(cm, dict) or cm.get("resourceType") != "ConceptMap":
            continue
        name = cm.get("name", "")
        key = next(
            (name[len(p):] for p in _CM_NAME_PREFIXES if name.startswith(p)), None
        )
        if not key:
            continue
        mapping = inverse.setdefault(key, {})
        for group in cm.get("group", []):
            for element in group.get("element", []):
                for target in element.get("target", []):
                    if target.get("code") is None or element.get("code") is None:
                        continue
                    mapping.setdefault(target["code"], []).append(element["code"])
    return inverse


def inverse_for(source_field: str, target_path: str, inverse_cms: dict) -> dict:
    """resolve the inverse ConceptMap for a mapping entry (target leaf, then source field)"""
    if not inverse_cms:
        return {}
    leaf = None
    if target_path:
        leaf = iv._element_name(target_path.split(".")[-1])
    field_leaf = source_field.rsplit(".", 1)[-1] if source_field else None
    return inverse_cms.get(leaf) or inverse_cms.get(field_leaf) or {}


def simplify_value(value):
    """reduce an extracted complex value to its source-shaped representative

    CodeableConcept/Coding nodes carry the mapped source value in their code
    (the forward generator maps coded fields at the CodeableConcept level);
    everything else is returned unchanged."""
    if isinstance(value, dict):
        codings = value.get("coding")
        if isinstance(codings, list):
            codes = [c.get("code") for c in codings if isinstance(c, dict) and c.get("code") is not None]
            if codes:
                return codes[0] if len(codes) == 1 else codes
        if "code" in value and isinstance(value.get("code"), str) and "system" in value:
            return value["code"]  # bare Coding
    return value


def translate_back(value, inverse: dict):
    """map an output code back to its source code via an inverted ConceptMap"""
    if not inverse or not isinstance(value, str) or value not in inverse:
        return value, None
    candidates = sorted(set(inverse[value]))
    if len(candidates) == 1:
        return candidates[0], None
    return value, candidates


def profile_ids_by_type(structure_definitions) -> dict:
    """FHIR base type -> {profile ids/names} for profile-rooted mapping paths"""
    by_type = {}
    for sd in structure_definitions or []:
        if not isinstance(sd, dict) or sd.get("resourceType") != "StructureDefinition":
            continue
        base_type = sd.get("type")
        if not base_type:
            continue
        ids = by_type.setdefault(base_type, set())
        for key in ("id", "name"):
            if sd.get(key):
                ids.add(sd[key])
    return by_type


def iter_resource_sets(payload):
    """yield one list of resources per logical record from a payload"""
    if isinstance(payload, dict) and payload.get("resourceType") == "Bundle":
        yield [
            e["resource"]
            for e in payload.get("entry", [])
            if isinstance(e, dict) and isinstance(e.get("resource"), dict)
        ]
    elif isinstance(payload, dict):
        yield [payload]
    elif isinstance(payload, list):
        if payload and all(
            isinstance(b, dict) and b.get("resourceType") == "Bundle" for b in payload
        ):
            for bundle in payload:
                yield from iter_resource_sets(bundle)
        else:
            yield [r for r in payload if isinstance(r, dict)]


def reverse_map_resources(
    resources, mapping_table: dict, profile_ids_map=None, inverse_cms=None
) -> dict:
    """build one flat source-shaped record from a set of resources"""

    record = {}
    notes = {}
    for source_field, target_value in mapping_table.items():
        target_path = mapping_target_path(target_value)
        if not target_path:
            continue
        flat_field = (
            source_field.split(".", 1)[1]
            if source_field.startswith("Sourcedefinition_")
            else source_field
        )
        inverse = inverse_for(source_field, target_path, inverse_cms or {})
        candidates = []
        for resource in resources:
            resource_type = resource.get("resourceType")
            profile_ids = (profile_ids_map or {}).get(resource_type)
            normalized = iv._applicable_target(
                target_path, resource_type, profile_ids=profile_ids
            )
            if normalized is None:
                continue
            for value in iv.extract_path(resource, normalized):
                candidates.append(simplify_value(value))
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            continue
        distinct = {str(c) for c in candidates}
        value = candidates[0] if len(distinct) == 1 else candidates
        ambiguous = None
        if isinstance(value, str):
            value, ambiguous = translate_back(value, inverse)
        if ambiguous:
            notes[flat_field] = {"ambiguous_inverse_candidates": ambiguous}
        elif len(distinct) > 1:
            notes[flat_field] = {"multiple_candidates": candidates}
        record[flat_field] = value
    if notes:
        record["_reverse_mapping"] = notes
    return record


def reverse_map_payload(
    payload, mapping_table: dict, profile_ids_map=None, inverse_cms=None
) -> list:
    """reverse-map a payload (Bundle / list of Bundles / resource(s)) to flat records"""
    return [
        reverse_map_resources(
            resources, mapping_table, profile_ids_map=profile_ids_map,
            inverse_cms=inverse_cms,
        )
        for resources in iter_resource_sets(payload)
        if resources
    ]


def roundtrip_report(records, source_records, mapping_table: dict) -> dict:
    """compare reverse-mapped records against the original flat source records"""
    if isinstance(source_records, dict):
        source_records = [source_records]
    flat_fields = {
        (k.split(".", 1)[1] if k.startswith("Sourcedefinition_") else k)
        for k in mapping_table
    }
    per_record = []
    tally = {}

    def _count(status):
        tally[status] = tally.get(status, 0) + 1

    for i, source in enumerate(source_records):
        record = records[i] if i < len(records) else {}
        fields = {}
        for field, expected in source.items():
            if field not in flat_fields:
                status = "not_mapped"
            elif field not in record:
                status = "missing"
            elif record[field] == expected:
                status = "exact"
            elif str(record[field]) == str(expected):
                status = "cast"
            else:
                status = "recovered_differs"
            fields[field] = {
                "status": status,
                "source": expected,
                "extracted": record.get(field),
            }
            _count(status)
        per_record.append(fields)
    return {"summary": tally, "records": per_record}
