"""Pure helpers for instance-based validation.

These functions have no controller or service dependencies — they operate only on
plain dicts (FHIR JSON) and are importable from the data layer directly.
"""

import logging
logger = logging.getLogger(__name__)
from helpers import utils


def is_definition_resource(resource_type: str) -> bool:
    """check if type is conformance type"""
    cls_ = utils.get_model_class(resource_type)
    if cls_ is None:
        return False
    model_fields = getattr(cls_, "model_fields", None) or {}
    return "url" in model_fields


def extract_path(resource: dict, path: str) -> list:
    """return all leaf values found at a dotted FHIR path within a resource"""
    if not isinstance(resource, dict) or not path:
        return []

    raw_segments = path.split(".")
    if raw_segments and _element_name(raw_segments[0])[:1].isupper():
        raw_segments = raw_segments[1:]

    current = [resource]
    for raw in raw_segments:
        segment = _element_name(raw)
        is_choice = raw.split(":", 1)[0].endswith("[x]")
        nxt = []
        for node in current:
            if isinstance(node, list):
                for item in node:
                    _collect(item, segment, nxt, is_choice)
            else:
                _collect(node, segment, nxt, is_choice)
        current = nxt
        if not current:
            break
    return current


def _element_name(segment: str) -> str:
    """reduce a mapping-path segment to its FHIR element name"""
    segment = segment.split(":", 1)[0]
    if segment.endswith("[x]"):
        segment = segment[:-3]
    return segment


def _collect(node, segment, out: list, is_choice: bool = False):
    """append node[segment] to out, flattening one level of list

    With ``is_choice``, also match the serialized choice-type keys
    (``effective[x]`` → ``effectiveDateTime``/``effectivePeriod``/…)."""
    if not isinstance(node, dict):
        return
    keys = [segment]
    if is_choice and segment not in node:
        keys = [
            k for k in node
            if k.startswith(segment) and k[len(segment):][:1].isupper()
        ]
    for key in keys:
        value = node.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            out.extend(value)
        else:
            out.append(value)


def strip_version(url) -> str:
    """Drop a trailing ``|version`` suffix from a canonical URL (e.g. meta.profile)."""
    if not isinstance(url, str):
        return ""
    return url.split("|", 1)[0]


def _applicable_target(target_path, resource_type: str, profile_ids=None):
    """return the base-type-rooted target path if it applies to this instance, else None"""
    if not isinstance(target_path, str) or not target_path:
        return None
    root = target_path.split(".", 1)[0]
    if root == resource_type:
        return target_path
    if profile_ids and root in profile_ids:
        rest = target_path[len(root) + 1 :]
        return f"{resource_type}.{rest}" if rest else resource_type
    return None


def reverse_extract(instance: dict, mapping_table: dict, profile_ids=None) -> dict:
    """build a flat source object from a FHIR instance using an inverted mapping table"""
    from mapping.rule_ir import mapping_target_path

    resource_type = instance.get("resourceType")
    flat = {}
    for source_field, target_value in mapping_table.items():
        target_path = mapping_target_path(target_value)
        normalized = _applicable_target(target_path, resource_type, profile_ids)
        if normalized is None:
            continue
        values = extract_path(instance, normalized)
        if not values:
            continue
        _set_path(flat, source_field, values[0] if len(values) == 1 else values)
    return flat


def compare_instance(
    expected: dict, actual: dict, mapping_table: dict, profile_ids=None
) -> dict:
    """compare an ``actual`` resource against an ``expected`` ground-truth instance over
    the mapped target paths (base-type- or ``profile_ids``-rooted)"""
    from mapping.rule_ir import mapping_target_path

    resource_type = expected.get("resourceType")
    matched = 0
    total = 0
    diffs = []
    for source_field, target_value in mapping_table.items():
        target_path = mapping_target_path(target_value)
        normalized = _applicable_target(target_path, resource_type, profile_ids)
        if normalized is None:
            continue
        exp = extract_path(expected, normalized)
        if not exp:
            continue  # not present in ground truth -> not applicable
        total += 1
        act = extract_path(actual, normalized)
        if not act:
            status = "missing_in_output"
        elif sorted(map(str, exp)) == sorted(map(str, act)):
            status = "match"
            matched += 1
        else:
            status = "mismatch"
        if status != "match":
            diffs.append(
                {
                    "field": source_field,
                    "path": target_path,
                    "status": status,
                    "expected": exp,
                    "actual": act,
                }
            )
    coverage = (matched / total) if total else None
    return {"coverage": coverage, "matched": matched, "total": total, "diffs": diffs}


def _profile_from_meta(resource: dict):
    """return the first profile URL from resource.meta.profile, or None if not present"""
    profiles = (resource.get("meta") or {}).get("profile") or []
    return profiles[0] if profiles else None


def synthesize_from_element_examples(sd: dict):
    """build a synthetic instance from a StructureDefinition's ``ElementDefinition.example``
    values, or return None if it carries none"""
    resource_type = sd.get("type")
    if not resource_type:
        return None
    instance = {"resourceType": resource_type}
    found = False
    for kind in ("snapshot", "differential"):
        for element in (sd.get(kind, {}) or {}).get("element", []):
            examples = element.get("example")
            path = element.get("path")
            if not examples or not path:
                continue
            value = _detype_value(examples[0])
            if value is None:
                continue
            if _set_path(instance, path, value):
                found = True
    return instance if found else None


def _detype_value(example: dict):
    """return the bare value from an ElementDefinition.example entry (its value[x] key)"""
    for key, value in example.items():
        if key.startswith("value"):
            return value
    return None


def _set_path(obj: dict, path: str, value) -> bool:
    """set a dotted path on a nested dict (best-effort, plain-dict nesting), creating dicts as needed"""
    segments = [_element_name(s) for s in path.split(".")]
    if segments and segments[0][:1].isupper():
        segments = segments[1:]
    if not segments:
        return False
    cursor = obj
    for segment in segments[:-1]:
        existing = cursor.get(segment)
        if not isinstance(existing, dict):
            existing = {}
            cursor[segment] = existing
        cursor = existing
    cursor[segments[-1]] = value
    return True


def discover_examples(
    resources,
    ig_resource=None,
    structure_definitions=None,
    include_element_examples=False,
) -> list:
    """resolve example instances from the JSON already present in a project"""
    resources = list(resources)
    index = {}
    for file, res in resources:
        if (
            isinstance(res, dict)
            and res.get("resourceType")
            and res.get("id") is not None
        ):
            index[f"{res['resourceType']}/{res['id']}"] = (file, res)

    records = []
    seen = set()

    # check ImplementationGuide.definition.resource (authoritative)
    if isinstance(ig_resource, dict):
        for entry in (ig_resource.get("definition", {}) or {}).get(
            "resource", []
        ) or []:
            is_example = entry.get("exampleBoolean") is True or bool(
                entry.get("exampleCanonical")
            )
            if not is_example:
                continue
            ref = (entry.get("reference", {}) or {}).get("reference")
            if not ref or ref not in index:
                if ref:
                    logger.warning(
                        "IG lists example %s but it was not found among loaded resources.",
                        ref,
                    )
                continue
            file, res = index[ref]
            profile = entry.get("exampleCanonical") or _profile_from_meta(res)
            records.append(
                {"resource": res, "profile_url": profile, "source": "ig", "file": file}
            )
            seen.add(ref)

    # standalone instance files
    for file, res in resources:
        if not isinstance(res, dict):
            continue
        resource_type = res.get("resourceType")
        if not resource_type or is_definition_resource(resource_type):
            continue
        key = f"{resource_type}/{res.get('id')}"
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "resource": res,
                "profile_url": _profile_from_meta(res),
                "source": "file",
                "file": file,
            }
        )

    # ElementDefinition.example -> synthetic instances (supplementary)
    if include_element_examples:
        for sd in structure_definitions or []:
            synth = synthesize_from_element_examples(sd)
            if synth:
                records.append(
                    {
                        "resource": synth,
                        "profile_url": sd.get("url"),
                        "source": "element-example",
                        "file": None,
                    }
                )

    return records
