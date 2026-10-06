"""calculate eval metrics (based on given artifacts), see docs/eval_overview.md"""

from __future__ import annotations

import difflib
import json
from typing import Any, Dict, Iterable, List, Optional

from eval.canonicalize import canonical_map, rule_level_prf1  # noqa: F401 (re-export)
from eval.normalize_fhir import behavioral_equal

try:
    from data_handling.instance_validation import extract_path
except Exception:
    extract_path = None


# define remaining placeholder count
_TODO_MARKERS = ("TODO", "PLACEHOLDER")


def _contains_todo(value: Any) -> bool:
    return isinstance(value, str) and any(m in value for m in _TODO_MARKERS)


def remaining_todo(structure_map: Dict[str, Any]) -> Dict[str, Any]:
    """count rules carrying a placeholder/TODO"""
    hits: List[str] = []

    def walk(rules: List[Dict[str, Any]], path: str) -> None:
        for rule in rules or []:
            name = rule.get("name", "?")
            for src in rule.get("source", []) or []:
                if _contains_todo(src.get("element")):
                    hits.append(f"{path}/{name}: source.element={src['element']}")
            for tgt in rule.get("target", []) or []:
                for p in tgt.get("parameter", []) or []:
                    for v in p.values():
                        if _contains_todo(v):
                            hits.append(f"{path}/{name}: param={v}")
            walk(rule.get("rule", []) or [], f"{path}/{name}")

    for group in structure_map.get("group", []) or []:
        walk(group.get("rule", []) or [], group.get("name", "group"))
    return {"todo_count": len(hits), "todo_rules": hits}

def _resources_by_type(output: Any) -> Dict[str, List[dict]]:
    """index every FHIR resource found anywhere in the output by resourceType"""
    by_type: Dict[str, List[dict]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for x in node:
                visit(x)
        elif isinstance(node, dict):
            rt = node.get("resourceType")
            if rt == "Bundle":
                for entry in node.get("entry", []) or []:
                    visit(entry.get("resource"))
            elif rt:
                by_type.setdefault(rt, []).append(node)

    visit(output)
    return by_type


def _as_text(value: Any) -> str:
    """stringify an output value the way the source record spells it"""

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _tolerant_values(resource: Any, path: str) -> set:
    """values reachable from path, representing how FHIR actually serialises it"""

    def scalars(node: Any) -> list:
        if isinstance(node, dict):
            return [v for child in node.values() for v in scalars(child)]
        if isinstance(node, (list, tuple)):
            return [v for child in node for v in scalars(child)]
        return [] if node is None else [node]

    def step(nodes: list, segment: str) -> list:
        name = segment.split(":", 1)[0]
        choice = name.endswith("[x]")
        base = name[:-3] if choice else name
        out = []
        for node in nodes:
            if isinstance(node, (list, tuple)):
                out.extend(step(list(node), segment))
                continue
            if not isinstance(node, dict):
                continue
            if base in node:
                out.append(node[base])
            elif choice or base not in node:
                out.extend(v for k, v in node.items() if k.startswith(base) and k != base)
        return out

    nodes = [resource]
    for segment in [seg for seg in path.split(".") if seg]:
        nodes = step(nodes, segment)
        if not nodes:
            return set()
    return {_as_text(v) for v in scalars(nodes)}


def data_preservation(
    mapping_table: Dict[str, str],
    source_data: List[dict],
    output: Any,
    concept_mapped_fields: Optional[Iterable[str]] = None,
    profile_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """check that every mapped source value reaches the output"""
    if extract_path is None:
        return {"error": "extract_path unavailable (src not importable)"}
    cm_fields = set(concept_mapped_fields or [])
    by_type = _resources_by_type(output)
    profile_types = profile_types or {}

    def _stamped_with(resources: List[dict], profile_id: str) -> List[dict]:
        out = []
        for r in resources:
            profiles = (r.get("meta") or {}).get("profile") or []
            if any(str(p).rstrip("/").endswith(profile_id) for p in profiles):
                out.append(r)
        return out or resources

    preserved, transformed, missing, unresolved = [], [], [], []
    for field, target in mapping_table.items():
        root, _, rest = target.partition(".")
        base_type = profile_types.get(root)
        if root in by_type:
            candidates = by_type[root]
            rel = target
        elif base_type is not None:
            candidates = _stamped_with(by_type.get(base_type, []), root)
            rel = rest
        elif root[:1].isupper():
            candidates = by_type.get(root, [])
            rel = target
        else:
            unresolved.append({"field": field, "target": target})
            continue
        keys = (field, field.rsplit(".", 1)[-1])
        src_vals = {
            _as_text(rec[k])
            for rec in source_data
            for k in keys
            if rec.get(k) not in (None, "")
        }
        if not src_vals:
            continue
        out_vals = set()
        for res in candidates:
            out_vals.update(_as_text(v) for v in extract_path(res, rel))
            out_vals |= _tolerant_values(res, rel.split(".", 1)[1] if "." in rel and rel.split(".", 1)[0][:1].isupper() else rel)
        record = {
            "field": field,
            "target": target,
            "source_values": sorted(src_vals),
            "found": sorted(src_vals & out_vals),
            "absent": sorted(src_vals - out_vals),
        }
        if field in cm_fields and out_vals:
            transformed.append(record)
        elif src_vals <= out_vals:
            preserved.append(record)
        else:
            missing.append(record)

    graded = len(preserved) + len(missing)
    return {
        "fields_total": len([f for f in mapping_table]),
        "fields_graded": graded,
        "fields_preserved": len(preserved),
        "fields_with_loss": len(missing),
        "fields_transform_mapped": len(transformed),
        "fields_unresolved_profile_root": len(unresolved),
        "preservation_pct": round(100.0 * len(preserved) / graded, 1) if graded else 100.0,
        "losses": missing,
        "transform_mapped": transformed,
        "unresolved": unresolved,
    }


def edit_distance(candidate: Dict[str, Any], golden: Dict[str, Any]) -> Dict[str, Any]:
    """edit distance between canonical forms (normalized) of two maps"""
    a = json.dumps(canonical_map(candidate), indent=1, sort_keys=True).splitlines()
    b = json.dumps(canonical_map(golden), indent=1, sort_keys=True).splitlines()
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return {
        "similarity": round(ratio, 4),
        "distance": round(1.0 - ratio, 4),
        "candidate_lines": len(a),
        "golden_lines": len(b),
    }


def behavioral_equivalence(out_candidate: Any, out_golden: Any) -> Dict[str, Any]:
    """behavioral equivalence of two transform outputs (M2)"""
    eq, diffs = behavioral_equal(out_candidate, out_golden)
    return {"equivalent": eq, "diff_count": len(diffs), "diffs": diffs[:50]}


def _registry_paths(mappable_fields: List[dict]) -> set:
    """all element paths present in a registry object field tree"""
    paths = set()

    def walk(fields):
        for f in fields or []:
            if not isinstance(f, dict):
                continue
            p = f.get("path")
            if p:
                paths.add(p)
            for key in ("children", "type_structure", "slices"):
                walk(f.get(key))
            for t in f.get("type", []) or []:
                if isinstance(t, dict):
                    walk(t.get("type_structure"))

    walk(mappable_fields)
    return paths


def _profile_element_paths(profile_sd: Dict[str, Any]) -> set:
    """element paths a profile declares and permits"""
    elements = (profile_sd.get("snapshot") or {}).get("element") or (
        profile_sd.get("differential") or {}
    ).get("element") or []
    prohibited = {
        el.get("path")
        for el in elements
        if str(el.get("max")) == "0" and el.get("path")
    }
    out = set()
    for el in elements:
        path = el.get("path") or el.get("id")
        if not path or "." not in path:
            continue
        if any(path == p or path.startswith(p + ".") for p in prohibited):
            continue
        out.add(path)
    return out


def extraction_completeness(
    profile_sd: Dict[str, Any], registry_obj: Dict[str, Any]
) -> Dict[str, Any]:
    """percentage of element paths of a profile represented in the registry (M8)"""
    prof = _profile_element_paths(profile_sd)
    reg = _registry_paths(registry_obj.get("mappable_fields") or [])
    represented = {p for p in prof if p in reg}
    missing = sorted(prof - represented)
    return {
        "profile_elements": len(prof),
        "represented": len(represented),
        "completeness_pct": round(100.0 * len(represented) / len(prof), 1) if prof else 100.0,
        "not_extracted": missing,
        "dropped_element_count": len(missing),
    }


def _profile_constructs(profile_sd: Dict[str, Any]) -> Dict[str, int]:
    """constructs detectors over a profile snapshot"""
    elements = (profile_sd.get("snapshot") or {}).get("element") or (
        profile_sd.get("differential") or {}
    ).get("element") or []
    c = {
        "slicing": 0,
        "fixed_or_pattern": 0,
        "binding": 0,
        "reference": 0,
        "extension": 0,
        "choice_x": 0,
        "complex_type": 0,
    }
    for el in elements:
        path = el.get("path", "")
        if el.get("slicing"):
            c["slicing"] += 1
        if any(k.startswith("fixed") or k.startswith("pattern") for k in el):
            c["fixed_or_pattern"] += 1
        if el.get("binding"):
            c["binding"] += 1
        types = el.get("type") or []
        codes = {t.get("code") for t in types if isinstance(t, dict)}
        if "Reference" in codes:
            c["reference"] += 1
        if "Extension" in codes or path.endswith(".extension") or ".extension:" in path:
            c["extension"] += 1
        if path.endswith("[x]"):
            c["choice_x"] += 1
        if codes and codes & {"CodeableConcept", "Coding", "Identifier", "Quantity", "Period", "Address", "HumanName"}:
            c["complex_type"] += 1
    return c


def _map_evidence(structure_map: Dict[str, Any]) -> Dict[str, bool]:
    """detection of which constructs the generated map actually emits"""
    blob = json.dumps(structure_map)
    ev = {
        "slicing": '"listMode"' in blob or '"listRuleId"' in blob,
        "fixed_or_pattern": False,
        "binding": '"translate"' in blob or "ConceptMap" in blob,
        "reference": '"reference"' in blob or "Reference" in blob,
        "extension": '"extension"' in blob,
        "choice_x": False,
        "complex_type": '"create"' in blob,
    }

    def walk(rules):
        for r in rules or []:
            for t in r.get("target", []) or []:
                tf = t.get("transform")
                if tf in (None, "copy"):
                    pass
                params = t.get("parameter", []) or []
                if tf and tf != "copy" and tf != "create":
                    for p in params:
                        if any(k.startswith("value") and not k == "valueId" for k in p):
                            ev["fixed_or_pattern"] = True
                el = t.get("element", "") or ""
                if any(el.endswith(s) for s in ("Quantity", "DateTime", "String", "Boolean", "Code")) and tf == "create":
                    ev["choice_x"] = True
            walk(r.get("rule", []) or [])

    for g in structure_map.get("group", []) or []:
        walk(g.get("rule", []) or [])
    return ev


def feature_matrix(
    profile_sd: Dict[str, Any], structure_map: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """per-construct {present_in_profile, emitted_in_map} matrix"""
    present = _profile_constructs(profile_sd)
    emitted = _map_evidence(structure_map) if structure_map else {k: False for k in present}
    matrix = {}
    for k, n in present.items():
        matrix[k] = {
            "present_in_profile": n,
            "emitted_in_map": bool(emitted.get(k)) if n else None,
        }
    return {"matrix": matrix, "note": "emission flags are heuristic"}


def graceful_degradation(
    profile_sd: Dict[str, Any], registry_obj: Dict[str, Any]
) -> Dict[str, Any]:
    """count (unreported) value drops"""
    comp = extraction_completeness(profile_sd, registry_obj)
    return {
        "silent_drops": comp["completeness_pct"] < 100.0,
        "dropped_element_count": len(comp["not_extracted"]),
        "dropped_elements": comp["not_extracted"],
    }
