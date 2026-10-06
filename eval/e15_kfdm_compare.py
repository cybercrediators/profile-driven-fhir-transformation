"""kfdm self-comparison eval script (compare original records vs generated mapping records)

Known/expected difference classes (declared by the user, classified, never
counted as substantive):
  - server ids / fullUrl / request — the original pipeline fixes server ids,
    ours emits urn:uuids + conditional POSTs. All ids and reference values are
    normalized to <ref:ResourceType> tokens before diffing.
  - identifier naming — e.g. QuestionnaireResponse.questionnaire links the
    questionnaire under a different name/canonical per pipeline.

Usage:
    PYTHONPATH=.:src python3 -m eval.e15_kfdm_compare [--record 2] [--offline OUTPUT.json]
Writes eval/reports/e15_kfdm_record<N>.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
_REPORTS = Path(__file__).resolve().parent / "reports"
_ORIGINAL = Path(__file__).resolve().parent / "kfdm_pipeline_bundle_example.json"

EXPECTED_PATH_CLASSES = {
    "questionnaire-link": re.compile(r"^QuestionnaireResponse\.questionnaire$"),
    "narrative": re.compile(r"\.text($|\.)") ,
}


def original_record_group(bundle: dict, rec: str) -> List[dict]:
    """The original pipeline's resources for one record: Patient/QR ``sl1-<rec>``,
    Observation ``sl1-<rec>0``, Conditions ``sl1-<rec>1..5``."""
    want = {
        ("Patient", f"sl1-{rec}"), ("QuestionnaireResponse", f"sl1-{rec}"),
        ("Observation", f"sl1-{rec}0"),
    } | {("Condition", f"sl1-{rec}{k}") for k in range(1, 10)}
    out = []
    for e in bundle.get("entry", []) or []:
        r = e.get("resource") or {}
        if (r.get("resourceType"), r.get("id")) in want:
            out.append(r)
    return out


def _reference_token(ref: str, id_to_type: Dict[str, str]) -> str:
    """Normalize any reference form (Type/id, bare id, urn:uuid) to <ref:Type>."""
    if ref.startswith("urn:uuid:"):
        return f"<ref:{id_to_type.get(ref, '?')}>"
    if "/" in ref:
        t, _ = ref.split("/", 1)
        return f"<ref:{t}>"
    if ref in id_to_type:
        return f"<ref:{id_to_type[ref]}>"
    return ref


def normalize_resource(res: dict, id_to_type: Dict[str, str]) -> dict:
    """Strip volatile/id-scheme fields; tokenize references."""
    def walk(o: Any) -> Any:
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if k in ("id", "meta", "text"):
                    continue
                if k == "reference" and isinstance(v, str):
                    out[k] = _reference_token(v, id_to_type)
                else:
                    out[k] = walk(v)
            return out
        if isinstance(o, list):
            return [walk(x) for x in o]
        return o

    return walk(res)


def _id_type_index(resources: List[dict], entries: Optional[List[dict]] = None) -> Dict[str, str]:
    idx = {}
    for r in resources:
        if r.get("id"):
            idx[str(r["id"])] = r.get("resourceType", "?")
    for e in entries or []:
        r = e.get("resource") or {}
        if e.get("fullUrl"):
            idx[str(e["fullUrl"])] = r.get("resourceType", "?")
    return idx


ORIG_CONDITION_SLOT_TOPIC = {
    "1": "bewegung", "2": "diabetes", "3": "herzkreislauf",
    "4": "other-disease", "5": "tumor",
}


def _condition_topic_orig(res: dict, rec: str) -> str:
    slot = str(res.get("id", ""))[len(f"sl1-{rec}"):]
    return ORIG_CONDITION_SLOT_TOPIC.get(slot, f"slot-{slot}")


def _condition_topic_ours(res: dict) -> str:
    for p in (res.get("meta") or {}).get("profile") or []:
        m = re.search(r"anamnese-([a-z-]+)", p)
        if m:
            return m.group(1)
    return "?"


def match_pairs(
    orig: List[dict], ours: List[dict], rec: str
) -> List[Tuple[str, Optional[dict], Optional[dict]]]:
    """Pair RAW resources (matching uses ids/meta that normalization strips)."""
    pairs: List[Tuple[str, Optional[dict], Optional[dict]]] = []
    by_type_o: Dict[str, List[dict]] = {}
    by_type_m: Dict[str, List[dict]] = {}
    for r in orig:
        by_type_o.setdefault(r.get("resourceType", "?"), []).append(r)
    for r in ours:
        by_type_m.setdefault(r.get("resourceType", "?"), []).append(r)
    for rt in sorted(set(by_type_o) | set(by_type_m)):
        o_list, m_list = by_type_o.get(rt, []), by_type_m.get(rt, [])
        if rt == "Condition":
            o_by = {_condition_topic_orig(r, rec): r for r in o_list}
            m_by = {_condition_topic_ours(r): r for r in m_list}
            for k in sorted(set(o_by) | set(m_by)):
                pairs.append((f"{rt}[{k}]", o_by.get(k), m_by.get(k)))
        else:
            for i in range(max(len(o_list), len(m_list))):
                label = rt if max(len(o_list), len(m_list)) == 1 else f"{rt}[{i}]"
                pairs.append((label,
                              o_list[i] if i < len(o_list) else None,
                              m_list[i] if i < len(m_list) else None))
    return pairs


def _list_keys(items: List[Any]) -> List[str]:
    """index a list by identity where FHIR gives it one, else by position"""
    positional = [f"[{i}]" for i in range(len(items))]
    if not items:
        return positional
    link_ids: List[str] = []
    for v in items:
        if not isinstance(v, dict) or v.get("linkId") is None:
            return positional
        link_ids.append(str(v["linkId"]))
    seen: Dict[str, int] = {}
    keys = []
    for link_id in link_ids:
        n = seen.get(link_id, 0)
        seen[link_id] = n + 1
        keys.append(f"[linkId={link_id}]" if n == 0 else f"[linkId={link_id}#{n}]")
    return keys


def flatten(o: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(o, dict):
        for k, v in sorted(o.items()):
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(o, list):
        for key, v in zip(_list_keys(o), o):
            out.update(flatten(v, f"{prefix}{key}"))
    else:
        out[prefix] = o
    return out


def _classify(path: str, resource_type: str) -> Optional[str]:
    logical = f"{resource_type}.{re.sub(r'\\[[^\\]]*\\]', '', path)}"
    for name, rx in EXPECTED_PATH_CLASSES.items():
        if rx.search(logical) or rx.search(path):
            return name
    return None


def diff_pair(label: str, orig: Optional[dict], ours: Optional[dict]) -> Dict[str, Any]:
    rt = (orig or ours or {}).get("resourceType", "?")
    if orig is None:
        return {"resource": label, "verdict": "ONLY_IN_OURS"}
    if ours is None:
        return {"resource": label, "verdict": "ONLY_IN_ORIGINAL"}
    fo, fm = flatten(orig), flatten(ours)
    only_o = sorted(set(fo) - set(fm))
    only_m = sorted(set(fm) - set(fo))
    changed = sorted(k for k in set(fo) & set(fm) if fo[k] != fm[k])

    def bucket(paths: List[str], values: Dict[str, Any]) -> Tuple[List[dict], List[dict]]:
        expected, substantive = [], []
        for p in paths:
            cls = _classify(p, rt)
            item = {"path": p, "value": values.get(p)}
            (expected if cls else substantive).append(
                {**item, **({"class": cls} if cls else {})}
            )
        return expected, substantive

    exp_o, sub_o = bucket(only_o, fo)
    exp_m, sub_m = bucket(only_m, fm)
    exp_c, sub_c = bucket(changed, {})
    for item in exp_c + sub_c:
        item["original"] = fo[item["path"]]
        item["ours"] = fm[item["path"]]
        item.pop("value", None)
    return {
        "resource": label,
        "verdict": "MATCH" if not (sub_o or sub_m or sub_c) else "DIFFERS",
        "substantive": {
            "only_in_original": sub_o,
            "only_in_ours": sub_m,
            "value_mismatch": sub_c,
        },
        "expected_class_diffs": exp_o + exp_m + exp_c,
    }


def generate_ours(record_id: str) -> Tuple[List[dict], dict]:
    from controller.bundle_service import BundleService
    from controller.pipeline_controller.pipeline_controller import PipelineController
    from helpers import utils

    conf = json.loads((_REPO / "conf" / "kfdm_e2e.json").read_text())
    pc = PipelineController(
        conf=conf, force_overwrite=False, create_references=True, minimal_mode=True
    )
    pc.prepare_matchbox_setup(force_upload=True)

    source = json.loads(
        (_REPO / "projects" / "kfdm_e2e" / "source_data" / "source_data.json").read_text()
    )
    records = source if isinstance(source, list) else [source]
    rec = next((r for r in records if str(r.get("record_id")) == record_id), None)
    if rec is None:
        raise SystemExit(
            f"record_id={record_id} not in kfdm_e2e source_data.json "
            f"(have: {[r.get('record_id') for r in records]})"
        )
    resources = pc.transform_data(dict(rec))
    if isinstance(resources, dict):
        resources = [resources]

    sm_data = []
    for f in (_REPO / "projects" / "kfdm_e2e" / "structure_maps").glob("*.json"):
        d = utils.get_json(f)
        if isinstance(d, dict) and d.get("resourceType") == "StructureMap":
            sm_data.append(d)
    bundle = BundleService.create_bundle(
        resources, structure_maps=sm_data, registry=pc.app_state.registry
    )
    return [e["resource"] for e in bundle.get("entry", []) if e.get("resource")], bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="2")
    ap.add_argument("--offline", help="path to a saved bundle for our side")
    args = ap.parse_args()

    original_bundle = json.loads(_ORIGINAL.read_text())
    orig_raw = original_record_group(original_bundle, args.record)
    if not orig_raw:
        raise SystemExit(f"no sl1-{args.record} group in {_ORIGINAL.name}")

    if args.offline:
        ours_bundle = json.loads(Path(args.offline).read_text())
        ours_raw = [e["resource"] for e in ours_bundle.get("entry", [])
                    if e.get("resource")]
    else:
        ours_raw, ours_bundle = generate_ours(args.record)

    idx_o = _id_type_index(orig_raw, original_bundle.get("entry"))
    idx_m = _id_type_index(ours_raw, ours_bundle.get("entry"))

    results = []
    for label, o_raw, m_raw in match_pairs(orig_raw, ours_raw, args.record):
        o = normalize_resource(o_raw, idx_o) if o_raw else None
        m = normalize_resource(m_raw, idx_m) if m_raw else None
        results.append(diff_pair(label, o, m))
    report = {
        "record": args.record,
        "original_resources": len(orig_raw),
        "our_resources": len(ours_raw),
        "results": results,
        "note": "ids/meta/narrative stripped; references tokenized to <ref:Type>; "
                "expected classes: " + ", ".join(EXPECTED_PATH_CLASSES),
    }
    _REPORTS.mkdir(parents=True, exist_ok=True)
    out = _REPORTS / f"e15_kfdm_record{args.record}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n== E15 kfdm record {args.record}: original {len(orig_raw)} vs ours {len(ours_raw)} resources")
    for r in results:
        v = r["verdict"]
        s = r.get("substantive", {})
        counts = (f"only-orig {len(s.get('only_in_original', []))}, "
                  f"only-ours {len(s.get('only_in_ours', []))}, "
                  f"mismatch {len(s.get('value_mismatch', []))}, "
                  f"expected-class {len(r.get('expected_class_diffs', []))}"
                  if v in ("MATCH", "DIFFERS") else "")
        print(f"  {r['resource']:24} {v:18} {counts}")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
