"""run synthetic generalizability ladder eval
Usage: PYTHONPATH=.:src python3 -m eval.synthetic_eval
(set a running matchbox instance first via MATCHBOX=<url>)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from eval import metrics

_REPO = Path(__file__).resolve().parent.parent
_SNAP_DIRS = [
    _REPO / "sushi-docker/fsh_shared/synthladder/temp/pages",
    _REPO / "sushi-docker/fsh_shared/synthladder/output",
    _REPO / "sushi-docker/fsh_shared/synthladder/fsh-generated/resources",
]
# define temporary path and default matchbox URL
_WORK = Path("/tmp/synthetic-eval")
MATCHBOX = os.environ.get("MATCHBOX", "http://localhost:8080/matchboxv3/")

# define FHIR shorthand profiles including the corresponding constructs
LADDER = [
    {
        "rung": "L1-scalar", "id": "synth-l1-patient", "type": "Patient",
        "mapping": {"familyName": "Patient.name.family"},
        "record": {"familyName": "Smith"},
    },
    {
        "rung": "L2-cardinality+complextype", "id": "synth-l2-patient", "type": "Patient",
        "mapping": {"idValue": "Patient.identifier.value", "birth": "Patient.birthDate"},
        "record": {"idValue": "p-1", "birth": "1980-01-01"},
    },
    {
        "rung": "L3-binding+fixed", "id": "synth-l3-patient", "type": "Patient",
        "mapping": {"gender": "Patient.gender"},
        "record": {"gender": "male"},
    },
    {
        "rung": "L4-slicing", "id": "synth-l4-patient", "type": "Patient",
        "mapping": {
            "mrn": "Patient.identifier:mrn.value",
            "mrnSys": "Patient.identifier:mrn.system",
        },
        "record": {"mrn": "MRN-9", "mrnSys": "http://example.org/mrn"},
    },
    {
        "rung": "L5-choice+reference", "id": "synth-l5-observation", "type": "Observation",
        "references": ["synth-l1-patient"],
        "mapping": {"status": "Observation.status", "obsCode": "Observation.code.coding.code",
                    "obsValue": "Observation.value[x]",
                    "familyName": "Patient.name.family"},
        "record": {"status": "final", "obsCode": "718-7", "obsValue": "5",
                   "familyName": "Smith"},
    },
    {
        "rung": "L6-extension+backbone", "id": "synth-l6-patient", "type": "Patient",
        "mapping": {"note": "Patient.extension.note", "contactFamily": "Patient.contact.name.family"},
        "record": {"note": "hi", "contactFamily": "Doe"},
    },
]


def _find_snapshot(profile_id: str):
    for d in _SNAP_DIRS:
        f = d / f"StructureDefinition-{profile_id}.json"
        if f.is_file():
            sd = json.loads(f.read_text())
            if (sd.get("snapshot") or {}).get("element"):
                return sd, f
    return None, None


def _scaffold(rung: dict, sd_path: Path) -> Path:
    proj = _WORK / rung["id"]
    if proj.exists():
        shutil.rmtree(proj)
    (proj / "input_profile").mkdir(parents=True)
    (proj / "source_data").mkdir(parents=True)
    (proj / "structure_maps").mkdir()
    (proj / "processed_resources").mkdir()
    (proj / "source_definitions").mkdir()
    (proj / "converted_data").mkdir()
    shutil.copy(sd_path, proj / "input_profile" / sd_path.name)
    for d in _SNAP_DIRS:
        ext = d / "StructureDefinition-synth-note.json"
        if ext.is_file():
            shutil.copy(ext, proj / "input_profile" / ext.name)
            break
    for profile_id in rung.get("references", []):
        _, ref_path = _find_snapshot(profile_id)
        if ref_path is not None:
            shutil.copy(ref_path, proj / "input_profile" / ref_path.name)
    (proj / "source_data" / "mapping_table.json").write_text(json.dumps(rung["mapping"], indent=2))
    (proj / "source_data" / "source_data.json").write_text(json.dumps([rung["record"]], indent=2))
    return proj


def _conf(rung: dict, proj: Path) -> Path:
    conf = {
        "project_path": str(proj) + "/",
        "profile_path": "",
        "resource_cache_path": "data/resource_cache/",
        "input_source_example": "source_data.json",
        "mapping_table_path": str(proj / "source_data" / "mapping_table.json"),
        "base_profile_url": "http://example.org/synthladder",
        "simplifier_api_url": "https://fhir.simplifier.net/R4/",
        "bundled_output": False,
        "external_cache_service": "DISK",
        "cache_args": {"cache_dir": str(proj / ".cache")},
        "matchbox_connection": {"url": MATCHBOX, "preload_location": ""},
        "terminology_server_uri": "https://tx.fhir.org/r4/",
        "root_resources": [],
    }
    p = _WORK / f"{rung['id']}.conf.json"
    p.write_text(json.dumps(conf, indent=2))
    return p


def run_rung(rung: dict) -> dict:
    sd, sd_path = _find_snapshot(rung["id"])
    if sd is None:
        return {"rung": rung["rung"], "error": "no snapshot found"}
    proj = _scaffold(rung, sd_path)
    conf = _conf(rung, proj)
    gen = subprocess.run(
        ["python", "src/main.py", "-c", str(conf), "pipeline", "run", "-f", "-msm",
         "-mt", str(proj / "source_data" / "mapping_table.json")],
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    maps = sorted((proj / "structure_maps").glob("*.json"))
    result = {"rung": rung["rung"], "profile": rung["id"], "generated": bool(maps)}
    if gen.returncode != 0 and not maps:
        result["gen_error"] = gen.stderr[-500:]
        return result
    def _targets_own_profile(path: Path) -> bool:
        try:
            obj = json.loads(path.read_text())
        except (OSError, ValueError):
            return False
        return any(
            s.get("mode") == "target" and s.get("url") == sd.get("url")
            for s in (obj.get("structure") or [])
        )

    own_maps = [m for m in maps if _targets_own_profile(m)] or maps
    sm = json.loads(own_maps[0].read_text()) if own_maps else None
    reg = None
    reg_files = sorted((proj / "processed_resources").iterdir())
    for rf in [f for f in reg_files if rung["id"] in f.name] or reg_files:
        obj = json.loads(rf.read_text())
        obj = json.loads(obj) if isinstance(obj, str) else obj
        if isinstance(obj, dict) and obj.get("data"):
            reg = obj
            break
    if sm:
        result["M5_todo"] = metrics.remaining_todo(sm)["todo_count"]
        result["M7_feature_matrix"] = metrics.feature_matrix(sd, sm)["matrix"]
    if reg:
        result["M8_extraction"] = metrics.extraction_completeness(sd, reg)["completeness_pct"]
        result["M9_silent_drops"] = metrics.graceful_degradation(sd, reg)["dropped_element_count"]
    if sm:
        result.update(_transform_and_validate(sd, rung["record"], proj))
    return result


def _transform_and_validate(sd: dict, record: dict, proj: Path) -> dict:
    """live conformance on Matchbox in BOTH modes (isolated + bundle)"""
    from controller.external_services.matchbox_controller import MatchboxController
    from controller.external_services.matchbox_transform_service import (
        MatchboxTransformService,
    )

    mb = MatchboxController({"url": MATCHBOX})
    try:
        mb.upload_structure_definition(sd)
        for d in ("input_profile", "source_definitions"):
            for f in (proj / d).glob("*.json"):
                obj = json.loads(f.read_text())
                if isinstance(obj, dict) and obj.get("resourceType") == "StructureDefinition":
                    mb.upload_structure_definition(obj)
        for f in (proj / "source_data" / "concept_maps").glob("*.json"):
            obj = json.loads(f.read_text())
            if isinstance(obj, dict) and obj.get("resourceType") == "ConceptMap":
                mb.upload_concept_map(obj)
        for f in sorted((proj / "structure_maps").glob("*.json")):
            obj = json.loads(f.read_text())
            if isinstance(obj, dict) and obj.get("resourceType") == "StructureMap":
                mb.upload_structure_map(obj)

        svc = MatchboxTransformService(mb, str(proj))

        def _validate(resource) -> tuple:
            oo = mb.validate_fhir_resources(resource, sd.get("url"))
            errs = [i for i in (oo.get("issue", []) if isinstance(oo, dict) else [])
                    if i.get("severity") in ("error", "fatal")]
            return (not errs), ("" if not errs else str(errs[0].get("diagnostics", ""))[:110])

        def _validate_all(payload) -> tuple:
            resources = (
                [e["resource"] for e in payload.get("entry", [])]
                if isinstance(payload, dict) and payload.get("resourceType") == "Bundle"
                else (payload if isinstance(payload, list) else [payload])
            )
            own = [r for r in resources if r.get("resourceType") == sd.get("type")]
            if not own:
                return False, f"no {sd.get('type')} resource in transform output"
            for r in own:
                ok, detail = _validate(r)
                if not ok:
                    return False, detail
            return True, ""

        single = svc.transform(dict(record), None, bundle=False)
        if not single:
            return {"P_transform": False, "P_detail": "no transform output"}
        bundle = svc.transform(dict(record), None, bundle=True)

        single_ok, single_detail = _validate_all(single)
        bundle_ok, bundle_detail = _validate_all(bundle)
        return {
            "P_transform": True,
            "P_single_validate": single_ok, "P_single_detail": single_detail,
            "P_bundle_validate": bundle_ok, "P_bundle_detail": bundle_detail,
        }
    except Exception as e:
        return {"P_transform": False, "P_detail": f"{type(e).__name__}: {e}"[:120]}


def main() -> int:
    _WORK.mkdir(parents=True, exist_ok=True)
    rows = [run_rung(r) for r in LADDER]
    (_REPO / "eval" / "reports").mkdir(parents=True, exist_ok=True)
    (_REPO / "eval" / "reports" / "E14_ladder.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{'rung':30} {'gen':4} {'M8%':6} {'M9drop':6} {'M5todo':6} "
          f"{'xform':6} {'val(single)':12} {'val(bundle)':12} detail")
    print("-" * 120)
    for r in rows:
        if r.get("error") or r.get("gen_error"):
            print(f"{r['rung']:30} ERROR {r.get('error') or r.get('gen_error','')[:60]}")
            continue
        detail = r.get("P_detail") or r.get("P_bundle_detail") or ""
        print(f"{r['rung']:30} {str(r['generated']):4} {str(r.get('M8_extraction','-')):6} "
              f"{str(r.get('M9_silent_drops','-')):6} {str(r.get('M5_todo','-')):6} "
              f"{str(r.get('P_transform','-')):6} {str(r.get('P_single_validate','-')):12} "
              f"{str(r.get('P_bundle_validate','-')):12} {detail}")
    print(f"\nReport: eval/reports/E14_ladder.json")
    print("(emission flags retained in JSON but are heuristic, the (P) transform/validate "
          "columns are the reliable where-it-holds/breaks signal.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
