#!/usr/bin/env python3
"""Direction-2 driver, run the generated, profile-driven maps on Marfoglia real records

Transforms `_data/records/<table>.records.json` through the generated StructureMaps

Usage:
  PYTHONPATH=.:src python3 eval/e10_marfoglia/transform_d2.py hospital_stay
  # regenerate maps first (after profile/mapping/composition changes)
  PYTHONPATH=.:src python3 eval/e10_marfoglia/transform_d2.py --regen patient
  # Patient table: exact behavioral parity vs ground truth (identifier/gender/birthDate)
  PYTHONPATH=.:src python3 eval/e10_marfoglia/transform_d2.py --parity patient
  # batch: per-stay resource rates vs ground truth + reference/ifNoneExist coverage
  PYTHONPATH=.:src python3 eval/e10_marfoglia/transform_d2.py --batch 30 hospital_stay

Matchbox on Port 8080. After --regen, `rm -f projects/marfoglia_full/structure_maps/*.json` first so the
generator (which does not clean stale maps) doesn't leave duplicate per-profile files.
"""
import json
import sys
import glob
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(REPO), str(REPO / "src")]

from controller.bundle_service import BundleService
from controller.pipeline_controller.pipeline_controller import PipelineController
from helpers import utils
import logging

logging.disable(logging.WARNING)

MB = __import__("os").environ.get("MATCHBOX_URL", "http://localhost:8080/matchboxv3/")
RECORDS = REPO / "eval/e10_marfoglia/_data/records"
GT = REPO / "eval/e10_marfoglia/_data/output"
PROJECT = REPO / "projects/marfoglia_full"

# Marfoglia stored-output totals over 1962 stays (post-HAPI ground truth).
GT_TOTAL = {"QuestionnaireResponse": 23568, "Observation": 11721, "Encounter": 1964,
            "Account": 1964, "Coverage": 1964, "CarePlan": 1964, "Goal": 1964,
            "Procedure": 1964, "Condition": 182, "DeviceRequest": 1837,
            "DeviceDefinition": 41, "Patient": 1008}
STAYS = 1962


def normalize(rec):
    """Pre-processing: flatten single-instance sub-models to top-level (so they default to
    whole-record + gating); leave repeated timepoints as sub-objects for the composition
    descriptor. Data shaping only — out of scope for the tool, would be an upstream NiFi step."""
    r = dict(rec)
    rename = {
        "hospitalization": {},
        "amputation": {"cause": "amputationCause", "date": "amputationDate", "side": "amputationSide"},
        "pain": {},
        "rehab": {"goal": "rehabGoal"},
    }
    for sub, rn in rename.items():
        obj = r.pop(sub, None)
        if isinstance(obj, dict):
            for k, v in obj.items():
                r[rn.get(k, k)] = v
    if r.get("anonymousId") is not None and r.get("admissionDate"):
        r["motuEncounterId"] = f"@{r['anonymousId']}${r['admissionDate']}"
    if r.get("prostheticKnee"):
        r["deviceIdentifier"] = r["prostheticKnee"]
    if r.get("firstDeliveryRenewal") == "Renewal":
        r["readmission"] = "R"
    return r


SCT = "http://snomed.info/sct"
AUTHORED_CMS = {
    "gender": {
        "source": "http://ahdbservices.it/CodeSystem/motu-sex",
        "target": "http://hl7.org/fhir/administrative-gender",
        "arrows": {"M": "male", "F": "female"},
    },
    "bodySite": {
        "source": "http://ahdbservices.it/CodeSystem/motu-amputation-side",
        "target": SCT,
        "arrows": {"L": "7771000", "R": "24028007"},
    },
    "reasonCode": {
        "source": "http://ahdbservices.it/CodeSystem/motu-amputation-cause",
        "target": SCT,
        "arrows": {"traumatic": "417746004", "vascular": "57662003",
                   "cancer": "363346000", "infectious": "40733004", "congenital": "66091009"},
    },
    "outcomeCode": {
        "source": "http://ahdbservices.it/CodeSystem/motu-rehab-goal",
        "target": SCT,
        "arrows": {"free_walk": "165252001", "aid1": "443392007",
                   "aid2": "443663000", "walker": "895488007"},
    },
    "event": {
        "groups": [
            {"source": "http://ahdbservices.it/CodeSystem/motu-near-fall", "target": SCT,
             "arrows": {"0": "1912002"}},
            {"source": "http://ahdbservices.it/CodeSystem/motu-near-fall",
             "target": "http://hl7.org/fhir/sid/icd-10", "arrows": {"1": "W18.40"}},
        ],
    },
}


def _cm_groups(spec):
    """Normalise an AUTHORED_CMS entry to a list of ConceptMap groups (single- or multi-group)."""
    groups = spec["groups"] if "groups" in spec else [spec]
    return [{
        "source": g["source"], "target": g["target"],
        "element": [{"code": s, "target": [{"code": t, "equivalence": "equivalent"}]}
                    for s, t in g["arrows"].items()],
    } for g in groups]


def reapply_concept_maps():
    """Author the real source→target arrows into the generated (identity-scaffold) ConceptMaps."""
    for field, spec in AUTHORED_CMS.items():
        for fp in glob.glob(str(PROJECT / f"source_data/concept_maps/cm-{field}-*.json")):
            d = json.load(open(fp))
            d["group"] = _cm_groups(spec)
            json.dump(d, open(fp, "w"), indent=2)


def seed_authored_concept_maps():
    """declare the fields whose source does not speak the target's code list"""
    out_dir = PROJECT / "source_data/concept_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    for field, spec in AUTHORED_CMS.items():
        if glob.glob(str(out_dir / f"cm-{field}-*.json")):
            continue
        (out_dir / f"cm-{field}-authored.json").write_text(json.dumps({
            "resourceType": "ConceptMap", "id": f"cm-{field}-authored",
            "url": f"http://ahdbservices.it/ConceptMap/cm-{field}-authored",
            "status": "draft", "group": _cm_groups(spec),
        }, indent=2))


def build_pc(regen):
    conf = json.loads((REPO / "conf/marfoglia_full.json").read_text())
    conf["matchbox_connection"] = dict(conf["matchbox_connection"])
    conf["matchbox_connection"]["url"] = MB
    pc = PipelineController(conf=conf, force_overwrite=regen, reprocess_profile=regen,
                            reprocess_structure_map=regen, reprocess_helper_definition=regen,
                            create_references=True, minimal_mode=True)
    if regen:
        print("[regen] regenerating source-def + maps ...")
        pc.run_source_def()
        seed_authored_concept_maps()
        pc.run_static_gen_sm()
    else:
        pc._ensure_processed()
    reapply_concept_maps()
    pc.prepare_matchbox_setup(force_upload=True)
    return pc


def load_maps(pc):
    sm = [utils.get_json(f) for f in pc.app_state.dataIO.get_structure_map_files() or []]
    return [x for x in sm if isinstance(x, dict) and x.get("resourceType") == "StructureMap"]


def bundle_for(pc, sm, rec):
    res = pc.transform_data(normalize(rec))
    return BundleService.create_bundle(res or [], structure_maps=sm, registry=pc.app_state.registry)


def patient_parity(pc, sm, n=30):
    idx = {}
    for line in (GT / "Patient.ndjson").open():
        line = line.strip()
        if line:
            p = json.loads(line)
            v = (p.get("identifier") or [{}])[0].get("value")
            if v:
                idx[str(v)] = {k: p[k] for k in ("identifier", "gender", "birthDate") if k in p}
    recs = json.loads((RECORDS / "patient.records.json").read_text())[:n]
    ok = 0
    diffs = []
    for rec in recs:
        b = bundle_for(pc, sm, rec)
        ours = next((e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Patient"), None)
        if not ours:
            continue
        a = {k: ours[k] for k in ("identifier", "gender", "birthDate") if k in ours}
        g = idx.get(str(rec["anonymousId"]))
        if g is None:
            continue
        if a == g:
            ok += 1
        else:
            diffs.append((rec["anonymousId"], a, g))
    print(f"Patient parity: {ok}/{len(recs)} exact (diffs: {len(diffs)})")
    for aid, a, g in diffs[:5]:
        print(f"  id={aid}\n    OURS: {a}\n    GT:   {g}")


def batch(pc, sm, table, n):
    recs = json.loads((RECORDS / f"{table}.records.json").read_text())[:n]
    agg = Counter(); ine = Counter(); refs_ok = refs_tot = 0
    for rec in recs:
        b = bundle_for(pc, sm, rec)
        for e in b["entry"]:
            r = e["resource"]; agg[r["resourceType"]] += 1
            if e.get("request", {}).get("ifNoneExist"):
                ine[r["resourceType"]] += 1
            for v in r.values():
                if isinstance(v, dict) and "reference" in v:
                    refs_tot += 1
                    refs_ok += str(v["reference"]).startswith("urn:uuid:")
    print(f"\n=== {n} {table} records → per-stay rate vs ground truth ===")
    print(f"{'type':24s} {'ours/stay':>10s} {'GT/stay':>9s}")
    for t in sorted(set(agg) | set(GT_TOTAL)):
        print(f"{t:24s} {agg[t]/n:10.2f} {GT_TOTAL.get(t,0)/STAYS:9.2f}")
    print(f"\nreferences wired (urn:uuid): {refs_ok}/{refs_tot} | ifNoneExist on: {dict(ine)}")


ALL_TABLES = ["patient", "knee", "drug", "fall", "hospital_stay"]


def dump_all(pc, sm, outdir, workers=int(__import__("os").environ.get("DUMP_WORKERS", "32"))):
    """Transform every record of all 5 tables to transaction Bundles, one ndjson per table (output layout so hapi_load.py --dir can POST them for the Q1-Q5 validation)"""
    from pathlib import Path as _P
    from concurrent.futures import ThreadPoolExecutor
    out = _P(outdir); out.mkdir(parents=True, exist_ok=True)
    bundle_for(pc, sm, json.loads((RECORDS / "patient.records.json").read_text())[0])
    for table in ALL_TABLES:
        recs = json.loads((RECORDS / f"{table}.records.json").read_text())
        fp = out / f"{table}.bundles.ndjson"
        ok = empty = 0
        with ThreadPoolExecutor(max_workers=workers) as ex, fp.open("w") as f:
            for b in ex.map(lambda r: bundle_for(pc, sm, r), recs):
                if b and b.get("entry"):
                    f.write(json.dumps(b) + "\n"); ok += 1
                else:
                    empty += 1
        print(f"  {table:14s} {ok} bundles ({empty} empty) -> {fp.name}", flush=True)


def main(argv):
    regen = "--regen" in argv
    do_parity = "--parity" in argv
    nbatch = None
    if "--batch" in argv:
        nbatch = int(argv[argv.index("--batch") + 1])
    dumpdir = None
    if "--dump" in argv:
        dumpdir = argv[argv.index("--dump") + 1]
    table = next((a for a in argv[1:] if not a.startswith("--")
                  and a not in (str(nbatch), dumpdir)), "hospital_stay")

    pc = build_pc(regen)
    sm = load_maps(pc)

    if dumpdir:
        dump_all(pc, sm, dumpdir)
    elif do_parity:
        patient_parity(pc, sm)
    elif nbatch:
        batch(pc, sm, table, nbatch)
    else:
        rec = json.loads((RECORDS / f"{table}.records.json").read_text())[0]
        b = bundle_for(pc, sm, rec)
        types = Counter(e["resource"]["resourceType"] for e in b["entry"])
        print(f"\n{table}[0] → {sum(types.values())} resources: {dict(sorted(types.items()))}")


if __name__ == "__main__":
    main(sys.argv)
