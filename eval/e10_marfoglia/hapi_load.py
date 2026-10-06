#!/usr/bin/env python3
"""Load resources into a HAPI FHIR server for the Q1-Q5 validation.

Two entry points:
  * ``put_ndjson`` — bulk-load the published ground-truth ndjson (references are already resolved
    to ``ResourceType/id``), preserving ids via PUT batch bundles. Used to anchor the query harness
    against a real server (`queries.py --source hapi` must then reproduce Table 7).
  * ``post_transactions`` — POST pipeline-produced transaction Bundles (``urn:uuid:`` placeholders +
    ``ifNoneExist``); HAPI resolves the UUIDs to real ids and dedups on the business keys.

  python3 eval/e10_marfoglia/hapi_load.py --base http://localhost:8089/fhir --gt   # load ground truth
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
GT_DIR = HERE / "_data" / "output"
# ground-truth types we query over (skip the huge QR/Observation files — no query needs them)
GT_TYPES = ["Patient", "Encounter", "DeviceRequest", "DeviceDefinition",
            "Medication", "MedicationStatement", "AdverseEvent"]


def _post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/fhir+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _prefix_ids(obj):
    """HAPI refuses purely numeric client-assigned ids, so prefix every id and every
    ``Type/<id>`` reference with ``g`` (consistent remap → the reference graph is preserved)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("id"), str) and obj["id"].isdigit():
            obj["id"] = "g" + obj["id"]
        r = obj.get("reference")
        if isinstance(r, str) and "/" in r:
            t, _, i = r.partition("/")
            if i.isdigit():
                obj["reference"] = f"{t}/g{i}"
        for v in obj.values():
            _prefix_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            _prefix_ids(v)
    return obj


def put_ndjson(base, types=GT_TYPES, chunk=400):
    """Load ndjson resources into HAPI via PUT batch bundles (client-assigned ids preserved)."""
    base = base.rstrip("/")
    total = 0
    for t in types:
        fp = GT_DIR / f"{t}.ndjson"
        if not fp.exists():
            continue
        res = [_prefix_ids(json.loads(l)) for l in fp.open() if l.strip()]
        for i in range(0, len(res), chunk):
            batch = {
                "resourceType": "Bundle", "type": "batch",
                "entry": [{"resource": r, "request": {"method": "PUT",
                                                      "url": f"{t}/{r['id']}"}} for r in res[i:i + chunk]],
            }
            _post(base, batch)
        total += len(res)
        print(f"  loaded {len(res):6d} {t}")
    print(f"total: {total} resources")
    return total


SHARED_DIR = HERE / "_data" / "shared"


def preload_shared(base):
    """POST the shared resources (Organizations, Questionnaires, ValueSet, CodeSystem) once, so the
    per-stay bundles' conditional references (``Organization?name=INAIL`` etc.) resolve to exactly
    one target. Marfoglia pre-uploads these to the server before running the pipeline."""
    base = base.rstrip("/")
    n = 0
    for fp in sorted(SHARED_DIR.glob("*.json")):
        r = json.loads(fp.read_text())
        rt = r.get("resourceType")
        r.pop("id", None)  # let HAPI assign; conditional refs match by name/url, not id
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{base}/{rt}", data=json.dumps(r).encode(), method="POST",
                headers={"Content-Type": "application/fhir+json"}), timeout=120)
            n += 1
        except Exception as e:
            print(f"  ! shared {fp.name}: {str(e)[:80]}")
    print(f"  preloaded {n} shared resources")


def _strip_self_refs(bundle):
    """Remove any reference element pointing at its own entry's fullUrl. Equivalent to the
    BundleService self-reference guard (fixed in-engine) — kept here so bundles dumped before the
    fix still load. A self-reference makes HAPI recurse forever resolving the transaction (500
    StackOverflowError)."""
    def scrub(node, own):
        if isinstance(node, dict):
            for k in list(node.keys()):
                v = node[k]
                if isinstance(v, dict) and v.get("reference") == own and set(v) <= {"reference", "type", "display"}:
                    del node[k]
                elif isinstance(v, list):
                    node[k] = [x for x in v if not (isinstance(x, dict) and x.get("reference") == own)]
                    for x in node[k]:
                        scrub(x, own)
                elif isinstance(v, dict):
                    scrub(v, own)
    for e in bundle.get("entry", []):
        own = e.get("fullUrl")
        if own:
            scrub(e.get("resource", {}), own)
    return bundle


def _fix_choice_refs(bundle):
    """Rename any choice element ``foo[x]`` that holds a reference to its typed form ``fooReference``
    (HAPI rejects the literal ``[x]`` placeholder — HAPI-1809). Fixed in-engine
    (``BundleService._collect_todo_refs`` now expands ``[x]`` → ``Reference`` like
    ``_iter_reference_fields``) — kept here so bundles dumped before the fix still load."""
    def scrub(node):
        if isinstance(node, dict):
            for k in list(node.keys()):
                v = node[k]
                if k.endswith("[x]") and isinstance(v, dict) and "reference" in v:
                    node[k[:-3] + "Reference"] = node.pop(k)
                else:
                    scrub(v)
        elif isinstance(node, list):
            for v in node:
                scrub(v)
    for e in bundle.get("entry", []):
        scrub(e.get("resource", {}))
    return bundle


def _strip_meta_profile(bundle):
    """Drop meta.profile before loading. HAPI stores it as a tag; concurrent transactions race to
    create the same tag row → HAPI-2023 'did not return a unique result'. The count queries don't
    need the profile, and the on-disk bundles retain meta.profile (the conformance claim stands)."""
    for e in bundle.get("entry", []):
        meta = e.get("resource", {}).get("meta")
        if isinstance(meta, dict):
            meta.pop("profile", None)
            if not meta:
                e["resource"].pop("meta", None)
    return bundle


def _fix_ine(bundle):
    """FHIR search encodes a space as ``%20``; the maps emit ``+`` (form-encoding), which HAPI's
    conditional-URL parser reads as a literal ``+`` → the create's ifNoneExist never matches its own
    resource (HAPI-0539/412). Any real ``+`` would be ``%2B``, so ``+``→``%20`` is a safe fix."""
    for e in bundle.get("entry", []):
        ine = e.get("request", {}).get("ifNoneExist")
        if isinstance(ine, str) and ("+" in ine or " " in ine):
            e["request"]["ifNoneExist"] = ine.replace("+", "%20").replace(" ", "%20")
    return bundle


def post_transactions(base, bundles, label="", timeout=120, workers=1):
    """POST each transaction Bundle to HAPI; returns (ok, failed).

    ``workers=1`` (default) is sequential — required where concurrent ifNoneExist conditional-creates
    would race and duplicate a shared resource that MATTERS for a count (Patient, the categorised
    DeviceDefinition). Once those masters are loaded, later tables can POST concurrently: their
    encounters have distinct business keys, their Patient/DeviceDefinition ifNoneExist only *match*
    (idempotent), and duplicate Medications don't change Q3's distinct-patient count."""
    base = base.rstrip("/")
    errs = {}
    ok = [0]
    t0 = time.monotonic()

    def one(b):
        try:
            _post(base, b, timeout=timeout)
            ok[0] += 1
        except Exception as e:
            k = str(e)[:120]
            errs[k] = errs.get(k, 0) + 1

    if workers <= 1:
        for b in bundles:
            one(b)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, bundles))
    failed = len(bundles) - ok[0]
    dt = time.monotonic() - t0
    rate = ok[0] / dt if dt > 0 else 0.0
    print(f"  {label:14s} posted {ok[0]} ok, {failed} failed  "
          f"[{dt:.0f}s, {rate:.2f} bundles/s]", flush=True)
    for k, c in sorted(errs.items(), key=lambda x: -x[1])[:3]:
        print(f"      [{c}x] {k}")
    return ok[0], failed


# Marfoglia's per-record transaction bundles. Load MASTER tables (full demographics / full device
# properties) BEFORE the transactional tables so their fuller resource wins the ifNoneExist dedup
# (the hospital_stay/fall bundles re-create Patient/Encounter/DeviceDefinition as identifier-only
# stubs that must match, not overwrite). hospital_stay before fall so the reAdmission-bearing
# Encounter is the one created.
DIRECTION_ORDER = ["patient", "knee", "hospital_stay", "drug", "fall"]
# masters loaded sequentially (their dedup establishes the canonical Patient / categorised
# DeviceDefinition); the rest may POST concurrently (see post_transactions docstring). fall AFTER
# hospital_stay so its re-created Encounter matches, never creates a duplicate.
SEQUENTIAL_TABLES = {"patient", "knee"}


def load_direction(base, tdir, order=DIRECTION_ORDER,
                   workers=int(os.environ.get("LOAD_WORKERS", "4"))):
    """POST all of a direction's per-table bundle ndjson files, in dependency order."""
    tdir = Path(tdir)
    preload_shared(base)
    grand = 0
    for table in order:
        fp = tdir / f"{table}.bundles.ndjson"
        if not fp.exists():
            print(f"  {table}: no {fp.name} — skip")
            continue
        bundles = [_strip_meta_profile(_fix_choice_refs(_strip_self_refs(_fix_ine(json.loads(l)))))
                   for l in fp.open() if l.strip()]
        w = 1 if table in SEQUENTIAL_TABLES else workers
        post_transactions(base, bundles, label=table, workers=w)
        grand += len(bundles)
    print(f"total bundles posted: {grand}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8089/fhir")
    ap.add_argument("--gt", action="store_true", help="load ground-truth ndjson")
    ap.add_argument("--dir", help="load per-table transaction bundles from this directory")
    a = ap.parse_args(argv)
    if a.gt:
        put_ndjson(a.base)
    if a.dir:
        load_direction(a.base, a.dir)


if __name__ == "__main__":
    main(sys.argv[1:])
