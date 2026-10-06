#!/usr/bin/env python3
"""Q1-Q5 demonstration-query validation (Marfoglia et al. 2025, Table 3/7).

The paper validates the MOTU→FHIR conversion by running five domain-expert queries on both the
source CSVs (pandas) and the FHIR output (SPARQL/Fuseki), and checking the counts match Table 7:

    Q1 = 85           patients who return for a 2nd hospitalization        (Encounter)
    Q2 = 34           mechanical-knee → electronic-knee transition          (Patient/DeviceRequest/DeviceDefinition)
    Q3 = 67           patients on anxiolytics/antidepressants (ATC N05A/N06A) (Patient/Medication/MedicationStatement)
    Q4 = 674          hospital stays from patients over 65                  (Patient/Encounter)
    Q5 = AMK11/FK79/LK26/MPK28 (=144)  falls per prosthesis category       (AdverseEvent/DeviceRequest/DeviceDefinition)

We query FHIR via HAPI's REST API (the paper also ran FHIRPath on HAPI); the relational joins the
paper expressed in SPARQL are done client-side over the pulled resources. The SAME query logic runs
against three collections so the numbers are comparable:
  * the published ground-truth ndjson (`--source gt`)  → must reproduce Table 7 (validates the logic)
  * a HAPI server holding a pipeline's uploaded output  (`--source hapi --base URL`)

References are keyed as ``ResourceType/id``; after a HAPI transaction upload the bundle's
``urn:uuid:`` placeholders are resolved to that form, so the same joins work pre/post upload.

  python3 eval/e10_marfoglia/queries.py --source gt                       # ground truth → Table 7
  python3 eval/e10_marfoglia/queries.py --source hapi --base http://localhost:8089/fhir
"""
import argparse
import json
import sys
import urllib.request
from collections import defaultdict, Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
GT_DIR = HERE / "_data" / "output"

TARGETS = {"Q1": 85, "Q2": 34, "Q3": 67, "Q4": 674,
           "Q5": {"AMK": 11, "FK": 79, "LK": 26, "MPK": 28}}

ATC = "http://www.whocc.no/atc"


# ---------------------------------------------------------------- loaders
def load_gt(types):
    """Load resources from the published ground-truth ndjson export."""
    coll = defaultdict(list)
    for t in types:
        fp = GT_DIR / f"{t}.ndjson"
        if not fp.exists():
            continue
        for line in fp.open():
            line = line.strip()
            if line:
                coll[t].append(json.loads(line))
    return coll


def load_hapi(base, types):
    """Pull every resource of each type from a HAPI server via paged REST search."""
    coll = defaultdict(list)
    for t in types:
        url = f"{base.rstrip('/')}/{t}?_count=1000"
        while url:
            with urllib.request.urlopen(url, timeout=120) as r:
                bundle = json.load(r)
            for e in bundle.get("entry", []):
                if "resource" in e:
                    coll[t].append(e["resource"])
            url = next((l["url"] for l in bundle.get("link", []) if l["relation"] == "next"), None)
    return coll


# ---------------------------------------------------------------- helpers
def ref(obj, key):
    """Return the ``ResourceType/id`` a resource's reference element points at (or None)."""
    v = obj.get(key)
    if isinstance(v, dict):
        return v.get("reference")
    return None


def is_renewal(enc):
    """A renewal (re-admission) stay: Marfoglia stamps ``Encounter.hospitalization.reAdmission``
    with code ``R``; first-delivery stays carry no ``hospitalization`` element at all."""
    for c in (enc.get("hospitalization", {}) or {}).get("reAdmission", {}).get("coding", []):
        if c.get("code") == "R":
            return True
    return False


def knee_category(dd):
    """Extract the MOTU prosthesis category (AMK/FK/LK/MPK) from a DeviceDefinition."""
    for prop in dd.get("property", []):
        codes = [c.get("code") for c in prop.get("type", {}).get("coding", [])]
        if "Category" in codes:
            for vc in prop.get("valueCode", []):
                for c in vc.get("coding", []):
                    if c.get("code"):
                        return c["code"]
    return None


def age_on(birth, on):
    """Whole years from ISO birthDate to ISO date ``on`` (period.start of the stay)."""
    b, o = date.fromisoformat(birth[:10]), date.fromisoformat(on[:10])
    return o.year - b.year - ((o.month, o.day) < (b.month, b.day))


# ---------------------------------------------------------------- queries
def q1(c):
    """Patients who came for an initial supply AND returned for a renewal — the intersection of
    {patients with a first-delivery stay} and {patients with a renewal stay} (Marfoglia's pandas
    intersection / SPARQL "R"-encounter + non-hospitalization-encounter for the same subject)."""
    first, renew = set(), set()
    for e in c["Encounter"]:
        pat = ref(e, "subject")
        if not pat:
            continue
        (renew if is_renewal(e) else first).add(pat)
    return len(first & renew)


def q2(c):
    """Patients who switched mechanical→electronic: a mechanical knee (category FK/LK/AMK) on a
    first-delivery stay AND an MPK knee on a renewal stay, same patient (Marfoglia's Q2)."""
    dd_cat = {f"DeviceDefinition/{d['id']}": knee_category(d) for d in c["DeviceDefinition"]}
    enc_renewal = {f"Encounter/{e['id']}": is_renewal(e) for e in c["Encounter"]}
    mech_first, mpk_renew = set(), set()
    for dr in c["DeviceRequest"]:
        pat, cat, renew = ref(dr, "subject"), dd_cat.get(ref(dr, "codeReference")), \
            enc_renewal.get(ref(dr, "encounter"))
        if not pat or cat is None:
            continue
        if cat in ("FK", "LK", "AMK") and renew is False:
            mech_first.add(pat)
        if cat == "MPK" and renew is True:
            mpk_renew.add(pat)
    return len(mech_first & mpk_renew)


def q3(c):
    """Distinct patients with a MedicationStatement for an ATC N05A/N06A drug."""
    fall_atc = set()
    for m in c["Medication"]:
        for cd in m.get("code", {}).get("coding", []):
            if cd.get("system") == ATC and str(cd.get("code", "")).startswith(("N05A", "N06A")):
                fall_atc.add(f"Medication/{m['id']}")
    pats = {ref(ms, "subject") for ms in c["MedicationStatement"]
            if ref(ms, "medicationReference") in fall_atc and ref(ms, "subject")}
    return len(pats)


def q4(c):
    """Hospital stays (Encounters) whose patient was over 65 at the stay start."""
    bd = {f"Patient/{p['id']}": p.get("birthDate") for p in c["Patient"]}
    n = 0
    for e in c["Encounter"]:
        start = (e.get("period") or {}).get("start")
        b = bd.get(ref(e, "subject"))
        if start and b and age_on(b, start) >= 65:
            n += 1
    return n


def q5(c):
    """Falls (AdverseEvents) grouped by the prosthesis category of that stay's DeviceRequest."""
    dd_cat = {f"DeviceDefinition/{d['id']}": knee_category(d) for d in c["DeviceDefinition"]}
    enc_cat = {}
    for dr in c["DeviceRequest"]:
        enc, cat = ref(dr, "encounter"), dd_cat.get(ref(dr, "codeReference"))
        if enc and cat:
            enc_cat[enc] = cat
    out = Counter()
    for ae in c["AdverseEvent"]:
        cat = enc_cat.get(ref(ae, "encounter"))
        if cat:
            out[cat] += 1
    return dict(out)


QUERIES = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}
NEEDED = ["Encounter", "DeviceRequest", "DeviceDefinition", "Patient",
          "Medication", "MedicationStatement", "AdverseEvent"]


def run(coll, label):
    print(f"\n=== Q1-Q5 on {label} ===")
    print(f"{'query':4s} {'result':>28s} {'target':>28s}  ok")
    allok = True
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        res, tgt = QUERIES[q](coll), TARGETS[q]
        if q == "Q5":
            res = {k: res.get(k, 0) for k in tgt}
        ok = res == tgt
        allok &= ok
        print(f"{q:4s} {str(res):>28s} {str(tgt):>28s}  {'✓' if ok else '✗'}")
    print(f"--> {'ALL MATCH' if allok else 'MISMATCH'}")
    return allok


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["gt", "hapi"], default="gt")
    ap.add_argument("--base", default="http://localhost:8089/fhir")
    ap.add_argument("--label", default=None)
    a = ap.parse_args(argv)
    if a.source == "gt":
        coll = load_gt(NEEDED)
        label = a.label or "ground-truth ndjson"
    else:
        coll = load_hapi(a.base, NEEDED)
        label = a.label or f"HAPI {a.base}"
    counts = {t: len(coll.get(t, [])) for t in NEEDED}
    print(f"loaded: {counts}")
    ok = run(coll, label)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
