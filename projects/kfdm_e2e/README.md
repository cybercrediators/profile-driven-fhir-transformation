# kfdm_e2e — FSH → FHIR e2e project (release smoke test)

Built 2026-07-02 from **FSH sources** (`data/fsh-profile`, the kfdm/WAVES IG) — unlike
`testing_kfdm_profile`, which started from pre-compiled JSON profiles. This project is the
subject of the scripted release smoke test (`scripts/e2e_kfdm_smoke.sh`) and the writeup's
pipeline (P) comparison.

## How it was built (reproduce with `scripts/e2e_kfdm_smoke.sh --full`)

1. `process-fsh data/fsh-profile --name kfdm_e2e` — sushi via docker image
   `sushi-docker-sushi:latest` (runs as the host user; a root-owned `fsh-generated/` from
   older runs breaks the IG publisher with EACCES).
2. **Snapshots**: `_genonce.sh` (IG publisher; `PATH=/tmp/shim` for the sushi shim; the
   terminal Jekyll failure is expected) → copy `data/fsh-profile/temp/pages/StructureDefinition-*.json`
   over `input_profile/`. Plain `sushi build` emits differential-only SDs, which degrade the
   generated maps to 6-rule stubs — snapshots are REQUIRED.
3. `source_data/`: REDCap codebook CSV (WAVES data dictionary, drives the REDCap plugin in
   csv mode), `mapping_table.json` (54 entries; prefix `Sourcedefinition_kfdm_e2e.`),
   `redcap_test.json` (3 records), `fhir_code_mapping.json`, `concept_maps/` (authored
   REDCap-code arrows; NB: ConceptMap `url`s embed the source-def name — rewritten from the
   testing_kfdm_profile copies).
4. Generate + upload: `run_source_def()` + `run_static_gen_sm()` → 8 StructureMaps
   (7 profiles + Questionnaire→QR) → `prepare_matchbox_setup(force_upload=True)`.

## Result

Each REDCap record → **transaction Bundle with 8 resources**: Patient (WAVESPatient) +
QuestionnaireResponse (Fragebogen-KFDM-WAVES) + 5 Condition (anamnese-*) + Observation
(anamnese-raucher), references wired via urn:uuid. The NiFi path
(`scripts/build_kfdm_nifi_flow.py`, process group `kfdm-e2e`) produces output identical to
the direct pipeline (modulo bundle uuids / `authored` timestamp).

## Pre-processing contract (upstream of the transformer)

* **One source record per FlowFile.** A REDCap API export → ConvertRecord (CSV→JSON) emits
  ONE FlowFile holding a JSON array of ALL records — the transformer rejects arrays
  (`'list' object has no attribute 'get'`). Insert SplitJson (`$[*]`) before the
  transformer, plus an UpdateAttribute setting a per-split `filename`
  (e.g. `record-${fragment.index}.json`) or PutFile's conflict strategy keeps only one file.
* REDCap exports unanswered fields as `""` — since 2026-07-02 the NiFi connector strips
  `""`/`null` fields itself on ingest (they mean "absent"; empty codes crash translate
  rules), so no separate normalize step is required. ConvertRecord's inferred types
  (numbers instead of strings) are tolerated by the transform.

## Still manual / user-side

* Replace the GenerateFlowFile stand-in with the REDCap API call (InvokeHTTP) + the
  normalization step in the NiFi flow.
* Comparison against the original hand-built poster implementation.

## Gotchas

* **Unix-socket bind mounts go stale**: the NiFi container mounts the socket FILE; when the
  bridge server restarts (recreates the socket), the container keeps the dead inode —
  restart `fsh-nifi-test` after (re)starting the bridge (the smoke script checks inodes).
* NiFi (this build) keys processor properties by DISPLAY name (`Custom Text`, not
  `generate-ff-custom-text`).
* The bridge's read-only `validate_setup` no longer fails on a missing IG package — the
  per-resource checks are the real prerequisite (transform-only matchbox can't install IGs).
