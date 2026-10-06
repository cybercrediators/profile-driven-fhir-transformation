# Evaluation documentation
Documentation of existing evaluation, current status, and corresponding results. Further, instructions for reproducing the evaluations are provided here.

# (Minimal) Sample Projects
- `example1`: contains a simple example of a project with a single resource (`Patient`) and the corresponding mapping table
- `example2`: contains multiple profiles on resources, a `CodeSystem` and a `ValueSet` and the corresponding mapping table

## Results

| metric | example_project1 | example_project2 |
|---|---|---|
| M0c reproducibility | 1/1 identical | 3/3 identical |
| M1 data preservation | 100.0% (4/4 graded of 4; 0 transform-mapped, 0 losses) | 100.0% (7/7 graded of 10; 3 transform-mapped, 0 losses) |
| M2 behavioral equivalence | self-consistent (offline) | self-consistent (offline) |
| M3 rule P/R/F1 | 1 map, TP 8, spurious 0, missing 0, F1 1.0 | 3 maps, TP 25, spurious 0, missing 0, F1 1.0 / 1.0 / 1.0 |
| M5 remaining TODO | 0 over 1 map | 1 over 3 maps |
| M6 edit distance | 0.0 | 0.0 / 0.0 / 0.0 |
| M7 feature matrix | 1 profile, 4 constructs present | 4 profiles, 11 constructs present |
| M8 extraction completeness | 100.0% (24 elements) | 100.0% x4 (40 elements) |
| M9 graceful degradation | 0 silent drops | 0 silent drops |

- (TODO is `Transform-example2-patient-profile/map-extension-observation: source.element=TODO-MAP-OBSERVATION_SOURCE`)
- correspoing metrics can be found [here](./eval_overview.md)

## Reproduce

- `example1`: PYTHONPATH=.:src python3 -m eval.run_eval example_project1 --regenerate
- `example2`: PYTHONPATH=.:src python3 -m eval.run_eval example_project2 --regenerate

Results can be found in `eval/reports/<project_name>.json`.

# Synthetic profiles (increasing feature complexity)

- 6 synthetic FHIR Shorthand (FSH) profiles with increasing complexity of FHIR constructs
- corresponding project (temporary) including required files are generated automatically and validated (isolated and bundle mode)

## Results

| profile | generated | M8 extraction | M9 silent drops | M5 TODO | transform | val (single) | val (bundle) |
|---|---|---|---|---|---|---|---|
| L1 scalar | ✓ | 100% | 0 | 1 | ✓ | ✓ | ✓ |
| L2 cardinality + complex type | ✓ | 100% | 0 | 1 | ✓ | ✓ | ✓ |
| L3 binding + fixed | ✓ | 100% | 0 | 1 | ✓ | ✓ | ✓ |
| L4 slicing | ✓ | 100% | 0 | 1 | ✓ | ✓ | ✓ |
| L5 choice[x] + reference | ✓ | 100% | 0 | 1 | ✓ | ✗ | ✓ |
| L6 extension + nested backbone | ✓ | 100% | 0 | 3 | ✓ | ✓ | ✓ |

## Reproduce
- use the `eval/synthetic_eval.py` script to generate and evaluate the synthetic profiles
- the script automatically will use the generator for the creation of the structuremaps and provide the corresponding results
- Important: a live matchbox instance is needed (set `MATCHBOX=<url>`)

# Real-world profile examples (simplifier)
- [ACME Base Profiles](https://simplifier.net/packages/acme.base.r4)
- [Capable Testing profiles](https://simplifier.net/packages/Capable.repository)
- [eTerminservice R4](https://simplifier.net/eterminservice-r4)
- [eVO13 (Heilmittelverordnung Muster 13)](https://simplifier.net/evo13)
- [Gematik HDDT](https://simplifier.net/packages/de.gematik.hddt)
- [Heavy Menstrual Bleeding (HMB)](https://simplifier.net/menstrual-bleeding)
- [UV IPS](https://simplifier.net/packages/hl7.fhir.uv.ips)
- [KDS Bildgebung](https://simplifier.net/medizininformatik-initiative-modul-bildgebung)
- [KDS Biobank](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.biobank)
- [KDS Consent](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.consent)
- [KDS Diagnose](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.diagnose)
- [KDS Fall](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.fall)
- [KDS ICU](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.icu)
- [KDS Laborbefund](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.laborbefund)
- [KDS Medikation](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.medikation)
- [KDS Molgen](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.molgen)
- [KDS Onkologie](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.onkologie)
- [KDS Person](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.person)
- [KDS Prozedur](https://simplifier.net/packages/de.medizininformatikinitiative.kerndatensatz.prozedur)
- [MedMij R4 Dental Care](https://simplifier.net/medmij-r4-dental-care)
- [NDHM India](https://simplifier.net/ndhm.in)
- [Nictiz CIO](https://simplifier.net/nictiz-r4-cio)
- [Ontario PS](https://simplifier.net/packages/ca.on.oh.patient-summary)
- [USCore](https://simplifier.net/packages/hl7.fhir.us.core)

## Results

21 `*_full_spec` projects, measured against a Matchbox instance.

| project | resources | transformed | val (single) | bundle clean | TODOs | M1 preservation |
|---|---|---|---|---|---|---|
| acme | 5 | 5/5 | 5/5 | 5/5 | 24 | 100.0% |
| capable | 9 | 9/9 | 7/9 | 7/9 | 6 | 100.0% |
| eterminservice | 3 | 3/3 | 2/3 | 2/3 | 5 | 100.0% |
| evo13 | 5 | 5/5 | 3/5 | 3/5 | 8 | 89.5% |
| hddt | 8 | 8/8 | 7/8 | 8/8 | 7 | 96.6% |
| hmb | 4 | 4/4 | 0/4 | 0/4 | 4 | 100.0% |
| kds_bildgebung | 11 | 11/11 | 11/11 | 10/11 | 47 | 100.0% |
| kds_biobank | 11 | 11/11 | 11/11 | 11/11 | 53 | 100.0% |
| kds_diagnose | 1 | 1/1 | 1/1 | 1/1 | 3 | 100.0% |
| kds_icu | 3 | 3/3 | 2/3 | 2/3 | 0 | 100.0% |
| kds_laborbefund | 3 | 3/3 | 3/3 | 3/3 | 3 | 100.0% |
| kds_medikation | 5 | 5/5 | 5/5 | 5/5 | 68 | 94.1% |
| kds_molgen | 16 | 16/16 | 16/16 | 15/16 | 68 | 100.0% |
| kds_onkologie | 3 | 3/3 | 3/3 | 3/3 | 15 | 90.0% |
| kds_person | 4 | 4/4 | 4/4 | 4/4 | 11 | 100.0% |
| kds_prozedur | 1 | 1/1 | 1/1 | 1/1 | 4 | 100.0% |
| medmij_dental | 6 | 6/6 | 6/6 | 6/6 | 5 | 100.0% |
| ndhm | 10 | 10/10 | 10/10 | 10/10 | 81 | 100.0% |
| nictiz_cio | 8 | 8/8 | 4/8 | 8/8 | 21 | 100.0% |
| ontario_ps | 14 | 14/14 | 12/14 | 11/14 | 47 | 98.2% |
| uscore | 2 | 2/2 | 2/2 | 2/2 | 22 | 100.0% |

- 132 resources, 0 transform errors
- Intended-profile `$validate`, single mode: 115/132. Bundle mode: 117/132 entries
- 502 remain placeholder TODO rules
- Extraction: 100% on every graded profile, 0 dropped elements. 13 profiles excluded as never processed (extensions no profile in a scoped project references).
- Data preservation: 6 losses across this set caused by the idenfiied reasons:

| project | field | source value |
|---|---|---|
| evo13 | `a9a791ad-a16e-4846-b54a-e1759c2692db.code.coding.system` | ['https://fhir.element44.de/E44_EVO13_CS_Heilmittel'] |
| evo13 | `e514744a-51b7-492f-9227-67376bf637fd.code.coding.system` | ['https://fhir.element44.de/E44_EVO13_CS_Heilmittel'] |
| hddt | `hddt-lung-reference-value.effective[x]` | ['2025-05-01'] |
| kds_medikation | `MedicationRequest.substitution.allowed[x]:allowedBoolean` | ['True'] |
| kds_onkologie | `mii-pr-onko-diagnose-primaertumor.verificationStatus` | ['confirmed'] |
| ontario_ps | `ca-on-ps-profile-medicationrequest.substitution.allowed[x]:allowed` | ['True'] |

- tow for the evo13 are the profile-pinned `code.coding.system` overriding the authored mapping
- kds_medikation and ontario_ps hold the string `'True'` where FHIR writes `true`
- kds_onkologie `verificationStatus` and hddt `effective[x]` emit nothing

## Reproduce
### Prerequisities
- all projects are built directly from the published simplifier package
- for profiles without provided `snapshots`, the HL7 validator is required to compile from differentials
- further, a live matchbox (and potentially `valkey` depending on the configured cache service) is required
- projects already contain used maps and source records already. The projects themselves have to be obtained individually!
- source structuremaps can be re-generated/derived by the profiles
- configs are provided, paths have to be adjusted accordingly

### Reproduce
- obtain the package (e.g. from `simplifier`) and init the project: `python src/main.py -c conf/<conf_name>.json init <package_folder_path>/` (DO NOT rename the project folder when using existing projects, since the source definitions and existing maps won't be named correctly otherwise)
- preserve the shipped structure maps (from `structure_maps/` directory)
- generate the maps first:
    - `python src/main.py -c conf/<project>.json pipeline run -f -mt projects/<project>/source_data/mapping_table.json -msm` (parse profiles, dervice source definitions, generate StructureMaps)
- transform an example record:
    - `python src/main.py -c conf/<project>.json pipeline prepare-matchbox -f` (upload to matchbox)
    - `python src/main.py -c conf/<project>.json pipeline validate` (transform and validate the source records)
- compute the metrics using the `eval/create_project_metrics.py` script:

```
PYTHONPATH=.:src python3 -m eval.create_project_metrics projects/<project> \
  --existing /tmp/existing-<project> \
  --output <transform result from step 5> \
  --report eval/reports/<project>.json
```

# (Unseen) real-world profiles (simplifier)
- for the following projects, the tool with release 0.0.6 was used (without modifications) to identify problems to fix
- reproduction of the results works similar to the previous profiles
- provided example instances were used as target concepts for source records and mapping tables

Project list:
- [UK Core R4](https://simplifier.net/packages/UK.Core.r4.v2)
- [AU Core](https://simplifier.net/packages/hl7.fhir.au.core)
- [ISiK Basismodul](https://simplifier.net/packages/de.gematik.isik-basismodul)
- [Point-of-Care Device (UV PoCD)](https://simplifier.net/packages/hl7.fhir.uv.pocd)
- [Swedish National Medication List](https://simplifier.net/packages/se.electronichealth.fhir.nll.r4)
- [gematik TI Verzeichnisdienst](https://simplifier.net/packages/de.gematik.fhir.directory)
- [HL7 Europe Laboratory Report](https://simplifier.net/packages/hl7.fhir.eu.laboratory)
- [KL Danish Rehabilitation (§140)](https://simplifier.net/packages/kl.dk.fhir.rehab)
- [IHE Radiology IMR](https://simplifier.net/packages/ihe.rad.imr)
- [BBMRI.de Biobanking](https://simplifier.net/packages/de.bbmri.fhir)
- [gematik ePA Medication](https://simplifier.net/packages/de.gematik.epa.medication)
- [BfArM DiGA/DiPA](https://simplifier.net/packages/fhir.bfarm.de)

## Results

| project | resources | transformed | val (single) | bundle clean | TODOs | M1 preservation |
|---|---|---|---|---|---|---|
| aucore | 25 | 24/25 | 24/24 | 22/24 | 175 | 88.6% |
| bbmri | 11 | 11/11 | 9/11 | 7/11 | 87 | 94.4% |
| bfarm | 7 | 6/7 | 4/6 | 0/3 | 145 | 65.6% |
| dkrehab | 8 | 8/8 | 2/8 | 3/8 | 16 | 97.0% |
| epamed | 9 | 9/9 | 6/9 | 5/9 | 216 | 89.3% |
| eulab | 10 | 10/10 | 6/10 | 5/10 | 98 | 94.4% |
| gdir | 12 | 12/12 | 10/12 | 10/12 | 253 | 96.6% |
| imr | 5 | 5/5 | 3/5 | 4/5 | 25 | 97.1% |
| isik | 27 | 27/27 | 23/27 | 7/27 | 187 | 93.4% |
| pocd | 13 | 13/13 | 11/13 | 8/13 | 250 | 94.5% |
| senll | 25 | 25/25 | 8/25 | 6/25 | 437 | 87.1% |
| ukcore | 31 | 31/31 | 27/30 | 26/30 | 522 | 99.2% |

- 183 resources expected, 181 transformed, 2 transform errors: 
    - `aucore`
    - `au-core-bloodpressure`
    - `bfarm` / `HealthAppQuestionnaire` each returned no output (engine errors), so neither reaches validation and both are absent from the values
- Intended-profile `$validate`:
  - single mode: 133/180 (73.9%)
- 2411 remaining TODO rules (numbers inflated by using non-minimal mode and therefore include all optional fields as well)
- Data preservation ranges from 65.6% (`bfarm`) to 99.2% (`ukcore`).
- `ukcore` produced 31 resources but validated 30

### Additional info: example-validation experiments

Each IG-provided examples validated against its own profiles (`bbmri` was not measured):
- `aucore` 64/64
- `eulab` 21/21
- `pocd` 10/10
- `epamed` 66/67
- `gdir` 17/18
- `ukcore` 28/32
- `isik` 25/50
- `imr` 6/14
- `dkrehab` 5/13
- `senll` 3/21
- `bfarm` 3/6

## Reproduce
see above, but without the `-msm` flag

# Reproduction of related/existing work
- rebuild both independent pipelines, the kfdm one and the published paper with its own acceptance queries (Marfoglia/MOTU)

## kfdm (id: E15)
- compared by output equivalence against the original pipeline (`eval/kfdm_pipeline_bundle_example.json`)

### Results (22 records sample records)
- Resources: 176 original, this 145
- Per-resource: 124 `DIFFERS`, 52 `ONLY_IN_ORIGINAL`, 21 `ONLY_IN_OURS`
- Field level: 747 only-in-original, 577 only-in-ours, 176 value mismatches (Patient 89, QuestionnaireResponse 51, Observation 18, Condition 18)
- All 22 QuestionnaireResponses validate
- `Condition.clinicalStatus` (min 1, required `condition-clinical` binding) with typed `$rules` in the mapping table

`cm-clinicalstatus-anamnese` (`1`/ja to `active`, `2`/`0`/nein to `inactive`):

- 59 of 67 `Conditions` create a `clinicalStatus` with an equal original, including the `system` and `display` not emitted by original
- remaining 8 are the `3 unbekannt` and blank answers
- `condition-clinical` has no code for an unknown clinical state, so they are deliberately left unmapped rather than given an invented value

Differences compared to the original:

- `Condition.clinicalStatus`, original codings have no `system` (110 Conditions)
- `Condition.code = "provisional"` (a `ConditionVerificationStatus` code in `Condition.code` which the profiles set to max = 0
- `Patient.birthDate`, original creates `0020-01-01`, matches neither the source `geburtsdatum` (`1982-02-20`) nor the recorded age
- `QuestionnaireResponse.status`, the original applies `amended` to all 15 complete records. REDCap `2 = Complete` states completion. This pipelin create maps `0/1 to in-progress`, `2 to completed` the split and the 7 `in-progress` match exactly

Remaining open classes:

- Derived `Observation.status` / `effective[x]` (26) have no provider in profile or source, the original hard-codes `status: final`
- `Patient` (22), 13 are the input profile binding `Patient.gender` required by `http://fhir.de/CodeSystem/gender-amtlich-de`. A `CodeSystem` canonical in the `valueSet` slot, and an illegal rebinding besides (R4 binds `administrative-gender` required, which may be limited). The rest are empty-answer records.
- Blank source answers abort the whole transform for a record (`translate` creates a null source code), a preprocessing concern, not a mapping one
- Items 2.3 / 2.4 / 2.9 declare `answerOption.valueCoding` without a `system` (3 of 21 entries)

### Reproduce
- a live Matchbox instance is required
- `eval/kfdm_pipeline_bundle_example.json` is the reference transaction bundle from the original
- profiles have no public package, so `projects/kfdm_e2e/input_profile/` contains it
- Generate the maps (use `-crm` for reference, `-msm` for minimal mode): `python src/main.py -c conf/kfdm_e2e.json pipeline run -f -msm -crm -mr projects/kfdm_e2e/source_data/mapping_table.json`
- Upload profiles/maps to Matchbox: `python src/main.py -c conf/kfdm_e2e.json pipeline prepare-matchbox -f`
- Transform a REDCap record into a bundle:

```
python src/main.py -c conf/kfdm_e2e.json server start
python src/main.py -c conf/kfdm_e2e.json client send-request -m transform_data \
  -p '{"bundle": true}' -d "$(python3 -c 'import json;print(json.dumps([r for r in json.load(open("projects/kfdm_e2e/source_data/source_data.json")) if str(r["record_id"])=="2"][0]))')" \
  > /tmp/kfdm_record2.json
```

- compare them and create metrics: `PYTHONPATH=.:src python3 -m eval.e15_kfdm_compare --record <n>` (writes `eval/reports/e15_kfdm_record<n>.json`)

## Marfoglia / MOTU (id: E10)

- [reference paper](https://www.sciencedirect.com/science/article/pii/S0010482525000952) queries run against our output
- created profiles based on the used resources

The MOTU dataset is used for the reproduction: Arcobelli V. A., Moscato S., Palumbo P., Marfoglia A., Nardini F., Randi P., Davalli A., Carbonaro A., Chiari L., Mellone S. (2024). MOTU data. MOTU on FHIR: A 10-year data collection on the clinical rehabilitation pathway of 1006 trans-femoral amputees. Zenodo. https://doi.org/10.5281/zenodo.10683153 — licensed under CC BY 4.0.

### Results

- Query parity 5/5: Q1 85/85, Q2 34/34, Q3 67/67, Q4 674/674,
  Q5 AMK 11 / FK 79 / LK 26 / MPK 28 (= 144).
- 6186 transaction bundles posted, 0 failed.
- Stored counts similar to paper figures: Encounter 1962, Patient 1006, MedicationStatement 3032, Medication 370, AdverseEvent 146.
- Generation determinism: 0 StructureMaps and 0 ConceptMaps drifted from the committed maps.

### Reproduce
- requires a live `matchbox` (Port: `8080`), and an HAPI FHIR server (Port: `8090`) with referential inetgrity on write disabled
- (sequential) writing to the hapi server locally takes around 1hr
- Steps:
    - Download artifacts linked in the [reference paper](https://www.sciencedirect.com/science/article/pii/S0010482525000952) to the `eval/e10_marfoglia/_data` folder
    - Preprocess the data using the `PYTHONPATH=.:src python eval/e10_marfoglia/preprocess.py` script
    - Generate maps from the thin FSH profiles and transform the records: `PYTHONPATH=.:src python eval/e10_marfoglia/transform_d2.py --dump eval/e10_marfoglia/_data/transformed_d2`
    - Load the bundles into the HAPI server: `PYTHONPATH=.: src python eval/e10_marfoglia/hapi_load.py --base http://localhost:8089/fhir --dir eval/e10_marfoglia/_data/transformed_d2`
    - Run the queries described in the paper against the HAPI server (reproduces Table 7 of the reference paper): `PYTHONPATH=.:src python eval/e10_marfoglia/queries.py --source hapi --base http://localhost:8089/fhir`

# LLM/agent fix mode (id: E18)
- uses the generated maps and attempts to repair against the acceptance classification
- tested on the 21 testing and 12 unseen projects
- for this experiment a local `qwen3.5:122b` model (temperature 0) was used

## Results

### 21 projects run:

- Blocking findings: 600 table, 600 agent baseline, 600 after -> no change, no regressions.
- 132 maps: 96 `blocked`, 27 `clean`, 9 `no-progress`.
- 18 attempts, 6 reached the acceptance gate, 0 accepted, 14 provider calls, 0 cache hits.
- 3 patch-policy rejections total: 1 `invalid-pointer`, 2 `out-of-scope-pointer`.
- The agent made no provider call at all in 13 of 21 projects.

Run-to-run stability (3 passes, all 21 projects, response cache disabled):

| pass | baseline | after | attempts | accepted | calls |
|---|---|---|---|---|---|
| 1 | 602 | 593 | 17 | 2 | 12 |
| 2 | 599 | 597 | 22 | 1 | 18 |
| 3 | 600 | 598 | 19 | 1 | 15 |

- The same 8 projects engaged the model in every pass; attempts varied 17 / 22 / 19.
- Per project across the three passes: 11/21 stable (identical behaviour and counts), 4/21 differ only by measurement drift ≤1, 6/21 vary in agent behaviour (capable, eterminservice, evo13, kds_onkologie, kds_person, nictiz_cio).
- `kds_person` accepted a repair in 3/3 passes, −2 blocking each time. `capable` accepted in 1/3 passes.

**Measurement noise floor (null arm, no extra runs):** in a run where the agent applied and staged
nothing, `baseline` and `after` measure the same unchanged maps, so their difference is pure
measurement noise. Across **81** such paired observations: 78 exactly 0, range −1..+1, stdev 0.19,
none beyond ±1. One further observation was excluded on recorded evidence (Matchbox died mid-run:
`matchbox_execution` 1.0 → 0.0, `validation_blocked_by_environment` 0 → 2, difference +11).
Effects of ±1 are therefore not interpretable; the −2 and −6 repairs are.

**Distribution of what is wrong (table arm, 21 projects):**

- 37 required-path gaps, of which **0** are classified map-fixable.
- 39 worklist items (map-fixable **and** blocking) out of 600 blocking findings.
- Failure classes: `source-paths` 87, `cross-map` 41, `engine` 29, `other` 26,
  `coverage-input-required` 24, `obligations` 1; 29 clean.
- On the 24 gap-bearing maps, `source-path-not-found` appears on 17.

## Reproduce

Needs a live Matchbox and provider configuration (`LLM_*` environment variables, see `.env.example`).
An arm says nothing until its baseline has been run for the same project.

```bash
PYTHONPATH=.:src python3 -m eval.e18_agent_modes <project> table --engine
PYTHONPATH=.:src python3 -m eval.e18_agent_modes <project> agent --engine
PYTHONPATH=.:src python3 -m eval.e18_aggregate --arms agent
```

For the stability figures, disable the response cache. Otherwise identical prompts replay identical
answers and the experiment reports perfect stability by construction:

```bash
LLM_CACHE_ENABLED=false PYTHONPATH=.:src python3 -m eval.e18_agent_modes <project> agent --engine
```

### 12 (unseen) projects

`eval.e18_agent_modes <project> {table,agent} --engine`, using the same 12 unseen projects.

| project | maps | blocking findings b→a | validate b→a | required cov. b→a | maps clean b→a | applied | calls |
|---|---|---|---|---|---|---|---|
| aucore | 25 | 46→46 | 1.0→1.0 | 1.0→1.0 | 4→4 | 0 | 2 |
| bbmri | 11 | 44→44 | 0.75→0.75 | 1.0→1.0 | 5→5 | 0 | 4 |
| bfarm | — | generation-failed | — | — | — | — | — |
| dkrehab | 8 | 26→23 | 0.375→0.375 | 0.9744→0.9744 | 2→2 | 1 | 8 |
| epamed | 9 | 149→149 | 0.6667→0.6667 | 1.0→1.0 | 0→0 | 0 | 6 |
| eulab | 10 | 65→64 | 0.6→0.6 | 0.9677→0.9677 | 1→1 | 1 | 4 |
| gdir | 12 | 160→160 | 1.0→1.0 | 1.0→1.0 | 0→0 | 0 | 4 |
| imr | 5 | 23→22 | 0.4→0.4 | 0.871→0.871 | 1→1 | 1 | 5 |
| isik | 27 | 118→117 | 0.7059→0.7059 | 1.0→1.0 | 1→1 | 1 | 12 |
| pocd | 13 | 74→74 | 1.0→1.0 | 1.0→1.0 | 0→0 | 0 | 2 |
| senll | 25 | 303→303 | 0.3478→0.3478 | 0.9213→0.9213 | 1→1 | 0 | 9 |
| ukcore | 31 | 46→41 | 0.88→0.92 | 0.9302→0.9535 | 9→10 | 2 | 4 |

- 176 maps, 60 provider calls, 6 patches applied
- Blocking findings: 1054 → 1043 (−11)
- Per-map outcomes: 
    - `blocked`: 125 (71%)
    - `no-progress`: 21
    - `clean`: 20
    - `accepted`: 6,
    - `exhausted`: 3
    - `limit-reached`: 1
- `ukcore` is the only project to improve on every axis (blocking −5, validate 0.88→0.92, coverage +2.3pp) from 2 patches and 4 calls
- `bfarm` produced no agent result at all, since it died in generation with `AttributeError: 'StructureDefinition' object has no attribute 'subjectType'` (`fml_creator/fml_questionnaire.py`)

#### Accepted Patch fixes

- all 6 accepted patches address a choice element (`[x]`) that has a mapping-table row but no rule in the generated map (`mapping-obligation-dropped`).

| map | element | card. | patch |
|---|---|---|---|
| UKCore-Immunization | `occurrence[x]` | 1..1 | add rule, `copy` |
| UKCore-Procedure | `performed[x]:performedDateTime` | 0..1 | add rule, `cast(…,'dateTime')` |
| imr-diagnosticreport | `effective[x]:effectiveDateTime` | 1..1 | add rule, `cast(…,'dateTime')` |
