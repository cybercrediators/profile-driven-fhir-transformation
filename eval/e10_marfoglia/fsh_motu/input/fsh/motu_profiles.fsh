// Complete MOTU thin-profile set (E10 Direction-2). Authored from their *ToBundle maps via
// eval/e10_marfoglia/extract_map_targets.py. Per the "author their fixed codings into the
// profile" decision: fixed domain codings -> pattern[x]; data fields -> left open + mapped via
// the per-resource mapping table; references (subject/encounter/...) are bundle-assembly and
// deferred. Displays omitted where the extractor could not cleanly isolate them (system+code
// are what the discriminator/conformance need).

Alias: $sct = http://snomed.info/sct
Alias: $icd10 = http://hl7.org/fhir/sid/icd-10
Alias: $v2-0203 = http://terminology.hl7.org/CodeSystem/v2-0203
Alias: $ucum = http://unitsofmeasure.org
Alias: $cond-clin = http://terminology.hl7.org/CodeSystem/condition-clinical
Alias: $v3act = http://terminology.hl7.org/CodeSystem/v3-ActCode
Alias: $adminGender = http://hl7.org/fhir/administrative-gender

CodeSystem: MotuProstheticKneeProperties
Id: MotuProstheticKneeProperties
Title: "MOTU Prosthetic Knee Properties"
* ^status = #draft
* #PatientMaximumWeight "Patient Maximum Weight"
* #PatientActivityLevel "Patient Activity"
* #Category "Knee Category"
* #MPK "Microprocessor-Controlled (MPK)"

// Amputation side/cause target value sets — extensional (enumerated SNOMED) so the generator
// treats the bound bodySite/reasonCode as translatable and emits a ConceptMap + $translate
// (mirroring gender). The source→target arrows (L→7771000, vascular→57662003, …) are authored
// into the generated ConceptMaps and preserved across regeneration (see transform_d2.py).
// Sex target value set. The base `Patient.gender` binding is a whole-CodeSystem include,
// which is not enumerable, so nothing triggers a ConceptMap and the source's `M`/`F` were
// copied through verbatim. Narrowing to the two codes the data uses is legal against the
// base *required* binding (a subset of administrative-gender) and is what makes the field
// translatable — the same mechanism the amputation value sets below rely on.
ValueSet: MotuSexVS
Id: motu-sex-vs
Title: "MOTU Sex (administrative-gender subset)"
* ^status = #draft
* $adminGender#male "Male"
* $adminGender#female "Female"

ValueSet: MotuAmputationSideVS
Id: motu-amputation-side-vs
Title: "MOTU Amputation Side (SNOMED bodySite)"
* ^status = #draft
* $sct#7771000 "Left side"
* $sct#24028007 "Right side"

ValueSet: MotuAmputationCauseVS
Id: motu-amputation-cause-vs
Title: "MOTU Amputation Cause (SNOMED reasonCode)"
* ^status = #draft
* $sct#417746004 "Injury"
* $sct#57662003 "Peripheral vascular disease"
* $sct#363346000 "Malignant neoplastic disease"
* $sct#40733004 "Infectious disease"
* $sct#66091009 "Congenital disease"

ValueSet: MotuRehabGoalVS
Id: motu-rehab-goal-vs
Title: "MOTU Rehab Goal (SNOMED Goal.outcomeCode)"
* ^status = #draft
* $sct#165252001 "No aid for walking"
* $sct#443392007 "Uses single crutch for walking"
* $sct#443663000 "Uses two crutches for walking"
* $sct#895488007 "Does mobilize using walker"

// nearFall → AdverseEvent.event spans TWO code systems: an actual fall is SNOMED, a near-fall is
// ICD-10. The mixed-system binding makes the generator emit translate→Coding (system travels with
// the code) instead of a single fixed system.
ValueSet: MotuFallEventVS
Id: motu-fall-event-vs
Title: "MOTU Fall Event (SNOMED + ICD-10 AdverseEvent.event)"
* ^status = #draft
* $sct#1912002 "Fall (event)"
* $icd10#W18.40 "Slipping, tripping and stumbling without falling, unspecified"

// ───────────────────────── Patient (MotuPatient) ─────────────────────────
Profile: MotuPatient
Parent: Patient
Id: motu-patient
Title: "MOTU Patient (thin)"
* identifier 1..*
* identifier.system = $v2-0203
* gender 0..1
* gender from MotuSexVS (required)   // enumerable binding drives the gender ConceptMap

// ───────────────────────── Medication (MotuDrug) ─────────────────────────
Profile: MotuMedication
Parent: Medication
Id: motu-medication
Title: "MOTU Medication (thin)"
* code 1..1
* code.coding 1..*
* code.coding.system = "http://www.whocc.no/atc"
* code.coding.code 1..1

// ───────────────────────── Encounter (Drug/Fall/HospitalStay) ─────────────
Profile: MotuEncounter
Parent: Encounter
Id: motu-encounter
Title: "MOTU Encounter (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* identifier 1..*
* identifier.system = "http://ahdbservices.it/CodeSystem/motu-encounter-id"
* status 1..1
* status = #finished
* class 1..1
* class = $v3act#IMP "inpatient encounter"
* type 0..*
* type.coding.system = "http://ncithesaurus-stage.nci.nih.gov"
* type.coding.code = #C25179
* period 0..1
* length 0..1
* length.system = $ucum
* length.code = #d
* length.unit = "day"
// Re-admission flag: Marfoglia stamps hospitalization.reAdmission = v2-0092#R on renewal stays and
// leaves it absent on first-delivery stays. Declaring the element here makes the generator emit the
// mapping rule (readmission -> code); the source field is set only for renewals (upstream), so the
// element is created only for renewals — matching the ground truth. Drives demonstration Q1/Q2.
* hospitalization.reAdmission.coding.system = "http://terminology.hl7.org/CodeSystem/v2-0092"

// ───────────────────────── MedicationStatement (MotuDrug) ─────────────────
Profile: MotuMedicationStatement
Parent: MedicationStatement
Id: motu-medication-statement
Title: "MOTU MedicationStatement (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* context 1..1
* context only Reference(Encounter)
* medication[x] 1..1
* medication[x] only Reference(Medication)
* status 1..1
* status = #completed

// ───────────────────────── AdverseEvent (MotuFall) ────────────────────────
Profile: MotuAdverseEvent
Parent: AdverseEvent
Id: motu-adverse-event
Title: "MOTU AdverseEvent (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
// event carries the fall/near-fall — bound to a mixed-system (SNOMED + ICD-10) VS so the generator
// emits a translate→Coding over the authored ConceptMap (nearFall 0→SNOMED fall, 1→ICD near-fall).
* event 0..1
* event from MotuFallEventVS (required)

// ───────────────────────── Condition (HospitalStay) ───────────────────────
Profile: MotuCondition
Parent: Condition
Id: motu-condition
Title: "MOTU Condition (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* clinicalStatus 0..1
* clinicalStatus.coding.system = $cond-clin
* clinicalStatus.coding.code = #active
* code 0..1
* code.coding.system = $sct
* code.coding.code = #282097004

// ───────────────────────── Procedure (HospitalStay) ───────────────────────
Profile: MotuProcedure
Parent: Procedure
Id: motu-procedure
Title: "MOTU Procedure (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #completed
* code 0..1
* code.coding.system = $sct
* code.coding.code = #81723002
// bodySite / reasonCode carry the amputation side / cause — bound to enumerable SNOMED value
// sets so the generator emits a $translate over a (regen-safe authored) ConceptMap that maps the
// source codes (L/R, vascular/traumatic/…) to SNOMED, reproducing Marfoglia's Procedure coding.
* bodySite 0..*
* bodySite from MotuAmputationSideVS (required)
* reasonCode 0..*
* reasonCode from MotuAmputationCauseVS (required)

// ───────────────────────── Goal (HospitalStay) ────────────────────────────
Profile: MotuGoal
Parent: Goal
Id: motu-goal
Title: "MOTU Goal (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
// outcomeCode carries the rehab goal — bound to an enumerable SNOMED VS so the generator emits a
// $translate over an authored ConceptMap (free_walk/aid1/aid2/walker → SNOMED), like bodySite/cause.
* outcomeCode 0..*
* outcomeCode from MotuRehabGoalVS (required)

// ───────────────────────── DeviceDefinition (MotuKnee, sliced) ─────────────
Profile: MotuDeviceDefinition
Parent: DeviceDefinition
Id: motu-device-definition
Title: "MOTU Device Definition (thin)"
* identifier 1..*
* identifier.system = "http://ahdbservices.it"
* deviceName 1..*
* deviceName.type = #model-name
* type 1..1
* type = $sct#109228008 "Knee joint prosthesis"
* contact 0..*
* contact.system = #url
* property ^slicing.discriminator.type = #pattern
* property ^slicing.discriminator.path = "type"
* property ^slicing.rules = #open
* property contains
    weight 0..1 and
    patientMaximumWeight 0..1 and
    patientActivityLevel 0..1 and
    category 0..1 and
    manualLock 0..1 and
    polycentric 0..1 and
    mpk 0..1
* property[weight].type = $sct#726527001 "Weight"
* property[weight].valueQuantity 1..1
* property[weight].valueQuantity.system = $ucum
* property[weight].valueQuantity.code = #/kg
* property[weight].valueQuantity.unit = "per kilogram"
* property[patientMaximumWeight].type = MotuProstheticKneeProperties#PatientMaximumWeight "Patient Maximum Weight"
* property[patientMaximumWeight].valueQuantity 1..1
* property[patientMaximumWeight].valueQuantity.system = $ucum
* property[patientMaximumWeight].valueQuantity.code = #kg
* property[patientMaximumWeight].valueQuantity.unit = "kilogram"
* property[patientActivityLevel].type = MotuProstheticKneeProperties#PatientActivityLevel "Patient Activity"
* property[patientActivityLevel].valueCode 1..1
* property[category].type = MotuProstheticKneeProperties#Category "Knee Category"
* property[category].valueCode 1..1
* property[manualLock].type = $sct#701702002 "Single-axis manual lock knee prosthesis"
* property[polycentric].type = $sct#701409005 "Polycentric mechanical knee prosthesis"
* property[mpk].type = MotuProstheticKneeProperties#MPK "Microprocessor-Controlled (MPK)"

// ───────────────────────── Observation: body weight (HospitalStay) ─────────
Profile: MotuWeightObservation
Parent: Observation
Id: motu-weight-observation
Title: "MOTU Weight Observation (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #registered
* category 0..*
* category.coding.system = "http://terminology.hl7.org/CodeSystem/observation-category"
* category.coding.code = #vital-signs
* code 1..1
* code.coding.system = "http://loinc.org"
* code.coding.code = #29463-7
* value[x] only Quantity
* valueQuantity.system = "http://unitsofmeasure.org"
* valueQuantity.code = #/kg
* valueQuantity.unit = "per kilogram"

// ───────────────────────── QuestionnaireResponse: RERMultif (item-sliced) ──
Profile: MotuRERMultifQR
Parent: QuestionnaireResponse
Id: motu-rermultif-qr
Title: "MOTU RERMultif QuestionnaireResponse (thin)"
// references — required so the generator emits resolve-reference rules
* source 1..1
* source only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #completed
* item ^slicing.discriminator.type = #value
* item ^slicing.discriminator.path = "linkId"
* item ^slicing.rules = #open
* item contains
    hfall 0..1 and
    fof 0..1 and
    drugCardio 0..1
* item[hfall].linkId = "1.1"
* item[hfall].answer 0..*
* item[hfall].answer.value[x] only integer
* item[fof].linkId = "1.2"
* item[fof].answer 0..*
* item[fof].answer.value[x] only integer
* item[drugCardio].linkId = "1.3"
* item[drugCardio].answer 0..*
* item[drugCardio].answer.value[x] only integer

// ───────────────────────── Observation: height (HospitalStay) ──────────────
Profile: MotuHeightObservation
Parent: Observation
Id: motu-height-observation
Title: "MOTU Height Observation (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #registered
* category.coding.system = "http://terminology.hl7.org/CodeSystem/observation-category"
* category.coding.code = #vital-signs
* code 1..1
* code.coding.system = "http://loinc.org"
* code.coding.code = #8302-2
* value[x] only Quantity
* valueQuantity.system = "http://unitsofmeasure.org"
* valueQuantity.code = #cm
* valueQuantity.unit = "centimeter"

// ───────────────────────── Observation: comorbidities (HospitalStay) ───────
Profile: MotuComorbiditiesObservation
Parent: Observation
Id: motu-comorbidities-observation
Title: "MOTU Comorbidities Observation (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #registered
* code 1..1
* code.coding.system = "http://snomed.info/sct"
* code.coding.code = #398192003
* value[x] only integer

// ───────────────────────── QuestionnaireResponse: Barthel (10 items) ───────
Profile: MotuBarthelQR
Parent: QuestionnaireResponse
Id: motu-barthel-qr
Title: "MOTU Barthel QuestionnaireResponse (thin)"
// references — required so the generator emits resolve-reference rules
* source 1..1
* source only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #completed
* item ^slicing.discriminator.type = #value
* item ^slicing.discriminator.path = "linkId"
* item ^slicing.rules = #open
* item contains
    hygiene 0..1 and wash 0..1 and nutrition 0..1 and dress 0..1 and
    intestinalIncontinence 0..1 and urinaryIncontinence 0..1 and toilet 0..1 and
    transfer 0..1 and walk 0..1 and stairs 0..1
* item[hygiene].linkId = "1.1"
* item[hygiene].answer.value[x] only integer
* item[wash].linkId = "1.2"
* item[wash].answer.value[x] only integer
* item[nutrition].linkId = "1.3"
* item[nutrition].answer.value[x] only integer
* item[dress].linkId = "1.4"
* item[dress].answer.value[x] only integer
* item[intestinalIncontinence].linkId = "1.5"
* item[intestinalIncontinence].answer.value[x] only integer
* item[urinaryIncontinence].linkId = "1.6"
* item[urinaryIncontinence].answer.value[x] only integer
* item[toilet].linkId = "1.7"
* item[toilet].answer.value[x] only integer
* item[transfer].linkId = "1.8"
* item[transfer].answer.value[x] only integer
* item[walk].linkId = "1.9"
* item[walk].answer.value[x] only integer
* item[stairs].linkId = "1.10"
* item[stairs].answer.value[x] only integer
Alias: $loinc = http://loinc.org

// ───────────────────────── Account (Hospitalization) ──────────────────────
Profile: MotuAccount
Parent: Account
Id: motu-account
Title: "MOTU Account (thin)"
* status 1..1
* status = #inactive
* subject 1..1
* subject only Reference(MotuPatient)
* coverage 1..*
* coverage.coverage 1..1
* coverage.coverage only Reference(MotuCoverage)

// ───────────────────────── Coverage (Hospitalization) ─────────────────────
Profile: MotuCoverage
Parent: Coverage
Id: motu-coverage
Title: "MOTU Coverage (thin)"
* status 1..1
* status = #active
* beneficiary 1..1
* beneficiary only Reference(MotuPatient)
* payor 1..*

// ───────────────────────── CarePlan (Rehab) ───────────────────────────────
Profile: MotuCarePlan
Parent: CarePlan
Id: motu-care-plan
Title: "MOTU CarePlan (thin)"
// references — required so the generator emits resolve-reference rules
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #completed
* intent 1..1
* intent = #order
* subject 1..1
* subject only Reference(MotuPatient)
* goal 1..*
* goal only Reference(MotuGoal)

// ───────────────────────── DeviceRequest (Knee) ───────────────────────────
Profile: MotuDeviceRequest
Parent: DeviceRequest
Id: motu-device-request
Title: "MOTU DeviceRequest (thin)"
// references — required so the generator emits resolve-reference rules
* encounter 1..1
* encounter only Reference(Encounter)
* intent 1..1
* intent = #order
* subject 1..1
* subject only Reference(MotuPatient)
* code[x] 1..1
// R4 (and R4B) forbid Reference(DeviceDefinition) on code[x] — codeReference is Reference(Device)
// only (CodeableReference(Device|DeviceDefinition) is R5-only). Marfoglia bends the R4 spec here.
// SUSHI's `only` keyword hard-errors on the off-spec target, so set the type directly with caret
// rules (these bypass `only`'s type validation): code[x] becomes Reference(motu-device-definition),
// keeping the inherited CodeableConcept branch. The generator then emits a resolve-reference rule
// and the bundle wires DeviceRequest.codeReference → the knee DeviceDefinition.
* code[x] ^type[0].code = "Reference"
* code[x] ^type[0].targetProfile = "http://ahdbservices.it/StructureDefinition/motu-device-definition"

// ───────────────────────── QuestionnaireResponse: Morse ───────────────────
Profile: MotuMorseQR
Parent: QuestionnaireResponse
Id: motu-morse-qr
Title: "MOTU Morse QuestionnaireResponse (thin)"
// references — required so the generator emits resolve-reference rules
* source 1..1
* source only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #completed
* item ^slicing.discriminator.type = #value
* item ^slicing.discriminator.path = "linkId"
* item ^slicing.rules = #open
* item contains
    hfall 0..1 and pathologies 0..1 and mobility 0..1 and
    endovenous 0..1 and transfer 0..1 and mental 0..1
* item[hfall].linkId = "1.1"
* item[hfall].answer.value[x] only integer
* item[pathologies].linkId = "1.2"
* item[pathologies].answer.value[x] only integer
* item[mobility].linkId = "1.3"
* item[mobility].answer.value[x] only integer
* item[endovenous].linkId = "1.4"
* item[endovenous].answer.value[x] only integer
* item[transfer].linkId = "1.5"
* item[transfer].answer.value[x] only integer
* item[mental].linkId = "1.6"
* item[mental].answer.value[x] only integer

// ───────────────────────── QuestionnaireResponse: AMP ─────────────────────
Profile: MotuAMPQR
Parent: QuestionnaireResponse
Id: motu-amp-qr
Title: "MOTU AMP QuestionnaireResponse (thin)"
// references — required so the generator emits resolve-reference rules
* source 1..1
* source only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #completed
* item ^slicing.discriminator.type = #value
* item ^slicing.discriminator.path = "linkId"
* item ^slicing.rules = #open
* item contains proNopro 0..1 and score 0..1 and kLevel 0..1
* item[proNopro].linkId = "1.1"
* item[proNopro].answer.value[x] only string   // categorical: "PRO" / "noPRO"
* item[score].linkId = "1.2"
* item[score].answer.value[x] only decimal     // AMP score, source carries fractional values
* item[kLevel].linkId = "1.3"
* item[kLevel].answer.value[x] only integer    // K-level 0–4

// ───────────────────────── QuestionnaireResponse: LCI ─────────────────────
Profile: MotuLCIQR
Parent: QuestionnaireResponse
Id: motu-lci-qr
Title: "MOTU LCI QuestionnaireResponse (thin)"
// references — required so the generator emits resolve-reference rules
* source 1..1
* source only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #completed
* item ^slicing.discriminator.type = #value
* item ^slicing.discriminator.path = "linkId"
* item ^slicing.rules = #open
* item contains score 0..1
* item[score].linkId = "1.1"
// decimal (not integer): the `score` source field maps into both AMP and LCI score items via one
// base-type-keyed mapping; AMP scores are fractional, so both answers use decimal (LCI's integer
// scores serialise losslessly as decimal).
* item[score].answer.value[x] only decimal

// ───────────────────────── Observation: TWT (components) ──────────────────
Profile: MotuTWTObservation
Parent: Observation
Id: motu-twt-observation
Title: "MOTU TWT Observation (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #registered
* code 1..1
* code.coding.system = $sct
* code.coding.code = #252478000
* component ^slicing.discriminator.type = #pattern
* component ^slicing.discriminator.path = "code"
* component ^slicing.rules = #open
* component contains time 0..1 and steps 0..1
* component[time].code = $loinc#55411-3 "Exercise duration"
* component[time].value[x] only Quantity
* component[steps].code = $loinc#55423-8 "Number of steps"
* component[steps].value[x] only Quantity

// ───────────────────────── Observation: Pain (components) ─────────────────
Profile: MotuPainObservation
Parent: Observation
Id: motu-pain-observation
Title: "MOTU Pain Observation (thin)"
// references — required so the generator emits resolve-reference rules
* subject 1..1
* subject only Reference(Patient)
* encounter 1..1
* encounter only Reference(Encounter)
* status 1..1
* status = #registered
* code 1..1
* code.coding.system = $loinc
* code.coding.code = #72514-3
* component ^slicing.discriminator.type = #pattern
* component ^slicing.discriminator.path = "code"
* component ^slicing.rules = #open
* component contains
    back 0..1 and contralateralLimb 0..1 and contralateralKnee 0..1 and
    stump 0..1 and phantomLimb 0..1
* component[back].code = $sct#77568009 "Structure of back of trunk"
* component[back].value[x] only Quantity
* component[contralateralLimb].code = $sct#66019005 "Limb structure"
* component[contralateralLimb].value[x] only Quantity
* component[contralateralKnee].code = $sct#72696002 "Knee region structure"
* component[contralateralKnee].value[x] only Quantity
* component[stump].code = $sct#38033009 "Stump"
* component[stump].value[x] only Quantity
* component[phantomLimb].code = $sct#193114000 "Phantom limb"
* component[phantomLimb].value[x] only Quantity
