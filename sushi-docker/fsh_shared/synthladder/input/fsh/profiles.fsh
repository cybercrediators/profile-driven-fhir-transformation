// E14 synthetic generalizability ladder — each rung adds one construct dimension.

// L1 — scalar required field
Profile: SynthL1Patient
Parent: Patient
Id: synth-l1-patient
Title: "L1 scalar"
* name 1..1
* name.family 1..1

// L2 — cardinality (0..*) + complex datatype (Identifier)
Profile: SynthL2Patient
Parent: Patient
Id: synth-l2-patient
Title: "L2 cardinality + complex type"
* identifier 1..*
* identifier.value 1..1
* birthDate 0..1

// L3 — coded element with required binding + fixed pattern coding
Profile: SynthL3Patient
Parent: Patient
Id: synth-l3-patient
Title: "L3 binding + fixed"
* gender 1..1
* maritalStatus 1..1
* maritalStatus = http://terminology.hl7.org/CodeSystem/v3-MaritalStatus#M

// L4 — slicing (discriminator on identifier.system) with fixed slice values
Profile: SynthL4Patient
Parent: Patient
Id: synth-l4-patient
Title: "L4 slicing"
* identifier ^slicing.discriminator[0].type = #value
* identifier ^slicing.discriminator[0].path = "system"
* identifier ^slicing.rules = #open
* identifier contains mrn 1..1 and ssn 0..1
* identifier[mrn].system 1..1
* identifier[mrn].system = "http://example.org/mrn"
* identifier[ssn].system = "http://example.org/ssn"

// L5 — choice[x] constrained + cross-resource reference
Profile: SynthL5Observation
Parent: Observation
Id: synth-l5-observation
Title: "L5 choice[x] + reference"
* status 1..1
* code 1..1
* value[x] only Quantity
* subject 1..1
* subject only Reference(SynthL1Patient)

// L6 — simple extension + nested backbone
Extension: SynthNote
Id: synth-note
Title: "L6 extension"
* value[x] only string

Profile: SynthL6Patient
Parent: Patient
Id: synth-l6-patient
Title: "L6 extension + nested backbone"
* extension contains SynthNote named note 0..1
* contact 0..*
* contact.name.family 1..1
