"""Reference wiring in BundleService, incl. the profile-typed-target fix.

A `Reference(SomeProfile)` carries the *profile id* as the target type in the
StructureMap rule documentation, but the present resource is stamped with the
base `resourceType`. `_wire_references` must resolve the profile id → base type
(via the registry-derived profile_type_map) or the reference is silently skipped.
"""

import pytest

from controller.bundle_service import BundleService

pytestmark = pytest.mark.unit


def _obs_pat():
    obs = {"resourceType": "Observation", "status": "final"}
    pat = {"resourceType": "Patient", "name": [{"family": "X"}]}
    return obs, pat


def test_profile_typed_reference_wires_with_profile_map():
    obs, pat = _obs_pat()
    urn = {id(obs): "urn:uuid:obs", id(pat): "urn:uuid:pat"}
    BundleService._wire_references(
        [obs, pat], [("Observation", "subject", "synth-l1-patient")], urn, {},
        {"synth-l1-patient": "Patient"},
    )
    assert obs.get("subject") == {"reference": "urn:uuid:pat"}


def test_profile_typed_reference_skipped_without_map():
    # Without the profile→type map the profile id matches no resourceType.
    obs, pat = _obs_pat()
    urn = {id(obs): "urn:uuid:obs", id(pat): "urn:uuid:pat"}
    BundleService._wire_references(
        [obs, pat], [("Observation", "subject", "synth-l1-patient")], urn, {}, {}
    )
    assert "subject" not in obs


def test_base_type_reference_still_wires():
    obs, pat = _obs_pat()
    urn = {id(obs): "urn:uuid:obs", id(pat): "urn:uuid:pat"}
    BundleService._wire_references(
        [obs, pat], [("Observation", "subject", "Patient")], urn, {}, {}
    )
    assert obs.get("subject") == {"reference": "urn:uuid:pat"}


def test_choice_target_picks_present_type():
    # "Patient|Group" — only Patient present, profile map empty.
    obs, pat = _obs_pat()
    urn = {id(obs): "urn:uuid:obs", id(pat): "urn:uuid:pat"}
    BundleService._wire_references(
        [obs, pat], [("Observation", "subject", "Patient|Group")], urn, {}, {}
    )
    assert obs.get("subject") == {"reference": "urn:uuid:pat"}


def test_build_profile_type_map_empty_without_registry():
    assert BundleService._build_profile_type_map(None) == {}


def test_reference_into_absent_optional_backbone_is_skipped():
    # Patient.link.other → Patient, but the transform produced no `link` backbone.
    # Wiring must NOT materialise an (incomplete, required-field-missing) link entry.
    pat = {"resourceType": "Patient", "name": [{"family": "X"}]}
    other = {"resourceType": "Patient", "name": [{"family": "Y"}]}
    urn = {id(pat): "urn:uuid:pat", id(other): "urn:uuid:other"}
    BundleService._wire_references(
        [pat, other], [("Patient", "link.other", "Patient")], urn, {}, {}
    )
    assert "link" not in pat  # backbone not created


def test_reference_into_present_backbone_still_wires():
    # If the backbone already exists, the nested reference wires normally.
    pat = {"resourceType": "Patient", "link": [{"type": "seealso"}]}
    rp = {"resourceType": "RelatedPerson"}
    urn = {id(pat): "urn:uuid:pat", id(rp): "urn:uuid:rp"}
    BundleService._wire_references(
        [pat, rp], [("Patient", "link.other", "RelatedPerson")], urn,
        {"Patient": {"link"}}, {},
    )
    assert pat["link"][0].get("other") == {"reference": "urn:uuid:rp"}
    assert pat["link"][0].get("type") == "seealso"  # existing content preserved


def test_choice_reference_field_expands_x_to_typed_name():
    # A choice element `code[x]` holding a Reference value must serialise as
    # `codeReference` (FHIR poly-property naming), not the literal `code[x]`.
    # `_iter_reference_fields` is the single point that must expand the leaf.
    fields = [
        {
            "path": "SomeResource.code[x]",
            "reference_target": "http://example.org/StructureDefinition/some-profile",
            "cardinality": {"min": 1, "max": "1"},
            "type": [{"code": "Reference"}, {"code": "CodeableConcept"}],
        }
    ]
    out = list(BundleService._iter_reference_fields(fields))
    assert len(out) == 1
    relative, raw, is_required, ref_only, extension_url = out[0]
    assert extension_url is None  # not inside an extension
    assert relative == "codeReference"  # not "code[x]"
    assert raw == "some-profile"
    assert is_required is True
    # mixed choice (Reference|CodeableConcept) → satisfiable without the reference
    assert ref_only is False


def test_ref_only_choice_still_expands_x():
    # A choice narrowed to only Reference (e.g. a `foo[x] only Reference(Bar)`) is
    # ref_only=True and must also expand to the typed property name.
    fields = [
        {
            "path": "SomeResource.thing[x]",
            "reference_target": "Bar",
            "cardinality": {"min": 1, "max": "1"},
            "type": [{"code": "Reference"}],
        }
    ]
    relative, raw, is_required, ref_only, _extension_url = next(
        iter(BundleService._iter_reference_fields(fields))
    )
    assert relative == "thingReference"
    assert ref_only is True


def test_collect_todo_refs_expands_choice_element_to_typed_reference():
    # The SM-declared wiring path parses the field path from the rule documentation;
    # a choice element (medication[x]) must expand to its typed form there too, or
    # _set_nested writes the literal `[x]` key (rejected by HAPI-1809) and the two
    # wiring paths (_iter_reference_fields vs TODO specs) disagree about the field.
    rule = {
        "name": "TODO-resolve-reference-MedicationStatement-medication",
        "documentation": (
            "Reference<MedicationStatement.medication[x]> → Medication "
            "— resolve via bundle assembler"
        ),
    }
    specs = []
    BundleService._collect_todo_refs(rule, specs)
    # Specs carry the declaring map's target profile (None when not supplied) so
    # _wire_references can scope a reference to the profile that declared it.
    assert specs == [("MedicationStatement", "medicationReference", "Medication", None)]


def _bundled_rule(**overrides):
    contract = {
        "sourceType": "Observation",
        "path": "performer",
        "targetTypes": ["Practitioner"],
        "targetProfile": None,
        "match": "byOrder",
        "sourceKey": None,
        "targetKey": None,
        "referenceMode": "urn",
        **overrides,
    }
    import json

    return {
        "name": "TODO-resolve-reference-performer",
        "documentation": "FHIRBRIDGE_REFERENCE:"
        + json.dumps(contract, sort_keys=True, separators=(",", ":")),
    }


def test_collect_bundled_reference_contract():
    specs = []
    BundleService._collect_todo_refs(_bundled_rule(match="singleton"), specs)
    assert specs == [
        (
            "Observation",
            "performer",
            "Practitioner",
            None,
            "bundled",
            "singleton",
            None,
            None,
            "urn",
            None,
        )
    ]


def test_identifier_correlation_wires_matching_target():
    first = {
        "resourceType": "Observation",
        "identifier": [{"value": "a"}],
    }
    second = {
        "resourceType": "Observation",
        "identifier": [{"value": "b"}],
    }
    practitioner_a = {
        "resourceType": "Practitioner",
        "identifier": [{"value": "a"}],
    }
    practitioner_b = {
        "resourceType": "Practitioner",
        "identifier": [{"value": "b"}],
    }
    resources = [first, second, practitioner_b, practitioner_a]
    urns = {
        id(practitioner_a): "urn:uuid:a",
        id(practitioner_b): "urn:uuid:b",
    }
    specs = []
    BundleService._collect_todo_refs(
        _bundled_rule(
            match="identifier",
            sourceKey="identifier.value",
            targetKey="identifier.value",
        ),
        specs,
    )
    BundleService._wire_references(resources, specs, urns)
    assert first["performer"] == {"reference": "urn:uuid:a"}
    assert second["performer"] == {"reference": "urn:uuid:b"}


def test_all_correlation_wires_repeated_multi_target_references():
    obs = {"resourceType": "Observation"}
    practitioner = {"resourceType": "Practitioner", "id": "p1"}
    organization = {"resourceType": "Organization", "id": "o1"}
    specs = [
        (
            "Observation",
            "performer",
            "Practitioner|Organization",
            None,
            "bundled",
            "all",
            None,
            None,
            "relative",
        )
    ]
    BundleService._wire_references(
        [obs, practitioner, organization], specs, {}, {"Observation": {"performer"}}
    )
    assert obs["performer"] == [
        {"reference": "Practitioner/p1"},
        {"reference": "Organization/o1"},
    ]


def test_singleton_policy_does_not_guess_between_multiple_targets():
    obs = {"resourceType": "Observation"}
    one = {"resourceType": "Practitioner"}
    two = {"resourceType": "Practitioner"}
    specs = [
        (
            "Observation",
            "performer",
            "Practitioner",
            None,
            "bundled",
            "singleton",
            None,
            None,
            "urn",
        )
    ]
    BundleService._wire_references(
        [obs, one, two],
        specs,
        {id(one): "urn:uuid:one", id(two): "urn:uuid:two"},
    )
    assert "performer" not in obs


def test_target_profile_scopes_same_base_type_candidates():
    obs = {"resourceType": "Observation"}
    wanted = {
        "resourceType": "Practitioner",
        "meta": {"profile": ["http://example.org/StructureDefinition/Wanted"]},
    }
    other = {
        "resourceType": "Practitioner",
        "meta": {"profile": ["http://example.org/StructureDefinition/Other"]},
    }
    specs = []
    BundleService._collect_todo_refs(
        _bundled_rule(
            targetProfile="http://example.org/StructureDefinition/Wanted",
            match="singleton",
        ),
        specs,
    )
    BundleService._wire_references(
        [obs, other, wanted],
        specs,
        {id(other): "urn:uuid:other", id(wanted): "urn:uuid:wanted"},
    )
    assert obs["performer"] == {"reference": "urn:uuid:wanted"}


def test_profile_sliced_reference_contract_wires_each_required_profile():
    complete_profile = "http://example.org/StructureDefinition/complete"
    measurement_profile = "http://example.org/StructureDefinition/measurement"
    reference_profile = "http://example.org/StructureDefinition/reference-value"
    complete = {
        "resourceType": "Observation",
        "meta": {"profile": [complete_profile]},
    }
    measurement = {
        "resourceType": "Observation",
        "meta": {"profile": [measurement_profile]},
    }
    reference = {
        "resourceType": "Observation",
        "meta": {"profile": [reference_profile]},
    }
    contract = _bundled_rule(
        sourceType="Observation",
        path="derivedFrom",
        targetTypes=[measurement_profile, reference_profile],
        targetProfiles=[measurement_profile, reference_profile],
        match="all",
    )
    specs = []
    BundleService._collect_todo_refs(
        contract,
        specs,
        {
            "measurement": "Observation",
            "reference-value": "Observation",
        },
        complete_profile,
    )

    BundleService._wire_references(
        [complete, measurement, reference],
        specs,
        {
            id(measurement): "urn:uuid:measurement",
            id(reference): "urn:uuid:reference",
        },
    )

    assert complete["derivedFrom"] == [
        {"reference": "urn:uuid:measurement"},
        {"reference": "urn:uuid:reference"},
    ]


def test_profile_sliced_reference_contract_is_collectable_from_structure_map():
    complete_profile = "http://example.org/StructureDefinition/complete"
    measurement_profile = "http://example.org/StructureDefinition/measurement"
    reference_profile = "http://example.org/StructureDefinition/reference-value"
    contract = _bundled_rule(
        sourceType="Observation",
        path="derivedFrom",
        targetTypes=[measurement_profile, reference_profile],
        targetProfiles=[measurement_profile, reference_profile],
        match="all",
    )
    structure_map = {
        "structure": [{"mode": "target", "url": complete_profile}],
        "group": [
            {
                "input": [
                    {
                        "name": "target",
                        "mode": "target",
                        "type": "Observation",
                    }
                ],
                "rule": [contract],
            }
        ],
    }

    specs = BundleService._specs_from_structure_maps([structure_map])

    assert len(specs) == 1
    assert specs[0][10] == (measurement_profile, reference_profile)


def test_reference_wiring_respects_prohibited_reference_child():
    profile = "http://example.org/StructureDefinition/display-only-subject"
    procedure = {
        "resourceType": "Procedure",
        "meta": {"profile": [profile]},
    }
    patient = {"resourceType": "Patient"}

    BundleService._wire_references(
        [procedure, patient],
        [("Procedure", "subject", "Patient", profile)],
        {id(patient): "urn:uuid:patient"},
        prohibited_map={profile: {"subject.reference"}},
    )

    assert "subject" not in procedure


def test_profile_constraints_survive_a_scheme_only_canonical_mismatch():
    """evo13 pins the http:// form of an https:// canonical on meta.profile.

    The prohibited/required maps are keyed by the published canonical, so an exact
    lookup misses and every max=0 path stops being enforced — the assembler would
    then wire elements the profile forbids.
    """
    from controller.bundle_service import BundleService

    resource = {"resourceType": "Communication",
                "meta": {"profile": ["http://example.org/StructureDefinition/X"]}}
    prohibited = {"https://example.org/StructureDefinition/X": {"subject"}}
    assert BundleService._wiring_forbidden(resource, "subject", prohibited) is True
    # an unrelated profile is still not matched
    assert BundleService._wiring_forbidden(
        resource, "subject",
        {"https://example.org/StructureDefinition/Y": {"subject"}}) is False


# --------------------------------------------------------------------------
# Slice-scoped contracts: a reference below a sliced repeating backbone.
#
# `Composition.section:sectionMedications.entry` and its five sibling sections all
# flatten to the runtime path `section.entry`. Without a selector the specs are
# byte-identical, `_specs_from_structure_maps` dedups them to one, and that one
# writes into `section[0]` — so five sections end up with no entry at all and the
# sixth gets the wrong references (ontario, 12 of its 22 bundle issues).
# --------------------------------------------------------------------------

_LOINC = "http://loinc.org"


def _section(code, title):
    return {"title": title, "code": {"coding": [{"system": _LOINC, "code": code}]}}


def _composition(*sections):
    return {"resourceType": "Composition", "status": "final", "section": list(sections)}


def _section_contract(code, target_profile, path="section.entry"):
    return _bundled_rule(
        sourceType="Composition",
        path=path,
        targetTypes=[target_profile],
        targetProfiles=[target_profile],
        match="all",
        selectors=[
            {
                "path": "section",
                "discriminator": "code",
                "value": {"coding": [{"system": _LOINC, "code": code}]},
            }
        ],
    )


def _specs(*rules):
    specs = []
    for rule in rules:
        BundleService._collect_todo_refs(rule, specs, {}, None)
    return specs


def test_slice_scoped_contracts_wire_into_their_own_section():
    """Each section must receive only the references its own slice requires.

    Sections are deliberately ordered so that neither the first nor the last
    element is the right answer for any contract — positional wiring cannot pass.
    """
    meds_profile = "http://example.org/StructureDefinition/medicationstatement"
    allergy_profile = "http://example.org/StructureDefinition/allergyintolerance"
    problem_profile = "http://example.org/StructureDefinition/condition"

    composition = _composition(
        _section("11450-4", "Problems"),
        _section("10160-0", "Medications"),
        _section("48765-2", "Allergies"),
    )
    medication = {"resourceType": "MedicationStatement",
                  "meta": {"profile": [meds_profile]}}
    allergy = {"resourceType": "AllergyIntolerance",
               "meta": {"profile": [allergy_profile]}}
    condition = {"resourceType": "Condition",
                 "meta": {"profile": [problem_profile]}}

    specs = _specs(
        _section_contract("10160-0", meds_profile),
        _section_contract("48765-2", allergy_profile),
        _section_contract("11450-4", problem_profile),
    )
    assert len(specs) == 3, "identical paths must stay distinct once scoped by slice"

    BundleService._wire_references(
        [composition, medication, allergy, condition],
        specs,
        {
            id(medication): "urn:uuid:med",
            id(allergy): "urn:uuid:allergy",
            id(condition): "urn:uuid:condition",
        },
        {},
        {"medicationstatement": "MedicationStatement",
         "allergyintolerance": "AllergyIntolerance",
         "condition": "Condition"},
    )

    by_title = {s["title"]: s for s in composition["section"]}
    assert by_title["Medications"]["entry"] == [{"reference": "urn:uuid:med"}]
    assert by_title["Allergies"]["entry"] == [{"reference": "urn:uuid:allergy"}]
    assert by_title["Problems"]["entry"] == [{"reference": "urn:uuid:condition"}]


def test_slice_selector_survives_a_structure_map_round_trip():
    profile = "http://example.org/StructureDefinition/condition"
    structure_map = {
        "structure": [{"mode": "target", "url": "http://example.org/comp"}],
        "group": [{"rule": [_section_contract("11450-4", profile)]}],
    }

    specs = BundleService._specs_from_structure_maps([structure_map])

    assert len(specs) == 1
    assert specs[0][1] == "section.entry"
    assert specs[0][11] == (
        ("section", "code", '{"coding":[{"code":"11450-4","system":"http://loinc.org"}]}'),
    )


def test_slice_selector_matching_no_container_wires_nothing():
    profile = "http://example.org/StructureDefinition/condition"
    composition = _composition(_section("10160-0", "Medications"))
    condition = {"resourceType": "Condition", "meta": {"profile": [profile]}}

    BundleService._wire_references(
        [composition, condition],
        _specs(_section_contract("11450-4", profile)),
        {id(condition): "urn:uuid:condition"},
        {},
        {"condition": "Condition"},
    )

    assert "entry" not in composition["section"][0]


def test_slice_selector_matching_several_containers_wires_nothing():
    """Two repeats satisfying one selector is ambiguous — guessing would be wrong."""
    profile = "http://example.org/StructureDefinition/condition"
    composition = _composition(
        _section("11450-4", "Problems"), _section("11450-4", "Problems again")
    )
    condition = {"resourceType": "Condition", "meta": {"profile": [profile]}}

    BundleService._wire_references(
        [composition, condition],
        _specs(_section_contract("11450-4", profile)),
        {id(condition): "urn:uuid:condition"},
        {},
        {"condition": "Condition"},
    )

    assert all("entry" not in section for section in composition["section"])


def test_slice_selector_matches_a_pattern_not_an_exact_value():
    """`pattern` discriminators constrain a subset — extra codings must still match."""
    profile = "http://example.org/StructureDefinition/condition"
    section = _section("11450-4", "Problems")
    section["code"]["text"] = "Problem list"
    section["code"]["coding"].insert(0, {"system": "http://snomed.info/sct", "code": "1"})
    composition = _composition(section)
    condition = {"resourceType": "Condition", "meta": {"profile": [profile]}}

    BundleService._wire_references(
        [composition, condition],
        _specs(_section_contract("11450-4", profile)),
        {id(condition): "urn:uuid:condition"},
        {},
        {"condition": "Condition"},
    )

    assert composition["section"][0]["entry"] == [{"reference": "urn:uuid:condition"}]


def test_contract_without_selectors_keeps_its_legacy_spec_shape():
    """Adding selectors must not change the tuple every existing consumer indexes."""
    specs = _specs(_bundled_rule())
    assert len(specs[0]) == 10


def test_slice_selector_on_a_missing_backbone_wires_nothing():
    """The sliced backbone may not have been generated at all."""
    profile = "http://example.org/StructureDefinition/condition"
    composition = {"resourceType": "Composition", "status": "final"}
    condition = {"resourceType": "Condition", "meta": {"profile": [profile]}}

    BundleService._wire_references(
        [composition, condition],
        _specs(_section_contract("11450-4", profile)),
        {id(condition): "urn:uuid:condition"},
        {},
        {"condition": "Condition"},
    )

    assert "section" not in composition


def test_malformed_selectors_reject_the_whole_contract():
    """A selector we cannot read must not degrade to first-element wiring."""
    import json

    rule = {
        "name": "TODO-resolve-reference-broken",
        "documentation": "FHIRBRIDGE_REFERENCE:" + json.dumps({
            "sourceType": "Composition",
            "path": "section.entry",
            "targetTypes": ["Condition"],
            "selectors": [{"path": "section"}],  # no discriminator, no value
        }),
    }
    specs = []
    BundleService._collect_todo_refs(rule, specs)
    assert specs == []


def test_extension_url_is_carried_from_the_owning_extension_element():
    # The parser records `extension_url` on the `…extension` element while the
    # required Reference lives on its `value[x]` child. Without carrying it down,
    # auto-wiring emits `{"valueReference": …}` with no `url` and the validator
    # rejects it ("Extension.url is required …") — dkrehab CarePlan.extension.
    fields = [
        {
            "path": "CarePlan.extension",
            "extension_url": "http://example.org/StructureDefinition/BasedOn",
            "cardinality": {"min": 1, "max": "1"},
            "children": [
                {
                    "path": "CarePlan.extension.value[x]",
                    "reference_target": "ServiceRequest",
                    "cardinality": {"min": 1, "max": "1"},
                    "type": [{"code": "Reference"}],
                }
            ],
        }
    ]
    out = list(BundleService._iter_reference_fields(fields))
    assert len(out) == 1
    relative, _raw, is_required, _ref_only, extension_url = out[0]
    assert relative == "extension.valueReference"
    assert is_required is True
    assert extension_url == "http://example.org/StructureDefinition/BasedOn"


def test_stamp_extension_url_fills_only_a_url_less_container():
    resource = {"extension": [{"valueReference": {"reference": "urn:uuid:1"}}]}
    BundleService._stamp_extension_url(
        resource, "extension.valueReference", "http://example.org/ext"
    )
    assert resource["extension"][0]["url"] == "http://example.org/ext"

    # an extension the transform already identified is left alone
    kept = {"extension": [{"url": "http://example.org/authored", "valueString": "x"}]}
    BundleService._stamp_extension_url(
        kept, "extension.valueString", "http://example.org/ext"
    )
    assert kept["extension"][0]["url"] == "http://example.org/authored"


def test_stamp_extension_url_targets_the_outermost_extension():
    # A complex extension wires as `extension.extension.valueReference`. The url in
    # hand is the *outer* extension's canonical (its sub-extensions are defined by
    # that extension's own SD), so the outer container is what must be stamped —
    # senll NLLDispensePaperPrescription reported `Extension.url is required` on it.
    resource = {"extension": [{"extension": [{"valueReference": {"reference": "urn:uuid:1"}}]}]}
    BundleService._stamp_extension_url(
        resource, "extension.extension.valueReference", "http://example.org/outer"
    )
    assert resource["extension"][0]["url"] == "http://example.org/outer"
    # the inner sub-extension is not given the outer's url
    assert "url" not in resource["extension"][0]["extension"][0]
