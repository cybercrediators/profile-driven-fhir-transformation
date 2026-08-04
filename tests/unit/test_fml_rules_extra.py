"""Unit tests for the Tier-4 correctness features that the offline golden corpus does not
exercise: meta.profile stamping (A), type-aware discriminator hints (D), and the
required-element coverage report (E)."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from mapping.fml_creator.fml_factory import FMLRuleFactory, _is_system_value_pseudo
from mapping.fml_map import StructureMapGenerator

pytestmark = pytest.mark.unit


@pytest.fixture
def factory():
    # These methods read no instance state, so bypass __init__.
    instance = object.__new__(FMLRuleFactory)
    instance.source_field_types = {}
    return instance


# ── Correctness-A: meta.profile stamping ──────────────────────────────────────
def test_meta_profile_rule_creates_meta_and_sets_profile(factory):
    url = "http://example.org/StructureDefinition/MyPatient"
    rule = factory.create_meta_profile_rule(url)
    # outer rule creates Meta on the target
    tgt = rule.target[0]
    assert tgt.element == "meta"
    assert tgt.transform == "create"
    assert tgt.parameter[0].valueString == "Meta"
    assert tgt.variable == "tgt-meta"
    # nested rule copies the profile url onto meta.profile
    nested = rule.rule[0]
    ntgt = nested.target[0]
    assert (ntgt.context, ntgt.element, ntgt.transform) == (
        "tgt-meta",
        "profile",
        "copy",
    )
    assert ntgt.parameter[0].valueString == url


def test_meta_profile_rule_respects_contexts(factory):
    rule = factory.create_meta_profile_rule(
        "u", source_context="src", target_context="tg"
    )
    assert rule.source[0].context == "src"
    assert rule.target[0].context == "tg"


@pytest.mark.parametrize(
    "path,field_type",
    [("Patient.id", "id"), ("Patient.language", "code")],
)
def test_explicit_base_primitive_mapping_is_emitted(factory, path, field_type):
    rule = factory.create_mappable_field_rule(
        {"path": path, "type": field_type},
        "Patient",
        "source",
        "target",
        automapped_mappings={path: f"Source.{path.rsplit('.', 1)[-1]}"},
    )
    assert rule is not None
    assert rule.target[0].element == path.rsplit(".", 1)[-1]
    assert rule.target[0].transform == "copy"


def test_mapped_string_emits_max_length_source_check(factory):
    rule = factory.create_mappable_field_rule(
        {
            "path": "Patient.identifier.value",
            "type": "string",
            "max_length": 12,
        },
        "Patient",
        "identifier",
        "target-identifier",
        automapped_mappings={"Patient.identifier.value": "Source.identifierValue"},
        parent_path="Patient.identifier",
    )
    assert rule.source[0].check == "($this.toString().length() <= 12)"


def test_mapped_numeric_emits_min_and_max_source_checks(factory):
    rule = factory.create_mappable_field_rule(
        {
            "path": "Observation.valueDecimal",
            "type": "decimal",
            "min_value": {"type": "Decimal", "value": "-2.5"},
            "max_value": {"type": "Decimal", "value": 10},
        },
        "Observation",
        "source",
        "target",
        automapped_mappings={"Observation.valueDecimal": "Source.measurement"},
    )
    assert rule.source[0].check == (
        "($this.toDecimal() >= -2.5) and ($this.toDecimal() <= 10)"
    )


def test_bounds_are_not_attached_to_todo_or_non_numeric_targets(factory):
    todo = factory.create_mappable_field_rule(
        {
            "path": "Patient.name",
            "type": "string",
            "max_length": 20,
        },
        "Patient",
        "source",
        "target",
    )
    assert todo.source[0].check is None

    date = factory.create_mappable_field_rule(
        {
            "path": "Patient.birthDate",
            "type": "date",
            "min_value": {"type": "Date", "value": "1900-01-01"},
        },
        "Patient",
        "source",
        "target",
        automapped_mappings={"Patient.birthDate": "Source.birthDate"},
    )
    assert date.source[0].check is None


def test_narrowed_choice_keeps_bound_check_on_real_provider(factory):
    rule = factory.create_mappable_field_rule(
        {
            "path": "Observation.value[x]",
            "type": [{"code": "decimal"}],
            "is_type_choice": True,
            "min_value": {"type": "Decimal", "value": 0},
        },
        "Observation",
        "source",
        "target",
        automapped_mappings={"Observation.value[x]:valueDecimal": "Source.measurement"},
    )
    narrowed = rule.rule[0]
    assert narrowed.source[0].element == "measurement"
    assert narrowed.source[0].check == "($this.toDecimal() >= 0)"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("Patient.contained", "contained-requires-typed-reference"),
        ("Patient.modifierExtension", "modifier-extension-requires-slice"),
    ],
)
def test_unsafe_unsliced_special_mapping_is_diagnosed(factory, path, expected):
    factory.diagnostics = []
    rule = factory.create_mappable_field_rule(
        {"path": path, "type": "Resource"},
        "Patient",
        "source",
        "target",
        automapped_mappings={path: "Source.value"},
    )
    assert rule is None
    assert factory.diagnostics[0]["code"] == expected


def test_explicit_xhtml_mapping_is_copied_and_diagnosed(factory):
    factory.diagnostics = []
    rule = factory.create_mappable_field_rule(
        {"path": "Patient.text.div", "type": "xhtml"},
        "Patient",
        "source",
        "narrative",
        automapped_mappings={"Patient.text.div": "Source.div"},
        parent_path="Patient.text",
    )
    assert rule.target[0].transform == "copy"
    assert factory.diagnostics[0]["code"] == "xhtml-content-authored"


def _display_only_reference_profile():
    return {
        "resourceType": "StructureDefinition",
        "type": "Procedure",
        "snapshot": {
            "element": [
                {"id": "Procedure", "path": "Procedure"},
                {
                    "id": "Procedure.subject",
                    "path": "Procedure.subject",
                    "min": 1,
                    "max": "1",
                    "type": [
                        {
                            "code": "Reference",
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Patient"
                            ],
                        }
                    ],
                },
                {
                    "id": "Procedure.subject.reference",
                    "path": "Procedure.subject.reference",
                    "min": 0,
                    "max": "0",
                    "type": [{"code": "string"}],
                },
                {
                    "id": "Procedure.subject.display",
                    "path": "Procedure.subject.display",
                    "min": 1,
                    "max": "1",
                    "type": [{"code": "string"}],
                },
            ]
        },
    }


def test_display_only_reference_suppresses_bundle_reference_placeholder(factory):
    factory.diagnostics = []
    factory._current_profile_sd = _display_only_reference_profile()
    factory._target_tree_cache = (None, None)
    field = {
        "path": "Procedure.subject",
        "type": [{"code": "Reference"}],
        "reference_target": "Patient",
        "cardinality": {"min": 1, "max": "1"},
    }

    rule = factory.create_mappable_field_rule(field, "Procedure", "source", "target")

    assert rule is None
    assert factory.diagnostics[-1]["code"] == (
        "reference-representation-requires-source"
    )


def test_display_only_reference_can_be_populated_from_authored_display(factory):
    factory.diagnostics = []
    factory._current_profile_sd = _display_only_reference_profile()
    factory._target_tree_cache = (None, None)
    field = {
        "path": "Procedure.subject",
        "type": [{"code": "Reference"}],
        "reference_target": "Patient",
        "cardinality": {"min": 1, "max": "1"},
    }

    rule = factory.create_mappable_field_rule(
        field,
        "Procedure",
        "source",
        "target",
        automapped_mappings={"Procedure.subject.display": "Source.patientLabel"},
    )

    assert rule.target[0].element == "subject"
    assert len(rule.rule) == 1
    assert rule.rule[0].source[0].element == "patientLabel"
    assert rule.rule[0].target[0].element == "display"
    assert "reference" not in rule.documentation


def test_unindexed_snapshot_slices_are_reported_once_per_profile(factory):
    factory.diagnostics = []
    factory._current_profile_id = "test-profile"
    sd = SimpleNamespace(
        snapshot=SimpleNamespace(
            element=[
                SimpleNamespace(
                    id="Patient.identifier:mrn",
                    path="Patient.identifier",
                    sliceName="mrn",
                    min=1,
                    max="1",
                ),
                SimpleNamespace(
                    id="Patient.identifier:known",
                    path="Patient.identifier",
                    sliceName="known",
                    min=0,
                    max="1",
                ),
                SimpleNamespace(
                    id="Patient.identifier:forbidden",
                    path="Patient.identifier",
                    sliceName="forbidden",
                    min=0,
                    max="0",
                ),
            ]
        )
    )

    factory.record_unindexed_snapshot_slices(
        sd,
        [
            {
                "id": "Patient.identifier",
                "slices": [{"id": "Patient.identifier:known"}],
            }
        ],
    )

    assert factory.diagnostics == [
        {
            "code": "snapshot-slices-not-indexed",
            "message": (
                "1 snapshot slice definition(s) were not retained as "
                "first-class parsed fields; owning constraints or explicit "
                "mappings may still recover them."
            ),
            "profile": "test-profile",
            "slice_ids": ["Patient.identifier:mrn"],
            "required_slice_ids": ["Patient.identifier:mrn"],
            "severity": "information",
        }
    ]


# ── Correctness-F: coded-leaf source resolution (binding-branch bug fix) ───────
# A coded field with an inherited/bound ValueSet routes through the translate /
# direct-copy branch. Users key the mapping at the *leaf* (code.coding.code), not
# at the CodeableConcept path, so the branch must check the leaf keys or it falls
# back to a TODO-SOURCE placeholder (the bug).
def test_coded_source_keys_include_codeableconcept_leaf(factory):
    field = {"path": "Observation.code", "type": "CodeableConcept"}
    keys = factory._coded_source_keys(field)
    assert keys[0] == "Observation.code"  # field path first (back-compat)
    assert "Observation.code.coding.code" in keys


def test_coded_source_keys_include_coding_leaf(factory):
    field = {"path": "Observation.value", "type": "Coding"}
    assert "Observation.value.code" in factory._coded_source_keys(field)


def test_resolve_coded_source_honours_leaf_keyed_mapping(factory):
    field = {"path": "Observation.code", "type": "CodeableConcept"}
    mappings = {"Observation.code.coding.code": "obsCode"}
    assert factory._resolve_coded_source(field, mappings) == "obsCode"


def test_resolve_coded_source_honours_field_path_mapping(factory):
    field = {"path": "Observation.code", "type": "CodeableConcept"}
    mappings = {"Observation.code": "obsCode"}
    assert factory._resolve_coded_source(field, mappings) == "obsCode"


def test_resolve_coded_source_returns_none_without_mapping(factory):
    field = {"path": "Observation.code", "type": "CodeableConcept"}
    assert factory._resolve_coded_source(field, {"Patient.gender": "g"}) is None


def _raw_codeable_concept(path, options=None):
    return {
        "path": path,
        "type": [
            {
                "code": "CodeableConcept",
                "type_structure": [
                    {
                        "path": f"{path}.coding",
                        "type": "Coding",
                        "type_structure": [
                            {"path": f"{path}.coding.code", "type": "code"},
                            {"path": f"{path}.coding.system", "type": "uri"},
                            {"path": f"{path}.coding.display", "type": "string"},
                        ],
                    },
                    {"path": f"{path}.text", "type": "string"},
                ],
            }
        ],
        "cardinality": {"min": 1, "max": "1"},
        "options": options or [],
        "valueSetUrl": "http://example.org/ValueSet/codes",
        "children": [],
    }


def test_raw_filter_bound_codeable_concept_uses_one_coded_builder(factory):
    field = _raw_codeable_concept(
        "Observation.code",
        [{"code": "FROM_CS", "system": "http://loinc.org"}],
    )

    rule = factory.create_mappable_field_rule(
        field,
        "Observation",
        "source",
        "target",
        automapped_mappings={"Observation.code.coding.code": "Source.obsCode"},
    )

    assert rule.name == "map-code-copy"
    assert rule.target[0].parameter[0].valueString == "CodeableConcept"
    assert len(rule.rule) == 1
    coding = rule.rule[0]
    assert coding.target[0].element == "coding"
    assert {child.target[0].element for child in coding.rule} == {"system", "code"}
    assert rule.source[0].element == "obsCode"


def test_unsliced_codeableconcept_populates_all_explicit_coding_leaves(factory):
    field = _raw_codeable_concept(
        "Procedure.code",
        [{"code": "FROM_CS", "system": "http://example.org/codes"}],
    )
    mappings = {
        "Procedure.code.coding.system": "Source.codeSystem",
        "Procedure.code.coding.code": "Source.code",
        "Procedure.code.coding.display": "Source.codeDisplay",
    }
    factory.explicit_mapping_targets = set(mappings)

    rule = factory.create_mappable_field_rule(
        field, "Procedure", "source", "target", automapped_mappings=mappings
    )

    assert rule.name == "map-code-authored-coded-leaves"
    coding = rule.rule[0]
    assert coding.target[0].element == "coding"
    assert {
        child.target[0].element: child.source[0].element for child in coding.rule
    } == {
        "system": "codeSystem",
        "code": "code",
        "display": "codeDisplay",
    }


def test_unsliced_codeableconcept_text_mapping_does_not_create_coding(factory):
    field = _raw_codeable_concept(
        "Goal.description",
        [{"code": "FROM_CS", "system": "http://example.org/codes"}],
    )
    mappings = {"Goal.description.text": "Source.goalText"}
    factory.explicit_mapping_targets = set(mappings)

    rule = factory.create_mappable_field_rule(
        field, "Goal", "source", "target", automapped_mappings=mappings
    )

    assert rule.name == "map-description-authored-coded-leaves"
    assert [child.target[0].element for child in rule.rule] == ["text"]
    assert rule.rule[0].source[0].element == "goalText"


def test_fixed_coding_leaves_take_precedence_over_example_binding(factory):
    field = _raw_codeable_concept(
        "ServiceRequest.serviceType",
        [{"system": "http://example.org", "code": "unrelated"}],
    )
    field["children"] = [
        {
            "path": "ServiceRequest.serviceType.coding",
            "children": [
                {
                    "path": "ServiceRequest.serviceType.coding.system",
                    "fixed_value": "http://snomed.info/sct",
                },
                {
                    "path": "ServiceRequest.serviceType.coding.code",
                    "fixed_value": "308335008",
                },
            ],
        }
    ]
    mappings = {"ServiceRequest.serviceType.coding.code": "Source.serviceType"}

    rule = factory.create_mappable_field_rule(
        field,
        "ServiceRequest",
        "source",
        "target",
        automapped_mappings=mappings,
    )

    assert rule.name == "map-serviceType-fixed-codings"
    coding_rules = rule.rule[0].rule
    literal_values = {
        child.target[0]
        .element: child.target[0]
        .parameter[0]
        .model_dump(exclude_none=True)
        for child in coding_rules
    }
    assert literal_values["system"] == {"valueString": "http://snomed.info/sct"}
    assert literal_values["code"] == {"valueString": "308335008"}
    assert all(child.target[0].transform == "copy" for child in coding_rules)


def test_reference_identifier_leaf_mappings_keep_reference_and_identifier_contexts(
    factory,
):
    factory.diagnostics = []
    field = {
        "path": "Procedure.subject",
        "type": [
            {
                "code": "Reference",
                "type_structure": [
                    {
                        "path": "Procedure.subject.identifier",
                        "type": "Identifier",
                        "type_structure": [
                            {
                                "path": "Procedure.subject.identifier.system",
                                "type": "uri",
                            },
                            {
                                "path": "Procedure.subject.identifier.value",
                                "type": "string",
                            },
                        ],
                    }
                ],
            }
        ],
        "reference_target": "Patient",
    }
    mappings = {
        "Procedure.subject.identifier.system": "Source.patientSystem",
        "Procedure.subject.identifier.value": "Source.patientId",
    }
    factory.explicit_mapping_targets = set(mappings)

    rule = factory.create_mappable_field_rule(
        field,
        "Procedure",
        "source",
        "target",
        automapped_mappings=mappings,
    )

    assert rule.target[0].element == "subject"
    identifier = rule.rule[0]
    assert identifier.target[0].context == rule.target[0].variable
    assert identifier.target[0].element == "identifier"
    assert {
        child.target[0].element: child.target[0].context for child in identifier.rule
    } == {
        "system": identifier.target[0].variable,
        "value": identifier.target[0].variable,
    }
    assert not any(child.target[0].context == "target" for child in identifier.rule)


def test_raw_bound_codeable_concept_without_provider_keeps_legacy_route(
    factory, monkeypatch
):
    field = _raw_codeable_concept(
        "Observation.code",
        [{"code": "FROM_CS", "system": "http://loinc.org"}],
    )

    def _unexpected_coded_route(*args, **kwargs):
        raise AssertionError("unmapped raw complex field must not change routes")

    monkeypatch.setattr(
        factory, "create_conditional_coding_rules", _unexpected_coded_route
    )
    rule = factory.create_mappable_field_rule(
        field,
        "Observation",
        "source",
        "target",
        automapped_mappings={},
    )

    assert rule.name == "map-code"


def test_raw_plain_complex_type_emits_mapped_embedded_children(factory):
    """Regression: non-sliced HumanName children disappeared after conv_mappable
    stopped mutating the registry field and no longer hoisted type_structure."""
    field = {
        "path": "Patient.name",
        "type": [
            {
                "code": "HumanName",
                "type_structure": [
                    {"path": "Patient.name.family", "type": "string"},
                    {"path": "Patient.name.given", "type": "string"},
                    {"path": "Patient.name.use", "type": "code"},
                ],
            }
        ],
        "cardinality": {"min": 1, "max": "*"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field,
        "Patient",
        "source",
        "target",
        automapped_mappings={
            "Patient.name.family": "Source.familyName",
            "Patient.name.given": "Source.givenName",
        },
    )

    assert rule.target[0].element == "name"
    assert rule.target[0].parameter[0].valueString == "HumanName"
    assert {nested.target[0].element for nested in rule.rule} == {"family", "given"}
    assert {nested.source[0].element for nested in rule.rule} == {
        "familyName",
        "givenName",
    }
    assert field.get("type_structure") is None  # generation view is non-mutating


def test_raw_sliced_codeable_concept_with_provider_keeps_slice_route(
    factory, monkeypatch
):
    field = _raw_codeable_concept(
        "Condition.code",
        [{"code": "FROM_CS", "system": "http://snomed.info/sct"}],
    )
    field["children"] = [
        {
            "path": "Condition.code.coding",
            "type": "Coding",
            "slices": [{"sliceName": "icd10-gm", "type": "Coding"}],
        }
    ]

    def _unexpected_coded_route(*args, **kwargs):
        raise AssertionError("sliced coded containers retain their slice route")

    monkeypatch.setattr(
        factory, "create_conditional_coding_rules", _unexpected_coded_route
    )
    rule = factory.create_mappable_field_rule(
        field,
        "Condition",
        "source",
        "target",
        automapped_mappings={"Condition.code.coding.code": "Source.diagnosisCode"},
    )

    assert rule.name == "map-code"


def test_raw_unexpandable_codeable_concept_copies_only_mapped_leaves(factory):
    field = _raw_codeable_concept("Task.code")
    mappings = {
        "Task.code.coding.code": "Source.taskCode",
        "Task.code.coding.system": "Source.taskSystem",
    }

    rule = factory.create_mappable_field_rule(
        field,
        "Task",
        "source",
        "target",
        automapped_mappings=mappings,
    )

    assert rule.name == "map-code"
    assert len(rule.rule) == 1
    coding = rule.rule[0]
    assert coding.target[0].element == "coding"
    leaves = {child.target[0].element: child for child in coding.rule}
    assert set(leaves) == {"code", "system"}
    assert leaves["code"].source[0].element == "taskCode"
    assert leaves["system"].source[0].element == "taskSystem"
    assert "TODO" not in str(rule.model_dump())


def test_raw_unexpandable_codeable_concept_does_not_invent_system(factory):
    field = _raw_codeable_concept("Task.code")

    rule = factory.create_mappable_field_rule(
        field,
        "Task",
        "source",
        "target",
        automapped_mappings={"Task.code.coding.code": "Source.taskCode"},
    )

    coding = rule.rule[0]
    assert [child.target[0].element for child in coding.rule] == ["code"]


def test_raw_enumerable_codeable_concept_does_not_also_recurse(factory, monkeypatch):
    field = _raw_codeable_concept(
        "Observation.code",
        [{"code": "1234-5", "system": "http://loinc.org"}],
    )
    monkeypatch.setattr(
        factory,
        "_generate_concept_map",
        lambda *args, **kwargs: "http://example.org/ConceptMap/code",
    )

    rule = factory.create_mappable_field_rule(
        field,
        "Observation",
        "source",
        "target",
        automapped_mappings={"Observation.code.coding.code": "Source.obsCode"},
    )

    assert rule.name == "map-code-translate"
    assert len(rule.rule) == 1
    assert rule.rule[0].name == "add-code-coding"


def test_raw_primitive_code_retains_scalar_route(factory, monkeypatch):
    field = {
        "path": "Observation.status",
        "type": [{"code": "code"}],
        "cardinality": {"min": 1, "max": "1"},
        "options": [{"code": "final", "system": "http://hl7.org/fhir"}],
        "children": [],
    }

    def _unexpected_coded_route(*args, **kwargs):
        raise AssertionError("raw primitive code must retain its scalar route")

    monkeypatch.setattr(
        factory, "create_conditional_coding_rules", _unexpected_coded_route
    )
    rule = factory.create_mappable_field_rule(
        field,
        "Observation",
        "source",
        "target",
        automapped_mappings={"Observation.status": "Source.status"},
    )

    assert rule.name == "map-status"
    assert rule.target[0].element == "status"
    assert rule.target[0].transform == "copy"


def test_exact_only_raw_choice_mapping_emits_concrete_primitive_target(factory):
    field = {
        "path": "MedicationStatement.effective[x]",
        "type": [{"code": "dateTime"}],
        "cardinality": {"min": 0, "max": "1"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field,
        "MedicationStatement",
        "source",
        "target",
        automapped_mappings={"MedicationStatement.effective[x]": "Source.medEffective"},
    )

    assert rule.name == "map-effective"
    assert rule.target == []
    assert rule.rule[0].target[0].element == "effectiveDateTime"
    assert rule.rule[0].target[0].transform == "cast"


def test_direct_raw_choice_mapping_narrows_from_source_primitive(factory):
    factory.source_field_types = {"allowed": "boolean"}
    field = {
        "path": "MedicationRequest.substitution.allowed[x]",
        "type": [{"code": "boolean"}, {"code": "CodeableConcept"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field,
        "MedicationRequest",
        "source",
        "target",
        automapped_mappings={
            "MedicationRequest.substitution.allowed[x]": "Source.allowed"
        },
        parent_path="MedicationRequest.substitution",
    )

    assert rule.rule[0].target[0].element == "allowedBoolean"
    assert rule.rule[0].target[0].transform == "copy"


def test_explicit_concrete_choice_mapping_overrides_ambiguous_source_type(factory):
    factory.source_field_types = {"allowed": "string"}
    field = {
        "path": "MedicationRequest.substitution.allowed[x]",
        "type": [{"code": "boolean"}, {"code": "CodeableConcept"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field,
        "MedicationRequest",
        "source",
        "target",
        automapped_mappings={
            "MedicationRequest.substitution.allowed[x]:allowedBoolean": "Source.allowed"
        },
        parent_path="MedicationRequest.substitution",
    )

    assert rule.rule[0].target[0].element == "allowedBoolean"
    assert rule.rule[0].target[0].transform == "copy"


def test_concrete_choice_parent_does_not_gate_on_a_todo_placeholder(factory):
    """The parent's source element is resolved from the bare ``[x]`` path, so a table
    addressing the concrete choice leaves it on the generated ``TODO_MAP_*`` fallback.
    A nested rule only runs when its parent matches, so leaving it there means
    ``$transform`` succeeds while the required choice element is silently absent
    (medikation 5/5 -> 3/5, biobank 11/11 -> 6/11 against live Matchbox)."""
    factory.source_field_types = {"statementEffective": "dateTime"}
    field = {
        "path": "MedicationStatement.effective[x]",
        "type": [{"code": "dateTime"}, {"code": "Period"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field,
        "MedicationStatement",
        "source",
        "target",
        automapped_mappings={
            "MedicationStatement.effective[x]:effectiveDateTime": (
                "Source.statementEffective"
            )
        },
    )

    parent_source = rule.source[0]
    assert (
        getattr(parent_source, "element", None) is None
    ), "a TODO placeholder on the parent gates the populated child"
    nested_source = rule.rule[0].source[0]
    assert nested_source.element == "statementEffective"
    assert rule.rule[0].target[0].element == "effectiveDateTime"


def test_scaffold_choice_without_a_provider_keeps_its_todo_source(factory):
    """The ungating must not strip scaffolding: with no provider there is no populated
    child to protect, and the TODO source is the author's fill-in point."""
    field = {
        "path": "MedicationStatement.effective[x]",
        "type": [{"code": "dateTime"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field, "MedicationStatement", "source", "target", automapped_mappings={}
    )

    if rule is not None:
        elements = [getattr(s, "element", None) for s in rule.source or []]
        assert any(
            isinstance(e, str) and e.startswith("TODO") for e in elements
        ), "an unmapped choice must remain a fillable scaffold"


def test_explicit_concrete_complex_choice_can_be_copied_as_a_whole(factory):
    factory.source_field_types = {"period": "string"}
    field = {
        "path": "Observation.effective[x]",
        "type": [{"code": "dateTime"}, {"code": "Period"}],
        "cardinality": {"min": 0, "max": "1"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field,
        "Observation",
        "source",
        "target",
        automapped_mappings={
            "Observation.effective[x]:effectivePeriod": "Source.period"
        },
    )

    target = rule.rule[0].target[0]
    assert target.element == "effectivePeriod"
    assert target.transform == "copy"
    assert target.parameter[0].valueId == "src-choiceval"


def test_explicit_choice_mapping_rejects_type_not_allowed_by_profile(factory, caplog):
    field = {
        "path": "Observation.effective[x]",
        "type": [{"code": "dateTime"}, {"code": "Period"}],
        "cardinality": {"min": 0, "max": "1"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field,
        "Observation",
        "source",
        "target",
        automapped_mappings={
            "Observation.effective[x]:effectiveQuantity": "Source.value"
        },
    )

    assert rule is None
    assert "unsupported choice type Quantity" in caplog.text


def test_direct_raw_choice_mapping_rejects_ambiguous_source_type(factory, caplog):
    factory.source_field_types = {"allowed": "string"}
    field = {
        "path": "MedicationRequest.substitution.allowed[x]",
        "type": [{"code": "boolean"}, {"code": "CodeableConcept"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [],
    }

    rule = factory.create_mappable_field_rule(
        field,
        "MedicationRequest",
        "source",
        "target",
        automapped_mappings={
            "MedicationRequest.substitution.allowed[x]": "Source.allowed"
        },
        parent_path="MedicationRequest.substitution",
    )

    assert rule is None
    assert "ambiguous" in caplog.text


def test_choice_resolution_uses_snapshot_narrowing_not_base_candidate_list(factory):
    factory._current_profile_sd = {
        "type": "Observation",
        "snapshot": {
            "element": [
                {"id": "Observation", "path": "Observation"},
                {
                    "id": "Observation.value[x]",
                    "path": "Observation.value[x]",
                    "type": [{"code": "Quantity"}],
                },
            ]
        },
    }
    factory._target_tree_cache = (None, None)
    field = {
        "id": "Observation.value[x]",
        "path": "Observation.value[x]",
        "type": [
            {"code": "string"},
            {"code": "boolean"},
            {"code": "Quantity"},
        ],
        "cardinality": {"min": 0, "max": "1"},
        "children": [],
    }

    assert (
        factory._resolve_choice_element(
            field,
            "value",
            "Observation.value[x]",
            {"Observation.value[x]": "Source.result"},
        )
        == "valueQuantity"
    )


@pytest.mark.parametrize(
    ("field_type", "fixed_value", "attribute", "literal"),
    [
        ("boolean", False, "valueBoolean", False),
        ("integer", 0, "valueInteger", 0),
        ("decimal", 0, "valueDecimal", Decimal("0")),
    ],
)
def test_falsy_fixed_primitive_is_emitted(
    factory, field_type, fixed_value, attribute, literal
):
    rule = factory.create_mappable_field_rule(
        {
            "path": "Observation.value[x]",
            "type": field_type,
            "fixed_value": fixed_value,
            "cardinality": {"min": 1, "max": "1"},
        },
        "Observation",
        "source",
        "target",
    )

    assert rule.target[0].element == f"value{factory._choice_suffix(field_type)}"
    parameter = rule.target[0].parameter[0]
    assert getattr(parameter, attribute) == literal
    assert parameter.valueString is None


def test_raw_constrained_coded_choice_merges_coding_leaf_rules(factory):
    field = _raw_codeable_concept("Observation.value[x]")
    mappings = {
        "Observation.value[x]": "Source.resultCode",
        "Observation.value[x].coding.code": "Source.resultCode",
        "Observation.value[x].coding.system": "Source.resultSystem",
    }

    rule = factory.create_mappable_field_rule(
        field,
        "Observation",
        "source",
        "target",
        automapped_mappings=mappings,
    )

    assert rule.target == []
    assert len(rule.rule) == 1
    value_rule = rule.rule[0]
    assert value_rule.target[0].parameter[0].valueString == "CodeableConcept"
    assert len(value_rule.rule) == 1
    coding = value_rule.rule[0]
    assert coding.target[0].element == "coding"
    assert {child.target[0].element for child in coding.rule} == {"code", "system"}


# ── Fixed coding.system wins over the binding-derived system ───────────────────
# A profile that pins `<coded field>.coding.system` (fixedUri/patternUri) states the
# target code system explicitly; it must override the system inferred from the bound
# ValueSet's compose, else a direct copy emits the wrong system.
FIXED_SYS = "http://example.org/CodeSystem/my-codes"


def _coded_cc_field_with_fixed_system(system):
    return {
        "path": "Observation.code",
        "type": "CodeableConcept",
        "children": [
            {
                "path": "Observation.code.coding",
                "children": [
                    {"path": "Observation.code.coding.code", "fixed_value": []},
                    {"path": "Observation.code.coding.system", "fixed_value": system},
                ],
            }
        ],
    }


def test_fixed_system_from_field_returns_fixed_uri(factory):
    field = _coded_cc_field_with_fixed_system(FIXED_SYS)
    assert factory._fixed_system_from_field(field) == FIXED_SYS


def test_fixed_system_from_field_none_when_unfixed(factory):
    field = _coded_cc_field_with_fixed_system([])  # empty fixed_value = not fixed
    assert factory._fixed_system_from_field(field) is None


def test_fixed_system_from_field_none_without_coding_child(factory):
    field = {"path": "Patient.gender", "type": "code"}
    assert factory._fixed_system_from_field(field) is None


def test_fixed_system_short_circuits_to_direct_copy(factory, monkeypatch):
    # With a fixed coding.system AND a source-mapped code, the dispatch must emit a
    # direct copy carrying the fixed system — not a $translate/ConceptMap rule.
    field = _coded_cc_field_with_fixed_system(FIXED_SYS)
    monkeypatch.setattr(factory, "_resolve_coded_source", lambda f, m: "src-code")
    captured = {}

    def _fake_direct_copy(f, fname, psc, ptc, system_uri, am):
        captured["system"] = system_uri
        return "DIRECT_COPY"

    monkeypatch.setattr(factory, "_create_direct_copy_rule", _fake_direct_copy)
    out = factory.create_conditional_coding_rules(
        field,
        "code",
        "src",
        "tgt",
        automapped_mappings={"Observation.code.coding.code": "src-code"},
    )
    assert out == "DIRECT_COPY"
    assert captured["system"] == FIXED_SYS


# ── Slice discriminator emitted from the slice's fixed child (e.g. property:weight.type) ──
def test_slice_child_fixed_rules_emits_discriminator(factory):
    children = [
        {
            "path": "DeviceDefinition.property.type",
            "type": [{"code": "CodeableConcept"}],
            "fixed_value": {
                "coding": [{"system": "http://example.org/cs", "code": "X1"}]
            },
        },
        # value element carries no fixed value -> must be ignored here
        {
            "path": "DeviceDefinition.property.valueQuantity",
            "type": [{"code": "Quantity"}],
            "fixed_value": [],
        },
    ]
    rules = factory._slice_child_fixed_rules(
        children, "tgt-prop-weight", "source", "property-weight"
    )
    assert len(rules) == 1  # only the fixed `.type` discriminator
    tgt = rules[0].target[0]
    assert tgt.element == "type"
    assert tgt.transform == "create"
    assert tgt.parameter[0].valueString == "CodeableConcept"
    # nested pattern pins the coding system + code
    dumped = str(rules[0].model_dump())
    assert "X1" in dumped and "example.org/cs" in dumped


# ── Fully-pinned plain CodeableConcept (system+code fixed, no source) emits as fixed coding ──
def _cc_field_with_fixed_coding(system=None, code=None):
    leaves = []
    if system is not None:
        leaves.append({"path": "Procedure.code.coding.system", "fixed_value": system})
    if code is not None:
        leaves.append({"path": "Procedure.code.coding.code", "fixed_value": code})
    return {
        "path": "Procedure.code",
        "type": "CodeableConcept",
        "children": [{"path": "Procedure.code.coding", "children": leaves}],
    }


# ── Primitive .value System.* pseudo-element is never emitted ──────────────────
def test_system_value_pseudo_detected():
    # the `.value` child FHIR adds under a primitive with children (e.g. birthDate.value)
    assert _is_system_value_pseudo(
        {
            "path": "Patient.birthDate.value",
            "type": [{"code": "http://hl7.org/fhirpath/System.Date"}],
        }
    )
    assert _is_system_value_pseudo(
        {
            "path": "Observation.effective.value",
            "type": "http://hl7.org/fhirpath/System.DateTime",
        }
    )


def test_system_value_pseudo_ignores_real_value_and_siblings():
    # a real value[x]/value element (FHIR datatype) and .extension/.id are NOT the pseudo-element
    assert not _is_system_value_pseudo(
        {"path": "Observation.valueQuantity.value", "type": [{"code": "decimal"}]}
    )
    assert not _is_system_value_pseudo(
        {"path": "Patient.birthDate.extension", "type": [{"code": "Extension"}]}
    )


def test_fixed_coding_from_leaves_requires_both_system_and_code(factory):
    full = _cc_field_with_fixed_coding("http://example.org/cs", "ABC")
    coding = factory._fixed_coding_from_leaves(full)
    assert coding == {"system": "http://example.org/cs", "code": "ABC"}
    # only system fixed -> not a fully-pinned coding
    assert (
        factory._fixed_coding_from_leaves(
            _cc_field_with_fixed_coding("http://example.org/cs")
        )
        is None
    )
    # neither fixed
    assert (
        factory._fixed_coding_from_leaves({"path": "Procedure.code", "children": []})
        is None
    )


# ── Correctness-D: type-aware discriminator hint ──────────────────────────────
@pytest.mark.parametrize("disc_type", ["value", "pattern"])
def test_discriminator_hint_emitted_for_value_based(factory, disc_type):
    slice_info = {
        "discriminator": {
            "path": "system",
            "type": disc_type,
            "value": "http://example.org/system",
        }
    }
    rule = factory.create_discriminator_hint_rule(slice_info, "src", "tgt")
    assert rule is not None
    assert rule.target[0].element == "system"
    assert rule.target[0].parameter[0].valueString == "http://example.org/system"
    assert rule.source[0].element is None


def test_discriminator_hint_without_value_is_diagnostic_not_todo(factory):
    slice_info = {"discriminator": {"path": "system", "type": "value"}}
    assert factory.create_discriminator_hint_rule(slice_info, "src", "tgt") is None
    assert factory.diagnostics[-1]["code"] == "missing-discriminator-provider"


@pytest.mark.parametrize("disc_type", ["type", "exists", "profile", "position"])
def test_discriminator_hint_skipped_for_non_value_based(factory, disc_type):
    # Non-value-based discriminators have no forward-generation value rule — skip them.
    slice_info = {"discriminator": {"path": "system", "type": disc_type}}
    assert factory.create_discriminator_hint_rule(slice_info, "src", "tgt") is None


def test_discriminator_hint_none_without_discriminator(factory):
    assert factory.create_discriminator_hint_rule({}, "src", "tgt") is None


# ── Correctness-E: required-element coverage report ───────────────────────────
def _gen():
    gen = object.__new__(StructureMapGenerator)
    gen._coverage = {}
    return gen


def _obj(profile_id, fields):
    return SimpleNamespace(data=SimpleNamespace(id=profile_id), mappable_fields=fields)


def test_coverage_counts_required_and_flags_unmapped():
    gen = _gen()
    fields = [
        {
            "path": "Patient.identifier",
            "cardinality": {"min": 1, "max": "1"},
        },  # required, unmapped
        {"path": "Patient.name", "is_required": True},  # required, mapped
        {"path": "Patient.gender", "cardinality": {"min": 0}},  # optional, ignored
    ]
    gen._record_coverage(_obj("p1", fields), "Patient", {"Patient.name"})
    cov = gen._coverage["p1"]
    assert cov["required_total"] == 2
    assert cov["required_unmapped"] == 1
    assert cov["unmapped_required_paths"] == ["Patient.identifier"]


def test_coverage_fixed_value_counts_as_covered():
    gen = _gen()
    fields = [
        {"path": "Observation.status", "is_required": True, "fixed_value": "final"}
    ]
    gen._record_coverage(_obj("p2", fields), "Observation", set())
    assert gen._coverage["p2"]["required_unmapped"] == 0


def test_coverage_mapped_descendant_covers_required_container():
    gen = _gen()
    # a required container is considered covered when a descendant is mapped
    fields = [
        {
            "path": "Patient.identifier",
            "is_required": True,
            "children": [{"path": "Patient.identifier.value", "is_required": True}],
        }
    ]
    gen._record_coverage(_obj("p3", fields), "Patient", {"Patient.identifier.value"})
    cov = gen._coverage["p3"]
    # both the container and the mapped child count as covered
    assert cov["required_unmapped"] == 0
    assert cov["required_total"] == 2


def test_coverage_fixed_descendant_covers_required_container():
    gen = _gen()
    # a required container whose descendant carries a fixed value is populated by the
    # generator's create+fixed rule chain (e.g. identifier.type.coding fixed system+code)
    fields = [
        {
            "path": "Patient.identifier.type",
            "is_required": True,
            "fixed_value": [],  # containers carry an empty (falsy) fixed_value
            "children": [
                {
                    "path": "Patient.identifier.type.coding",
                    "is_required": True,
                    "fixed_value": [],
                    "children": [
                        {
                            "path": "Patient.identifier.type.coding.code",
                            "is_required": True,
                            "fixed_value": "MIN",
                        }
                    ],
                }
            ],
        }
    ]
    gen._record_coverage(_obj("p4", fields), "Patient", set())
    cov = gen._coverage["p4"]
    assert cov["required_total"] == 3
    assert cov["required_unmapped"] == 0


def _snapshot_element(element_id, path, min_cardinality, **kwargs):
    return SimpleNamespace(
        id=element_id,
        path=path,
        min=min_cardinality,
        max=kwargs.pop("max", "*"),
        sliceName=kwargs.pop("sliceName", None),
        **kwargs,
    )


def test_snapshot_coverage_is_slice_aware_and_separates_active_from_latent():
    """Regression for KDS Person's partially populated Strassenanschrift slice."""
    elements = [
        _snapshot_element("Patient", "Patient", 0),
        _snapshot_element("Patient.address", "Patient.address", 0),
        _snapshot_element(
            "Patient.address:Strassenanschrift",
            "Patient.address",
            0,
            sliceName="Strassenanschrift",
        ),
        _snapshot_element(
            "Patient.address:Strassenanschrift.line", "Patient.address.line", 1
        ),
        _snapshot_element(
            "Patient.address:Strassenanschrift.country", "Patient.address.country", 1
        ),
        _snapshot_element(
            "Patient.address:Strassenanschrift.city", "Patient.address.city", 1
        ),
        _snapshot_element(
            "Patient.address:Strassenanschrift.postalCode",
            "Patient.address.postalCode",
            1,
        ),
        _snapshot_element("Patient.communication", "Patient.communication", 0),
        _snapshot_element(
            "Patient.communication.language", "Patient.communication.language", 1
        ),
        _snapshot_element("Patient.identifier", "Patient.identifier", 1),
        _snapshot_element(
            "Patient.identifier:pid",
            "Patient.identifier",
            1,
            sliceName="pid",
        ),
        _snapshot_element(
            "Patient.identifier:pid.system", "Patient.identifier.system", 1
        ),
        _snapshot_element(
            "Patient.identifier:pid.value", "Patient.identifier.value", 1
        ),
    ]
    obj = SimpleNamespace(
        data=SimpleNamespace(
            id="kds-person",
            snapshot=SimpleNamespace(element=elements),
        ),
        # Deliberately incomplete: snapshot-derived coverage must not depend on it.
        mappable_fields=[
            {"path": "Patient.communication.language", "is_required": True}
        ],
    )
    mapped = {
        "Patient.address:Strassenanschrift.city",
        "Patient.address:Strassenanschrift.postalCode",
        "Patient.identifier:pid.value",
    }

    gen = _gen()
    gen._record_coverage(obj, "Patient", mapped)
    cov = gen._coverage["kds-person"]

    assert cov["static_required_total"] == 9
    assert cov["required_total"] == 8
    assert cov["latent_required_paths"] == ["Patient.communication.language"]
    assert cov["unmapped_required_paths"] == [
        "Patient.address:Strassenanschrift.line",
        "Patient.address:Strassenanschrift.country",
        "Patient.identifier:pid.system",
    ]


def test_snapshot_coverage_counts_profile_fixed_provider():
    elements = [
        _snapshot_element("Observation", "Observation", 0),
        _snapshot_element(
            "Observation.status", "Observation.status", 1, fixedCode="final"
        ),
    ]
    obj = SimpleNamespace(
        data=SimpleNamespace(
            id="fixed-observation", snapshot=SimpleNamespace(element=elements)
        ),
        mappable_fields=[],
    )

    gen = _gen()
    gen._record_coverage(obj, "Observation", set())

    entry = gen._coverage["fixed-observation"]["requirement_manifest"][0]
    assert entry["provider"] == "profile-fixed"
    assert entry["status"] == "covered"


def test_snapshot_coverage_expands_partial_pattern_on_slice_root():
    elements = [
        _snapshot_element("Patient", "Patient", 0),
        _snapshot_element("Patient.address", "Patient.address", 0),
        _snapshot_element(
            "Patient.address:street",
            "Patient.address",
            0,
            sliceName="street",
            patternAddress={"type": "both"},
        ),
        _snapshot_element("Patient.address:street.type", "Patient.address.type", 1),
        _snapshot_element("Patient.address:street.line", "Patient.address.line", 1),
    ]
    obj = SimpleNamespace(
        data=SimpleNamespace(
            id="pattern-address", snapshot=SimpleNamespace(element=elements)
        ),
        mappable_fields=[],
    )

    gen = _gen()
    gen._record_coverage(obj, "Patient", {"Patient.address:street.city"})

    manifest = {
        entry["id"]: entry
        for entry in gen._coverage["pattern-address"]["requirement_manifest"]
    }
    assert manifest["Patient.address:street.type"]["provider"] == "profile-fixed"
    assert manifest["Patient.address:street.line"]["provider"] == "missing"


# ── Correctness-A guard: meta.profile only when the snapshot declares meta ────
def _obj_with_snapshot(res_type, snapshot_paths):
    """A res_obj whose StructureDefinition snapshot declares the given paths."""
    elements = [SimpleNamespace(path=p) for p in snapshot_paths]
    return SimpleNamespace(
        data=SimpleNamespace(
            id=f"{res_type}-profile",
            url=f"http://example.org/{res_type}",
            snapshot=SimpleNamespace(element=elements),
        )
    )


def test_meta_emitted_when_snapshot_declares_meta():
    gen = _gen()
    obj = _obj_with_snapshot("Patient", ["Patient", "Patient.meta", "Patient.name"])
    assert gen._target_declares_meta(obj, "Patient") is True


def test_meta_skipped_when_snapshot_omits_meta():
    # minimal/incomplete snapshot (like the example2 Organization profile) — matchbox
    # would reject a meta rule with "Unrecognised name meta on <Type>".
    gen = _gen()
    obj = _obj_with_snapshot("Organization", ["Organization", "Organization.name"])
    assert gen._target_declares_meta(obj, "Organization") is False


def test_meta_emitted_when_no_snapshot_present():
    # no snapshot → matchbox generates a complete one at load time → safe to emit
    gen = _gen()
    obj = SimpleNamespace(data=SimpleNamespace(id="x", url="u", snapshot=None))
    assert gen._target_declares_meta(obj, "Observation") is True


# ── Reference subtree: the profile's children, not the base type expansion ────
#
# ``type_structure`` is the base FHIR Reference expanded from fhir.resources, so it
# carries no profile constraints. The profile's own fixed values (and its max=0
# prunings) live in ``children``. Reading the wrong one silently drops e.g. a
# ``fixedUri`` on ``practitioner.identifier.system`` or a ``fixedString`` on
# ``subject.display``, which then fail validation as missing required elements.
def _reference_factory():
    factory = object.__new__(FMLRuleFactory)
    factory.source_field_types = {}
    factory.source_field_max = {}
    factory.collection_rules = {}
    factory.diagnostics = []
    factory._current_profile_id = None
    factory._current_profile_sd = None
    factory._target_tree_cache = (None, None)
    captured = {}

    def _capture(_type, provided, **kwargs):
        captured["provided"] = provided
        return []

    factory.create_field_rules = _capture
    factory._apply_repeating_list_modes = lambda *a, **k: None
    return factory, captured


def _rule():
    from fhir.resources.R4B.structuremap import StructureMapGroupRule

    return StructureMapGroupRule.model_construct(name="map-subject")


def _paths(provided):
    return {c["path"] for c in provided}


def test_reference_subtree_prefers_the_profile_children_over_the_base_expansion():
    factory, captured = _reference_factory()
    field = {
        "path": "Procedure.subject",
        # base Reference expansion: no fixed values anywhere
        "type_structure": [{"path": "Procedure.subject.reference", "fixed_value": []}],
        # profile view: subject.display is pinned, the rest is prohibited (max=0)
        # and has already been pruned away by the element parser
        "children": [
            {"path": "Procedure.subject.display", "fixed_value": "Versicherter"},
            {"path": "Procedure.subject.extension", "fixed_value": []},
        ],
    }
    factory._reference_descendant_rule(
        _rule(), field, "subject", "subject", "source", "target", {}
    )
    assert _paths(captured["provided"]) == {"Procedure.subject.display"}


def test_reference_subtree_keeps_a_fixed_value_nested_under_identifier():
    factory, captured = _reference_factory()
    field = {
        "path": "PractitionerRole.practitioner",
        "children": [
            {
                "path": "PractitionerRole.practitioner.identifier",
                "fixed_value": [],
                "children": [
                    {
                        "path": "PractitionerRole.practitioner.identifier.system",
                        "fixed_value": "http://fhir.de/NamingSystem/kbv/lanr",
                    },
                    {
                        "path": "PractitionerRole.practitioner.identifier.value",
                        "fixed_value": [],
                    },
                ],
            }
        ],
    }
    factory._reference_descendant_rule(
        _rule(),
        field,
        "practitioner",
        "practitioner",
        "source",
        "target",
        {"PractitionerRole.practitioner.identifier.value": "Src.practIdValue"},
    )
    identifier = captured["provided"][0]
    assert identifier["path"] == "PractitionerRole.practitioner.identifier"
    kept = {c["path"] for c in identifier["children"]}
    # the authored .value AND the profile-fixed .system both survive
    assert kept == {
        "PractitionerRole.practitioner.identifier.system",
        "PractitionerRole.practitioner.identifier.value",
    }


# ── meta.profile precedence: a pinned canonical outranks the SD url ───────────
def _sd_obj(url, elements):
    return SimpleNamespace(
        data=SimpleNamespace(
            id="x", url=url,
            snapshot=SimpleNamespace(
                element=[SimpleNamespace(**e) for e in elements])))


def test_meta_profile_prefers_a_canonical_pinned_on_the_snapshot():
    # evo13 pins the http:// form on an SD whose own canonical is https://.
    gen = _gen()
    obj = _sd_obj(
        "https://fhir.element44.de/E44_EVO13_PR_CorrectionRequest",
        [{"path": "Communication.meta"},
         {"path": "Communication.meta.profile",
          "fixedCanonical": "http://fhir.element44.de/E44_EVO13_PR_CorrectionRequest"}],
    )
    assert gen._snapshot_fixed_meta_profile(obj, "Communication") == (
        "http://fhir.element44.de/E44_EVO13_PR_CorrectionRequest")


def test_meta_profile_falls_back_to_the_sd_url_when_nothing_is_pinned():
    gen = _gen()
    obj = _sd_obj("http://example.org/StructureDefinition/P",
                  [{"path": "Patient.meta"}, {"path": "Patient.meta.profile"}])
    assert gen._snapshot_fixed_meta_profile(obj, "Patient") is None


# ── Deferred references below a sliced repeating backbone (ontario) ───────────
#
# `Composition.section` is sliced on a `code` pattern; each section slice carries a
# required `entry` sub-slice discriminated by `profile:resolve()`. The runtime path
# of every one of them is `section.entry`, so the contract has to say which section
# it means or six contracts collapse into one write against `section[0]`.

_LOINC = "http://loinc.org"


def _entry_child(slice_name, entry_slice, target_profiles):
    return {
        "path": "Composition.section.entry",
        "id": f"Composition.section:{slice_name}.entry",
        "cardinality": {"min": 1, "max": "*"},
        "reference_target": "http://hl7.org/fhir/StructureDefinition/Resource|4.0.1",
        "type": [{"code": "Reference"}],
        "slicing": {"discriminators": [{"path": "resolve()", "type": "profile"}]},
        "slices": [
            {
                "path": "Composition.section.entry",
                "id": f"Composition.section:{slice_name}.entry:{entry_slice}",
                "sliceName": entry_slice,
                "cardinality": {"min": 1, "max": "*"},
                "type": [{"code": "Reference", "targetProfile": list(target_profiles)}],
            }
        ],
    }


def _section_slice(slice_name, code, entry_slice, target_profiles, pinned_code=True):
    code_child = {
        "path": "Composition.section.code",
        "id": f"Composition.section:{slice_name}.code",
        "cardinality": {"min": 1, "max": "1"},
        "type": [{"code": "CodeableConcept"}],
    }
    if pinned_code:
        code_child["fixed_value"] = {"coding": [{"system": _LOINC, "code": code}]}
        code_child["fixed_kind"] = "pattern"
    return {
        "path": "Composition.section",
        "id": f"Composition.section:{slice_name}",
        "sliceName": slice_name,
        "slice_identity": f"Composition.section:{slice_name}",
        "cardinality": {"min": 1, "max": "1"},
        "type": [{"code": "BackboneElement"}],
        "children": [code_child, _entry_child(slice_name, entry_slice, target_profiles)],
    }


def _section_parent():
    return {
        "path": "Composition.section",
        "cardinality": {"min": 0, "max": "*"},
        "type": [{"code": "BackboneElement"}],
        "slicing": {
            "discriminators": [{"path": "code", "type": "pattern"}],
            "rules": "open",
        },
    }


def _contracts(rule):
    import json

    found = []

    def walk(node):
        doc = node.documentation or ""
        marker = "FHIRBRIDGE_REFERENCE:"
        if doc.startswith(marker):
            found.append(json.loads(doc[len(marker):]))
        for child in node.rule or []:
            walk(child)

    walk(rule)
    return found


def test_reference_under_a_sliced_backbone_is_scoped_to_its_own_slice(factory):
    factory.diagnostics = []
    parent = _section_parent()
    slice_field = _section_slice(
        "sectionMedications", "10160-0", "medicationInformation",
        ["http://ontariohealth.ca/fhir/StructureDefinition/ca-on-ps-profile-medicationstatement",
         "http://ontariohealth.ca/fhir/StructureDefinition/ca-on-ps-profile-medicationrequest"],
    )
    rule = factory._create_slice_instance_rule(
        parent, slice_field, "Composition", "source", "target", {},
    )
    contracts = _contracts(rule)
    assert len(contracts) == 1, "sliced reference did not produce a correlated contract"
    contract = contracts[0]
    assert contract["path"] == "section.entry"
    assert contract["selectors"] == [
        {
            "path": "section",
            "discriminator": "code",
            "value": {"coding": [{"system": _LOINC, "code": "10160-0"}]},
        }
    ]
    # Both profiles the slice accepts must be offered, not just the first.
    assert len(contract["targetProfiles"]) == 2


def test_sibling_sections_produce_distinguishable_contracts(factory):
    """Identical contracts are deduped by `_specs_from_structure_maps`."""
    factory.diagnostics = []
    parent = _section_parent()
    rules = [
        factory._create_slice_instance_rule(
            parent,
            _section_slice(name, code, entry, ["http://example.org/p"]),
            "Composition", "source", "target", {},
        )
        for name, code, entry in (
            ("sectionMedications", "10160-0", "medicationInformation"),
            ("sectionAllergies", "48765-2", "allergyOrIntolerance"),
        )
    ]
    contracts = [c for rule in rules for c in _contracts(rule)]
    assert len(contracts) == 2
    assert contracts[0]["selectors"] != contracts[1]["selectors"]
    assert contracts[0]["path"] == contracts[1]["path"] == "section.entry"


def test_sliced_reference_with_no_target_profile_is_reported(factory):
    """A required entry slice with no target profile cannot be correlated.

    The fallback is an unscoped reference, which cannot satisfy the profile
    slices — so the gap is reported rather than left to look like success.
    (An *unpinned* discriminator never reaches here: the slice itself is
    rejected earlier as `ambiguous-slice-selection`.)
    """
    factory.diagnostics = []
    parent = _section_parent()
    slice_field = _section_slice(
        "sectionMedications", "10160-0", "medicationInformation", [],
    )
    rule = factory._create_slice_instance_rule(
        parent, slice_field, "Composition", "source", "target", {},
    )
    assert _contracts(rule) == []
    assert any(
        d.get("code") == "sliced-reference-not-deferrable"
        for d in factory.diagnostics
    )


def test_fixed_complex_value_on_a_choice_element_keeps_the_x_base(factory):
    """`create` names the type, so the element must not repeat it.

    Emitting element `valueCodeableConcept` together with `create CodeableConcept`
    makes the engine look for `valueCodeableConceptCodeableConcept` and abort the
    whole transform (matchbox: "Unable to find type ... with path Extension.value[x]").
    """
    rule = factory._create_fixed_value_rule(
        "CodeableConcept",
        {"coding": [{"system": "http://snomed.info/sct", "code": "420134006"}]},
        "value[x]",
        "ext-type",
        "source",
    )
    target = rule.target[0]
    assert target.element == "value"
    assert target.transform == "create"
    assert target.parameter[0].valueString == "CodeableConcept"
    # the rule/variable names still carry the concrete type for readability
    assert rule.name == "set-fixed-valueCodeableConcept"


def test_fixed_scalar_on_a_choice_element_still_uses_the_concrete_name(factory):
    """A `copy` carries no type parameter, so the element must be concrete."""
    rule = factory._create_fixed_value_rule(
        "string", "fixed text", "value[x]", "ext-note", "source",
    )
    assert rule.target[0].element == "valueString"
    assert rule.target[0].transform == "copy"


# ── optional pattern-only elements ────────────────────────────────────────────
def _patterned_child(path, parent_required=True):
    """A pinned, optional element — the shape of `Patient.identifier.system`."""
    return {
        "path": path,
        "id": path,
        "type": [{"code": "uri"}],
        "cardinality": {"min": 0, "max": "1"},
        "is_required": False,
        "is_pattern": True,
        "fixed_kind": "pattern",
        "fixed_value": "http://terminology.hl7.org/CodeSystem/v2-0203",
        "options": [],
        "children": [],
    }


def test_pinned_component_of_a_created_element_is_emitted(factory):
    # `Patient.identifier` is required and separately mapped, so an Identifier is
    # written regardless; the profile pins `identifier.system` precisely so that every
    # Identifier it produces carries that system. Dropping it emitted a bare
    # `{"value": "1"}` where the profile — and the reference output — say otherwise.
    rule = factory.create_mappable_field_rule(
        _patterned_child("Patient.identifier.system"),
        "Patient",
        "source",
        "tgt-identifier",
        automapped_mappings={},
        parent_path="Patient.identifier",
        # `identifier.value` is mapped, so an Identifier really is being written.
        siblings_have_content=True,
    )
    assert rule is not None
    assert rule.target[0].element == "system"
    assert (
        rule.target[0].parameter[0].valueString
        == "http://terminology.hl7.org/CodeSystem/v2-0203"
    )


def test_standalone_optional_pattern_element_is_still_suppressed(factory):
    # No parent context is being built here, so emitting the element would create it
    # purely because the profile patterns it — inventing data the record lacks.
    rule = factory.create_mappable_field_rule(
        _patterned_child("Patient.maritalStatus"),
        "Patient",
        "source",
        "target",
        automapped_mappings={},
        parent_path=None,
    )
    assert rule is None


def test_pinned_component_of_an_otherwise_empty_structure_is_suppressed(factory):
    # `Procedure.outcome.coding` has no mapped or fixed sibling. Emitting its pinned
    # `system` alone writes a Coding carrying a system and no code, which matches no
    # member of the required value set and fails the whole instance.
    rule = factory.create_mappable_field_rule(
        _patterned_child("Procedure.outcome.coding.system"),
        "Procedure",
        "source",
        "tgt-coding",
        automapped_mappings={},
        parent_path="Procedure.outcome.coding",
        siblings_have_content=False,
    )
    assert rule is None
