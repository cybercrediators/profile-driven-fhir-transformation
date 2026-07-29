"""Unit tests for the slice/choice-population mixin (`_SliceRulesMixin` / fml_slice.py).

Covers: populating a narrowed value[x] choice (primitive and complex, including a dotted
sub-path), creating a slice instance (primitive and complex, including fixed patterns, a
child discriminator, and dotted sub-paths), `_slice_child_fixed_rules`, and the generic
sub-path builder `_emit_subpath_population` in isolation (profile-declared children winning
over generic fhir.resources introspection, and the value[x] renaming rule).

All type information comes from the real (offline, no-network) `fhir.resources` package via
`get_complex_type_fields` — no spec facts are hand-invented here.
"""

import pytest

from mapping.fml_creator.fml_factory import FMLRuleFactory

pytestmark = pytest.mark.unit


@pytest.fixture
def factory():
    # The slice/choice mixin methods under test read no instance state beyond the pure
    # helpers they call, so bypass __init__.
    instance = object.__new__(FMLRuleFactory)
    instance.diagnostics = []
    instance.source_field_types = {}
    instance.source_field_max = {}
    instance._target_tree_cache = (None, None)
    return instance


# ── _create_choice_populate_rule: primitive choice type ────────────────────────
def test_choice_populate_primitive_with_mapping(factory):
    rule = factory._create_choice_populate_rule(
        "value", "integer", [("value", "srcInt")], "src", "tgt", {"max": "1"}
    )
    assert rule.name == "map-value-integer"
    tgt = rule.target[0]
    assert tgt.element == "valueInteger"
    assert tgt.transform == "cast"
    assert tgt.parameter[-1].valueString == "integer"
    assert rule.source[0].element == "srcInt"
    assert rule.source[0].variable == "src-choiceval"


def test_choice_populate_primitive_without_mapping_has_no_transform(factory):
    rule = factory._create_choice_populate_rule("value", "boolean", [], "src", "tgt", {})
    tgt = rule.target[0]
    assert tgt.element == "valueBoolean"
    assert rule.source[0].element is None
    assert getattr(tgt, "transform", None) is None


# ── _create_choice_populate_rule: complex choice type, direct sub-fields ───────
def test_choice_populate_complex_type_direct_subfields(factory):
    rule = factory._create_choice_populate_rule(
        "value", "Quantity",
        [("value", "srcVal"), ("unit", "srcUnit")],
        "src", "tgt", {},
    )
    outer = rule.target[0]
    assert outer.transform == "create"
    assert outer.parameter[0].valueString == "Quantity"
    assert outer.variable == "tgt-value-Quantity"

    names = [r.name for r in rule.rule]
    assert names == ["set-value-value", "set-value-unit"]
    # Quantity.value is decimal (cast); Quantity.unit is string (copy) — leaf type is
    # resolved from the datatype's own fields, not assumed.
    assert rule.rule[0].target[0].transform == "cast"
    assert rule.rule[1].target[0].transform == "copy"


def test_choice_populate_complex_type_dotted_subfield_delegates_to_subpath_builder(factory):
    rule = factory._create_choice_populate_rule(
        "value", "CodeableConcept", [("coding.code", "srcCode")], "src", "tgt", {}
    )
    coding_rule = rule.rule[0]
    assert coding_rule.name == "set-value-coding"
    assert coding_rule.target[0].transform == "create"
    assert coding_rule.target[0].parameter[0].valueString == "Coding"
    code_rule = coding_rule.rule[0]
    assert code_rule.name == "set-value-coding-code"
    assert code_rule.target[0].transform == "copy"
    assert code_rule.source[0].element == "srcCode"


def test_choice_populate_ignores_whole_placeholder_when_descendants_are_mapped(factory):
    rule = factory._create_choice_populate_rule(
        "item",
        "CodeableConcept",
        [
            ("coding.code", "ingredientCode"),
            ("", "TODO_MAP_INGREDIENT_ITEM_SOURCE"),
            ("coding.system", "ingredientSystem"),
        ],
        "src",
        "tgt",
        {},
    )

    assert rule.target[0].element == "item"
    assert rule.target[0].transform == "create"
    assert all(
        target.element
        for child in rule.rule
        for target in (child.target or [])
    )
    assert factory.diagnostics[-1]["code"] == "whole-complex-provider-shadowed"


# ── _create_slice_instance_rule ─────────────────────────────────────────────────
def test_coding_slice_of_codeableconcept_is_deferred(factory):
    parent_field = {"path": "Observation.category", "type": [{"code": "CodeableConcept"}]}
    slice_field = {
        "path": "Observation.category.coding:loinc",
        "sliceName": "loinc",
        "type": [{"code": "Coding"}],
    }
    assert (
        factory._create_slice_instance_rule(parent_field, slice_field, "Observation", "src", "tgt", {})
        is None
    )


def _sliced_coding_field(rules="open"):
    slicing = {
        "discriminators": [{"type": "pattern", "path": "$this"}],
        "ordered": False,
        "rules": rules,
    }
    code_child = {
        "path": "Condition.code.coding.code",
        "id": "Condition.code.coding.code",
        "type": [{"code": "code"}],
        "cardinality": {"min": 0, "max": "1"},
        "fixed_value": [],
    }
    unrelated_extension = {
        "path": "Condition.code.coding.display",
        "id": "Condition.code.coding.display",
        "type": [{"code": "string"}],
        "cardinality": {"min": 0, "max": "1"},
        "fixed_value": [],
        "children": [
            {
                "path": "Condition.code.coding.display.extension",
                "id": "Condition.code.coding.display.extension:translation",
                "sliceName": "translation",
                "type": [{"code": "Extension"}],
                "cardinality": {"min": 0, "max": "1"},
                "fixed_value": [],
                "children": [
                    {
                        "path": "Condition.code.coding.display.extension.url",
                        "id": (
                            "Condition.code.coding.display.extension:"
                            "translation.url"
                        ),
                        "type": [{"code": "uri"}],
                        "fixed_value": (
                            "http://hl7.org/fhir/StructureDefinition/translation"
                        ),
                    }
                ],
            }
        ],
    }
    coding_slice = {
        "path": "Condition.code.coding",
        "id": "Condition.code.coding:icd10",
        "slice_identity": "Condition.code.coding:icd10",
        "sliceName": "icd10",
        "type": [{"code": "Coding"}],
        "cardinality": {"min": 0, "max": "1"},
        "fixed_value": [],
        "slicing": slicing,
        "children": [dict(code_child)],
    }
    flattened_descendant_slice = dict(unrelated_extension["children"][0])
    flattened_descendant_slice.update(
        {
            "path": "Condition.code.coding.display.extension",
            "id": "Condition.code.coding.display.extension:translation",
            "slice_identity": (
                "Condition.code.coding.display.extension:translation"
            ),
            "sliceName": "translation",
        }
    )
    return {
        "path": "Condition.code.coding",
        "id": "Condition.code.coding",
        "type": [{"code": "Coding"}],
        "cardinality": {"min": 0, "max": "*"},
        "fixed_value": [],
        "slicing": slicing,
        "children": [code_child, unrelated_extension],
        # The second entry mirrors the parser's flattened descendant slice.
        # It belongs below display.extension and must not be hoisted here.
        "slices": [coding_slice, flattened_descendant_slice],
    }


def test_open_slicing_emits_one_generic_entry_for_unsliced_provider(factory):
    """An unqualified Coding mapping is a valid non-slice entry in open slicing."""
    field = _sliced_coding_field()

    rules = factory.create_field_rules(
        "Condition",
        [field],
        "src-code",
        "tgt-code",
        automapped_mappings={
            "Condition.code.coding.code": "Source.problemCode",
        },
        parent_path="Condition.code",
    )

    assert [rule.name for rule in rules] == ["map-coding"]
    assert rules[0].target[0].element == "coding"
    assert rules[0].target[0].parameter[0].valueString == "Coding"
    assert [child.name for child in rules[0].rule] == ["map-code"]
    assert rules[0].rule[0].source[0].element == "problemCode"
    assert rules[0].rule[0].target[0].element == "code"
    assert {
        diagnostic["code"] for diagnostic in factory.diagnostics
    } == {"deferred-descendant-slice"}


def test_open_unsliced_entry_keeps_base_fixed_leaf_but_not_slice_fixed_leaf(factory):
    field = _sliced_coding_field()
    field["children"].append(
        {
            "path": "Condition.code.coding.system",
            "id": "Condition.code.coding.system",
            "type": [{"code": "uri"}],
            "cardinality": {"min": 1, "max": "1"},
            "fixed_value": "http://example.org/base-system",
        }
    )
    # A named-slice constraint can appear in the flattened parser tree with the
    # same path. It must not constrain the ordinary open-slicing entry.
    field["children"].append(
        {
            "path": "Condition.code.coding.version",
            "id": "Condition.code.coding:icd10.version",
            "type": [{"code": "string"}],
            "cardinality": {"min": 0, "max": "1"},
            "fixed_value": "slice-only-version",
        }
    )

    rules = factory.create_field_rules(
        "Condition",
        [field],
        "src-code",
        "tgt-code",
        automapped_mappings={
            "Condition.code.coding.code": "Source.problemCode",
        },
        parent_path="Condition.code",
    )

    assert len(rules) == 1
    assert [child.target[0].element for child in rules[0].rule] == [
        "code",
        "system",
    ]


def test_open_at_end_places_generic_entry_after_selected_slices(factory):
    field = _sliced_coding_field("openAtEnd")

    rules = factory.create_field_rules(
        "Condition",
        [field],
        "src-code",
        "tgt-code",
        automapped_mappings={
            "Condition.code.coding:icd10.code": "Source.icd10Code",
            "Condition.code.coding.code": "Source.problemCode",
        },
        parent_path="Condition.code",
    )

    assert [rule.name for rule in rules] == ["map-coding-icd10", "map-coding"]
    assert rules[0].rule[-1].source[0].element == "icd10Code"
    assert rules[1].rule[0].source[0].element == "problemCode"


def test_closed_slicing_rejects_unsliced_provider_with_diagnostic(factory):
    field = _sliced_coding_field("closed")

    rules = factory.create_field_rules(
        "Condition",
        [field],
        "src-code",
        "tgt-code",
        automapped_mappings={
            "Condition.code.coding.code": "Source.problemCode",
        },
        parent_path="Condition.code",
    )

    assert rules == []
    assert factory.diagnostics[0] == {
        "code": "unsliced-provider-for-closed-slicing",
        "message": (
            "Unsliced mapping target Condition.code.coding cannot create an "
            "entry in closed slicing. Use a slice-qualified mapping target."
        ),
        "profile": "unknown",
        "path": "Condition.code.coding",
        "provider_keys": ["Condition.code.coding.code"],
        "slicing_rules": "closed",
        "severity": "error",
    }
    assert {
        diagnostic["code"] for diagnostic in factory.diagnostics[1:]
    } == {"deferred-descendant-slice"}


def test_slice_qualified_path_is_not_mistaken_for_descendant_path(factory):
    slicing = {
        "discriminators": [{"type": "pattern", "path": "$this.code"}],
        "rules": "open",
    }
    parent = {
        "path": "Observation.component",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 2, "max": "*"},
        "slicing": slicing,
        "slices": [
            {
                "path": "Observation.component:type",
                "id": "Observation.component:type",
                "sliceName": "type",
                "type": [{"code": "BackboneElement"}],
                "cardinality": {"min": 1, "max": "1"},
                "children": [
                    {
                        "path": "Observation.component.code",
                        "id": "Observation.component:type.code",
                        "type": [{"code": "CodeableConcept"}],
                        "cardinality": {"min": 1, "max": "1"},
                        "fixed_value": {"text": "type"},
                    }
                ],
            },
            {
                "path": "Observation.component:result",
                "id": "Observation.component:result",
                "sliceName": "result",
                "type": [{"code": "BackboneElement"}],
                "cardinality": {"min": 1, "max": "1"},
                "children": [
                    {
                        "path": "Observation.component.code",
                        "id": "Observation.component:result.code",
                        "type": [{"code": "CodeableConcept"}],
                        "cardinality": {"min": 1, "max": "1"},
                        "fixed_value": {"text": "result"},
                    }
                ],
            },
        ],
    }

    rules = factory.create_field_rules(
        "Observation", [parent], "src", "tgt", automapped_mappings={}
    )

    assert [rule.name for rule in rules] == [
        "map-component-type",
        "map-component-result",
    ]
    assert not any(
        diagnostic["code"] == "deferred-descendant-slice"
        for diagnostic in factory.diagnostics
    )


def test_primitive_slice_optional_without_subs_is_dropped(factory):
    slice_field = {
        "path": "Patient.name.given:first",
        "sliceName": "first",
        "type": [{"code": "string"}],
        "cardinality": {"min": 0},
    }
    parent_field = {"path": "Patient.name.given", "type": [{"code": "string"}]}
    assert (
        factory._create_slice_instance_rule(parent_field, slice_field, "Patient", "src", "tgt", {})
        is None
    )


def test_primitive_slice_required_without_subs_has_no_source(factory):
    slice_field = {
        "path": "Patient.name.given:first",
        "sliceName": "first",
        "type": [{"code": "string"}],
        "cardinality": {"min": 1},
    }
    parent_field = {"path": "Patient.name.given", "type": [{"code": "string"}]}
    rule = factory._create_slice_instance_rule(parent_field, slice_field, "Patient", "src", "tgt", {})
    assert rule is not None
    tgt = rule.target[0]
    assert tgt.element == "given"
    assert tgt.listMode == ["share"]
    assert rule.source[0].element is None
    assert getattr(tgt, "transform", None) is None


def test_primitive_slice_with_subs_gets_typed_transform(factory):
    slice_field = {
        "path": "Patient.name.given:first",
        "sliceName": "first",
        "type": [{"code": "string"}],
        "cardinality": {"min": 0},
    }
    parent_field = {"path": "Patient.name.given", "type": [{"code": "string"}]}
    automapped = {"Patient.name.given:first.raw": "Source.firstNameRaw"}
    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Patient", "src", "tgt", automapped
    )
    tgt = rule.target[0]
    assert tgt.element == "given"
    assert tgt.transform == "copy"
    assert rule.source[0].element == "firstNameRaw"
    assert rule.source[0].variable == "src-slice"


# ── Complex slice: fixed pattern + child discriminator + plain/dotted subs ──────
def test_complex_slice_instance_full_shape(factory):
    slice_field = {
        "path": "Patient.identifier:mrn",
        "sliceName": "mrn",
        "type": [{"code": "Identifier"}],
        "cardinality": {"min": 0, "max": "1"},
        "fixed_value": {"use": "official"},
        "children": [
            {
                "path": "Patient.identifier:mrn.type",
                "type": [{"code": "CodeableConcept"}],
                "fixed_value": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                            "code": "MR",
                        }
                    ]
                },
            },
            {
                "path": "Patient.identifier:mrn.system",
                "type": [{"code": "uri"}],
                "fixed_value": "http://hospital.example.org/mrn",
            },
        ],
    }
    parent_field = {"path": "Patient.identifier", "type": [{"code": "Identifier"}]}
    automapped = {
        "Patient.identifier:mrn.value": "Source.mrnValue",
        "Patient.identifier:mrn.period.start": "Source.mrnStart",
        # Redundant ancestor of period.start — must be filtered out, not emitted twice.
        "Patient.identifier:mrn.period": "Source.mrnPeriodContainer",
    }
    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Patient", "src", "tgt", automapped
    )
    assert rule is not None
    assert rule.name == "map-identifier-mrn"
    outer = rule.target[0]
    assert outer.element == "identifier"
    assert outer.transform == "create"
    assert outer.parameter[0].valueString == "Identifier"
    assert outer.listMode == ["share"]

    names = [r.name for r in rule.rule]
    assert names == [
        "set-identifier-mrn-use",  # top-level fixed_value pattern
        "set-identifier-mrn-type",  # discriminator child (fixed CodeableConcept)
        "set-identifier-mrn-system",  # scalar fixed child
        "set-identifier-mrn-value",  # plain mapped sub
        "set-identifier-mrn-period",  # dotted mapped sub -> nested create+cast
    ]

    use_rule = rule.rule[0]
    assert use_rule.target[0].transform == "copy"
    assert use_rule.target[0].parameter[0].valueString == "official"

    type_rule = rule.rule[1]
    assert type_rule.target[0].transform == "create"
    dumped = str(type_rule.model_dump())
    assert "MR" in dumped and "v2-0203" in dumped

    system_rule = rule.rule[2]
    assert system_rule.target[0].transform == "copy"
    assert system_rule.target[0].parameter[0].valueString == "http://hospital.example.org/mrn"

    value_rule = rule.rule[3]
    assert value_rule.target[0].transform == "copy"
    assert value_rule.source[0].element == "mrnValue"

    period_rule = rule.rule[4]
    assert period_rule.target[0].transform == "create"
    assert period_rule.target[0].parameter[0].valueString == "Period"
    start_rule = period_rule.rule[0]
    assert start_rule.name == "set-identifier-mrn-period-start"
    assert start_rule.target[0].transform == "cast"
    assert start_rule.target[0].parameter[-1].valueString == "dateTime"
    assert start_rule.source[0].element == "mrnStart"


def test_required_narrowed_choice_child_uses_concrete_element_name(factory):
    parent = {
        "path": "Communication.payload",
        "type": [{"code": "BackboneElement"}],
    }
    slice_field = {
        "path": "Communication.payload",
        "id": "Communication.payload:responseProposalContraIndication",
        "sliceName": "responseProposalContraIndication",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [
            {
                "path": "Communication.payload.content[x]",
                "id": (
                    "Communication.payload:responseProposalContraIndication"
                    ".content[x]"
                ),
                "type": [{"code": "string"}],
                "is_type_choice": True,
                "cardinality": {"min": 1, "max": "1"},
                "is_required": True,
                "fixed_value": [],
            }
        ],
    }

    rule = factory._create_slice_instance_rule(
        parent, slice_field, "Communication", "src", "tgt", {}
    )

    content = next(
        child for child in rule.rule if child.name.endswith("-content")
    )
    assert content.source[0].element.startswith("TODO-MAP-")
    assert content.target[0].element == "contentString"
    assert content.target[0].transform == "copy"


def test_active_slice_emits_todos_for_unprovided_required_siblings(factory):
    slice_field = {
        "path": "Patient.address:Strassenanschrift",
        "sliceName": "Strassenanschrift",
        "type": [{"code": "Address"}],
        "cardinality": {"min": 0, "max": "*"},
        "fixed_value": {"type": "both"},
        "children": [
            {
                "path": "Patient.address.line",
                "type": [{"code": "string"}],
                "cardinality": {"min": 1, "max": "3"},
                "is_required": True,
                "fixed_value": [],
            },
            {
                "path": "Patient.address.city",
                "type": [{"code": "string"}],
                "cardinality": {"min": 1, "max": "1"},
                "is_required": True,
                "fixed_value": [],
            },
            {
                "path": "Patient.address.country",
                "type": [{"code": "string"}],
                "cardinality": {"min": 1, "max": "1"},
                "is_required": True,
                "fixed_value": [],
            },
            {
                "path": "Patient.address.type",
                "type": [{"code": "code"}],
                "cardinality": {"min": 1, "max": "1"},
                "is_required": True,
                "fixed_value": [],
            },
        ],
    }
    parent_field = {"path": "Patient.address", "type": [{"code": "Address"}]}
    automapped = {
        "Patient.address:Strassenanschrift.city": "Source.city",
    }

    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Patient", "src", "tgt", automapped
    )

    by_name = {nested.name: nested for nested in rule.rule}
    assert by_name["set-address-Strassenanschrift-city"].source[0].element == "city"
    assert (
        by_name["set-address-Strassenanschrift-line"].source[0].element
        == "TODO-MAP-ADDRESS-STRASSENANSCHRIFT-LINE_SOURCE"
    )
    assert (
        by_name["set-address-Strassenanschrift-country"].source[0].element
        == "TODO-MAP-ADDRESS-STRASSENANSCHRIFT-COUNTRY_SOURCE"
    )
    # The root pattern supplies type=both; it must not get a contradictory TODO.
    assert [name for name in by_name if name == "set-address-Strassenanschrift-type"] == [
        "set-address-Strassenanschrift-type"
    ]
    assert by_name["set-address-Strassenanschrift-type"].source[0].element is None

def test_complex_slice_fixed_value_model_dump_object_is_normalized(factory):
    class _FixedIdentifier:
        def model_dump(self, exclude_none=True):
            return {"use": "secondary"}

    slice_field = {
        "path": "Patient.identifier:alt",
        "sliceName": "alt",
        "type": [{"code": "Identifier"}],
        "cardinality": {"min": 1},
        "fixed_value": _FixedIdentifier(),
    }
    parent_field = {"path": "Patient.identifier", "type": [{"code": "Identifier"}]}
    rule = factory._create_slice_instance_rule(parent_field, slice_field, "Patient", "src", "tgt", {})
    assert rule.rule[0].target[0].parameter[0].valueString == "secondary"


# ── _slice_child_fixed_rules ─────────────────────────────────────────────────────
def test_slice_child_fixed_rules_scalar_fixed_value(factory):
    children = [{"path": "Patient.identifier:mrn.system", "type": [{"code": "uri"}], "fixed_value": "http://x"}]
    rules = factory._slice_child_fixed_rules(children, "tgt-mrn", "source", "identifier-mrn")
    assert len(rules) == 1
    tgt = rules[0].target[0]
    assert tgt.element == "system"
    assert tgt.transform == "copy"
    assert tgt.parameter[0].valueString == "http://x"


def test_slice_child_fixed_rules_skips_empty_fixed_value(factory):
    children = [{"path": "Patient.identifier:mrn.value", "type": [{"code": "string"}], "fixed_value": []}]
    assert factory._slice_child_fixed_rules(children, "tgt-mrn", "source", "identifier-mrn") == []


def test_slice_child_fixed_rules_boolean_uses_typed_parameter(factory):
    children = [{"path": "Patient.deceased:flag.deceasedBoolean", "type": [{"code": "boolean"}], "fixed_value": True}]
    rules = factory._slice_child_fixed_rules(children, "tgt-flag", "source", "deceased-flag")
    assert rules[0].target[0].parameter[0].valueBoolean is True
    assert rules[0].target[0].parameter[0].valueString is None
    children[0]["fixed_value"] = False
    rules = factory._slice_child_fixed_rules(children, "tgt-flag", "source", "deceased-flag")
    assert rules[0].target[0].parameter[0].valueBoolean is False
    assert rules[0].target[0].parameter[0].valueString is None


# ── _emit_subpath_population in isolation ───────────────────────────────────────
def test_emit_subpath_population_two_level_generic_introspection(factory):
    r = factory._emit_subpath_population(
        "tgt-code", "coding.code", "srcCode", "CodeableConcept", "code", "src"
    )
    assert r.name == "set-code-coding"
    assert r.target[0].transform == "create"
    assert r.target[0].parameter[0].valueString == "Coding"
    inner = r.rule[0]
    assert inner.name == "set-code-coding-code"
    assert inner.target[0].transform == "copy"
    assert inner.source[0].element == "srcCode"


def test_emit_subpath_population_prefers_profile_declared_children(factory):
    # The profile narrows Period.start to `date` here; that must win over the generic
    # fhir.resources Period.start (`dateTime`).
    root_children = [{"path": "Observation.effective.start", "type": [{"code": "date"}]}]
    r = factory._emit_subpath_population(
        "tgt-eff", "start", "srcStart", "Period", "eff", "src", root_children=root_children
    )
    assert r.name == "set-eff-start"
    assert r.target[0].element == "start"
    assert r.target[0].transform == "cast"
    assert r.target[0].parameter[-1].valueString == "date"


def test_emit_subpath_population_renames_value_x_child(factory):
    root_children = [{"path": "Observation.effective.value[x]", "type": [{"code": "integer"}]}]
    r = factory._emit_subpath_population(
        "tgt-eff", "value", "srcVal", "SomeType", "eff", "src", root_children=root_children
    )
    assert r.target[0].element == "valueInteger"
    assert r.target[0].transform == "cast"


def test_emit_subpath_population_returns_none_when_unresolvable(factory):
    # First segment can't be resolved against a primitive root type and there's more path
    # left to walk -> no rule can be built.
    assert (
        factory._emit_subpath_population("tgt-x", "foo.bar", "srcFoo", "string", "x", "src")
        is None
    )


# ── regression: plain slice sub-fields get type-aware transforms, not a bare copy ──
def test_slice_instance_plain_subfield_numeric_leaf_gets_cast(factory):
    parent_field = {"path": "Observation.component", "type": [{"code": "Quantity"}]}
    slice_field = {
        "path": "Observation.component:weight",
        "sliceName": "weight",
        "type": [{"code": "Quantity"}],
        "cardinality": {"min": 0},
    }
    automapped = {
        "Observation.component:weight.value": "Source.weightValue",
        "Observation.component:weight.unit": "Source.weightUnit",
    }
    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Observation", "src", "tgt", automapped
    )
    by_name = {r.name: r for r in rule.rule}
    # Quantity.value is decimal -> cast; Quantity.unit is string -> copy. Mirrors
    # _create_choice_populate_rule; a bare copy of a non-string source crashes the engine.
    val = by_name["set-component-weight-value"].target[0]
    unit = by_name["set-component-weight-unit"].target[0]
    assert val.transform == "cast"
    assert val.parameter[-1].valueString == "decimal"
    assert unit.transform == "copy"


def test_slice_instance_profile_declared_child_type_wins(factory):
    parent_field = {"path": "Observation.component", "type": [{"code": "Quantity"}]}
    slice_field = {
        "path": "Observation.component:score",
        "sliceName": "score",
        "type": [{"code": "Quantity"}],
        "cardinality": {"min": 0},
        "children": [
            {"path": "Observation.component.value", "type": [{"code": "integer"}]}
        ],
    }
    automapped = {"Observation.component:score.value": "Source.scoreValue"}
    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Observation", "src", "tgt", automapped
    )
    val = {r.name: r for r in rule.rule}["set-component-score-value"].target[0]
    assert val.transform == "cast"
    # the profile-constrained child type (integer) wins over Quantity.value's decimal
    assert val.parameter[-1].valueString == "integer"


# ── extension slice of a slice instance (the biobank Organization contact case) ──
def test_slice_instance_extension_slice_delegates_to_extension_rule(factory):
    # A mapped `extension:<name>` sub-path must NOT become a literal element name
    # (`element = "extension:rolle"` crashes the engine: "Cannot set property
    # extension:rolle on contact"). It resolves the canonical from the current
    # profile snapshot and emits url + typed value via the extension machinery.
    from types import SimpleNamespace

    ext_canonical = "http://example.org/StructureDefinition/organization-contact-rolle"
    factory._current_profile_sd = SimpleNamespace(
        snapshot=SimpleNamespace(
            element=[
                SimpleNamespace(id="Organization.contact:forschungskontakt", type=None),
                SimpleNamespace(
                    id="Organization.contact:forschungskontakt.extension:rolle",
                    type=[SimpleNamespace(profile=[ext_canonical])],
                ),
            ]
        )
    )
    # create_extension_rule resolves the value type from the extension SD via the
    # registry; keep it unresolvable here -> falls back to the string placeholder.
    factory.app_state = SimpleNamespace(
        registry=SimpleNamespace(registry_objects={}),
        cache=SimpleNamespace(
            get_resource_from_cache=lambda u: None,
            add_resource_to_cache=lambda o: None,
        ),
        conf={"resource_cache_path": "/nonexistent-dir-xyz"},
        dataIO=None,
    )
    factory.map_url = "http://example.org/fml"
    factory.overwrite = False
    factory.plugins = []
    factory.source_field_types = {}

    parent_field = {"path": "Organization.contact", "type": [{"code": "BackboneElement"}]}
    slice_field = {
        "path": "Organization.contact:forschungskontakt",
        "sliceName": "forschungskontakt",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 0},
    }
    automapped = {
        "Organization.contact:forschungskontakt.extension:rolle": "Source.contactRole",
    }
    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Organization", "src", "tgt", automapped
    )
    by_name = {r.name: r for r in rule.rule}
    # no rule may target the literal slice-marker element
    for r in rule.rule:
        for t in r.target or []:
            assert ":" not in (t.element or ""), f"slice marker leaked: {t.element}"
    ext_rule = by_name["map-extension-rolle-contact-forschungskontakt"]
    assert ext_rule.target[0].element == "extension"
    url_rule, value_rule = ext_rule.rule
    assert url_rule.target[0].element == "url"
    assert url_rule.target[0].parameter[0].valueString == ext_canonical
    assert value_rule.source[0].element == "contactRole"


def test_slice_instance_extension_slice_unresolvable_canonical_is_dropped(factory):
    # No profile SD available -> the sub-rule cannot be built; it must be DROPPED
    # (logged), never emitted with the invalid slice-marker element name.
    factory._current_profile_sd = None
    parent_field = {"path": "Organization.contact", "type": [{"code": "BackboneElement"}]}
    slice_field = {
        "path": "Organization.contact:forschungskontakt",
        "sliceName": "forschungskontakt",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 0},
    }
    automapped = {
        "Organization.contact:forschungskontakt.extension:rolle": "Source.contactRole",
    }
    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Organization", "src", "tgt", automapped
    )
    for r in rule.rule or []:
        for t in r.target or []:
            assert ":" not in (t.element or "")


def test_required_nested_extension_slice_is_emitted_under_backbone_slice(factory):
    from types import SimpleNamespace

    canonical = "http://example.org/StructureDefinition/contact-role"
    exact_id = "Organization.contact:forschungskontakt.extension:rolle"
    factory._current_profile_sd = SimpleNamespace(
        snapshot=SimpleNamespace(
            element=[
                SimpleNamespace(
                    id=exact_id,
                    type=[SimpleNamespace(profile=[canonical])],
                )
            ]
        )
    )
    factory.app_state = SimpleNamespace(
        registry=SimpleNamespace(registry_objects={}),
        cache=SimpleNamespace(
            get_resource_from_cache=lambda url: None,
            add_resource_to_cache=lambda resource: None,
        ),
        conf={"resource_cache_path": "/nonexistent-dir-xyz"},
        dataIO=None,
    )
    factory.map_url = "http://example.org/fml"
    factory.overwrite = False
    factory.plugins = []

    parent_field = {
        "path": "Organization.contact",
        "type": [{"code": "BackboneElement"}],
    }
    slice_field = {
        "path": "Organization.contact",
        "id": "Organization.contact:forschungskontakt",
        "slice_identity": "Organization.contact:forschungskontakt",
        "sliceName": "forschungskontakt",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 1, "max": "*"},
        "children": [
            {
                "path": "Organization.contact.extension",
                "id": "Organization.contact:forschungskontakt.extension",
                "type": [{"code": "Extension"}],
                "cardinality": {"min": 1, "max": "*"},
                "slices": [
                    {
                        "path": "Organization.contact.extension",
                        "id": exact_id,
                        "slice_identity": exact_id,
                        "sliceName": "rolle",
                        "type": [
                            {
                                "code": "Extension",
                                "profile_canonical": [canonical],
                            }
                        ],
                        "cardinality": {"min": 1, "max": "1"},
                    }
                ],
            }
        ],
    }

    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Organization", "src", "tgt", {}
    )

    extension = next(
        child for child in rule.rule if child.name.startswith("map-extension-rolle")
    )
    assert extension.source[0].element.startswith("TODO-MAP-")
    assert extension.target[0].context == "tgt-contact-forschungskontakt"
    assert extension.target[0].element == "extension"
    assert extension.rule[0].target[0].parameter[0].valueString == canonical
    assert not any(
        target.element == "extension"
        for child in rule.rule
        if child is not extension
        for target in (child.target or [])
    )


def test_required_extension_below_primitive_choice_uses_primitive_context(factory):
    from types import SimpleNamespace

    canonical = "http://example.org/StructureDefinition/content-codeable-concept"
    exact_id = (
        "Communication.payload:responseProposalContraIndication"
        ".content[x].extension:contentCodeableConcept"
    )
    factory._current_profile_sd = SimpleNamespace(
        snapshot=SimpleNamespace(
            element=[
                SimpleNamespace(
                    id=exact_id,
                    type=[SimpleNamespace(profile=[canonical])],
                )
            ]
        )
    )
    factory.app_state = SimpleNamespace(
        registry=SimpleNamespace(registry_objects={}),
        cache=SimpleNamespace(
            get_resource_from_cache=lambda url: None,
            add_resource_to_cache=lambda resource: None,
        ),
        conf={"resource_cache_path": "/nonexistent-dir-xyz"},
        dataIO=None,
    )
    factory.map_url = "http://example.org/fml"
    factory.overwrite = False
    factory.plugins = []

    parent = {
        "path": "Communication.payload",
        "type": [{"code": "BackboneElement"}],
    }
    slice_field = {
        "path": "Communication.payload",
        "id": "Communication.payload:responseProposalContraIndication",
        "sliceName": "responseProposalContraIndication",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [
            {
                "path": "Communication.payload.content[x]",
                "id": (
                    "Communication.payload:responseProposalContraIndication"
                    ".content[x]"
                ),
                "type": [{"code": "string"}],
                "is_type_choice": True,
                "cardinality": {"min": 1, "max": "1"},
                "is_required": True,
                "children": [
                    {
                        "path": "Communication.payload.content[x].extension",
                        "id": (
                            "Communication.payload:responseProposalContraIndication"
                            ".content[x].extension"
                        ),
                        "type": [{"code": "Extension"}],
                        "cardinality": {"min": 1, "max": "*"},
                        "slices": [
                            {
                                "path": (
                                    "Communication.payload.content[x].extension"
                                ),
                                "id": exact_id,
                                "slice_identity": exact_id,
                                "sliceName": "contentCodeableConcept",
                                "type": [{"code": "Extension"}],
                                "cardinality": {"min": 1, "max": "1"},
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rule = factory._create_slice_instance_rule(
        parent, slice_field, "Communication", "src", "tgt", {}
    )

    content = next(child for child in rule.rule if child.name.endswith("-content"))
    assert content.target[0].element == "contentString"
    assert content.target[0].variable == (
        "tgt-payload-responseProposalContraIndication-content"
    )
    extension = content.rule[0]
    assert extension.target[0].context == content.target[0].variable
    assert extension.target[0].element == "extension"
    assert extension.rule[0].target[0].parameter[0].valueString == canonical


# ── G3: sibling leaves of one repeating sub-element merge into a single entry ───
def test_slice_instance_repeating_subelement_leaves_merge(factory):
    """telecom.system + telecom.value under a slice instance must populate ONE
    ContactPoint, not one entry per leaf. Two entries yield a system-only and a
    value-only ContactPoint, both failing cpt-2 ('system required if value
    provided') — the biobank Organization forschungskontakt failure (G3)."""
    parent_field = {"path": "Patient.contact", "type": [{"code": "BackboneElement"}]}
    slice_field = {
        "path": "Patient.contact:primary",
        "sliceName": "primary",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 0},
        "children": [
            {"path": "Patient.contact.telecom", "type": [{"code": "ContactPoint"}]},
        ],
    }
    automapped = {
        "Patient.contact:primary.telecom.system": "Source.phoneSystem",
        "Patient.contact:primary.telecom.value": "Source.phoneValue",
    }
    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Patient", "src", "tgt", automapped
    )
    telecom_rules = [r for r in (rule.rule or []) if r.name == "set-contact-primary-telecom"]
    assert len(telecom_rules) == 1, "telecom leaves must merge into ONE ContactPoint entry"
    leaves = sorted(c.target[0].element for c in (telecom_rules[0].rule or []))
    assert leaves == ["system", "value"]


# ── G5: a reference-typed required leaf in a backbone slice emits a wireable
#        TODO-resolve-reference rule (not an inert create('Reference')) ─────────
def test_slice_instance_reference_leaf_emits_resolve_reference(factory):
    """Composition.section:diagRep.entry (Reference, min=1, targeting a profile)
    must be emitted as a ``TODO-resolve-reference`` rule the bundle assembler
    collects and wires array-aware — not a plain create('Reference') whose phantom
    source never fires, leaving section.entry empty (G5, deep-backbone ref wiring)."""
    parent_field = {"path": "Composition.section", "type": [{"code": "BackboneElement"}]}
    target_profile = (
        "https://www.medizininformatik-initiative.de/fhir/ext/modul-bildgebung/"
        "StructureDefinition/mii-pr-bildgebung-radiologischer-befund"
    )
    slice_field = {
        "path": "Composition.section",
        "sliceName": "diagRep",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [
            {
                "path": "Composition.section.entry",
                "type": [{"code": "Reference"}],
                "reference_target": target_profile,
                "is_required": True,
                "cardinality": {"min": 1, "max": "*"},
            },
        ],
    }
    rule = factory._create_slice_instance_rule(
        parent_field, slice_field, "Composition", "src", "tgt", {}
    )
    assert rule is not None
    entry_rules = [
        r for r in (rule.rule or [])
        if (r.name or "").lower().startswith("todo-resolve-reference-")
    ]
    assert len(entry_rules) == 1, "reference leaf must emit a TODO-resolve-reference rule"
    er = entry_rules[0]
    # documentation must be parseable by BundleService._collect_todo_refs:
    #   Reference<Composition.section.entry> → <target> — resolve via bundle assembler
    assert "Reference<Composition.section.entry>" in er.documentation
    assert target_profile in er.documentation
    assert er.target is None
    # it must NOT also be emitted as a plain create('Reference') set rule
    assert not any(
        r.name == "set-section-diagRep-entry" for r in (rule.rule or [])
    )


def test_required_slice_with_unresolved_discriminator_is_omitted(factory):
    parent = {
        "path": "Observation.component",
        "type": [{"code": "BackboneElement"}],
        "slicing": {
            "discriminators": [{"type": "value", "path": "code"}],
            "ordered": False,
            "rules": "closed",
        },
    }
    slice_field = {
        "path": "Observation.component",
        "id": "Observation.component:heart-rate",
        "sliceName": "heart-rate",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [
            {
                "path": "Observation.component.code",
                "type": [{"code": "CodeableConcept"}],
                "cardinality": {"min": 1, "max": "1"},
                "fixed_value": [],
            }
        ],
    }

    assert (
        factory._create_slice_instance_rule(
            parent, slice_field, "Observation", "src", "tgt", {}
        )
        is None
    )
    assert factory.diagnostics[-1]["code"] == "ambiguous-slice-selection"
    assert factory.diagnostics[-1]["slicing_rules"] == "closed"


def test_exists_discriminator_accepts_required_canonical_extension(factory):
    canonical = (
        "http://nictiz.nl/fhir/5.0/StructureDefinition/"
        "extension-Communication.payload.contentCodeableConcept"
    )
    factory._current_profile_sd = {
        "snapshot": {
            "element": [
                {
                    "id": (
                        "Communication.payload:responseProposalContraIndication"
                        ".content[x].extension:contentCodeableConcept"
                    ),
                    "path": "Communication.payload.content[x].extension",
                    "min": 1,
                    "max": "1",
                    "type": [{"code": "Extension", "profile": [canonical]}],
                }
            ]
        }
    }
    parent = {
        "path": "Communication.payload",
        "type": [{"code": "BackboneElement"}],
        "slicing": {
            "discriminators": [
                {
                    "type": "exists",
                    "path": f"content.extension(url='{canonical}')",
                }
            ],
            "rules": "open",
        },
    }
    slice_field = {
        "path": "Communication.payload",
        "id": "Communication.payload:responseProposalContraIndication",
        "sliceName": "responseProposalContraIndication",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 1, "max": "1"},
    }

    assert factory._slice_selection_is_safe(
        parent, slice_field, explicit_provider=False
    )
    assert not factory.diagnostics


def test_multiple_discriminators_are_checked_together(factory):
    parent = {
        "path": "Patient.identifier",
        "type": [{"code": "Identifier"}],
        "slicing": {
            "discriminators": [
                {"type": "value", "path": "system"},
                {"type": "exists", "path": "period"},
            ],
            "ordered": False,
            "rules": "open",
        },
    }
    slice_field = {
        "path": "Patient.identifier",
        "id": "Patient.identifier:mrn",
        "sliceName": "mrn",
        "type": [{"code": "Identifier"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [
            {
                "path": "Patient.identifier.system",
                "type": [{"code": "uri"}],
                "cardinality": {"min": 1, "max": "1"},
                "fixed_value": "http://example.org/mrn",
            },
            {
                "path": "Patient.identifier.period",
                "type": [{"code": "Period"}],
                "cardinality": {"min": 0, "max": "0"},
                "is_prohibited": True,
                "fixed_value": [],
            },
        ],
    }

    rule = factory._create_slice_instance_rule(
        parent, slice_field, "Patient", "src", "tgt", {}
    )

    assert rule is not None
    assert not factory.diagnostics


def test_value_discriminator_accepts_profile_fixed_descendants(factory):
    """HMB fixes Coding leaves below its value:coding discriminator (N1)."""
    factory._current_profile_sd = {
        "resourceType": "StructureDefinition",
        "type": "Observation",
        "snapshot": {
            "element": [
                {
                    "id": "Observation",
                    "path": "Observation",
                    "min": 0,
                    "max": "*",
                },
                {
                    "id": "Observation.category",
                    "path": "Observation.category",
                    "min": 1,
                    "max": "*",
                    "type": [{"code": "CodeableConcept"}],
                },
                {
                    "id": "Observation.category:obstetrics",
                    "path": "Observation.category",
                    "sliceName": "obstetrics",
                    "min": 1,
                    "max": "1",
                    "type": [{"code": "CodeableConcept"}],
                },
                {
                    "id": "Observation.category:obstetrics.coding",
                    "path": "Observation.category.coding",
                    "min": 1,
                    "max": "1",
                    "type": [{"code": "Coding"}],
                },
                {
                    "id": "Observation.category:obstetrics.coding.system",
                    "path": "Observation.category.coding.system",
                    "min": 0,
                    "max": "1",
                    "type": [{"code": "uri"}],
                    "fixedUri": "http://terminology.hl7.org/CodeSystem/observation-category",
                },
                {
                    "id": "Observation.category:obstetrics.coding.code",
                    "path": "Observation.category.coding.code",
                    "min": 0,
                    "max": "1",
                    "type": [{"code": "code"}],
                    "patternCode": "social-history",
                },
                {
                    "id": "Observation.category:obstetrics.coding.display",
                    "path": "Observation.category.coding.display",
                    "min": 0,
                    "max": "1",
                    "type": [{"code": "string"}],
                    "patternString": "Social History",
                },
            ]
        },
    }
    parent = {
        "path": "Observation.category",
        "type": [{"code": "CodeableConcept"}],
        "slicing": {
            "discriminators": [{"type": "value", "path": "coding"}],
            "ordered": False,
            "rules": "open",
        },
    }
    slice_field = {
        "path": "Observation.category",
        "id": "Observation.category:obstetrics",
        "sliceName": "obstetrics",
        "type": [{"code": "CodeableConcept"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [
            {
                "path": "Observation.category.coding",
                "id": "Observation.category:obstetrics.coding",
                "type": [{"code": "Coding"}],
                "cardinality": {"min": 1, "max": "1"},
                "fixed_value": [],
            }
        ],
    }

    rule = factory._create_slice_instance_rule(
        parent, slice_field, "Observation", "src", "tgt", {}
    )

    assert rule is not None
    assert not factory.diagnostics
    coding = next(child for child in rule.rule if child.target[0].element == "coding")
    fixed_leaves = {
        child.target[0].element: child.target[0].parameter[0].valueString
        for child in coding.rule
    }
    assert fixed_leaves == {
        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
        "code": "social-history",
        "display": "Social History",
    }


def test_position_discriminator_requires_ordered_and_explicit_rule(factory):
    factory.source_field_max = {"firstValue": "1"}
    parent = {
        "path": "Observation.component",
        "type": [{"code": "BackboneElement"}],
        "slicing": {
            "discriminators": [{"type": "position", "path": "$this"}],
            "ordered": True,
            "rules": "open",
        },
    }
    slice_field = {
        "path": "Observation.component",
        "id": "Observation.component:first",
        "sliceName": "first",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 0, "max": "1"},
        "children": [],
    }

    assert (
        factory._create_slice_instance_rule(
            parent, slice_field, "Observation", "src", "tgt", {}
        )
        is None
    )
    rule = factory._create_slice_instance_rule(
        parent,
        slice_field,
        "Observation",
        "src",
        "tgt",
        {"Observation.component:first.value": "firstValue"},
    )
    assert rule is not None
    assert factory.diagnostics[-1]["code"] == "explicit-slice-selection"

    parent["slicing"]["ordered"] = False
    assert (
        factory._create_slice_instance_rule(
            parent,
            slice_field,
            "Observation",
            "src",
            "tgt",
            {"Observation.component:first.value": "firstValue"},
        )
        is None
    )
    assert factory.diagnostics[-1]["code"] == "invalid-position-slicing"


def test_position_discriminator_rejects_repeating_source(factory):
    factory.source_field_max = {"components": "*"}
    parent = {
        "path": "Observation.component",
        "type": [{"code": "BackboneElement"}],
        "slicing": {
            "discriminators": [{"type": "position", "path": "$this"}],
            "ordered": True,
            "rules": "open",
        },
    }
    slice_field = {
        "path": "Observation.component",
        "id": "Observation.component:first",
        "sliceName": "first",
        "type": [{"code": "BackboneElement"}],
        "cardinality": {"min": 0, "max": "1"},
        "children": [],
    }

    assert (
        factory._create_slice_instance_rule(
            parent,
            slice_field,
            "Observation",
            "src",
            "tgt",
            {"Observation.component:first.value": "components"},
        )
        is None
    )
    assert factory.diagnostics[-1]["code"] == "ambiguous-position-source"


def test_type_profile_exists_and_resolve_discriminator_boundaries(factory):
    choice_parent = {
        "path": "Observation.value[x]",
        "type": [{"code": "string"}, {"code": "Quantity"}],
    }
    typed_slice = {
        "path": "Observation.value[x]",
        "type": [{"code": "Quantity"}],
    }
    assert factory._profile_defines_discriminator(
        choice_parent, typed_slice, {"type": "type", "path": "$this"}
    )

    profiled_slice = {
        "path": "Observation.subject",
        "type": [
            {
                "code": "Reference",
                "targetProfile": ["http://example.org/StructureDefinition/MyPatient"],
            }
        ],
    }
    assert not factory._profile_defines_discriminator(
        {"path": "Observation.subject", "type": [{"code": "Reference"}]},
        profiled_slice,
        {"type": "profile", "path": "$this"},
    )

    exists_slice = {
        "path": "Observation.component",
        "children": [
            {
                "path": "Observation.component.value[x]",
                "cardinality": {"min": 0, "max": "0"},
            }
        ],
    }
    assert factory._profile_defines_discriminator(
        {"path": "Observation.component"},
        exists_slice,
        {"type": "exists", "path": "value[x]"},
    )

    resolving_parent = {
        "path": "Observation.subject",
        "slicing": {
            "discriminators": [
                {"type": "profile", "path": "resolve()"}
            ],
            "ordered": False,
            "rules": "open",
        },
    }
    assert not factory._slice_selection_is_safe(
        resolving_parent, profiled_slice, explicit_provider=False
    )
    assert factory._slice_selection_is_safe(
        resolving_parent, profiled_slice, explicit_provider=True
    )


def test_create_field_rules_emits_reslice_once_by_exact_identity(factory):
    parent = {
        "path": "Patient.identifier",
        "type": [{"code": "Identifier"}],
        "slicing": {
            "discriminators": [{"type": "value", "path": "system"}],
            "ordered": False,
            "rules": "open",
        },
        "slices": [
            {
                "path": "Patient.identifier",
                "id": "Patient.identifier:national",
                "slice_identity": "Patient.identifier:national",
                "sliceName": "national",
                "type": [{"code": "Identifier"}],
                "cardinality": {"min": 1, "max": "*"},
                "children": [],
                "slices": [
                    {
                        "path": "Patient.identifier",
                        "id": "Patient.identifier:national/ssn",
                        "slice_identity": "Patient.identifier:national/ssn",
                        "sliceName": "national/ssn",
                        "type": [{"code": "Identifier"}],
                        "cardinality": {"min": 1, "max": "1"},
                        "slicing": {
                            "discriminators": [
                                {"type": "value", "path": "system"}
                            ],
                            "ordered": False,
                            "rules": "open",
                        },
                        "children": [
                            {
                                "path": "Patient.identifier.system",
                                "type": [{"code": "uri"}],
                                "cardinality": {"min": 1, "max": "1"},
                                "fixed_value": "urn:ssn",
                            },
                            {
                                "path": "Patient.identifier.value",
                                "type": [{"code": "string"}],
                                "cardinality": {"min": 1, "max": "1"},
                                "fixed_value": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }

    rules = factory.create_field_rules(
        "Patient",
        [parent],
        "src",
        "tgt",
        automapped_mappings={
            "Patient.identifier:national/ssn.value": "ssn"
        },
    )

    assert [rule.name for rule in rules] == ["map-identifier-national-ssn"]
