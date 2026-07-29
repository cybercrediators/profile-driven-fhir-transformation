"""Unit tests for the extension-rule mixin (`_ExtensionRulesMixin` / fml_extension.py).

Covers: simple (leaf value[x]) extensions, the Reference-valued extension branch, the
modifierExtension/slice naming rules, and the complex-extension path where sub-extension
slices are resolved from a (fake, in-registry) Extension StructureDefinition and recursively
turned into nested rules — including the min>=1-vs-mapped inclusion gate. All resolution goes
through an in-memory registry so no cache/network/local-package fallback is ever exercised.
"""

from types import SimpleNamespace

import pytest

from mapping.fml_creator.fml_factory import FMLRuleFactory
from data_handling.data_io import DataIO

pytestmark = pytest.mark.unit


class _FakeRegistry:
    def __init__(self, objects=None):
        self.registry_objects = objects or {}


class _FakeDataIO:
    """Records store_project_file calls; ConceptMap files never pre-exist."""

    ProjectFolders = DataIO.ProjectFolders

    def __init__(self):
        self.stored = []

    def project_file_exists(self, folder, filename):
        return False

    def store_project_file(self, folder, filename, content, mode="STR", overwrite=False):
        self.stored.append((folder, filename, content))


def _app_state(registry_objects=None):
    return SimpleNamespace(
        registry=_FakeRegistry(registry_objects),
        cache=SimpleNamespace(
            get_resource_from_cache=lambda u: None,
            add_resource_to_cache=lambda o: None,
        ),
        # A nonexistent path keeps get_resource_from_local_package a no-op (os.walk on a
        # missing dir just yields nothing) — guards against ever falling through to a real
        # network resolver if a lookup is unexpectedly not satisfied by the registry.
        conf={"resource_cache_path": "/nonexistent-dir-xyz"},
        dataIO=_FakeDataIO(),
    )


@pytest.fixture
def factory():
    f = object.__new__(FMLRuleFactory)
    f.app_state = _app_state()
    f.map_url = "http://example.org/fml"
    f.overwrite = False
    f.plugins = []
    f.source_field_types = {}
    return f


# ── Simple (leaf) extension value ──────────────────────────────────────────────
def test_simple_extension_scalar_value_mapped(factory):
    field = {
        "path": "Patient.extension",
        "type": "Extension",
        "extension_url": "http://example.org/StructureDefinition/simple-ext",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={"Patient.extension": "Source.flag"},
    )
    assert rule is not None
    assert rule.target[0].element == "extension"
    assert rule.target[0].transform == "create"
    assert rule.target[0].parameter[0].valueString == "Extension"

    url_rule, value_rule = rule.rule
    assert url_rule.target[0].element == "url"
    assert (
        url_rule.target[0].parameter[0].valueString
        == "http://example.org/StructureDefinition/simple-ext"
    )

    vt = value_rule.target[0]
    assert vt.element == "valueString"  # no value_type override -> defaults to string
    assert vt.transform == "copy"
    assert value_rule.source[0].element == "flag"  # local_element_name stripped "Source."


def test_extension_value_type_override_drives_transform(factory):
    field = {
        "path": "Patient.extension",
        "type": "Extension",
        "value_type": "boolean",
        "extension_url": "http://example.org/StructureDefinition/bool-ext",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={"Patient.extension": "Source.flag"},
    )
    value_rule = rule.rule[1]
    assert value_rule.target[0].element == "valueBoolean"
    assert value_rule.target[0].transform == "copy"


def test_complex_value_type_creates_generic_value_element(factory):
    field = {
        "path": "Patient.extension",
        "type": "Extension",
        "value_type": "Quantity",
        "extension_url": "http://example.org/StructureDefinition/qty-ext",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={"Patient.extension": "Source.qty"},
    )
    value_rule = rule.rule[1]
    vt = value_rule.target[0]
    assert vt.element == "value"
    assert vt.transform == "create"
    assert vt.parameter[0].valueString == "Quantity"


def test_reference_valued_extension_emits_dependent_rule(factory):
    field = {
        "path": "Patient.extension",
        "type": "Extension",
        "value_type": "Reference",
        "reference_target": "Practitioner",
        "extension_url": "http://example.org/StructureDefinition/ref-ext",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={"Patient.extension": "Source.ref"},
    )
    value_rule = rule.rule[1]
    vt = value_rule.target[0]
    assert vt.element == "valueReference"
    assert vt.transform == "create"
    assert vt.parameter[0].valueString == "Reference"
    dep = value_rule.dependent[0]
    assert dep.name == "ReferencePractitioner"
    assert dep.variable == ["srcVal", "valRef"]


def test_modifier_extension_targets_modifier_extension_element(factory):
    field = {
        "path": "Patient.modifierExtension",
        "type": "Extension",
        "extension_url": "http://example.org/StructureDefinition/mod-ext",
    }
    rule = factory.create_extension_rule(
        field, "modifierExtension", "src", "tgt",
        automapped_mappings={"Patient.modifierExtension": "Source.flag"},
    )
    assert rule.target[0].element == "modifierExtension"


def test_modifier_extension_definition_reroutes_plain_extension_slice(factory):
    url = "http://example.org/StructureDefinition/modifier-by-definition"
    factory.app_state = _app_state(
        {
            url: SimpleNamespace(
                data={
                    "snapshot": {
                        "element": [
                            {"id": "Extension", "isModifier": True},
                            {
                                "id": "Extension.value[x]",
                                "type": [{"code": "string"}],
                            },
                        ]
                    }
                }
            )
        }
    )
    field = {
        "path": "Procedure.extension",
        "type": "Extension",
        "sliceName": "PerformerAbbreviation",
        "extension_url": url,
    }

    rule = factory.create_extension_rule(
        field,
        "extension:PerformerAbbreviation",
        "src",
        "tgt",
        automapped_mappings={"Procedure.extension": "Source.abbreviation"},
    )

    assert rule.target[0].element == "modifierExtension"
    assert any(
        diagnostic["code"] == "modifier-extension-rerouted"
        for diagnostic in factory.diagnostics
    )


# ── Skip / not-yet-mappable cases ──────────────────────────────────────────────
def test_non_slice_extension_without_source_is_skipped(factory):
    # A non-slice extension with a resolvable URL but no mapped source would only ever
    # produce an unusable TODO rule -> the factory drops it entirely.
    field = {
        "path": "Patient.extension",
        "type": "Extension",
        "extension_url": "http://example.org/StructureDefinition/simple-ext",
    }
    assert factory.create_extension_rule(field, "extension", "src", "tgt") is None


def test_non_slice_extension_without_resolvable_url_is_skipped(factory):
    field = {"path": "Patient.extension", "type": "Extension"}
    assert factory.create_extension_rule(field, "extension", "src", "tgt") is None


def test_slice_extension_proceeds_even_with_todo_source(factory):
    # Slice extensions are allowed through with a TODO source placeholder (unlike the
    # non-slice case above) — naming reflects the slice name.
    field = {
        "path": "Patient.extension:myExt",
        "type": "Extension",
        "sliceName": "myExt",
        "extension_url": "http://example.org/StructureDefinition/my-ext",
    }
    rule = factory.create_extension_rule(field, "extension:myExt", "src", "tgt")
    assert rule is not None
    assert rule.name == "map-extension-myExt"
    value_rule = rule.rule[1]
    assert value_rule.source[0].element.startswith("TODO")


def test_required_slice_without_provider_is_reported(factory):
    field = {
        "path": "Task.extension:InvoiceSent",
        "type": "Extension",
        "sliceName": "InvoiceSent",
        "extension_url": "http://example.org/StructureDefinition/invoice-sent",
        "value_type": "boolean",
        "cardinality": {"min": 1, "max": "1"},
    }

    rule = factory.create_extension_rule(
        field,
        "extension:InvoiceSent",
        "src",
        "tgt",
        automapped_mappings={},
    )

    assert rule is not None
    assert rule.source[0].element.startswith("TODO")
    assert factory.diagnostics == [
        {
            "code": "required-extension-provider-missing",
            "message": (
                "Required extension slice Task.extension:InvoiceSent has no "
                "authored source provider. Its URL is known, but its semantic "
                "value cannot be inferred from the target profile."
            ),
            "profile": "unknown",
            "path": "Task.extension:InvoiceSent",
            "extension_url": (
                "http://example.org/StructureDefinition/invoice-sent"
            ),
            "severity": "error",
        }
    ]


def test_required_extension_container_without_any_provider_is_reported(factory):
    field = {
        "id": "Procedure.extension",
        "path": "Procedure.extension",
        "type": "Extension",
        "cardinality": {"min": 1, "max": "*"},
        "slices": [
            {
                "id": "Procedure.extension:PerformerAbbreviation",
                "path": "Procedure.extension",
                "type": "Extension",
                "sliceName": "PerformerAbbreviation",
                "extension_url": (
                    "http://example.org/StructureDefinition/performer-abbreviation"
                ),
                "value_type": "string",
                "cardinality": {"min": 0, "max": "1"},
            }
        ],
    }

    factory.create_field_rules(
        "Procedure",
        [field],
        "src",
        "tgt",
        automapped_mappings={},
    )

    diagnostic = next(
        item
        for item in factory.diagnostics
        if item["code"] == "required-extension-container-provider-missing"
    )
    assert diagnostic["path"] == "Procedure.extension"
    assert diagnostic["minimum"] == 1


# ── Complex extension: sub-extension slices resolved from the extension's own SD ───
_COMPLEX_EXT_URL = "http://example.org/StructureDefinition/complex-ext"


def _complex_ext_sd():
    return {
        "snapshot": {
            "element": [
                {"id": "Extension"},
                {
                    "id": "Extension.extension:partA",
                    "min": 1,
                    "max": "1",
                    "type": [{"code": "Extension"}],
                },
                {"id": "Extension.extension:partA.value[x]", "type": [{"code": "string"}]},
                {
                    "id": "Extension.extension:partB",
                    "min": 0,
                    "max": "1",
                    "type": [{"code": "Extension"}],
                },
                {"id": "Extension.extension:partB.value[x]", "type": [{"code": "integer"}]},
            ]
        }
    }


def _complex_ext_field():
    return {
        "path": "Patient.extension:complexExt",
        "type": "Extension",
        "sliceName": "complexExt",
        "extension_url": _COMPLEX_EXT_URL,
        "children": [
            {
                "path": "Patient.extension:complexExt.extension",
                "slices": [
                    {
                        "sliceName": "partA",
                        "path": "Patient.extension:complexExt.extension:partA",
                        "type": "Extension",
                    },
                    {
                        "sliceName": "partB",
                        "path": "Patient.extension:complexExt.extension:partB",
                        "type": "Extension",
                    },
                ],
            },
            {"path": "Patient.extension:complexExt.url", "fixed_value": _COMPLEX_EXT_URL},
        ],
    }


def test_complex_extension_includes_required_and_mapped_sub_extensions(factory):
    factory.app_state = _app_state({_COMPLEX_EXT_URL: SimpleNamespace(data=_complex_ext_sd())})
    automapped = {
        # partA is required (min=1) so it's included even though unmapped; partB (min=0)
        # is only included *because* it's mapped.
        "Patient.extension:complexExt.extension:partB": "Source.partBVal",
    }
    rule = factory.create_extension_rule(
        _complex_ext_field(), "extension:complexExt", "src", "ext-outer",
        automapped_mappings=automapped,
    )
    assert rule is not None
    assert rule.name == "map-extension-complexExt"
    # url rule + partA + partB (the plain ".extension"/".url" children are filtered out)
    assert len(rule.rule) == 3
    url_rule, part_a_rule, part_b_rule = rule.rule
    assert url_rule.name == "set-extension-url-complexExt"

    assert part_a_rule.name == "map-extension-partA"
    a_value_rule = part_a_rule.rule[1]
    assert a_value_rule.target[0].element == "valueString"  # spec-derived value_type
    assert a_value_rule.source[0].element.startswith("TODO")  # unmapped, but still emitted

    assert part_b_rule.name == "map-extension-partB"
    b_value_rule = part_b_rule.rule[1]
    assert b_value_rule.target[0].element == "valueInteger"  # spec-derived value_type
    assert b_value_rule.source[0].element == "partBVal"  # mapped source, local name


def test_complex_extension_drops_unmapped_optional_sub_extension(factory):
    factory.app_state = _app_state({_COMPLEX_EXT_URL: SimpleNamespace(data=_complex_ext_sd())})
    # partB is optional (min=0) and not mapped this time -> must be dropped.
    rule = factory.create_extension_rule(
        _complex_ext_field(), "extension:complexExt", "src", "ext-outer",
        automapped_mappings=None,
    )
    names = [r.name for r in rule.rule]
    assert names == ["set-extension-url-complexExt", "map-extension-partA"]


# ── Coded extension value (pre-resolved _value_binding) -> ConceptMap + translate ──
def test_coded_extension_value_generates_concept_map_and_translate_rule(factory):
    field = {
        "path": "Patient.extension:statusExt",
        "type": "Extension",
        "sliceName": "statusExt",
        "extension_url": "http://example.org/StructureDefinition/status-ext",
        "_value_binding": {
            "value_type": "code",
            "value_set": "http://example.org/ValueSet/status-vs",
            "options": [
                {"code": "active", "system": "http://example.org/CodeSystem/status"},
                {"code": "done", "system": "http://example.org/CodeSystem/status"},
            ],
        },
    }
    rule = factory.create_extension_rule(field, "extension:statusExt", "src", "tgt")
    assert rule is not None
    value_rule = rule.rule[1]
    vt = value_rule.target[0]
    assert vt.element == "valueCode"
    assert vt.transform == "translate"
    cm_url = vt.parameter[1].valueString
    # ConceptMap id/URL is generated from the sub-extension's var_suffix (here the slice
    # name "statusExt"), not from the value-element name ("value").
    assert cm_url.startswith(f"{factory.map_url}/ConceptMap/cm-statusExt-")
    # the ConceptMap was actually persisted
    assert any("cm-statusExt-" in name for _, name, _ in factory.app_state.dataIO.stored)


# ── Value-leaf slice whose children were pruned to url-only (kfdm E15 class-3) ──
# Minimal-mode pruning leaves only the fixed `.url` child under a mapped extension
# slice (the unmapped value[x] is dropped from the tree). The value assignment must
# still be emitted — url-only children are NOT a complex extension.
_YESNO_EXT_URL = "http://example.org/StructureDefinition/yes-no-unknown-extension"


def _yesno_ext_sd():
    return {
        "snapshot": {
            "element": [
                {"id": "Extension"},
                {
                    "id": "Extension.value[x]",
                    "type": [{"code": "code"}],
                    "binding": {
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/yesnodontknow",
                    },
                },
            ]
        }
    }


def _pruned_coded_slice_field():
    return {
        "path": "Condition.extension:existance",
        "type": "Extension",
        "sliceName": "existance",
        "extension_url": _YESNO_EXT_URL,
        "children": [
            {
                "path": "Condition.extension:existance.url",
                "id": "Extension.url",
                "fixed_value": _YESNO_EXT_URL,
            }
        ],
    }


def test_pruned_coded_slice_still_emits_translate_rule(factory, monkeypatch):
    factory.app_state = _app_state({_YESNO_EXT_URL: SimpleNamespace(data=_yesno_ext_sd())})
    options = [
        {"code": "Y", "display": "Yes", "system": "http://terminology.hl7.org/CodeSystem/v2-0136"},
        {"code": "N", "display": "No", "system": "http://terminology.hl7.org/CodeSystem/v2-0136"},
    ]
    monkeypatch.setattr(
        "mapping.fml_creator.fml_extension.expand_valueset",
        lambda vs, args, app_state: options,
    )
    rule = factory.create_extension_rule(
        _pruned_coded_slice_field(), "extension:existance", "src", "tgt",
        automapped_mappings={"Condition.extension:existance": "Source.diabetes"},
    )
    names = [r.name for r in rule.rule]
    assert names[0] == "set-extension-url-existance"
    assert len(rule.rule) == 2, f"value rule missing: {names}"
    value_rule = rule.rule[1]
    vt = value_rule.target[0]
    assert vt.element == "valueCode"
    assert vt.transform == "translate"
    assert value_rule.source[0].element == "diabetes"
    # ConceptMap persisted so the plugin arrows have a regen-safe home
    assert any("cm-existance-" in name for _, name, _ in factory.app_state.dataIO.stored)


def test_pruned_primitive_slice_gets_concrete_value_type(factory):
    # years-of-smoking shape: value[x] is integer, no binding — placeholder must
    # target valueInteger (not the valueExtension/valueString fallback guesses).
    ext_url = "http://example.org/StructureDefinition/years-of-smoking"
    sd = {
        "snapshot": {
            "element": [
                {"id": "Extension"},
                {"id": "Extension.value[x]", "type": [{"code": "integer"}]},
            ]
        }
    }
    factory.app_state = _app_state({ext_url: SimpleNamespace(data=sd)})
    field = {
        "path": "Observation.extension:years",
        "type": "Extension",
        "sliceName": "years",
        "extension_url": ext_url,
        "children": [
            {
                "path": "Observation.extension:years.url",
                "id": "Extension.url",
                "fixed_value": ext_url,
            }
        ],
    }
    rule = factory.create_extension_rule(
        field, "extension:years", "src", "tgt",
        automapped_mappings={"Observation.extension:years": "Source.rauchen_jahre"},
    )
    names = [r.name for r in rule.rule]
    assert len(rule.rule) == 2, f"value rule missing: {names}"
    value_rule = rule.rule[1]
    assert value_rule.target[0].element == "valueInteger"
    assert value_rule.target[0].transform == "copy"
    assert value_rule.source[0].element == "rauchen_jahre"


def _bool_ext_sd():
    return {
        "snapshot": {
            "element": [
                {"id": "Extension"},
                {"id": "Extension.value[x]", "type": [{"code": "boolean"}]},
            ]
        }
    }


def _bool_slice_field(ext_url):
    return {
        "path": "Condition.extension:activity",
        "type": "Extension",
        "sliceName": "activity",
        "extension_url": ext_url,
        "children": [
            {"path": "Condition.extension:activity.url", "id": "Extension.url",
             "fixed_value": ext_url}
        ],
    }


def test_boolean_mismatch_emits_guarded_literal_rules(factory):
    # REDCap yesno "0"/"1" strings into a boolean value[x]: engine `copy` keeps
    # the source type and engine `cast` hard-fails on '0' — must emit guarded
    # literal true/false rules instead.
    ext_url = "http://example.org/StructureDefinition/yes-no-extension"
    factory.app_state = _app_state({ext_url: SimpleNamespace(data=_bool_ext_sd())})
    factory.source_field_types = {"bewegung": "string"}
    rule = factory.create_extension_rule(
        _bool_slice_field(ext_url), "extension:activity", "src", "tgt",
        automapped_mappings={"Condition.extension:activity": "Source.bewegung"},
    )
    names = [r.name for r in rule.rule]
    assert names == [
        "set-extension-url-activity",
        "set-extension-value-activity-true",
        "set-extension-value-activity-false",
    ]
    true_rule, false_rule = rule.rule[1], rule.rule[2]
    assert true_rule.source[0].condition == "$this = '1' or $this = 'true'"
    assert true_rule.target[0].transform == "copy"
    assert true_rule.target[0].parameter[0].valueBoolean is True
    assert false_rule.source[0].condition == "$this = '0' or $this = 'false'"
    assert false_rule.target[0].parameter[0].valueBoolean is False


def test_boolean_matching_source_type_stays_plain_copy(factory):
    ext_url = "http://example.org/StructureDefinition/yes-no-extension"
    factory.app_state = _app_state({ext_url: SimpleNamespace(data=_bool_ext_sd())})
    factory.source_field_types = {"bewegung": "boolean"}
    rule = factory.create_extension_rule(
        _bool_slice_field(ext_url), "extension:activity", "src", "tgt",
        automapped_mappings={"Condition.extension:activity": "Source.bewegung"},
    )
    names = [r.name for r in rule.rule]
    assert names == ["set-extension-url-activity", "set-extension-value-activity"]
    assert rule.rule[1].target[0].element == "valueBoolean"
    assert rule.rule[1].target[0].transform == "copy"


def test_primitive_mismatch_emits_cast(factory):
    # decimal-typed source into an integer value[x] → cast(src, 'integer')
    # (engine-verified: matchbox executes cast to integer/dateTime).
    ext_url = "http://example.org/StructureDefinition/years-of-smoking"
    sd = {
        "snapshot": {
            "element": [
                {"id": "Extension"},
                {"id": "Extension.value[x]", "type": [{"code": "integer"}]},
            ]
        }
    }
    factory.app_state = _app_state({ext_url: SimpleNamespace(data=sd)})
    factory.source_field_types = {"rauchen_jahre": "decimal"}
    field = {
        "path": "Observation.extension:years",
        "type": "Extension",
        "sliceName": "years",
        "extension_url": ext_url,
        "children": [
            {"path": "Observation.extension:years.url", "id": "Extension.url",
             "fixed_value": ext_url}
        ],
    }
    rule = factory.create_extension_rule(
        field, "extension:years", "src", "tgt",
        automapped_mappings={"Observation.extension:years": "Source.rauchen_jahre"},
    )
    value_rule = rule.rule[1]
    vt = value_rule.target[0]
    assert vt.element == "valueInteger"
    assert vt.transform == "cast"
    assert vt.parameter[0].valueId == "srcValue"
    assert vt.parameter[1].valueString == "integer"
    # non-string target with a mapped source keeps the empty-source guard
    assert value_rule.source[0].condition == "$this != ''"


def test_pruned_reference_slice_stays_url_only(factory):
    # body-weight shape: value[x] is Reference (contained-resource boundary) with
    # no resolvable target — must NOT gain an empty `create('Reference')` value.
    ext_url = "http://example.org/StructureDefinition/body-weight-extension"
    sd = {
        "snapshot": {
            "element": [
                {"id": "Extension"},
                {"id": "Extension.value[x]", "type": [{"code": "Reference"}]},
            ]
        }
    }
    factory.app_state = _app_state({ext_url: SimpleNamespace(data=sd)})
    field = {
        "path": "Patient.extension:bodyweight",
        "type": "Extension",
        "sliceName": "bodyweight",
        "extension_url": ext_url,
        "children": [
            {
                "path": "Patient.extension:bodyweight.url",
                "id": "Extension.url",
                "fixed_value": ext_url,
            }
        ],
    }
    rule = factory.create_extension_rule(
        field, "extension:bodyweight", "src", "tgt",
        automapped_mappings={"Patient.extension:bodyweight": "Source.gewicht"},
    )
    assert [r.name for r in rule.rule] == ["set-extension-url-bodyweight"]


def test_complex_extension_with_url_only_children_gets_no_value_rule(factory):
    # A complex extension (sub-extension slices, value[x] prohibited) must NOT
    # gain a value rule from the value-leaf fallback.
    factory.app_state = _app_state({_COMPLEX_EXT_URL: SimpleNamespace(data=_complex_ext_sd())})
    field = _complex_ext_field()
    field["children"].append(
        {"path": "Patient.extension:complexExt.url", "id": "Extension.url",
         "fixed_value": _COMPLEX_EXT_URL}
    )
    rule = factory.create_extension_rule(
        field, "extension:complexExt", "src", "ext-outer",
        automapped_mappings=None,
    )
    names = [r.name for r in rule.rule]
    assert names == ["set-extension-url-complexExt", "map-extension-partA"]


# ── documentation cap (R1, evaluation_results.md §M) ───────────────────────────
def test_extension_doc_string_is_capped_for_huge_type_structures(factory):
    # a complex extension's serialized type structure can reach MBs and blow
    # matchbox's 2 MB request limit — the doc must be truncated with a pointer
    huge_type = [{"code": "Extension", "profile": [[{"path": f"Extension.x{i}",
                  "description": "y" * 50} for i in range(2000)]]}]
    field = {
        "path": "ImagingStudy.extension",
        "type": huge_type,
        "extension_url": "http://example.org/StructureDefinition/huge-ext",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={"ImagingStudy.extension": "Source.flag"},
    )
    assert len(rule.documentation) < 5000
    assert (
        "truncated; full structure in the registry entry for "
        "http://example.org/StructureDefinition/huge-ext" in rule.documentation
    )


def test_extension_doc_string_untouched_when_small(factory):
    field = {
        "path": "Patient.extension",
        "type": "Extension",
        "extension_url": "http://example.org/StructureDefinition/simple-ext",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={"Patient.extension": "Source.flag"},
    )
    assert "truncated" not in rule.documentation
    assert "Type: Extension" in rule.documentation


# ── G1: required extension slice emission (mapped-but-not-emitted) ──────────────
def test_extension_slice_qualified_mapping_key_resolves(factory):
    """A required extension slice is keyed in the mapping table by its
    slice-qualified path (``Condition.extension:Feststellungsdatum``) while
    ``field["path"]`` is the bare base (``Condition.extension``). The outer
    ``create`` rule must resolve its source from the slice-qualified key, else
    it never fires and the required extension is absent (G1)."""
    field = {
        "path": "Condition.extension",
        "type": "Extension",
        "sliceName": "Feststellungsdatum",
        "value_type": "dateTime",
        "extension_url": "http://hl7.org/fhir/StructureDefinition/condition-assertedDate",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={
            "Condition.extension:Feststellungsdatum": "Source.feststellung"
        },
    )
    assert rule is not None
    # outer create rule must have a REAL source element (not the TODO sentinel)
    assert rule.source[0].element == "feststellung"
    assert not rule.source[0].element.startswith("TODO")
    value_rule = rule.rule[1]
    assert value_rule.source[0].element == "feststellung"


def test_extension_url_resolved_from_type_profile(factory):
    """A simple extension slice with no explicit ``extension_url`` carries its
    canonical as its type profile; the url rule must use it instead of emitting
    ``TODO_EXTENSION_URL`` (G1)."""
    field = {
        "path": "Condition.extension",
        "sliceName": "Feststellungsdatum",
        "type": [
            {
                "code": "Extension",
                "profile": [
                    "http://hl7.org/fhir/StructureDefinition/condition-assertedDate"
                ],
            }
        ],
        "value_type": "dateTime",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={
            "Condition.extension:Feststellungsdatum": "Source.feststellung"
        },
    )
    assert rule is not None
    url_rule = rule.rule[0]
    assert url_rule.target[0].element == "url"
    assert (
        url_rule.target[0].parameter[0].valueString
        == "http://hl7.org/fhir/StructureDefinition/condition-assertedDate"
    )


def test_extension_non_slice_base_path_lookup_unchanged(factory):
    """Regression guard: a non-slice extension still resolves via its bare base
    path (the pre-existing behavior must not change)."""
    field = {
        "path": "Patient.extension",
        "type": "Extension",
        "extension_url": "http://example.org/StructureDefinition/simple-ext",
    }
    rule = factory.create_extension_rule(
        field, "extension", "src", "tgt",
        automapped_mappings={"Patient.extension": "Source.flag"},
    )
    assert rule.source[0].element == "flag"
    assert rule.rule[1].source[0].element == "flag"


def _codeable_extension_field():
    """A sliced extension whose value[x] is a CodeableConcept with a coding
    (code + system) — mirrors onkologie Procedure.extension:Intention. The
    children are parsed under the BARE base path (no slice qualifier), exactly
    as the profile parser emits them."""
    return {
        "path": "Procedure.extension",
        "sliceName": "Intention",
        "extension_url": "http://example.org/StructureDefinition/intention-ext",
        "type": [
            {
                "code": "Extension",
                "profile": ["http://example.org/StructureDefinition/intention-ext"],
            }
        ],
        "children": [
            {"path": "Procedure.extension.url", "fixed_value": "x"},
            {
                "path": "Procedure.extension.value[x]",
                "type": [
                    {
                        "code": "CodeableConcept",
                        "type_structure": [
                            {
                                "path": "Procedure.extension.value[x].coding",
                                "type": "Coding",
                                "type_structure": [
                                    {
                                        "path": "Procedure.extension.value[x].coding.code",
                                        "type": "code",
                                    },
                                    {
                                        "path": "Procedure.extension.value[x].coding.system",
                                        "type": "uri",
                                    },
                                    {
                                        "path": "Procedure.extension.value[x].coding.display",
                                        "type": "string",
                                    },
                                ],
                            },
                            {"path": "Procedure.extension.value[x].text", "type": "string"},
                        ],
                    }
                ],
                "cardinality": {"min": 1, "max": "1"},
                "options": [],
                "valueSetUrl": "http://example.org/ValueSet/codes",
                "children": [],
            },
        ],
    }


def test_coded_extension_value_rekeys_slice_qualified_mapping(factory):
    """G8: a coded CodeableConcept extension value mapped via a slice-qualified
    key (``…extension:Intention.value[x].coding.code``) must be re-keyed onto the
    bare child path so the value is actually emitted, not left url-only."""
    field = _codeable_extension_field()
    rule = factory.create_extension_rule(
        field,
        "extension",
        "source",
        "tgt",
        automapped_mappings={
            # leaf targets + the ancestor keys the mapping layer injects
            # (_apply_custom_mapping_table) so intermediate backbones get created.
            "Procedure.extension:Intention.value[x]": "Source.procIntention",
            "Procedure.extension:Intention.value[x].coding": "Source.procIntention",
            "Procedure.extension:Intention.value[x].coding.code": "Source.procIntention",
            "Procedure.extension:Intention.value[x].coding.system": "Source.procIntentionSystem",
        },
    )
    assert rule is not None
    # The outer create + url rule must be followed by a value-populating rule —
    # not just [create, url] (which is the hollow url-only regression).
    assert len(rule.rule) > 1, "coded extension value was not emitted (url-only)"
    flat = str([r.model_dump(exclude_none=True) for r in rule.rule])
    assert "procIntention" in flat, "mapped value source never resolved onto the value"
    assert "TODO-MAP-INTENTION_SOURCE" not in flat
