"""Characterization tests for the pure type/name helpers in the mapping layer.

These pin the exact behaviour of the primitive/type tables and name utilities that the
Tier-1 (spec-derivation) and Tier-2 (de-duplication) refactors touch, so that a behaviour-
preserving change is provably behaviour-preserving and an intended spec-correction is the
only thing that moves an assertion.
"""

from types import SimpleNamespace

import pytest

from mapping.fml_creator import fml_helper as H
from mapping.fml_creator.fml_factory import FMLRuleFactory

pytestmark = pytest.mark.unit


# ── is_primitive_type ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "type_name,expected",
    [
        ("string", True),
        ("String", True),  # case-insensitive
        ("dateTime", True),
        ("datetime", True),
        ("base64Binary", True),
        ("code", True),
        ("uuid", True),
        ("Quantity", False),
        ("CodeableConcept", False),
        ("Extension", False),
        ("", False),
        (None, False),
        # FHIRPath System.* primitives are treated as primitive (must be cast/copied,
        # never create()d — matchbox cannot create System.DateTime).
        ("System.String", True),
        ("http://hl7.org/fhirpath/System.DateTime", True),
    ],
)
def test_is_primitive_type(type_name, expected):
    assert H.is_primitive_type(type_name) is expected


def test_primitive_set_matches_spec_lowercased():
    """The hand-maintained PRIMITIVE_TYPES set is exactly the parser's spec-derived
    PRIMITIVES, lowercased. (Tier 1 unifies these — this guards the equivalence.)"""
    from parser.resource_parser.fhir_type_introspection import PRIMITIVES

    assert H.PRIMITIVE_TYPES == {p.lower() for p in PRIMITIVES}


# ── get_transform_for_type ───────────────────────────────────────────────────
@pytest.mark.parametrize(
    "type_name,expected_transform",
    [
        ("string", "copy"),
        ("boolean", "copy"),
        ("code", "copy"),
        ("uuid", "copy"),
        ("base64binary", "copy"),
        ("date", "cast"),
        ("dateTime", "cast"),
        ("time", "cast"),
        ("instant", "cast"),
        ("integer", "cast"),
        ("decimal", "cast"),
        ("positiveint", "cast"),
        ("Quantity", "create"),
        ("CodeableConcept", "create"),
        ("Reference", "create"),
    ],
)
def test_get_transform_for_type(type_name, expected_transform):
    assert H.get_transform_for_type(type_name)["transform"] == expected_transform


def test_get_transform_cast_carries_type_param():
    info = H.get_transform_for_type("dateTime", "srcVar")
    assert info["transform"] == "cast"
    # cast params: [valueId(source), valueString(type)]
    assert info["parameters"][-1].valueString == "dateTime"


def test_copy_cast_partition_covers_all_primitives():
    """Every R4B primitive has an explicit value-producing transform."""
    assert (H.COPYABLE_PRIMITIVES | H.CASTABLE_PRIMITIVES) == (
        H.PRIMITIVE_TYPES
    )
    assert not (H.COPYABLE_PRIMITIVES & H.CASTABLE_PRIMITIVES)


# ── infer_extension_value_type ───────────────────────────────────────────────
@pytest.mark.parametrize(
    "field_type,expected",
    [
        ("string", "valueString"),
        ("integer", "valueInteger"),
        ("boolean", "valueBoolean"),
        ("code", "valueCode"),
        ("CodeableConcept", "valueCodeableConcept"),
        ("Reference", "valueReference"),
        ("Quantity", "valueQuantity"),
    ],
)
def test_infer_extension_value_type_correct_cases(field_type, expected):
    assert H.infer_extension_value_type(field_type, {}) == expected


@pytest.mark.parametrize(
    "field_type,correct_value_x",
    [
        ("dateTime", "valueDateTime"),
        ("base64Binary", "valueBase64Binary"),
        ("unsignedInt", "valueUnsignedInt"),
        ("positiveInt", "valuePositiveInt"),
    ],
)
def test_infer_extension_value_type_multicase_primitives(field_type, correct_value_x):
    """Multi-case primitives must keep their camelCase. The former ``str.capitalize()``
    lowercased the tail and produced INVALID names (valueDatetime, valueBase64binary, ...);
    Tier 1 routes this through the spec-derived canonical suffix, fixing the casing."""
    assert H.infer_extension_value_type(field_type, {}) == correct_value_x


def test_infer_extension_value_type_uses_value_type_field_override():
    # Isolates the field['value_type'] override path from the multi-case casing bug
    # (covered separately above): an integer override has no ambiguous casing.
    assert (
        H.infer_extension_value_type("Extension", {"value_type": "integer"})
        == "valueInteger"
    )


# ── clean_field_name ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name,expected",
    [
        ("value[x]", "value-x"),
        ("data_absent_reason", "data-absent-reason"),
        ("foo.bar", "foo-bar"),
        ("--x--", "x"),
    ],
)
def test_clean_field_name(name, expected):
    assert H.clean_field_name(name) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (3.5, "3.5"),
        ("MR", "MR"),
    ],
)
def test_fixed_scalar_literal(value, expected):
    # Booleans must take FHIR lexical form, not Python's str() capitalization.
    assert H.fixed_scalar_literal(value) == expected


# ── factory pure helpers (_choice_suffix / _as_local_element) ─────────────────
@pytest.fixture
def factory():
    # _choice_suffix and _as_local_element read no instance state, so bypass __init__.
    return object.__new__(FMLRuleFactory)


@pytest.mark.parametrize(
    "choice_type,expected",
    [
        ("boolean", "Boolean"),
        ("dateTime", "DateTime"),
        ("datetime", "DateTime"),
        ("base64Binary", "Base64Binary"),
        ("unsignedInt", "UnsignedInt"),
        ("code", "Code"),
        ("Quantity", "Quantity"),
        ("CodeableConcept", "CodeableConcept"),
        ("", ""),
    ],
)
def test_choice_suffix(factory, choice_type, expected):
    assert factory._choice_suffix(choice_type) == expected


@pytest.mark.parametrize(
    "source_id,expected",
    [
        ("Patient.name", "name"),
        ("field", "field"),
        ("A.B.C", "C"),
        ("TODO_MAP_X", "TODO_MAP_X"),  # TODO sentinels pass through unchanged
        ("", ""),
    ],
)
def test_as_local_element(factory, source_id, expected):
    assert factory._as_local_element(source_id) == expected


def test_as_local_element_delegates_to_shared_helper(factory):
    # The factory method and the questionnaire creator now share one implementation.
    assert factory._as_local_element("Obs.code") == H.local_element_name("Obs.code")


# ── factory pure helpers (_apply_repeating_list_modes / _reference_base_type) ─
@pytest.mark.parametrize("max_card", ["*", "n"])
def test_apply_repeating_list_modes_sets_share_mode(factory, max_card):
    target = SimpleNamespace(listMode=None, listRuleId=None)
    factory._apply_repeating_list_modes({"max": max_card}, source=None, target=target, rule_name="r1")
    assert target.listMode == ["share"]
    assert target.listRuleId == "r1"


def test_apply_repeating_list_modes_noop_for_single_cardinality(factory):
    target = SimpleNamespace(listMode=None, listRuleId=None)
    factory._apply_repeating_list_modes({"max": "1"}, source=None, target=target, rule_name="r1")
    assert target.listMode is None
    assert target.listRuleId is None


def test_reference_base_type_none_for_falsy_or_non_str_url(factory):
    assert factory._reference_base_type(None) is None
    assert factory._reference_base_type("") is None
    assert factory._reference_base_type(123) is None


def test_reference_base_type_none_when_unresolved(monkeypatch, factory):
    import mapping.fml_creator.fml_factory as fml_factory_module

    monkeypatch.setattr(fml_factory_module, "resolve_url", lambda url, app_state: None)
    factory.app_state = SimpleNamespace()
    assert factory._reference_base_type("http://x/StructureDefinition/Foo") is None


def test_reference_base_type_none_on_resolve_exception(monkeypatch, factory):
    import mapping.fml_creator.fml_factory as fml_factory_module

    def raising_resolve(url, app_state):
        raise RuntimeError("boom")

    monkeypatch.setattr(fml_factory_module, "resolve_url", raising_resolve)
    factory.app_state = SimpleNamespace()
    assert factory._reference_base_type("http://x/StructureDefinition/Foo") is None


def test_reference_base_type_returns_type_from_resolved_sd(monkeypatch, factory):
    import mapping.fml_creator.fml_factory as fml_factory_module

    monkeypatch.setattr(
        fml_factory_module, "resolve_url", lambda url, app_state: SimpleNamespace(type="Condition")
    )
    factory.app_state = SimpleNamespace()
    assert factory._reference_base_type("http://x/StructureDefinition/MinimalCondition3") == "Condition"


def test_reference_base_type_unwraps_list_type(monkeypatch, factory):
    import mapping.fml_creator.fml_factory as fml_factory_module

    monkeypatch.setattr(
        fml_factory_module, "resolve_url", lambda url, app_state: SimpleNamespace(type=["Condition"])
    )
    factory.app_state = SimpleNamespace()
    assert factory._reference_base_type("http://x/StructureDefinition/MinimalCondition3") == "Condition"


# ── shared type normalization helpers (Tier 2 de-duplication) ─────────────────
@pytest.mark.parametrize(
    "field_type,expected",
    [
        (None, []),
        ("Extension", ["Extension"]),
        (["string"], ["string"]),
        ({"code": "Reference"}, ["Reference"]),
        ([{"code": "Reference"}, {"code": "canonical"}], ["Reference", "canonical"]),
        ([{"profile": ["x"]}], []),  # type dict without a code is dropped
        ([{"code": "Extension"}, "boolean"], ["Extension", "boolean"]),
    ],
)
def test_type_codes(field_type, expected):
    assert H.type_codes(field_type) == expected


@pytest.mark.parametrize(
    "field_type,is_ext,is_ref",
    [
        ("Extension", True, False),
        ("Reference", False, True),
        ([{"code": "Extension", "profile": ["p"]}], True, False),
        ([{"code": "Reference", "targetProfile": ["p"]}], False, True),
        ("string", False, False),
        (None, False, False),
    ],
)
def test_is_extension_and_reference_type(field_type, is_ext, is_ref):
    assert H.is_extension_type(field_type) is is_ext
    assert H.is_reference_type(field_type) is is_ref


# ── is_meaningful_modifier_extension ─────────────────────────────────────────
def test_is_meaningful_modifier_extension_non_dict_returns_false():
    assert H.is_meaningful_modifier_extension("not-a-dict") is False


def test_is_meaningful_modifier_extension_wrong_path_returns_false():
    assert H.is_meaningful_modifier_extension({"path": "Patient.extension"}) is False


@pytest.mark.parametrize(
    "field,expected",
    [
        ({"path": "Patient.modifierExtension"}, False),
        ({"path": "Patient.modifierExtension", "sliceName": "myslice"}, True),
        ({"path": "Patient.modifierExtension", "slices": [{}]}, True),
        ({"path": "Patient.modifierExtension", "extension_url": "http://x"}, True),
    ],
)
def test_is_meaningful_modifier_extension_slice_markers(field, expected):
    assert H.is_meaningful_modifier_extension(field) is expected


# ── attr ──────────────────────────────────────────────────────────────────────
def test_attr_reads_from_dict():
    assert H.attr({"path": "Patient.name"}, "path") == "Patient.name"


def test_attr_reads_default_when_missing_from_dict():
    assert H.attr({}, "path", "fallback") == "fallback"


def test_attr_reads_from_object_via_getattr():
    obj = SimpleNamespace(path="Patient.name")
    assert H.attr(obj, "path") == "Patient.name"


def test_attr_reads_default_when_missing_from_object():
    obj = SimpleNamespace()
    assert H.attr(obj, "path", "fallback") == "fallback"


# ── canonical_primitive / fhir_type_suffix ───────────────────────────────────
@pytest.mark.parametrize(
    "type_code,expected",
    [
        ("string", "string"),
        ("STRING", "string"),
        ("dateTime", "dateTime"),
        ("datetime", "dateTime"),
        ("CodeableConcept", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_primitive(type_code, expected):
    assert H.canonical_primitive(type_code) == expected


@pytest.mark.parametrize(
    "type_code,expected",
    [
        ("string", "String"),
        ("dateTime", "DateTime"),
        ("datetime", "DateTime"),
        ("", ""),
        (None, ""),
        # non-primitive types pass through unchanged except for the first-letter uppercase
        ("CodeableConcept", "CodeableConcept"),
        ("myCustomType", "MyCustomType"),
    ],
)
def test_fhir_type_suffix(type_code, expected):
    assert H.fhir_type_suffix(type_code) == expected


# ── infer_extension_value_type: base_type normalization branches ────────────
def test_infer_extension_value_type_list_value_type_override():
    assert (
        H.infer_extension_value_type("Extension", {"value_type": ["dateTime"]})
        == "valueDateTime"
    )


def test_infer_extension_value_type_empty_list_defaults_to_string():
    assert (
        H.infer_extension_value_type("Extension", {"value_type": []}) == "valueString"
    )


def test_infer_extension_value_type_dict_value_type_override():
    assert (
        H.infer_extension_value_type("Extension", {"value_type": {"code": "integer"}})
        == "valueInteger"
    )


def test_infer_extension_value_type_dict_without_code_defaults_to_string():
    assert (
        H.infer_extension_value_type("Extension", {"value_type": {}}) == "valueString"
    )


def test_infer_extension_value_type_non_string_defaults_to_string():
    assert (
        H.infer_extension_value_type("Extension", {"value_type": 123}) == "valueString"
    )


# ── parse_slice_info ──────────────────────────────────────────────────────────
def test_parse_slice_info_no_slice_marker():
    is_slice, info = H.parse_slice_info("name", {})
    assert is_slice is False
    assert info == {}


def test_parse_slice_info_colon_in_field_name():
    is_slice, info = H.parse_slice_info("identifier:mrn", {})
    assert is_slice is True
    assert info == {"base_info": "identifier", "slice_name": "mrn"}


def test_parse_slice_info_colon_with_empty_slice_part():
    # "identifier:".split(":") -> ["identifier", ""]; len(parts) > 1 so the (empty)
    # second part is used verbatim -- the "unknown" default only applies when there
    # is no second part at all (no colon in the name).
    is_slice, info = H.parse_slice_info("identifier:", {})
    assert is_slice is True
    assert info == {"base_info": "identifier", "slice_name": ""}


def test_parse_slice_info_slicename_without_colon():
    is_slice, info = H.parse_slice_info("identifier", {"sliceName": "mrn"})
    assert is_slice is True
    assert info == {"base_info": "identifier", "slice_name": "mrn"}


def test_parse_slice_info_with_discriminator():
    field = {
        "sliceName": "mrn",
        "slicing": {"discriminator": [{"path": "type", "type": "pattern"}]},
        "fixed_value": "MR",
    }
    is_slice, info = H.parse_slice_info("identifier", field)
    assert is_slice is True
    assert info["discriminator"] == {
        "path": "type",
        "type": "pattern",
        "value": "MR",
    }


# ── find_reference_fields ─────────────────────────────────────────────────────
def test_find_reference_fields_simple():
    fields = [
        {"type": "Reference", "reference_target": "Patient"},
        {"type": "string"},
    ]
    assert H.find_reference_fields(fields) == {"Patient"}


def test_find_reference_fields_skips_reference_without_target():
    fields = [{"type": "Reference"}]
    assert H.find_reference_fields(fields) == set()


def test_find_reference_fields_recurses_into_children_and_slices():
    fields = [
        {
            "type": "BackboneElement",
            "children": [{"type": "Reference", "reference_target": "Organization"}],
        },
        {
            "type": "BackboneElement",
            "slices": [{"type": "Reference", "reference_target": "Practitioner"}],
        },
    ]
    assert H.find_reference_fields(fields) == {"Organization", "Practitioner"}


# ── conv_mappable ──────────────────────────────────────────────────────────
def make_app_state(registry_hit=True, resolve_result=None):
    registry = SimpleNamespace(get_obj_by_name=lambda u: registry_hit)
    app_state = SimpleNamespace(registry=registry)
    return app_state


def test_conv_mappable_multi_type_choice():
    field = {
        "path": "Observation.value[x]",
        "type": [{"code": "string"}, {"code": "boolean"}],
    }
    conv = H.conv_mappable(make_app_state(), field)
    assert conv["type"] == "choice"
    assert conv["choice_types"] == ["string", "boolean"]
    assert conv["is_type_choice"] is True


def test_conv_mappable_single_type_non_reference():
    field = {"path": "Patient.name", "type": [{"code": "HumanName"}]}
    conv = H.conv_mappable(make_app_state(), field)
    assert conv["type"] == "HumanName"
    assert "reference_target" not in conv


def test_conv_mappable_does_not_mutate_parser_field():
    field = {"path": "Patient.name", "type": [{"code": "HumanName"}]}
    conv = H.conv_mappable(make_app_state(), field)

    assert conv is not field
    assert conv["type"] == "HumanName"
    assert field["type"] == [{"code": "HumanName"}]


def test_conv_mappable_reference_with_target_profile(monkeypatch):
    field = {
        "path": "Observation.subject",
        "type": [
            {
                "code": "Reference",
                "targetProfile": ["http://hl7.org/fhir/StructureDefinition/Patient"],
            }
        ],
    }

    def fake_resolve_url(url, app_state):
        return SimpleNamespace(type="Patient")

    monkeypatch.setattr(H, "resolve_url", fake_resolve_url)

    conv = H.conv_mappable(make_app_state(registry_hit=True), field)
    assert conv["type"] == "Reference"
    assert conv["reference_target"] == "Patient"
    assert conv["reference_target_profile"] == "http://hl7.org/fhir/StructureDefinition/Patient"
    assert conv["reference_target_in_registry"] is True


def test_conv_mappable_reference_falls_back_to_url_tail_when_unresolved(monkeypatch):
    field = {
        "path": "Observation.subject",
        "type": [
            {
                "code": "Reference",
                "targetProfile": ["http://hl7.org/fhir/StructureDefinition/Patient"],
            }
        ],
    }

    monkeypatch.setattr(H, "resolve_url", lambda url, app_state: None)

    conv = H.conv_mappable(make_app_state(registry_hit=False), field)
    assert conv["reference_target"] == "Patient"
    assert conv["reference_target_in_registry"] is False


def test_conv_mappable_reference_dedups_candidates_across_profiles(monkeypatch):
    field = {
        "path": "Observation.subject",
        "type": [
            {
                "code": "Reference",
                "targetProfile": [
                    "http://x/StructureDefinition/Patient",
                    "http://y/StructureDefinition/Patient",
                    "http://z/StructureDefinition/Group",
                ],
            }
        ],
    }
    monkeypatch.setattr(H, "resolve_url", lambda url, app_state: None)

    conv = H.conv_mappable(make_app_state(), field)
    assert conv["reference_target"] == "Patient|Group"


def test_conv_mappable_carries_profile_and_type_structure():
    field = {
        "path": "Patient.extension",
        "type": [{"code": "Extension", "profile": ["profileA"], "type_structure": ["ts"]}],
    }
    conv = H.conv_mappable(make_app_state(), field)
    assert conv["profile_structure"] == ["profileA"]
    assert conv["type_structure"] == ["ts"]


def test_conv_mappable_empty_type_list_defaults_to_string():
    field = {"path": "Patient.foo", "type": []}
    conv = H.conv_mappable(make_app_state(), field)
    assert conv["type"] == "string"


def test_conv_mappable_string_type_field():
    field = {"path": "Patient.foo", "type": "code"}
    conv = H.conv_mappable(make_app_state(), field)
    assert conv["type"] == "code"


def test_conv_mappable_no_type_key_defaults_to_string():
    field = {"path": "Patient.foo"}
    conv = H.conv_mappable(make_app_state(), field)
    assert conv["type"] == "string"


# ── flatten_profile_fields ───────────────────────────────────────────────────
def test_flatten_profile_fields_skips_base_meta_paths():
    fields = [
        {"path": "Patient.id", "type": "id"},
        {"path": "Patient.name", "type": [{"code": "HumanName"}]},
    ]
    results = H.flatten_profile_fields(make_app_state(), "Patient", fields)
    paths = [r["path"] for r in results]
    assert "Patient.id" not in paths
    assert "Patient.name" in paths


def test_flatten_profile_fields_includes_explicit_base_or_narrative_paths():
    fields = [
        {"path": "Patient.id", "type": "id"},
        {
            "path": "Patient.text",
            "type": [{"code": "Narrative"}],
        },
        {"path": "Patient.contained", "type": [{"code": "Resource"}]},
    ]
    results = H.flatten_profile_fields(
        make_app_state(),
        "Patient",
        fields,
        include_special_paths={"Patient.id", "Patient.text.div"},
    )
    assert [item["path"] for item in results] == ["Patient.id", "Patient.text"]


def test_xhtml_uses_copy_transform():
    info = H.get_transform_for_type("xhtml", "src-div")
    assert info["transform"] == "copy"
    assert info["parameters"][0].valueId == "src-div"


def test_flatten_profile_fields_keeps_meaningful_modifier_extension():
    fields = [
        {
            "path": "Patient.modifierExtension",
            "type": [{"code": "Extension"}],
            "sliceName": "my-slice",
        }
    ]
    results = H.flatten_profile_fields(make_app_state(), "Patient", fields)
    assert len(results) == 1
    assert results[0]["path"] == "Patient.modifierExtension"


def test_flatten_profile_fields_derives_extension_url_and_children():
    fields = [
        {
            "path": "Patient.extension",
            "type": [
                {
                    "code": "Extension",
                    "profile": [
                        [
                            {"path": "Extension.url", "fixed_value": "http://x/ext"},
                            {"path": "Extension.value[x]", "type": [{"code": "string"}]},
                        ]
                    ],
                }
            ],
        }
    ]
    results = H.flatten_profile_fields(make_app_state(), "Patient", fields)
    assert len(results) == 1
    top = results[0]
    assert top["extension_url"] == "http://x/ext"
    # children includes every nested field (the .url slot too, not just .value[x])
    child_paths = [c["path"] for c in top["children"]]
    assert child_paths == ["Patient.extension.url", "Patient.extension.value[x]"]


def test_flatten_profile_fields_processes_slices():
    fields = [
        {
            "path": "Patient.identifier",
            "type": [{"code": "Identifier"}],
            "slices": [{"sliceName": "mrn", "path": "Patient.identifier", "type": [{"code": "Identifier"}]}],
        }
    ]
    results = H.flatten_profile_fields(make_app_state(), "Patient", fields)
    paths = [r["path"] for r in results]
    assert "Patient.identifier" in paths
    assert "Patient.identifier:mrn" in paths


# ── process_slice ─────────────────────────────────────────────────────────
def test_process_slice_non_dict_returns_empty_list():
    assert H.process_slice(make_app_state(), {"path": "Patient.foo"}, "not-a-dict") == []


def test_process_slice_builds_sliced_path():
    field = {"path": "Patient.identifier"}
    sl = {"sliceName": "mrn", "type": [{"code": "Identifier"}]}
    result = H.process_slice(make_app_state(), field, sl)
    assert len(result) == 1
    assert result[0]["path"] == "Patient.identifier:mrn"


def test_process_slice_defaults_unknown_slice_name():
    field = {"path": "Patient.identifier"}
    sl = {"type": [{"code": "Identifier"}]}
    result = H.process_slice(make_app_state(), field, sl)
    assert result[0]["path"] == "Patient.identifier:unknown"


def test_process_slice_extracts_inline_profile_and_discriminator():
    field = {
        "path": "Patient.extension",
        "slicing_type": "value",
        "discriminator_path": "url",
    }
    sl = {
        "sliceName": "my-ext",
        "type": [{"code": "Extension", "profile": ["inline-profile"]}],
    }
    result = H.process_slice(make_app_state(), field, sl)
    conv_slice = result[0]
    assert conv_slice["profile_structure"] == ["inline-profile"]
    assert conv_slice["slicing"]["discriminator"][0] == {"path": "url", "type": "value"}
    assert conv_slice["discriminator_path"] == "url"
    assert conv_slice["slicing_type"] == "value"


def test_process_slice_preserves_snapshot_children_for_unresolved_profile():
    """A canonical-only profile reference must not erase children parsed from the snapshot."""
    children = [
        {
            "id": "Patient.address:Strassenanschrift.line",
            "path": "Patient.address.line",
            "type": [{"code": "string"}],
            "is_required": True,
        },
        {
            "id": "Patient.address:Strassenanschrift.country",
            "path": "Patient.address.country",
            "type": [{"code": "string"}],
            "is_required": True,
        },
    ]
    field = {"path": "Patient.address", "type": [{"code": "Address"}]}
    sl = {
        "sliceName": "Strassenanschrift",
        "type": [
            {
                "code": "Address",
                "profile": ["http://example.org/StructureDefinition/GermanAddress"],
            }
        ],
        "children": children,
    }

    result = H.process_slice(make_app_state(), field, sl)

    assert [child["id"] for child in result[0]["children"]] == [
        "Patient.address:Strassenanschrift.line",
        "Patient.address:Strassenanschrift.country",
    ]
    assert sl["children"] == children


def test_flatten_profile_fields_does_not_truncate_slice_tree():
    slice_field = {
        "sliceName": "Strassenanschrift",
        "type": [
            {
                "code": "Address",
                "profile": ["http://example.org/StructureDefinition/GermanAddress"],
            }
        ],
        "children": [
            {
                "id": "Patient.address:Strassenanschrift.line",
                "path": "Patient.address.line",
                "type": [{"code": "string"}],
                "is_required": True,
            }
        ],
    }
    fields = [
        {
            "path": "Patient.address",
            "type": [{"code": "Address"}],
            "slices": [slice_field],
        }
    ]

    flattened = H.flatten_profile_fields(make_app_state(), "Patient", fields)
    street_address = next(
        field for field in flattened if field["path"] == "Patient.address:Strassenanschrift"
    )

    assert street_address["children"][0]["id"] == "Patient.address:Strassenanschrift.line"
    assert slice_field["children"][0]["id"] == "Patient.address:Strassenanschrift.line"


def test_process_slice_extracts_extension_url_and_children_from_profile_structure():
    field = {"path": "Patient.extension"}
    sl = {
        "sliceName": "my-ext",
        "profile_structure": [
            [
                {"path": "Extension.url", "fixed_value": "http://x/ext"},
                {"path": "Extension.value[x]", "type": [{"code": "string"}]},
            ]
        ],
    }
    result = H.process_slice(make_app_state(), field, sl)
    conv_slice = result[0]
    assert conv_slice["extension_url"] == "http://x/ext"
    child_paths = [c["path"] for c in conv_slice["children"]]
    assert child_paths == ["Patient.extension:my-ext.url", "Patient.extension:my-ext.value[x]"]
