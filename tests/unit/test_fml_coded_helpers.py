"""Unit tests for the pure/leaf helper methods on _CodedRulesMixin
(mapping/fml_creator/fml_coded.py), driven through FMLRuleFactory (which mixes it in).

Only self-contained helpers that can be exercised with small dict fixtures are
covered here -- full map-generation is exercised by tests/integration/test_golden_regen.py.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

import mapping.fml_creator.fml_coded as fml_coded_module
from mapping.fml_creator.fml_factory import FMLRuleFactory

pytestmark = pytest.mark.unit


@pytest.fixture
def factory():
    """Bypass __init__ (heavy: app_state, registries, ...) for pure helper methods."""
    return object.__new__(FMLRuleFactory)


# ── _coded_source_keys ────────────────────────────────────────────────────────
def test_coded_source_keys_no_path_returns_empty(factory):
    assert factory._coded_source_keys({}) == []


def test_coded_source_keys_default_type_is_field_path_only(factory):
    assert factory._coded_source_keys({"path": "Observation.code"}) == [
        "Observation.code"
    ]


def test_coded_source_keys_codeable_concept_adds_leaf_paths(factory):
    keys = factory._coded_source_keys(
        {"path": "Observation.code", "type": "CodeableConcept"}
    )
    assert keys == [
        "Observation.code",
        "Observation.code.coding.code",
        "Observation.code.coding",
        "Observation.code.text",
    ]


def test_coded_source_keys_accepts_raw_parser_type_shape(factory):
    keys = factory._coded_source_keys(
        {"path": "Observation.code", "type": [{"code": "CodeableConcept"}]}
    )
    assert "Observation.code.coding.code" in keys


def test_coded_source_keys_coding_adds_code_leaf(factory):
    keys = factory._coded_source_keys({"path": "Observation.code", "type": "Coding"})
    assert keys == ["Observation.code", "Observation.code.code"]


# ── _resolve_coded_source ─────────────────────────────────────────────────────
def test_resolve_coded_source_no_mappings_returns_none(factory):
    assert factory._resolve_coded_source({"path": "Observation.code"}, None) is None
    assert factory._resolve_coded_source({"path": "Observation.code"}, {}) is None


def test_resolve_coded_source_matches_field_path(factory):
    field = {"path": "Observation.code", "type": "code"}
    mappings = {"Observation.code": "src.code_field"}
    assert factory._resolve_coded_source(field, mappings) == "code_field"


def test_resolve_coded_source_matches_leaf_key_over_missing_field_path(factory):
    field = {"path": "Observation.code", "type": "CodeableConcept"}
    mappings = {"Observation.code.coding.code": "src.raw_code"}
    assert factory._resolve_coded_source(field, mappings) == "raw_code"


def test_resolve_coded_source_no_match_returns_none(factory):
    field = {"path": "Observation.code", "type": "code"}
    assert factory._resolve_coded_source(field, {"Other.path": "x"}) is None


# ── _options_are_filter_based ─────────────────────────────────────────────────
def test_options_are_filter_based_empty_list_is_false(factory):
    assert factory._options_are_filter_based([]) is False


def test_options_are_filter_based_all_filter_sentinels(factory):
    options = [{"code": "FILTER:is-a"}, {"code": "FROM_CS"}]
    assert factory._options_are_filter_based(options) is True


def test_options_are_filter_based_mixed_is_false(factory):
    options = [{"code": "FILTER:is-a"}, {"code": "1234-5"}]
    assert factory._options_are_filter_based(options) is False


# ── _system_from_options ──────────────────────────────────────────────────────
def test_system_from_options_returns_first_system(factory):
    options = [
        {"code": "a"},
        {"code": "b", "system": "http://sys1"},
        {"code": "c", "system": "http://sys2"},
    ]
    assert factory._system_from_options(options) == "http://sys1"


def test_system_from_options_none_when_no_system_present(factory):
    assert factory._system_from_options([{"code": "a"}]) is None


# ── _resolve_valueset_compose_includes / _bound_valueset_is_intensional / _system_from_valueset ──
def test_resolve_valueset_compose_includes_no_url_returns_empty(factory):
    assert factory._resolve_valueset_compose_includes({}) == []


def test_resolve_valueset_compose_includes_resolve_failure_returns_empty(
    monkeypatch, factory
):
    def raising_resolve(url, app_state):
        raise RuntimeError("boom")

    monkeypatch.setattr(fml_coded_module, "resolve_url", raising_resolve)
    factory.app_state = SimpleNamespace()
    assert (
        factory._resolve_valueset_compose_includes({"valueSetUrl": "http://vs"}) == []
    )


def test_resolve_valueset_compose_includes_unresolved_returns_empty(
    monkeypatch, factory
):
    monkeypatch.setattr(fml_coded_module, "resolve_url", lambda url, app_state: None)
    factory.app_state = SimpleNamespace()
    assert (
        factory._resolve_valueset_compose_includes({"valueSetUrl": "http://vs"}) == []
    )


def test_resolve_valueset_compose_includes_no_compose_returns_empty(
    monkeypatch, factory
):
    monkeypatch.setattr(fml_coded_module, "resolve_url", lambda url, app_state: {})
    factory.app_state = SimpleNamespace()
    assert (
        factory._resolve_valueset_compose_includes({"valueSetUrl": "http://vs"}) == []
    )


def test_resolve_valueset_compose_includes_returns_include_list(monkeypatch, factory):
    vs = {"compose": {"include": [{"system": "http://sys1"}]}}
    monkeypatch.setattr(fml_coded_module, "resolve_url", lambda url, app_state: vs)
    factory.app_state = SimpleNamespace()
    result = factory._resolve_valueset_compose_includes({"valueSetUrl": "http://vs"})
    assert result == [{"system": "http://sys1"}]


def test_bound_valueset_is_intensional_true_when_filter_present(monkeypatch, factory):
    vs = {"compose": {"include": [{"filter": [{"property": "concept"}]}]}}
    monkeypatch.setattr(fml_coded_module, "resolve_url", lambda url, app_state: vs)
    factory.app_state = SimpleNamespace()
    assert factory._bound_valueset_is_intensional({"valueSetUrl": "http://vs"}) is True


def test_bound_valueset_is_intensional_false_without_filter(monkeypatch, factory):
    vs = {"compose": {"include": [{"system": "http://sys1"}]}}
    monkeypatch.setattr(fml_coded_module, "resolve_url", lambda url, app_state: vs)
    factory.app_state = SimpleNamespace()
    assert factory._bound_valueset_is_intensional({"valueSetUrl": "http://vs"}) is False


def test_system_from_valueset_returns_first_system(monkeypatch, factory):
    vs = {"compose": {"include": [{}, {"system": "http://sys1"}]}}
    monkeypatch.setattr(fml_coded_module, "resolve_url", lambda url, app_state: vs)
    factory.app_state = SimpleNamespace()
    assert factory._system_from_valueset({"valueSetUrl": "http://vs"}) == "http://sys1"


def test_system_from_valueset_none_when_no_includes(factory):
    factory.app_state = SimpleNamespace()
    assert factory._system_from_valueset({}) is None


# ── _fixed_system_from_field ──────────────────────────────────────────────────
def test_fixed_system_from_field_no_children_returns_none(factory):
    assert factory._fixed_system_from_field({}) is None


def test_fixed_system_from_field_finds_coding_system_leaf():
    field = {
        "children": [
            {
                "path": "Observation.code.coding",
                "children": [
                    {
                        "path": "Observation.code.coding.system",
                        "fixed_value": "http://loinc.org",
                    }
                ],
            }
        ]
    }
    assert FMLRuleFactory._fixed_system_from_field(field) == "http://loinc.org"


def test_fixed_system_from_field_ignores_non_coding_children():
    field = {
        "children": [
            {"path": "Observation.code.text", "children": []},
        ]
    }
    assert FMLRuleFactory._fixed_system_from_field(field) is None


def test_fixed_system_from_field_ignores_empty_fixed_value():
    field = {
        "children": [
            {
                "path": "Observation.code.coding",
                "children": [
                    {"path": "Observation.code.coding.system", "fixed_value": ""}
                ],
            }
        ]
    }
    assert FMLRuleFactory._fixed_system_from_field(field) is None


# ── _fixed_coding_from_leaves ──────────────────────────────────────────────────
def test_fixed_coding_from_leaves_requires_both_system_and_code():
    field = {
        "children": [
            {
                "path": "Procedure.code.coding",
                "children": [
                    {
                        "path": "Procedure.code.coding.system",
                        "fixed_value": "http://snomed.info/sct",
                    },
                ],
            }
        ]
    }
    assert FMLRuleFactory._fixed_coding_from_leaves(field) is None


def test_fixed_coding_from_leaves_builds_full_coding():
    field = {
        "children": [
            {
                "path": "Procedure.code.coding",
                "children": [
                    {
                        "path": "Procedure.code.coding.system",
                        "fixed_value": "http://snomed.info/sct",
                    },
                    {"path": "Procedure.code.coding.code", "fixed_value": "81723002"},
                    {
                        "path": "Procedure.code.coding.display",
                        "fixed_value": "Appendectomy",
                    },
                ],
            }
        ]
    }
    assert FMLRuleFactory._fixed_coding_from_leaves(field) == {
        "system": "http://snomed.info/sct",
        "code": "81723002",
        "display": "Appendectomy",
    }


def test_fixed_coding_from_leaves_no_children_returns_none():
    assert FMLRuleFactory._fixed_coding_from_leaves({}) is None


# ── _fixed_coding_slices ───────────────────────────────────────────────────────
def test_fixed_coding_slices_empty_without_coding_child():
    assert FMLRuleFactory._fixed_coding_slices({}) == []


def test_fixed_coding_slices_collects_fixed_dicts():
    field = {
        "children": [
            {
                "path": "Observation.category.coding",
                "slices": [
                    {"fixed_value": {"system": "http://sys1", "code": "c1"}},
                    {"fixed_value": {"code": None}},  # no code -> dropped
                    "not-a-dict",  # ignored
                ],
            }
        ]
    }
    result = FMLRuleFactory._fixed_coding_slices(field)
    assert result == [{"system": "http://sys1", "code": "c1"}]


def test_fixed_coding_slices_unwraps_model_dump(factory):
    class FakeCoding:
        def model_dump(self, exclude_none=True):
            return {"system": "http://sys1", "code": "c1"}

    field = {
        "children": [
            {
                "path": "Observation.category.coding",
                "slices": [{"fixed_value": FakeCoding()}],
            }
        ]
    }
    result = FMLRuleFactory._fixed_coding_slices(field)
    assert result == [{"system": "http://sys1", "code": "c1"}]


def test_fixed_coding_slices_drops_unreadable_model_dump():
    class BrokenCoding:
        def model_dump(self, exclude_none=True):
            raise RuntimeError("boom")

    field = {
        "children": [
            {
                "path": "Observation.category.coding",
                "slices": [{"path": "x", "fixed_value": BrokenCoding()}],
            }
        ]
    }
    assert FMLRuleFactory._fixed_coding_slices(field) == []


# ── Fixed complex-value emission ─────────────────────────────────────────────
def test_fixed_coding_raw_type_list_is_created_not_stringified(factory):
    """Expanded datatype children can retain the parser's list-form type."""
    fixed = {
        "system": "http://snomed.info/sct",
        "code": "789279006",
        "display": "Clavien-Dindo classification grade",
    }

    rule = factory._create_fixed_value_rule(
        [{"code": "Coding"}],
        fixed,
        "coding",
        "tgt-code",
        "src-code",
    )

    target = rule.target[0]
    assert target.element == "coding"
    assert target.transform == "create"
    assert target.parameter[0].valueString == "Coding"
    assert {child.target[0].element for child in rule.rule} == {
        "system",
        "code",
        "display",
    }
    assert str(fixed) not in str(rule.model_dump())


def test_fixed_quantity_is_emitted_recursively_not_stringified(factory):
    rule = factory._create_fixed_value_rule(
        [{"code": "Quantity"}],
        {"value": 12, "unit": "mg"},
        "valueQuantity",
        "target",
        "source",
    )

    target = rule.target[0]
    assert target.transform == "create"
    assert target.parameter[0].valueString == "Quantity"
    children = {child.target[0].element: child.target[0] for child in rule.rule}
    assert children["value"].parameter[0].valueDecimal == Decimal("12")
    assert children["unit"].parameter[0].valueString == "mg"
    assert str({"value": 12, "unit": "mg"}) not in str(rule.model_dump())


def test_fixed_identifier_emits_nested_period(factory):
    rule = factory._create_fixed_value_rule(
        [{"code": "Identifier"}],
        {
            "use": "official",
            "system": "urn:example",
            "value": "123",
            "period": {"start": "2020-01-01"},
        },
        "identifier",
        "target",
        "source",
    )

    children = {child.target[0].element: child for child in rule.rule}
    period = children["period"]
    assert period.target[0].transform == "create"
    assert period.target[0].parameter[0].valueString == "Period"
    assert period.rule[0].target[0].element == "start"
    assert period.rule[0].target[0].parameter[0].valueString == "2020-01-01"


def test_fixed_repeating_primitive_children_get_distinct_rules(factory):
    rule = factory._create_fixed_value_rule(
        [{"code": "HumanName"}],
        {"family": "Smith", "given": ["Alice", "A"]},
        "name",
        "target",
        "source",
    )

    given = [child for child in rule.rule if child.target[0].element == "given"]
    assert [child.name for child in given] == [
        "add-fixed-name-given-0",
        "add-fixed-name-given-1",
    ]
    assert [child.target[0].parameter[0].valueString for child in given] == [
        "Alice",
        "A",
    ]


def test_profile_prohibited_complex_child_is_not_emitted(factory):
    factory.diagnostics = []
    field = {
        "type": [
            {
                "code": "Identifier",
                "type_structure": [
                    {
                        "path": "Identifier.system",
                        "type": "uri",
                        "cardinality": {"min": 0, "max": "0"},
                    }
                ],
            }
        ]
    }
    rule = factory._create_fixed_value_rule(
        field["type"],
        {"system": "urn:forbidden", "value": "123"},
        "identifier",
        "target",
        "source",
        field=field,
    )

    assert {child.target[0].element for child in rule.rule} == {"value"}
    assert factory.diagnostics[0]["code"] == "complex-fixed-child-prohibited"


@pytest.mark.parametrize(
    "field_type,value,attribute,expected",
    [
        ("boolean", False, "valueBoolean", False),
        ("integer", 0, "valueInteger", 0),
        ("decimal", 1.5, "valueDecimal", Decimal("1.5")),
    ],
)
def test_profile_fixed_scalar_rule_uses_typed_parameter(
    factory, field_type, value, attribute, expected
):
    rule = factory._create_fixed_value_rule(
        [{"code": field_type}],
        value,
        "value",
        "target",
        "source",
    )

    parameter = rule.target[0].parameter[0]
    assert getattr(parameter, attribute) == expected
    assert parameter.valueString is None


# ── _infer_pattern_type ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [
        ({"coding": [{"code": "a"}]}, "CodeableConcept"),
        ({"system": "http://x", "code": "a"}, "Coding"),
        ({"code": "a"}, None),
        ({}, None),
    ],
)
def test_infer_pattern_type(value, expected):
    assert FMLRuleFactory._infer_pattern_type(value) == expected


# ── _existing_concept_map_is_authored ─────────────────────────────────────────
class FakeDataIO:
    class ProjectFolders:
        CONCEPT_MAPS = "source_data/concept_maps"

    def __init__(self, existing=None, raise_exc=None):
        self.existing = existing
        self.raise_exc = raise_exc

    def load_project_file(self, folder, filename):
        if self.raise_exc:
            raise self.raise_exc
        return self.existing


def test_existing_concept_map_is_authored_read_failure_returns_false(factory):
    factory.app_state = SimpleNamespace(
        dataIO=FakeDataIO(raise_exc=FileNotFoundError())
    )
    assert factory._existing_concept_map_is_authored("cm-1") is False


def test_existing_concept_map_is_authored_non_dict_returns_false(factory):
    factory.app_state = SimpleNamespace(dataIO=FakeDataIO(existing="not-a-dict"))
    assert factory._existing_concept_map_is_authored("cm-1") is False


def test_existing_concept_map_is_authored_identity_map_returns_false(factory):
    existing = {
        "group": [
            {
                "source": "http://sys1",
                "target": "http://sys1",
                "element": [{"code": "a", "target": [{"code": "a"}]}],
            }
        ]
    }
    factory.app_state = SimpleNamespace(dataIO=FakeDataIO(existing=existing))
    assert factory._existing_concept_map_is_authored("cm-1") is False


def test_existing_concept_map_is_authored_true_when_group_source_differs_from_target(
    factory,
):
    existing = {
        "group": [{"source": "http://sys1", "target": "http://sys2", "element": []}]
    }
    factory.app_state = SimpleNamespace(dataIO=FakeDataIO(existing=existing))
    assert factory._existing_concept_map_is_authored("cm-1") is True


def test_existing_concept_map_is_authored_true_when_element_target_code_differs(
    factory,
):
    existing = {
        "group": [
            {
                "source": "http://sys1",
                "target": "http://sys1",
                "element": [{"code": "a", "target": [{"code": "b"}]}],
            }
        ]
    }
    factory.app_state = SimpleNamespace(dataIO=FakeDataIO(existing=existing))
    assert factory._existing_concept_map_is_authored("cm-1") is True
