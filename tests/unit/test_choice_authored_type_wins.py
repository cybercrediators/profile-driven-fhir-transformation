"""An authored concrete choice type must survive source-type narrowing.

Source field types are inferred from the source document's JSON, where `dateTime`,
`code`, `uri`, `id` and `markdown` are all indistinguishable from `string`. When a
profile allows both a string and a non-string primitive — `Immunization.occurrence[x]`
is `dateTime | string` — narrowing on the inferred type collapsed the choice to
`string`. The author's concrete `occurrence[x]:occurrenceDateTime` row was then
rejected as an unsupported choice type and *the entire rule was dropped*, leaving a
mandatory 1..1 element absent from the map with no error anywhere.

Inference is for what the author did not state, so an explicit concrete variant wins.
"""


import pytest

from fhir.resources.R4B.structuremap import (
    StructureMapGroupRule,
    StructureMapGroupRuleSource,
)

from mapping.fml_creator.fml_factory import FMLRuleFactory

pytestmark = pytest.mark.unit


@pytest.fixture
def factory():
    instance = object.__new__(FMLRuleFactory)
    instance.diagnostics = []
    instance.source_field_types = {"occ": "string"}  # a dateTime arrives as JSON string
    instance.source_field_max = {}
    instance._target_tree_cache = (None, None)
    instance.collection_rules = {}
    instance._current_profile_id = None
    instance._current_profile_sd = None
    return instance


def _occurrence_field():
    """Immunization.occurrence[x] — mandatory, dateTime or string."""
    return {
        "path": "Immunization.occurrence[x]",
        "id": "Immunization.occurrence[x]",
        "name": "occurrence[x]",
        "choice_types": ["dateTime", "string"],
        "type": [{"code": "dateTime"}, {"code": "string"}],
        "cardinality": {"min": 1, "max": "1"},
        "children": [],
    }


def _build(factory, automapped):
    # Production populates the rule's source before delegating to the choice builder.
    rule = StructureMapGroupRule.model_construct(
        name="map-occurrence",
        source=[
            StructureMapGroupRuleSource.model_construct(
                context="source",
                element="TODO_MAP_OCCURRENCE",
                variable="src-occurrence",
            )
        ],
        target=[],
        rule=[],
    )
    factory._choice_field_rule(
        field=_occurrence_field(),
        rule=rule,
        field_name="occurrence[x]",
        var_suffix="occurrence",
        source_element="TODO_MAP_OCCURRENCE",
        parent_source_context="source",
        parent_target_context="target",
        cardinality={"min": 1, "max": "1"},
        is_slice=False,
        slice_info=None,
        automapped_mappings=automapped,
    )
    return rule


def _targets(rule):
    found = []

    def walk(node):
        for target in getattr(node, "target", None) or []:
            if getattr(target, "element", None):
                found.append(target.element)
        for child in getattr(node, "rule", None) or []:
            walk(child)

    walk(rule)
    return found


def test_authored_concrete_choice_is_not_narrowed_away(factory):
    """The dateTime the author named survives, so the mandatory rule is produced.

    Both rows are present, as in the UK Core table that exposed this: the *bare*
    row is what feeds the source type into the narrowing, and the *concrete* row
    is what the narrowing then rejects.
    """
    returned = _build(
        factory,
        {
            "Immunization.occurrence[x]": "Src.occ",
            "Immunization.occurrence[x]:occurrenceDateTime": "Src.occ",
        },
    )
    assert returned is not None, (
        "rule dropped: the authored dateTime was narrowed away by the source's "
        "JSON string type, then rejected as an unsupported choice type"
    )
    assert "Choice types: dateTime" in (returned.documentation or ""), (
        returned.documentation
    )
    emitted = [child.name for child in (returned.rule or [])]
    assert any("dateTime" in name for name in emitted), emitted
    assert "occurrence" in _targets(returned), _targets(returned)


def test_unstated_choice_still_uses_the_source_type(factory):
    """With no concrete variant authored, inference from the source is retained."""
    returned = _build(factory, {"Immunization.occurrence[x]": "Src.occ"})
    assert returned is not None
    assert "Choice types: string" in (returned.documentation or ""), (
        returned.documentation
    )
