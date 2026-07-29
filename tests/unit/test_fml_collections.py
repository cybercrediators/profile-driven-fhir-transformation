"""Collection/list-mode and repeated-backbone correlation tests."""

from types import SimpleNamespace

import pytest
from fhir.resources.R4B.structuremap import StructureMapGroupRule

from mapping.fml_creator.fml_factory import FMLRuleFactory
from mapping.fml_map import StructureMapGenerator
from mapping.rule_ir import (
    CollectionRuleSpec,
    SOURCE_LIST_MODES,
    TARGET_LIST_MODES,
    parse_collection_rule,
)

pytestmark = pytest.mark.unit


def _field(path, field_type, maximum="1", children=None):
    return {
        "path": path,
        "type": field_type,
        "cardinality": {"min": 0, "max": maximum},
        "children": children or [],
    }


def _factory():
    factory = object.__new__(FMLRuleFactory)
    factory.source_field_types = {}
    factory.source_field_max = {}
    factory.collection_rules = {}
    factory.diagnostics = []
    factory._current_profile_id = None
    factory._current_profile_sd = None
    factory._target_tree_cache = (None, None)
    return factory


def _generator(factory=None):
    generator = object.__new__(StructureMapGenerator)
    generator.factory = factory or SimpleNamespace(
        _choice_suffix=lambda value: value[0].upper() + value[1:]
    )
    generator.mapping_diagnostics = []
    generator.current_profile_name = "patient-profile"
    return generator


@pytest.mark.parametrize("mode", sorted(SOURCE_LIST_MODES))
def test_rule_ir_accepts_every_r4_source_list_mode(mode):
    spec = parse_collection_rule(
        "Source.items",
        {"target": "Patient.identifier", "sourceListMode": mode},
    )
    assert spec.source_list_mode == mode


@pytest.mark.parametrize("mode", sorted(TARGET_LIST_MODES))
def test_rule_ir_accepts_every_r4_target_list_mode(mode):
    spec = parse_collection_rule(
        "Source.items",
        {"target": "Patient.identifier", "targetListMode": mode},
    )
    assert spec.target_list_modes == (mode,)
    assert bool(spec.list_rule_id) is (mode == "share")


def test_rule_ir_rejects_unknown_modes_and_unscoped_rule_id():
    with pytest.raises(ValueError, match="sourceListMode"):
        parse_collection_rule(
            "Source.items",
            {"target": "Patient.identifier", "sourceListMode": "middle"},
        )
    with pytest.raises(ValueError, match="only meaningful"):
        parse_collection_rule(
            "Source.items",
            {"target": "Patient.identifier", "listRuleId": "ids"},
        )


def test_invalid_typed_mode_is_diagnosed_and_omitted_by_generator():
    factory = _factory()
    generator = _generator(factory)
    target = _field("Patient.identifier", "Identifier", "*")

    _, mappings = generator._apply_custom_mapping_table(
        {
            "Source.identifiers": {
                "target": "Patient.identifier",
                "sourceListMode": "middle",
            }
        },
        [
            {
                "id": "Source.identifiers",
                "path": "Source.identifiers",
                "max": "*",
            }
        ],
        [target],
        "Patient",
        "patient-profile",
    )

    assert factory.collection_rules["Patient.identifier"].invalid is True
    assert factory.create_mappable_field_rule(
        target,
        "Patient",
        "source",
        "target",
        automapped_mappings=mappings,
    ) is None
    assert generator.mapping_diagnostics[0]["code"] == "invalid-collection-rule"


def test_document_envelope_keys_do_not_enter_legacy_mapping_parser():
    factory = _factory()
    generator = _generator(factory)
    paths, mappings = generator._apply_custom_mapping_table(
        {
            "$imports": ["http://example.org/StructureMap/common"],
            "$rules": [{"name": "explicit"}],
            "Source.name": "Patient.name.text",
        },
        [{"id": "Source.name", "path": "Source.name"}],
        [_field("Patient.name.text", "string")],
        "Patient",
        "patient-profile",
    )
    assert "Patient.name.text" in paths
    assert mappings["Patient.name.text"] == "Source.name"
    assert not any(
        diagnostic["code"] == "invalid-collection-rule"
        for diagnostic in generator.mapping_diagnostics
    )


def test_collection_parent_correlates_all_children_in_one_target_repetition():
    factory = _factory()
    generator = _generator(factory)
    identifier = _field(
        "Patient.identifier",
        "Identifier",
        "*",
        [
            _field("Patient.identifier.system", "uri"),
            _field("Patient.identifier.value", "string"),
        ],
    )
    source_fields = [
        {"id": "Source.identifiers", "path": "Source.identifiers", "max": "*"},
        {
            "id": "Source.identifiers.system",
            "path": "Source.identifiers.system",
            "max": "1",
        },
        {
            "id": "Source.identifiers.value",
            "path": "Source.identifiers.value",
            "max": "1",
        },
    ]
    table = {
        "Source.identifiers": {
            "target": "Patient.identifier",
            "correlation": {"sourceKey": "system", "targetKey": "system"},
        },
        "Source.identifiers.system": "Patient.identifier.system",
        "Source.identifiers.value": "Patient.identifier.value",
    }

    _, mappings = generator._apply_custom_mapping_table(
        table, source_fields, [identifier], "Patient", "patient-profile"
    )
    rule = factory.create_mappable_field_rule(
        identifier,
        "Patient",
        "source",
        "target",
        automapped_mappings=mappings,
    )

    # No source items means no rule execution; one/many items execute this same
    # outer rule once per item and create one target Identifier for each.
    assert rule.source[0].element == "identifiers"
    assert rule.source[0].check == "system.count() = 1"
    assert rule.target[0].element == "identifier"
    assert rule.target[0].listMode is None
    assert {nested.source[0].element for nested in rule.rule} == {"system", "value"}
    assert {nested.target[0].context for nested in rule.rule} == {"tgt-identifier"}
    StructureMapGroupRule(**rule.model_dump(exclude_none=True))


def test_explicit_list_modes_override_legacy_target_share():
    factory = _factory()
    factory.collection_rules["Patient.identifier"] = CollectionRuleSpec(
        source="Source.identifiers",
        target="Patient.identifier",
        source_list_mode="last",
        target_list_modes=("first", "share", "collate"),
        list_rule_id="identifier-correlation",
    )
    field = _field("Patient.identifier", "Identifier", "*")

    rule = factory.create_mappable_field_rule(
        field,
        "Patient",
        "source",
        "target",
        automapped_mappings={"Patient.identifier": "Source.identifiers"},
    )

    assert rule.source[0].listMode == "last"
    assert rule.target[0].listMode == ["first", "share", "collate"]
    assert rule.target[0].listRuleId == "identifier-correlation"


def test_nested_collections_keep_each_level_in_its_own_iteration_scope():
    factory = _factory()
    factory.collection_rules = {
        "Patient.contact": CollectionRuleSpec(
            source="Source.contacts", target="Patient.contact"
        ),
        "Patient.contact.telecom": CollectionRuleSpec(
            source="Source.contacts.telecoms",
            target="Patient.contact.telecom",
            source_list_mode="not_last",
            target_list_modes=("last",),
        ),
    }
    telecom = _field(
        "Patient.contact.telecom",
        "ContactPoint",
        "*",
        [_field("Patient.contact.telecom.value", "string")],
    )
    contact = _field("Patient.contact", "BackboneElement", "*", [telecom])
    mappings = {
        "Patient.contact": "Source.contacts",
        "Patient.contact.telecom": "Source.contacts.telecoms",
        "Patient.contact.telecom.value": "Source.contacts.telecoms.value",
    }

    rule = factory.create_mappable_field_rule(
        contact,
        "Patient",
        "source",
        "target",
        automapped_mappings=mappings,
    )

    nested_collection = rule.rule[0]
    assert rule.source[0].element == "contacts"
    assert nested_collection.source[0].context == "src-contact"
    assert nested_collection.source[0].element == "telecoms"
    assert nested_collection.source[0].listMode == "not_last"
    assert nested_collection.target[0].context == "tgt-contact"
    assert nested_collection.target[0].listMode == ["last"]
    assert nested_collection.rule[0].source[0].context == "src-telecom"
    assert nested_collection.rule[0].source[0].element == "value"


def test_collection_modes_and_context_apply_to_an_ordered_slice():
    factory = _factory()
    factory.collection_rules["Patient.identifier:mrn"] = CollectionRuleSpec(
        source="Source.identifiers",
        target="Patient.identifier:mrn",
        source_list_mode="first",
        target_list_modes=("first", "collate"),
        source_key="value",
        target_key="value",
    )
    parent = {
        "path": "Patient.identifier",
        "type": [{"code": "Identifier"}],
        "slicing": {
            "ordered": True,
            "rules": "open",
            "discriminators": [{"type": "value", "path": "system"}],
        },
    }
    sliced = {
        "path": "Patient.identifier:mrn",
        "sliceName": "mrn",
        "type": [{"code": "Identifier"}],
        "cardinality": {"min": 0, "max": "*"},
        "children": [_field("Patient.identifier:mrn.value", "string")],
    }

    rule = factory._create_slice_instance_rule(
        parent,
        sliced,
        "Patient",
        "source",
        "target",
        {
            "Patient.identifier:mrn": "Source.identifiers",
            "Patient.identifier:mrn.value": "Source.identifiers.value",
        },
    )

    assert rule.source[0].element == "identifiers"
    assert rule.source[0].listMode == "first"
    assert rule.source[0].check == "value.count() = 1"
    assert rule.target[0].listMode == ["first", "collate"]
    assert rule.rule[0].source[0].context == "src-slice"
    assert rule.rule[0].source[0].element == "value"


def test_missing_declared_key_marks_collection_invalid_and_omits_rule():
    factory = _factory()
    generator = _generator(factory)
    target = _field("Patient.identifier", "Identifier", "*")
    source_fields = [
        {"id": "Source.identifiers", "path": "Source.identifiers", "max": "*"}
    ]

    _, mappings = generator._apply_custom_mapping_table(
        {
            "Source.identifiers": {
                "target": "Patient.identifier",
                "correlation": {
                    "sourceKey": "missing",
                    "targetKey": "system",
                },
            }
        },
        source_fields,
        [target],
        "Patient",
        "patient-profile",
    )

    assert factory.collection_rules["Patient.identifier"].invalid is True
    assert factory.create_mappable_field_rule(
        target,
        "Patient",
        "source",
        "target",
        automapped_mappings=mappings,
    ) is None
    assert {
        diagnostic["code"] for diagnostic in generator.mapping_diagnostics
    } == {"collection-source-key-not-found"}


@pytest.mark.parametrize(
    "source_max,target_max,code",
    [
        ("1", "*", "collection-source-not-repeating"),
        ("*", "1", "collection-target-not-repeating"),
    ],
)
def test_collection_requires_repeating_source_and_target(
    source_max, target_max, code
):
    factory = _factory()
    generator = _generator(factory)
    target = _field("Patient.identifier", "Identifier", target_max)

    generator._apply_custom_mapping_table(
        {
            "Source.identifiers": {
                "target": "Patient.identifier",
                "sourceListMode": "only_one",
            }
        },
        [
            {
                "id": "Source.identifiers",
                "path": "Source.identifiers",
                "max": source_max,
            }
        ],
        [target],
        "Patient",
        "patient-profile",
    )

    assert factory.collection_rules["Patient.identifier"].invalid is True
    assert code in {
        diagnostic["code"] for diagnostic in generator.mapping_diagnostics
    }
