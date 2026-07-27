"""Unit tests for the PipelinePlugin base class (plugins/base.py)."""

import pytest

from plugins.base import PipelinePlugin

pytestmark = pytest.mark.unit


class _MinimalPlugin(PipelinePlugin):
    """Concrete subclass that overrides nothing but the required plugin_id,
    so the base class' default pass-through hook implementations run."""

    @property
    def plugin_id(self) -> str:
        return "minimal"


def test_pipeline_plugin_is_abstract():
    with pytest.raises(TypeError):
        PipelinePlugin({})


def test_subclass_without_plugin_id_cannot_be_instantiated():
    class _Incomplete(PipelinePlugin):
        pass

    with pytest.raises(TypeError):
        _Incomplete({})


def test_minimal_plugin_stores_config():
    plugin = _MinimalPlugin({"key": "value"})
    assert plugin.config == {"key": "value"}
    assert plugin.plugin_id == "minimal"


def test_default_post_source_def_is_pass_through():
    plugin = _MinimalPlugin({})
    source_definition = {"resourceType": "StructureDefinition"}
    result = plugin.post_source_def(source_definition, [{"id": "Source.field"}])
    assert result is source_definition


def test_default_enrich_automapping_is_pass_through():
    plugin = _MinimalPlugin({})
    automapping = {"Source.field": "Patient.name"}
    result = plugin.enrich_automapping(automapping)
    assert result is automapping


def test_default_generate_concept_map_returns_none():
    plugin = _MinimalPlugin({})
    assert plugin.generate_concept_map("field", {}, None) is None
    assert plugin.generate_concept_map("field", {}, {"resourceType": "ConceptMap"}) is None


def test_default_get_field_metadata_returns_none():
    plugin = _MinimalPlugin({})
    assert plugin.get_field_metadata("any_field") is None
