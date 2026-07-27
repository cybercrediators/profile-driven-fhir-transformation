"""Unit tests for Registry and RegistryObject."""

from types import SimpleNamespace

import pytest

from data_handling.registry.registry import Registry
from data_handling.registry.registry_object import RegistryObject

pytestmark = pytest.mark.unit


def _obj(url, res_type="StructureDefinition", mappable_fields=None):
    """Build a minimal fake FHIR object with a .url attribute."""
    return SimpleNamespace(url=url, res_type=res_type, mappable_fields=mappable_fields or [])


@pytest.fixture
def registry():
    return Registry()


# --------------------------------------------------------------------------- #
# add_fhir_object
# --------------------------------------------------------------------------- #


def test_add_fhir_object_returns_registry_object(registry):
    obj = _obj("http://x/sd/Patient")
    result = registry.add_fhir_object(obj, "StructureDefinition")
    assert isinstance(result, RegistryObject)
    assert result.data is obj
    assert result.res_type == "StructureDefinition"


def test_add_fhir_object_stores_by_url(registry):
    obj = _obj("http://x/sd/Patient")
    registry.add_fhir_object(obj, "StructureDefinition")
    assert "http://x/sd/Patient" in registry.registry_objects


def test_add_fhir_object_without_url_raises(registry):
    no_url = SimpleNamespace(name="no-url")
    with pytest.raises(ReferenceError):
        registry.add_fhir_object(no_url)


def test_add_fhir_object_duplicate_raises(registry):
    obj = _obj("http://x/sd/Patient")
    registry.add_fhir_object(obj, "StructureDefinition")
    with pytest.raises(ReferenceError):
        registry.add_fhir_object(obj, "StructureDefinition")


# --------------------------------------------------------------------------- #
# get_obj_by_name
# --------------------------------------------------------------------------- #


def test_get_obj_by_name_found(registry):
    obj = _obj("http://x/sd/Patient")
    registry.add_fhir_object(obj, "StructureDefinition")
    result = registry.get_obj_by_name("http://x/sd/Patient")
    assert result is not None
    assert result.data is obj


def test_get_obj_by_name_missing_returns_none(registry):
    assert registry.get_obj_by_name("http://x/nonexistent") is None


# --------------------------------------------------------------------------- #
# get_all_obj_by_type / get_all_obj_names_by_type
# --------------------------------------------------------------------------- #


def test_get_all_obj_by_type_filters_correctly(registry):
    registry.add_fhir_object(_obj("http://x/sd/A"), "StructureDefinition")
    registry.add_fhir_object(_obj("http://x/cm/A"), "ConceptMap")
    registry.add_fhir_object(_obj("http://x/sd/B"), "StructureDefinition")

    sds = registry.get_all_obj_by_type("StructureDefinition")
    assert len(sds) == 2
    cms = registry.get_all_obj_by_type("ConceptMap")
    assert len(cms) == 1


def test_get_all_obj_names_by_type_empty(registry):
    assert registry.get_all_obj_names_by_type("StructureMap") == []


# --------------------------------------------------------------------------- #
# add_to_used_by
# --------------------------------------------------------------------------- #


def test_add_to_used_by_updates_set(registry):
    registry.add_fhir_object(_obj("http://x/sd/Base"), "StructureDefinition")
    registry.add_to_used_by("http://x/sd/Base", "http://x/sd/Derived")
    base = registry.get_obj_by_name("http://x/sd/Base")
    assert "http://x/sd/Derived" in base.used_by


def test_add_to_used_by_multiple(registry):
    registry.add_fhir_object(_obj("http://x/sd/Base"), "StructureDefinition")
    registry.add_to_used_by("http://x/sd/Base", "http://x/sd/A")
    registry.add_to_used_by("http://x/sd/Base", "http://x/sd/B")
    assert len(registry.get_obj_by_name("http://x/sd/Base").used_by) == 2


def test_add_to_used_by_unknown_src_is_noop(registry):
    registry.add_to_used_by("http://x/nonexistent", "http://x/sd/Other")  # must not raise


# --------------------------------------------------------------------------- #
# get_all_mappable_fields / get_all_mappable_fields_of_roots
# --------------------------------------------------------------------------- #


def test_get_all_mappable_fields_structure(registry):
    obj = _obj("http://x/sd/Patient", mappable_fields=["name", "birthDate"])
    reg_obj = registry.add_fhir_object(obj, "StructureDefinition")
    reg_obj.mappable_fields = ["name", "birthDate"]

    result = registry.get_all_mappable_fields()
    assert result == [{"http://x/sd/Patient": ["name", "birthDate"]}]


def test_get_all_mappable_fields_of_roots_only_root(registry):
    obj_root = _obj("http://x/sd/Root")
    obj_leaf = _obj("http://x/sd/Leaf")
    ro = registry.add_fhir_object(obj_root, "StructureDefinition")
    registry.add_fhir_object(obj_leaf, "StructureDefinition")
    ro.set_root()

    result = registry.get_all_mappable_fields_of_roots()
    keys = [list(d.keys())[0] for d in result]
    assert "http://x/sd/Root" in keys
    assert "http://x/sd/Leaf" not in keys


# --------------------------------------------------------------------------- #
# get_all_unprocessed_objects
# --------------------------------------------------------------------------- #


def test_get_all_unprocessed_objects_initially_all(registry):
    registry.add_fhir_object(_obj("http://x/sd/A"), "StructureDefinition")
    registry.add_fhir_object(_obj("http://x/sd/B"), "StructureDefinition")
    unprocessed = registry.get_all_unprocessed_objects()
    assert set(unprocessed) == {"http://x/sd/A", "http://x/sd/B"}


def test_get_all_unprocessed_objects_excludes_processed(registry):
    registry.add_fhir_object(_obj("http://x/sd/A"), "StructureDefinition")
    reg_obj = registry.add_fhir_object(_obj("http://x/sd/B"), "StructureDefinition")
    reg_obj.set_processed()

    unprocessed = registry.get_all_unprocessed_objects()
    assert unprocessed == ["http://x/sd/A"]


# --------------------------------------------------------------------------- #
# RegistryObject
# --------------------------------------------------------------------------- #


def test_registry_object_initial_state():
    obj = SimpleNamespace(url="http://x/sd/X")
    ro = RegistryObject(obj, "StructureDefinition")
    assert ro.processed == 0
    assert ro.is_root is False
    assert ro.used_by == set()
    assert ro.mappable_fields == []


def test_registry_object_set_processed():
    ro = RegistryObject(SimpleNamespace(url="http://x"), "StructureDefinition")
    assert not ro.is_processed()
    ro.set_processed()
    assert ro.is_processed()


def test_registry_object_set_root():
    ro = RegistryObject(SimpleNamespace(url="http://x"), "StructureDefinition")
    ro.set_root()
    assert ro.is_root is True
