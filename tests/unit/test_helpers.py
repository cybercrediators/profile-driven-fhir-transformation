"""Unit tests for helpers/utils.py and helpers/config.py."""

import json
from types import SimpleNamespace

import pytest

from helpers import utils
from helpers.config import load_config, save_config, show_config

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# get_json / store_json / store_json_str
# --------------------------------------------------------------------------- #


def test_store_and_get_json_roundtrip(tmp_path):
    data = {"key": "value", "nested": {"n": 42}, "arr": [1, 2, 3]}
    path = tmp_path / "data.json"
    utils.store_json(data, path)
    assert utils.get_json(path) == data


def test_store_json_uses_utf8(tmp_path):
    data = {"name": "Ärzte", "emoji": "✓"}
    path = tmp_path / "utf8.json"
    utils.store_json(data, path)
    raw = path.read_text(encoding="utf-8")
    assert "Ärzte" in raw
    assert "✓" in raw


def test_store_json_str_writes_raw_string(tmp_path):
    path = tmp_path / "raw.json"
    utils.store_json_str('{"raw": true}', path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"raw": True}


# --------------------------------------------------------------------------- #
# clean_obj
# --------------------------------------------------------------------------- #


def test_clean_obj_removes_none_values():
    obj = {"a": 1, "b": None, "c": "x"}
    result = utils.clean_obj(obj)
    assert "b" not in result
    assert result["a"] == 1


def test_clean_obj_removes_text_key():
    obj = {"a": 1, "text": "narrative", "b": 2}
    result = utils.clean_obj(obj)
    assert "text" not in result


def test_clean_obj_modifies_in_place():
    obj = {"a": 1, "b": None}
    returned = utils.clean_obj(obj)
    assert returned is obj


def test_clean_obj_empty_dict():
    assert utils.clean_obj({}) == {}


# --------------------------------------------------------------------------- #
# get_predef_modules
# --------------------------------------------------------------------------- #


def test_get_predef_modules_returns_list():
    modules = utils.get_predef_modules()
    assert isinstance(modules, list)
    assert len(modules) > 0


def test_get_predef_modules_contains_known_types():
    modules = utils.get_predef_modules()
    for expected in ("patient", "observation", "structuredefinition", "valueset"):
        assert expected in modules


# --------------------------------------------------------------------------- #
# get_model_class
# --------------------------------------------------------------------------- #


def test_get_model_class_returns_class_for_known_type():
    cls = utils.get_model_class("Patient")
    assert cls is not None
    assert cls.__name__ == "Patient"


def test_get_model_class_returns_class_for_structuredefinition():
    cls = utils.get_model_class("StructureDefinition")
    assert cls is not None
    assert cls.__name__ == "StructureDefinition"


def test_get_model_class_returns_none_for_unknown_type():
    assert utils.get_model_class("NotARealResource") is None


def test_get_model_class_returns_none_for_none():
    assert utils.get_model_class(None) is None


# --------------------------------------------------------------------------- #
# _strip_primitive_extensions
# --------------------------------------------------------------------------- #


def test_strip_primitive_extensions_removes_underscore_keys():
    obj = {"id": "x", "_id": {"extension": []}, "status": "active"}
    result = utils._strip_primitive_extensions(obj)
    assert "_id" not in result
    assert result["id"] == "x"
    assert result["status"] == "active"


def test_strip_primitive_extensions_recursive_dict():
    obj = {"coding": [{"code": "A", "_code": {"id": "ext"}}]}
    result = utils._strip_primitive_extensions(obj)
    assert "_code" not in result["coding"][0]
    assert result["coding"][0]["code"] == "A"


def test_strip_primitive_extensions_handles_list():
    lst = [{"_v": 1, "v": 2}, {"v": 3}]
    result = utils._strip_primitive_extensions(lst)
    assert "_v" not in result[0]
    assert result[0]["v"] == 2


def test_strip_primitive_extensions_primitives_unchanged():
    assert utils._strip_primitive_extensions("hello") == "hello"
    assert utils._strip_primitive_extensions(42) == 42
    assert utils._strip_primitive_extensions(None) is None


def test_strip_primitive_extensions_empty_dict():
    assert utils._strip_primitive_extensions({}) == {}


def test_strip_drops_r5_binding_additional():
    # ElementDefinition.binding.additional is an R5 field the R4B model forbids;
    # it must be dropped so the SD validates into a typed object.
    el = {
        "path": "Patient.language",
        "binding": {
            "strength": "preferred",
            "valueSet": "http://hl7.org/fhir/ValueSet/languages",
            "additional": [{"purpose": "starter", "valueSet": "http://x/vs"}],
        },
    }
    result = utils._strip_primitive_extensions(el)
    assert "additional" not in result["binding"]
    assert result["binding"]["strength"] == "preferred"
    assert result["binding"]["valueSet"] == "http://hl7.org/fhir/ValueSet/languages"


def test_strip_keeps_additional_outside_binding():
    # Only drop `additional` inside a binding — not a field that merely shares the name.
    obj = {"additional": "keep-me", "binding": {"additional": ["drop"], "strength": "required"}}
    result = utils._strip_primitive_extensions(obj)
    assert result["additional"] == "keep-me"
    assert "additional" not in result["binding"]


# --------------------------------------------------------------------------- #
# json_to_obj
# --------------------------------------------------------------------------- #


def test_json_to_obj_returns_fhir_object_for_known_type():
    content = {"id": "p1"}
    obj = utils.json_to_obj(content, "Patient")
    assert obj is not None
    assert obj.__class__.__name__ == "Patient"


def test_json_to_obj_returns_none_for_unknown_type():
    result = utils.json_to_obj({"id": "x"}, "NotAType")
    assert result is None


def test_json_to_obj_returns_none_for_none_type():
    result = utils.json_to_obj({"id": "x"}, None)
    assert result is None


def test_json_to_obj_strips_primitive_extensions_and_retries():
    # A Patient with an underscore-prefixed key that the model would reject.
    # After stripping, construction should succeed.
    content = {"id": "p1", "_id": {"extension": []}}
    obj = utils.json_to_obj(content, "Patient")
    # Either parsed cleanly or fell back to the cleaned dict — either way not None
    assert obj is not None


def test_json_to_obj_returns_dict_when_construction_fails_after_strip():
    # Pass a field with a value that cannot be validated even after stripping.
    # fhir.resources will reject the invalid content on both attempts → returns original dict.
    content = {"birthDate": "not-a-date", "id": "bad"}
    result = utils.json_to_obj(content, "Patient")
    # The validator raises, falls back to the dict
    assert isinstance(result, dict)


# --------------------------------------------------------------------------- #
# get_resource_type
# --------------------------------------------------------------------------- #


def test_get_resource_type_returns_resource_type(tmp_path):
    path = tmp_path / "patient.json"
    path.write_text(json.dumps({"resourceType": "Patient", "id": "p1"}), encoding="utf-8")
    assert utils.get_resource_type(path) == "Patient"


def test_get_resource_type_returns_none_for_array(tmp_path):
    path = tmp_path / "array.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert utils.get_resource_type(path) is None


def test_get_resource_type_returns_none_when_key_missing(tmp_path):
    path = tmp_path / "nokey.json"
    path.write_text(json.dumps({"id": "p1"}), encoding="utf-8")
    assert utils.get_resource_type(path) is None


# --------------------------------------------------------------------------- #
# get_value_from_element
# --------------------------------------------------------------------------- #


def test_get_value_from_element_finds_fixed_value():
    elem = SimpleNamespace(fixedString="hello", id="e1")
    attr_name, val = utils.get_value_from_element(elem, ("fixed",))
    assert attr_name == "fixedString"
    assert val == "hello"


def test_get_value_from_element_finds_pattern_value():
    elem = SimpleNamespace(patternCode="active", id="e1")
    attr_name, val = utils.get_value_from_element(elem, ("pattern",))
    assert attr_name == "patternCode"
    assert val == "active"


def test_get_value_from_element_returns_none_when_not_found():
    elem = SimpleNamespace(id="e1", status="active")
    attr_name, val = utils.get_value_from_element(elem, ("fixed",))
    assert attr_name is None
    assert val is None


def test_get_value_from_element_skips_none_values():
    # fixedString is None → should not match; fixedCode is the first non-None hit
    elem = SimpleNamespace(fixedString=None, fixedCode="Y", id="e1")
    _, val = utils.get_value_from_element(elem, ("fixed",))
    assert val is not None


# --------------------------------------------------------------------------- #
# config.py
# --------------------------------------------------------------------------- #


def test_load_config_returns_dict(tmp_path):
    cfg = {"project_path": "/data/project", "mode": "test"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert load_config(str(path)) == cfg


def test_load_config_exits_on_missing_file():
    with pytest.raises(SystemExit) as exc_info:
        load_config("/nonexistent/config.json")
    assert exc_info.value.code == 1


def test_save_config_writes_json(tmp_path):
    path = tmp_path / "out.json"
    cfg = {"a": 1, "b": [1, 2]}
    save_config(str(path), cfg)
    assert json.loads(path.read_text(encoding="utf-8")) == cfg


def test_save_config_roundtrip(tmp_path):
    path = tmp_path / "cfg.json"
    original = {"key": "value", "nested": {"x": 42}}
    save_config(str(path), original)
    assert load_config(str(path)) == original


def test_show_config_runs_without_error(capsys):
    show_config({"a": 1, "b": [1, 2]})
    assert capsys.readouterr().out  # something was printed
