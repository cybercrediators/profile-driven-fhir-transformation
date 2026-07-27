"""Unit tests for mapping/fml_structure.py — the "helper map" generator that turns an
arbitrary JSON sample into a placeholder (logical) StructureDefinition snapshot so
arbitrary source data can act as a StructureMap source.

Covers: the JSON->FHIR-type inference (`map_fhir_type`), the recursive ElementDefinition
builder (`create_json_element_definition`) for nested objects, lists of objects, lists of
primitives, and scalars, and the `generate_helper_map` orchestration (fresh generation,
reuse of a previously-generated helper map, and the input-not-found/invalid-input errors) —
using a minimal in-memory DataIO fake so the tests never touch the real filesystem project
structure.
"""

from pathlib import Path

import pytest

from data_handling.app_state import AppState
from data_handling.data_io import DataIO
from mapping import fml_structure as FS

pytestmark = pytest.mark.unit


# ── map_fhir_type ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [
        (True, "boolean"),
        (False, "boolean"),
        (3, "integer"),
        (3.5, "decimal"),
        ("hello", "string"),
        (None, "string"),
    ],
)
def test_map_fhir_type(value, expected):
    assert FS.map_fhir_type(value) == expected


def test_map_fhir_type_bool_before_int():
    # bool is a subclass of int in Python -> booleans must not be misclassified as integer.
    assert FS.map_fhir_type(True) == "boolean"


# ── construct_element_definition ─────────────────────────────────────────────────
def test_construct_element_definition_defaults():
    from fhir.resources.R4B.elementdefinition import ElementDefinitionType

    el = FS.construct_element_definition(
        "Model.name", "Model.name", ElementDefinitionType.model_construct(code="string")
    )
    assert el.id == "Model.name"
    assert el.path == "Model.name"
    assert el.min == 0
    assert el.max == "1"
    assert el.type[0].code == "string"


# ── create_json_element_definition: recursive structure ─────────────────────────
def _paths(elements):
    return [(e.path, e.type[0].code, e.max) for e in elements]


def test_create_json_element_definition_scalar_leaf():
    out = []
    FS.create_json_element_definition("Model.age", 30, out)
    assert len(out) == 1
    assert out[0].path == "Model.age"
    assert out[0].type[0].code == "integer"
    assert out[0].max == "1"


def test_create_json_element_definition_nested_object():
    out = []
    FS.create_json_element_definition("Model.address", {"city": "X", "zip": "12345"}, out)
    paths = _paths(out)
    assert ("Model.address", "Element", "1") in paths
    assert ("Model.address.city", "string", "1") in paths
    assert ("Model.address.zip", "string", "1") in paths


def test_create_json_element_definition_list_of_objects_uses_first_sample():
    out = []
    data = [{"system": "sys1", "value": "v1"}, {"system": "sys2", "value": "v2"}]
    FS.create_json_element_definition("Model.identifiers", data, out)
    paths = _paths(out)
    assert ("Model.identifiers", "Element", "*") in paths
    # Children are derived from the first list item only.
    assert ("Model.identifiers.system", "string", "1") in paths
    assert ("Model.identifiers.value", "string", "1") in paths
    assert len(out) == 3  # parent + 2 children, not 5


def test_create_json_element_definition_list_of_primitives_has_no_children():
    out = []
    FS.create_json_element_definition("Model.tags", ["a", "b"], out)
    assert len(out) == 1
    assert out[0].path == "Model.tags"
    assert out[0].max == "*"


# ── generate_helper_map orchestration ────────────────────────────────────────────
class _FakeDataIO:
    """Minimal in-memory stand-in exercising exactly the DataIO surface
    generate_helper_map touches (no real project-folder filesystem needed)."""

    ProjectFolders = DataIO.ProjectFolders

    def __init__(self, project_dir, *, processed=False, existing_helper_map=None):
        self.project_dir = project_dir
        self._processed = processed
        self._existing = existing_helper_map
        self.stored = []

    def check_processed_helper_maps(self):
        return self._processed

    def load_project_file(self, folder, filename):
        return self._existing

    def store_project_file(self, folder, filename, content, mode="STR", overwrite=False):
        self.stored.append((folder, filename, content))


def _app_state(tmp_path, **kwargs):
    return AppState(conf={}, registry=None, cache=None, dataIO=_FakeDataIO(tmp_path, **kwargs))


def test_generate_helper_map_fresh_generation(tmp_path):
    input_json = tmp_path / "sample.json"
    input_json.write_text(
        '{"name": "Alice", "age": 30, "active": true, '
        '"address": {"city": "Springfield", "zip": "12345"}, '
        '"tags": ["a", "b"]}'
    )
    app_state = _app_state(tmp_path, processed=False)

    sd = FS.generate_helper_map(input_json, app_state, "model1", "http://x/model1", "model1")

    assert sd.id == "model1"
    assert sd.url == "http://x/model1"
    assert sd.type == "Model1"
    assert sd.kind == "logical"
    assert sd.derivation == "specialization"

    paths = [e.path for e in sd.snapshot.element]
    assert "Model1" in paths  # root element
    assert "Model1.name" in paths
    assert "Model1.address.city" in paths
    assert "Model1.tags" in paths

    # persisted exactly once
    assert len(app_state.dataIO.stored) == 1
    folder, filename, _content = app_state.dataIO.stored[0]
    assert filename == "model1.json"


def test_generate_helper_map_reuses_existing_when_not_overwriting(tmp_path):
    existing = {
        "resourceType": "StructureDefinition",
        "id": "model1",
        "url": "http://x/model1",
        "name": "model1",
        "status": "draft",
        "kind": "logical",
        "abstract": False,
        "type": "Model1",
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/Element",
        "derivation": "specialization",
    }
    app_state = _app_state(tmp_path, processed=True, existing_helper_map=existing)
    input_json = tmp_path / "unused.json"  # never read when reusing

    sd = FS.generate_helper_map(
        input_json, app_state, "model1", "http://x/model1", "model1", overwrite=False
    )
    assert sd.id == "model1"
    assert sd.url == "http://x/model1"
    # nothing (re-)stored, no regeneration happened
    assert app_state.dataIO.stored == []


def test_generate_helper_map_reads_from_project_source_data_when_input_missing(tmp_path):
    # input_json doesn't exist at the given path, but a same-named file does exist under
    # the project's source_data folder -> generate_helper_map falls back to that.
    source_data_dir = tmp_path / DataIO.ProjectFolders.SOURCE_DATA.value
    source_data_dir.mkdir(parents=True)
    (source_data_dir / "sample.json").write_text('{"count": 1}')

    app_state = _app_state(tmp_path, processed=False)
    missing_path = tmp_path / "does-not-exist" / "sample.json"

    sd = FS.generate_helper_map(missing_path, app_state, "model2", "http://x/model2", "model2")
    paths = [e.path for e in sd.snapshot.element]
    assert "Model2.count" in paths


def test_generate_helper_map_missing_input_raises(tmp_path):
    app_state = _app_state(tmp_path, processed=False)
    missing_path = tmp_path / "really-missing.json"
    with pytest.raises(FileNotFoundError):
        FS.generate_helper_map(missing_path, app_state, "model3", "http://x/model3", "model3")


def test_generate_helper_map_empty_list_input_raises(tmp_path):
    input_json = tmp_path / "empty.json"
    input_json.write_text("[]")
    app_state = _app_state(tmp_path, processed=False)
    with pytest.raises(ValueError):
        FS.generate_helper_map(input_json, app_state, "model4", "http://x/model4", "model4")


def test_generate_helper_map_scalar_input_raises_type_error(tmp_path):
    input_json = tmp_path / "scalar.json"
    input_json.write_text("42")
    app_state = _app_state(tmp_path, processed=False)
    with pytest.raises(TypeError):
        FS.generate_helper_map(input_json, app_state, "model5", "http://x/model5", "model5")
