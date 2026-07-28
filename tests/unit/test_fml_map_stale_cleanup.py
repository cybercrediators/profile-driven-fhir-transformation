"""Stale-file cleanup after StructureMap generation (fml_map).

A removed/renamed profile (or a shifted index prefix on a fresh reprocess) leaves the
previously generated SM file behind; it would still be uploaded and transform against a
stale source model. `_cleanup_stale_structure_maps` deletes exactly the generated-named
files this run did not produce — and nothing else.
"""

from types import SimpleNamespace

import pytest

from mapping.fml_map import StructureMapGenerator
from helpers.utils import fhir_name_token

pytestmark = pytest.mark.unit


def _generator(tmp_path, map_name="structure_map_proj"):
    data_io = SimpleNamespace(
        ProjectFolders=SimpleNamespace(
            STRUCTURE_MAPS=SimpleNamespace(value="structure_maps")
        ),
        project_dir=tmp_path,
        delete_file=lambda rel: (tmp_path / rel).unlink(),
    )
    smg = object.__new__(StructureMapGenerator)
    smg.app_state = SimpleNamespace(dataIO=data_io)
    smg.map_name = map_name
    return smg


def test_stale_generated_maps_are_deleted_current_and_foreign_kept(tmp_path):
    sm_dir = tmp_path / "structure_maps"
    sm_dir.mkdir()
    (sm_dir / "001_structure_map_proj-patient.json").write_text("{}")
    # stale: same index as a current file but a profile no longer generated
    (sm_dir / "001_structure_map_proj-removed-profile.json").write_text("{}")
    # stale: index shifted on regen
    (sm_dir / "003_structure_map_proj-patient.json").write_text("{}")
    # hand-authored / foreign names must never be touched
    (sm_dir / "custom_handwritten_map.json").write_text("{}")
    (sm_dir / "002_structure_map_OTHERproj-x.json").write_text("{}")

    smg = _generator(tmp_path)
    current = [SimpleNamespace(name="001_structure_map_proj-patient")]
    smg._cleanup_stale_structure_maps(current)

    remaining = {f.name for f in sm_dir.iterdir()}
    assert remaining == {
        "001_structure_map_proj-patient.json",
        "custom_handwritten_map.json",
        "002_structure_map_OTHERproj-x.json",
    }


def test_cleanup_is_noop_without_structure_maps_dir(tmp_path):
    smg = _generator(tmp_path)
    smg._cleanup_stale_structure_maps([SimpleNamespace(name="001_structure_map_proj-a")])


def test_fhir_name_token_and_structure_map_invariant_normalization():
    target = SimpleNamespace(
        context="target", contextType=None, element="active"
    )
    rule = SimpleNamespace(name="map-active", target=[target], rule=[])
    sm = SimpleNamespace(
        name=fhir_name_token("001_structure-map-profile"),
        group=[SimpleNamespace(rule=[rule])],
    )

    StructureMapGenerator._normalize_and_validate_structure_map(sm)

    assert sm.name == "Map_001_structure_map_profile"
    assert target.contextType == "variable"


def test_structure_map_invariant_rejects_target_element_without_context():
    bad = SimpleNamespace(
        name="bad", target=[SimpleNamespace(context=None, element="active")], rule=[]
    )
    sm = SimpleNamespace(name="ValidMap", group=[SimpleNamespace(rule=[bad])])

    with pytest.raises(ValueError, match="without a context"):
        StructureMapGenerator._normalize_and_validate_structure_map(sm)
