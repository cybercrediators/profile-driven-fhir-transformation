"""Stale-file cleanup after StructureMap generation (fml_map).

A removed/renamed profile (or a shifted index prefix on a fresh reprocess) leaves the
previously generated SM file behind; it would still be uploaded and transform against a
stale source model. `_cleanup_stale_structure_maps` deletes exactly the generated-named
files this run did not produce — and nothing else.
"""

from types import SimpleNamespace
import json

import pytest

from fhir.resources.R4B.quantity import Quantity

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


def test_todo_sanitizer_keeps_source_scaffolds_and_removes_unsafe_todos():
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "PatientProfile"
    generator.mapping_diagnostics = []
    source_scaffold = SimpleNamespace(
        name="map-required",
        source=[
            SimpleNamespace(
                context="source",
                element="TODO_MAP_REQUIRED",
                variable="src-required",
            )
        ],
        target=[
            SimpleNamespace(
                context="target",
                element="status",
                transform="copy",
                parameter=None,
            )
        ],
        dependent=None,
        rule=[],
    )
    unsafe_target_todo = SimpleNamespace(
        name="map-unsafe-target",
        source=[SimpleNamespace(context="source", element="status")],
        target=[
            SimpleNamespace(
                context="target",
                element="TODO_TARGET_STATUS",
                transform="copy",
                parameter=None,
            )
        ],
        dependent=None,
        rule=[],
    )
    unsafe_source_expression = SimpleNamespace(
        name="map-unsafe-check",
        source=[
            SimpleNamespace(
                context="source",
                element="status",
                check="TODO_DEFINE_SOURCE_CHECK",
            )
        ],
        target=[
            SimpleNamespace(
                context="target",
                element="status",
                transform="copy",
                parameter=None,
            )
        ],
        dependent=None,
        rule=[],
    )
    deferred_reference = SimpleNamespace(
        name="TODO-resolve-reference-Patient-generalPractitioner",
        documentation="Reference<Patient.generalPractitioner> -> Practitioner",
        source=[SimpleNamespace(element=None)],
        target=None,
        dependent=None,
        rule=[],
    )
    valid = SimpleNamespace(
        name="map-active",
        source=[SimpleNamespace(element="active")],
        target=[SimpleNamespace(element="active", parameter=None)],
        dependent=None,
        rule=[source_scaffold, unsafe_target_todo, unsafe_source_expression],
    )
    group = SimpleNamespace(name="Transform-Patient", rule=[valid, deferred_reference])
    structure_map = SimpleNamespace(group=[group])

    generator._sanitize_todo_rules(structure_map)

    assert group.rule == [valid, deferred_reference]
    assert valid.rule == [source_scaffold]
    assert source_scaffold.source[0].element == "TODO_MAP_REQUIRED"
    assert {item["code"] for item in generator.mapping_diagnostics} == {
        "unresolved-map-placeholder",
        "deferred-reference",
    }


def test_profile_facets_become_json_safe_diagnostics():
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "ObservationProfile"
    generator.mapping_diagnostics = []
    obj = SimpleNamespace(
        mappable_fields=[
            {
                "path": "Observation.value[x]",
                "id": "Observation.value[x]",
                "default_value": {
                    "type": "Quantity",
                    "value": Quantity(value=2, unit="mg"),
                },
                "constraints": [
                    {
                        "key": "obs-1",
                        "severity": "error",
                        "expression": "dataAbsentReason.empty() or value.empty()",
                    }
                ],
                "is_modifier": True,
            }
        ]
    )

    generator._record_profile_facet_diagnostics(obj)

    json.dumps(generator.mapping_diagnostics)
    assert {item["code"] for item in generator.mapping_diagnostics} == {
        "target-default-value",
        "target-fhirpath-constraint",
        "target-modifier-element",
    }
    assert all(
        item["profile"] == "ObservationProfile"
        for item in generator.mapping_diagnostics
    )


def test_coverage_report_is_written_when_only_diagnostics_exist():
    stored = {}
    generator = object.__new__(StructureMapGenerator)
    generator._coverage = {}
    generator.mapping_diagnostics = [
        {
            "code": "ambiguous-slice-selection",
            "message": "explicit selection required",
        }
    ]
    generator.map_name = "diagnostic-map"
    generator.app_state = SimpleNamespace(
        dataIO=SimpleNamespace(
            ProjectFolders=SimpleNamespace(SOURCE_DATA="source_data"),
            store_project_file=lambda folder, name, body, **kwargs: stored.update(
                {"folder": folder, "name": name, "body": body}
            ),
        )
    )

    generator._save_coverage_report()

    assert stored["name"] == "diagnostic-map_coverage.json"
    report = json.loads(stored["body"])
    assert report["summary"]["profiles"] == 0
    assert report["mapping_diagnostics"][0]["code"] == (
        "ambiguous-slice-selection"
    )
