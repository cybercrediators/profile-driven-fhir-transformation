import json

import pytest

from fhir_spec.context import (
    FHIRSpecContext,
    activate_fhir_spec_context,
    clear_fhir_spec_context,
)
from mapping.rule_ir import (
    CONTEXT_TYPES,
    GROUP_TYPE_MODES,
    INPUT_MODES,
    MAPPING_TRANSFORMS,
    PARAMETER_VALUE_TYPES,
    SOURCE_LIST_MODES,
    STRUCTURE_MODES,
    TARGET_LIST_MODES,
)


pytestmark = pytest.mark.unit


_FILES = {
    "source_list_modes": "CodeSystem-map-source-list-mode.json",
    "target_list_modes": "CodeSystem-map-target-list-mode.json",
    "transforms": "CodeSystem-map-transform.json",
    "structure_modes": "CodeSystem-map-model-mode.json",
    "input_modes": "CodeSystem-map-input-mode.json",
    "context_types": "CodeSystem-map-context-type.json",
    "group_type_modes": "CodeSystem-map-group-type-mode.json",
}

_VALUES = {
    "source_list_modes": ["first", "only_one"],
    "target_list_modes": ["first", "share"],
    "transforms": ["copy", "fixture-transform"],
    "structure_modes": ["source", "target"],
    "input_modes": ["source", "target"],
    "context_types": ["type", "variable"],
    "group_type_modes": ["none", "types"],
}


def _core_package(tmp_path, release="R4"):
    version = "4.0.1" if release == "R4" else "4.3.0"
    name = "hl7.fhir.r4.core" if release == "R4" else "hl7.fhir.r4b.core"
    package = tmp_path / release / "package"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": version,
                "fhirVersions": [version],
            }
        ),
        encoding="utf-8",
    )
    for key, filename in _FILES.items():
        (package / filename).write_text(
            json.dumps(
                {
                    "resourceType": "CodeSystem",
                    "concept": [{"code": code} for code in _VALUES[key]],
                }
            ),
            encoding="utf-8",
        )
    (package / "StructureDefinition-StructureMap.json").write_text(
        json.dumps(
            {
                "resourceType": "StructureDefinition",
                "snapshot": {
                    "element": [
                        {
                            "path": (
                                "StructureMap.group.rule.target.parameter.value[x]"
                            ),
                            "type": [
                                {"code": "id"},
                                {"code": "string"},
                                {"code": "boolean"},
                                {"code": "integer"},
                                {"code": "decimal"},
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    for type_code in ("string", "CodeableConcept"):
        (package / f"StructureDefinition-{type_code}.json").write_text(
            "{}", encoding="utf-8"
        )
    return package


@pytest.fixture(autouse=True)
def _clear_context():
    clear_fhir_spec_context()
    yield
    clear_fhir_spec_context()


@pytest.mark.parametrize(
    ("release", "version", "package_name"),
    [
        ("R4", "4.0.1", "hl7.fhir.r4.core"),
        ("R4B", "4.3.0", "hl7.fhir.r4b.core"),
    ],
)
def test_loads_release_specific_capabilities_from_core_package(
    tmp_path, release, version, package_name
):
    context = FHIRSpecContext.from_core_package(
        _core_package(tmp_path, release),
        declared_fhir_version=version,
    )
    assert context.release == release
    assert context.core_package_name == package_name
    assert context.capabilities.transforms == {"copy", "fixture-transform"}
    assert context.capabilities.parameter_value_types == {
        "id",
        "string",
        "boolean",
        "integer",
        "decimal",
    }


def test_public_capability_sets_follow_active_context(tmp_path):
    context = FHIRSpecContext.from_core_package(_core_package(tmp_path))
    activate_fhir_spec_context(context)

    assert set(SOURCE_LIST_MODES) == set(_VALUES["source_list_modes"])
    assert set(TARGET_LIST_MODES) == set(_VALUES["target_list_modes"])
    assert set(MAPPING_TRANSFORMS) == set(_VALUES["transforms"])
    assert set(STRUCTURE_MODES) == set(_VALUES["structure_modes"])
    assert set(INPUT_MODES) == set(_VALUES["input_modes"])
    assert set(CONTEXT_TYPES) == set(_VALUES["context_types"])
    assert set(GROUP_TYPE_MODES) == set(_VALUES["group_type_modes"])
    assert set(PARAMETER_VALUE_TYPES) == {
        "id",
        "string",
        "boolean",
        "integer",
        "decimal",
    }


def test_core_package_release_mismatch_is_rejected(tmp_path):
    package = _core_package(tmp_path, "R4B")
    with pytest.raises(ValueError, match="requires core package"):
        FHIRSpecContext.from_core_package(
            package, declared_fhir_version="4.0.1"
        )


def test_r4_guard_accepts_r4_type_and_rejects_r4b_only_type(tmp_path):
    context = FHIRSpecContext.from_core_package(_core_package(tmp_path, "R4"))
    base = {
        "resourceType": "StructureDefinition",
        "url": "http://example.org/StructureDefinition/test",
        "snapshot": {
            "element": [
                {
                    "id": "Observation.code",
                    "path": "Observation.code",
                    "type": [{"code": "CodeableConcept"}],
                }
            ]
        },
    }
    context.assert_profile_compatible(base)

    base["snapshot"]["element"][0]["type"] = [{"code": "CodeableReference"}]
    with pytest.raises(ValueError, match="CodeableReference.*not defined"):
        context.assert_profile_compatible(base)


def test_project_release_defaults_to_r4_and_detects_r4b(tmp_path):
    r4 = _core_package(tmp_path, "R4")
    project = tmp_path / "project-r4"
    (project / "input_profile").mkdir(parents=True)
    default_context = FHIRSpecContext.for_project(
        project, {"fhir_core_package_path": str(r4)}
    )
    assert default_context.release == "R4"

    r4b = _core_package(tmp_path, "R4B")
    project_b = tmp_path / "project-r4b"
    (project_b / "input_profile").mkdir(parents=True)
    (project_b / "input_profile" / "package.json").write_text(
        json.dumps({"fhirVersions": ["4.3.0"]}), encoding="utf-8"
    )
    r4b_context = FHIRSpecContext.for_project(
        project_b, {"fhir_core_package_path": str(r4b)}
    )
    assert r4b_context.release == "R4B"


def test_conflicting_project_and_config_releases_are_rejected(tmp_path):
    project = tmp_path / "project"
    (project / "input_profile").mkdir(parents=True)
    (project / "input_profile" / "package.json").write_text(
        json.dumps({"fhirVersions": ["4.3.0"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="conflicts"):
        FHIRSpecContext.for_project(
            project,
            {
                "fhir_version": "4.0.1",
                "fhir_core_package_path": str(_core_package(tmp_path, "R4")),
            },
        )
