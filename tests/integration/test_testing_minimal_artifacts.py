from pathlib import Path

import pytest


@pytest.mark.integration
def test_testing_minimal_artifacts_exist():
    base = Path(__file__).resolve().parents[2] / "projects" / "testing_minimal"
    assert base.exists(), "testing_minimal project folder is missing"

    expected_files = [
        base / "input_profile" / "StructureDefinition-minimal-patient-profile.json",
        base / "structure_maps" / "structure_map_testing_minimal.json",
        base / "source_data" / "source_data.json",
        base / "source_data" / "source_obj.json",
        base / "source_definitions" / "source-definition-testing-minimal.json",
        base / "source_definitions" / "source-definition-testing-minimal2.json",
        base / "processed_resources",
        base / "testing_minimal.tgz",
    ]

    missing = [str(p) for p in expected_files if not p.exists()]
    assert not missing, f"Missing expected testing_minimal artifacts: {missing}"

    # processed_resources should not be empty
    assert (base / "processed_resources").stat().st_size > 0
