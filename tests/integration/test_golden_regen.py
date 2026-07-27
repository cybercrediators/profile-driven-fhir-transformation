"""Golden-regen gate as a committed test: regenerate source-def + StructureMaps (+
ConceptMaps, coverage report) for the golden example projects into a temp copy and
content-compare against the goldens on disk. Generation is deterministic, so any
unintended change to the parser/mapping layer shows up as a diff here. Offline —
everything resolves from the repo (profiles, local package cache); no matchbox,
no valkey (DISK cache in a tmp dir).

The numeric map-file prefix (``001_``…) depends on registry iteration order, which
can differ between filesystems — comparison is keyed by the name suffix instead.
"""
import json
import re
import shutil
from pathlib import Path

import pytest

from controller.pipeline_controller.pipeline_controller import PipelineController

REPO = Path(__file__).resolve().parents[2]
GOLDEN_PROJECTS = ["example_project1", "example_project2"]


def _regen(project: str, tmp_path: Path) -> Path:
    """Copy the project to tmp and regenerate its artifacts there; return the copy."""
    work = tmp_path / project
    shutil.copytree(REPO / "projects" / project, work)
    for f in (work / "structure_maps").glob("*.json"):
        f.unlink()

    conf = json.loads((REPO / "conf" / f"{project}.json").read_text())
    conf["project_path"] = str(work)
    conf["mapping_table_path"] = str(work / "source_data" / "mapping_table.json")
    conf["resource_cache_path"] = str(REPO / "data" / "resource_cache") + "/"
    conf["external_cache_service"] = "DISK"
    conf["cache_args"] = {"cache_dir": str(tmp_path / "url_cache")}

    pc = PipelineController(
        conf=conf,
        force_overwrite=True,
        reprocess_profile=True,
        reprocess_helper_definition=True,
        reprocess_structure_map=True,
        create_references=True,
        minimal_mode=True,
    )
    assert pc.run_source_def(), f"{project}: source-def generation failed"
    assert pc.run_static_gen_sm(), f"{project}: StructureMap generation failed"
    return work


def _maps_by_suffix(project_dir: Path) -> dict:
    """StructureMap JSON content keyed by filename without the numeric prefix."""
    out = {}
    for f in (project_dir / "structure_maps").glob("*.json"):
        suffix = f.name.split("_", 1)[1] if f.name[:1].isdigit() else f.name
        assert suffix not in out, f"duplicate map suffix {suffix} in {project_dir}"
        content = json.loads(f.read_text())
        # the map's `name` embeds the same order-dependent numeric prefix as the filename
        if isinstance(content.get("name"), str):
            content["name"] = re.sub(r"^\d{3}_", "", content["name"])
        out[suffix] = content
    return out


def _concept_maps(project_dir: Path) -> dict:
    cm_dir = project_dir / "source_data" / "concept_maps"
    if not cm_dir.is_dir():
        return {}
    return {f.name: json.loads(f.read_text()) for f in cm_dir.glob("*.json")}


@pytest.mark.integration
@pytest.mark.parametrize("project", GOLDEN_PROJECTS)
def test_regen_reproduces_golden_artifacts(project, tmp_path):
    work = _regen(project, tmp_path)

    golden_dir = REPO / "projects" / project
    golden_maps = _maps_by_suffix(golden_dir)
    regen_maps = _maps_by_suffix(work)
    assert set(regen_maps) == set(golden_maps)
    for suffix in golden_maps:
        assert regen_maps[suffix] == golden_maps[suffix], (
            f"{project}: regenerated StructureMap differs from golden: {suffix}"
        )

    assert _concept_maps(work) == _concept_maps(golden_dir), (
        f"{project}: regenerated ConceptMaps differ from goldens"
    )

    cov_name = f"structure_map_{project}_coverage.json"
    golden_cov = golden_dir / "source_data" / cov_name
    if golden_cov.is_file():
        regen_cov = work / "source_data" / cov_name
        assert json.loads(regen_cov.read_text()) == json.loads(golden_cov.read_text()), (
            f"{project}: regenerated coverage report differs from golden"
        )
