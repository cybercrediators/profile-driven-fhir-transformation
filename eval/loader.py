"""artifact loading + offline map regeneration for the evaluation runner"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_registry_obj(path: Path) -> Optional[dict]:
    """load a jsonpickle processed_resources object as a plain dict"""
    try:
        raw = json.loads(path.read_text())
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None


def _target_profile_url(sm: Dict[str, Any]) -> Optional[str]:
    for s in sm.get("structure", []) or []:
        if s.get("mode") == "target":
            return s.get("url")
    return None


@dataclass
class ProjectArtifacts:
    """everything the metric layer needs for one project (from disk)"""

    name: str
    project_dir: Path
    maps: List[dict] = field(default_factory=list)
    output: Any = None
    mapping_table: Dict[str, str] = field(default_factory=dict)
    source_data: List[dict] = field(default_factory=list)
    profiles: Dict[str, dict] = field(default_factory=dict)
    registry: Dict[str, dict] = field(default_factory=dict)
    concept_mapped_fields: List[str] = field(default_factory=list)
    regenerated_maps: Optional[List[dict]] = None

    def profile_id_for_map(self, sm: dict) -> Optional[str]:
        """resolve which profile id a map targets"""
        url = _target_profile_url(sm)
        for pid, sd in self.profiles.items():
            if sd.get("url") == url:
                return pid
        return None


def load_project(name: str) -> ProjectArtifacts:
    pdir = _REPO / "projects" / name
    art = ProjectArtifacts(name=name, project_dir=pdir)

    sm_dir = pdir / "structure_maps"
    if sm_dir.is_dir():
        art.maps = [_load_json(p) for p in sorted(sm_dir.glob("*.json"))]

    out = pdir / "converted_data" / "expected_output.json"
    if out.is_file():
        art.output = _load_json(out)

    mt = pdir / "source_data" / "mapping_table.json"
    if mt.is_file():
        art.mapping_table = _load_json(mt)

    sd = pdir / "source_data" / "source_data.json"
    if sd.is_file():
        data = _load_json(sd)
        art.source_data = data if isinstance(data, list) else [data]

    prof_dir = pdir / "input_profile"
    if prof_dir.is_dir():
        for p in sorted(prof_dir.glob("*.json")):
            sdj = _load_json(p)
            if not isinstance(sdj, dict) or sdj.get("resourceType") != "StructureDefinition":
                continue
            pid = sdj.get("id") or p.stem
            art.profiles[pid] = sdj

    reg_dir = pdir / "processed_resources"
    if reg_dir.is_dir():
        for p in sorted(reg_dir.iterdir()):
            if p.is_file():
                obj = _load_registry_obj(p)
                if isinstance(obj, dict):
                    art.registry[p.name] = obj

    cm_dir = pdir / "source_data" / "concept_maps"
    if cm_dir.is_dir() and cm_dir.glob("*.json"):
        art.concept_mapped_fields = _detect_concept_mapped_fields(art)

    return art


def _detect_concept_mapped_fields(art: ProjectArtifacts) -> List[str]:
    """Best-effort: source fields whose mapped target is a coded element and for which a ConceptMap exists in the project"""
    coded_endings = ("gender", "status", "code")

    def _is_coded(target: Any) -> bool:
        if not isinstance(target, str) or not target:
            return False
        leaf = target.rsplit(".", 1)[-1].split(":", 1)[0].lower()
        return leaf.endswith(coded_endings)

    return [f for f, t in art.mapping_table.items() if _is_coded(t)]


def regenerate_offline(name: str, workdir: Path) -> List[dict]:
    """regenerate StructureMaps for a project offline (no Matchbox upload)"""
    src = _REPO / "projects" / name
    dst = workdir / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for sub in ("structure_maps", "source_data/concept_maps"):
        d = dst / sub
        if d.is_dir():
            for f in d.iterdir():
                f.unlink()

    conf = _load_json(_REPO / "conf" / f"{name}.json")
    conf["project_path"] = str(dst) + "/"
    confpath = workdir / f"{name}.conf.json"
    confpath.write_text(json.dumps(conf, indent=2))

    mt = dst / "source_data" / "mapping_table.json"
    cmd = [
        "python", "src/main.py", "-c", str(confpath),
        "pipeline", "run", "-f", "-msm", "-mt", str(mt),
    ]
    env = dict(os.environ)
    # timeout set to longest running pipeline atm
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, timeout=2400, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"offline regeneration of {name} failed:\n{r.stderr[-2000:]}")

    out_dir = dst / "structure_maps"
    return [_load_json(p) for p in sorted(out_dir.glob("*.json"))]

def profile_type_aliases(profiles: Dict[str, dict]) -> Dict[str, str]:
    """map every name a mapping table might root a target on to its base type"""

    out: Dict[str, str] = {}
    for pid, sd in (profiles or {}).items():
        base = sd.get("type")
        if not base:
            continue
        url = str(sd.get("url") or "")
        for alias in (pid, sd.get("id"), sd.get("name"),
                      url.rstrip("/").rsplit("/", 1)[-1] if url else None):
            if alias:
                out.setdefault(str(alias), base)
    return out

def find_registry(registry: Dict[str, dict], sd: dict, pid: str):
    """Locate registry object for a profile by given names"""

    url = str(sd.get("url") or "")
    aliases = [pid, sd.get("id"), sd.get("name"),
               url.rstrip("/").rsplit("/", 1)[-1] if url else None]
    for alias in aliases:
        if alias and alias in registry:
            return registry[alias]
    for alias in aliases:
        if not alias:
            continue
        for name, reg in registry.items():
            if alias in name or name in alias:
                return reg
    return None
