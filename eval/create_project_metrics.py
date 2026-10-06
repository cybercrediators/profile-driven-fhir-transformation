"""compute the offline metrics for one project directory

Usage:
    PYTHONPATH=.:src python3 -m eval.create_project_metrics projects/<project> \
        [--existing DIR] [--output FILE] [--report FILE]

--existing defines the copy of the shipped maps taken before regeneration
--output is a transform result to compare M1 against
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval import canonicalize, metrics


def _load(path: Path) -> Any:
    obj = json.loads(path.read_text())
    # registry objects are sometimes stored as a JSON-encoded string
    return json.loads(obj) if isinstance(obj, str) else obj


def _maps_by_target(directory: Path) -> Dict[str, dict]:
    """StructureMaps of a directory, identified by the profile url they target"""
    out: Dict[str, dict] = {}
    for f in sorted(directory.glob("*.json")):
        sm = _load(f)
        if not isinstance(sm, dict) or sm.get("resourceType") != "StructureMap":
            continue
        for s in sm.get("structure") or []:
            if s.get("mode") == "target" and s.get("url"):
                out[s["url"]] = sm
    return out


def _profiles(project: Path) -> List[dict]:
    out = []
    for f in sorted((project / "input_profile").glob("*.json")):
        sd = _load(f)
        if isinstance(sd, dict) and sd.get("resourceType") == "StructureDefinition":
            out.append(sd)
    return out


def _registry_for(project: Path, sd: dict) -> Optional[dict]:
    """locate a profile registry object"""
    reg_dir = project / "processed_resources"
    if not reg_dir.is_dir():
        return None
    keys = [k for k in (sd.get("id"), (sd.get("url") or "").rsplit("/", 1)[-1]) if k]
    for key in keys:
        candidate = reg_dir / key
        if candidate.is_file():
            return _load(candidate)
    lowered = {p.name.lower(): p for p in reg_dir.iterdir() if p.is_file()}
    for key in keys:
        hit = lowered.get(key.lower())
        if hit is not None:
            return _load(hit)
    return None


def collect(project: Path, existing_dir: Optional[Path], output: Optional[Path]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"project": project.name, "metrics": {}}
    m = report["metrics"]
    fresh = _maps_by_target(project / "structure_maps")

    if fresh:
        m["M5_remaining_todo"] = {
            url: metrics.remaining_todo(sm)["todo_count"] for url, sm in fresh.items()
        }
        m["M5_total"] = sum(m["M5_remaining_todo"].values())

    if existing_dir is not None:
        existing = _maps_by_target(existing_dir)
        shared = sorted(set(existing) & set(fresh))
        m["M3_rule_prf1"] = {
            url: canonicalize.rule_level_prf1(fresh[url], existing[url]) for url in shared
        }
        m["M6_edit_distance"] = {
            url: metrics.edit_distance(fresh[url], existing[url]) for url in shared
        }
        drifted = [url for url in shared if m["M3_rule_prf1"][url]["f1"] != 1.0]
        drifted += [f"{url}: present in only one set" for url in set(existing) ^ set(fresh)]
        m["M0c_reproducibility"] = {
            "maps_compared": len(set(existing) | set(fresh)),
            "identical": len(shared) - len([u for u in shared if m["M3_rule_prf1"][u]["f1"] != 1.0]),
            "drifted": drifted,
            "rule_identical": not drifted,
        }

    table = project / "source_data" / "mapping_table.json"
    source = project / "source_data" / "source_data.json"
    if output is not None and table.is_file() and source.is_file():
        m["M1_data_preservation"] = metrics.data_preservation(
            _load(table), _load(source), _load(output)
        )

    feature, extraction, degradation = {}, {}, {}
    for sd in _profiles(project):
        url = sd.get("url") or ""
        key = sd.get("id") or url.rsplit("/", 1)[-1] or "?"
        sm = fresh.get(url)
        feature[key] = metrics.feature_matrix(sd, sm)["matrix"]
        reg = _registry_for(project, sd)
        if reg is None:
            continue
        extraction[key] = metrics.extraction_completeness(sd, reg)["completeness_pct"]
        degradation[key] = metrics.graceful_degradation(sd, reg)["dropped_element_count"]
    if feature:
        m["M7_feature_matrix"] = feature
    if extraction:
        m["M8_extraction_completeness"] = extraction
    if degradation:
        m["M9_graceful_degradation"] = degradation

    return report


def summarize(report: Dict[str, Any]) -> str:
    m = report["metrics"]
    lines = [f"\n=== {report['project']} ==="]
    if "M5_total" in m:
        lines.append(f"  M5 remaining TODO: {m['M5_total']} over {len(m['M5_remaining_todo'])} maps")
    if "M0c_reproducibility" in m:
        rc = m["M0c_reproducibility"]
        lines.append(
            f"  M0c reproducibility: rule_identical={rc['rule_identical']} "
            f"({rc['identical']}/{rc['maps_compared']})"
        )
        for d in rc["drifted"]:
            lines.append(f"    drifted: {d}")
    else:
        lines.append("  M0c/M3/M6: skipped (no --existing)")
    if "M1_data_preservation" in m:
        dp = m["M1_data_preservation"]
        lines.append(
            f"  M1 data preservation: {dp['fields_preserved']}/{dp['fields_graded']} "
            f"({dp['preservation_pct']}%), transform-mapped={dp['fields_transform_mapped']}"
        )
    else:
        lines.append("  M1: skipped (no --output)")
    for key, label in (("M8_extraction_completeness", "M8 extraction"),
                       ("M9_graceful_degradation", "M9 silent drops")):
        for pid, value in (m.get(key) or {}).items():
            lines.append(f"  {label}[{pid}]: {value}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="offline metrics for one project directory")
    ap.add_argument("project", type=Path, help="path to the project folder")
    ap.add_argument("--existing", type=Path, default=None,
                    help="directory holding the shipped maps, for M0c/M3/M6")
    ap.add_argument("--output", type=Path, default=None,
                    help="a transform result to grade M1 against")
    ap.add_argument("--report", type=Path, default=None,
                    help="where to write the JSON report (default: no file)")
    args = ap.parse_args()

    if not args.project.is_dir():
        ap.error(f"no such project directory: {args.project}")

    report = collect(args.project, args.existing, args.output)
    print(summarize(report))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nReport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
