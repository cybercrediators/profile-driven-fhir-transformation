"""run evaluation based on metrics
Usage:
    python -m eval.run_eval [project ...] [--regenerate]
    python -m eval.run_eval
Reports stored in eval/reports/<project>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from eval import loader, metrics

_REPORTS = Path(__file__).resolve().parent / "reports"

# define example projects as default
DEFAULT_PROJECTS = ["example_project1", "example_project2"]


def _map_name(sm: Dict[str, Any]) -> str:
    return sm.get("id") or sm.get("name") or "?"


def evaluate_project(name: str, regenerate: bool = False) -> Dict[str, Any]:
    art = loader.load_project(name)
    report: Dict[str, Any] = {"project": name, "metrics": {}}
    m = report["metrics"]

    # check remaining placeholders
    if art.maps:
        m["M5_remaining_todo"] = {
            _map_name(sm): metrics.remaining_todo(sm) for sm in art.maps
        }

    # check data preservation
    if art.mapping_table and art.source_data and art.output is not None:
        m["M1_data_preservation"] = metrics.data_preservation(
            art.mapping_table, art.source_data, art.output, art.concept_mapped_fields,
            profile_types=loader.profile_type_aliases(art.profiles),
        )

    # check equivalence (needs matchbox connection!)
    if art.output is not None:
        m["M2_behavioral_equivalence"] = {
            "measured": False,
            "normaliser_deterministic": metrics.behavioral_equivalence(
                art.output, art.output
            )["equivalent"],
            "note": "offline run: behavioural equivalence needs a live "
            "head-to-head transform of the generated vs comparator map (E16 / E10b)",
        }

    # check feature coverage per (profile and map)
    if art.profiles:
        e8, e7, e9 = {}, {}, {}
        # map profile id to its generated map
        map_for_profile = {}
        for sm in art.maps:
            pid = art.profile_id_for_map(sm)
            if pid:
                map_for_profile[pid] = sm
        for pid, sd in art.profiles.items():
            reg = art.registry.get(pid) or _match_registry(art.registry, pid)
            if reg:
                e8[pid] = metrics.extraction_completeness(sd, reg)
                e9[pid] = metrics.graceful_degradation(sd, reg)
            e7[pid] = metrics.feature_matrix(sd, map_for_profile.get(pid))
        if e8:
            m["M8_extraction_completeness"] = e8
        if e9:
            m["M9_graceful_degradation"] = e9
        if e7:
            m["M7_feature_matrix"] = e7

    # check reproducibility, rule-based metrics, distance compared to regeneration
    if regenerate:
        try:
            workdir = _REPORTS / "_regen"
            workdir.mkdir(parents=True, exist_ok=True)
            regen = loader.regenerate_offline(name, workdir)
            art.regenerated_maps = regen
            m["M0c_reproducibility"] = _reproducibility(art.maps, regen)
            m["M3_rule_prf1"], m["M6_edit_distance"] = _candidate_vs_comparator(art.maps, regen)
        except Exception as e:
            m["regeneration_error"] = str(e)

    _REPORTS.mkdir(parents=True, exist_ok=True)
    (_REPORTS / f"{name}.json").write_text(json.dumps(report, indent=2))
    return report


def _match_registry(registry: Dict[str, dict], pid: str):
    for name, obj in registry.items():
        if pid in name or name in pid:
            return obj
    return None


def _by_target_url(maps: List[dict]) -> Dict[str, dict]:
    out = {}
    for sm in maps:
        for s in sm.get("structure", []) or []:
            if s.get("mode") == "target" and s.get("url"):
                out[s["url"]] = sm
    return out


def _reproducibility(comparator: List[dict], regen: List[dict]) -> Dict[str, Any]:
    """compare rule-identical regenerated maps paired by target SD url"""
    g = _by_target_url(comparator)
    r = _by_target_url(regen)
    identical, drifted = [], []
    for url in sorted(set(g) | set(r)):
        if url not in g or url not in r:
            drifted.append(f"{url}: present in only one run")
            continue
        ga = metrics.rule_level_prf1(r[url], g[url])
        if ga["f1"] == 1.0:
            identical.append(url)
        else:
            drifted.append(f"{url}: F1={ga['f1']}")
    return {
        "maps_compared": len(set(g) | set(r)),
        "identical": len(identical),
        "drifted": drifted,
        "rule_identical": not drifted,
    }


def _candidate_vs_comparator(comparator: List[dict], regen: List[dict]):
    g = _by_target_url(comparator)
    r = _by_target_url(regen)
    prf, dist = {}, {}
    for url in sorted(set(g) & set(r)):
        prf[url] = metrics.rule_level_prf1(r[url], g[url])
        dist[url] = metrics.edit_distance(r[url], g[url])
    return prf, dist


def _summarize(report: Dict[str, Any]) -> str:
    m = report["metrics"]
    lines = [f"\n=== {report['project']} ==="]
    if "M1_data_preservation" in m:
        dp = m["M1_data_preservation"]
        lines.append(
            f"  M1 data preservation: {dp['fields_preserved']}/{dp['fields_graded']} "
            f"({dp['preservation_pct']}%), transform-mapped={dp['fields_transform_mapped']}"
        )
    if "M5_remaining_todo" in m:
        total = sum(v["todo_count"] for v in m["M5_remaining_todo"].values())
        lines.append(f"  M5 remaining TODO/placeholders: {total}")
    if "M8_extraction_completeness" in m:
        for pid, v in m["M8_extraction_completeness"].items():
            lines.append(f"  M8 extraction[{pid}]: {v['completeness_pct']}% ({v['represented']}/{v['profile_elements']})")
    if "M9_graceful_degradation" in m:
        drops = sum(v["dropped_element_count"] for v in m["M9_graceful_degradation"].values())
        lines.append(f"  M9 silent-drop element count (target 0): {drops}")
    if "M0c_reproducibility" in m:
        rc = m["M0c_reproducibility"]
        lines.append(f"  M0c reproducibility: rule_identical={rc['rule_identical']} ({rc['identical']}/{rc['maps_compared']})")
    if "regeneration_error" in m:
        lines.append(f"  ! regeneration error: {m['regeneration_error'][:200]}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    regenerate = "--regenerate" in argv
    projects = [a for a in argv if not a.startswith("--")] or DEFAULT_PROJECTS
    for name in projects:
        report = evaluate_project(name, regenerate=regenerate)
        print(_summarize(report))
    print(f"\nReports written to {_REPORTS}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
