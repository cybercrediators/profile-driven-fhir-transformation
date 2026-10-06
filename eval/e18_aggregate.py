"""E18 aggregation comparison, per case

Reads the e18_<project>_<id>.json reports and checks if maps were unchanged, improved, regressed, or unresolved.

Usage::

    PYTHONPATH=.:src python3 -m eval.e18_aggregate
    PYTHONPATH=.:src python3 -m eval.e18_aggregate --json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPORTS = Path(__file__).resolve().parent / "reports"

DEFAULT_BASELINE_FOR = {
    "deterministic-automap": "table",
    "llm-automap": "deterministic-automap",
    "agent": "table",
}

COMPARED = (
    ("compiler_acceptance", "higher"),
    ("required_path_coverage", "higher"),
    ("matchbox_execution", "higher"),
    ("profile_validation", "higher"),
    ("maps_clean", "higher"),
    ("blocking_findings", "lower"),
)

PER_MAP_COMPARED = (
    ("compiler_accepted", "higher"),
    ("transform_ok", "higher"),
    ("validate_ok", "higher"),
    ("covered_required", "higher"),
    ("satisfied_obligations", "higher"),
    ("required_gaps", "lower"),
    ("blocking_findings", "lower"),
)


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_reports() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """load reports from eval/reports/"""

    found: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in sorted(_REPORTS.glob("e18_*.json")):
        report = _load(path)
        if not isinstance(report, dict) or report.get("experiment") != "E18":
            continue
        project, arm = report.get("project"), report.get("arm")
        if not project or not arm:
            continue
        found[(str(project), str(arm))] = report
    return found


def _result_block(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """check what was produced by the agent state where there is one"""

    return report.get("after") or report.get("baseline")


def _moves(left: Dict[str, Any], right: Dict[str, Any], compared) -> Dict[str, Any]:
    """check which of compared moved between two measurement blocks"""

    moves: List[Dict[str, Any]] = []
    unmeasured: List[str] = []
    for key, direction in compared:
        before, after = left.get(key), right.get(key)
        if before is None or after is None:
            if before != after:
                unmeasured.append(key)
            continue
        if before == after:
            continue
        better = after > before if direction == "higher" else after < before
        moves.append({"metric": key, "before": before, "after": after, "better": better})
    return {
        "improved": [move for move in moves if move["better"]],
        "regressed": [move for move in moves if not move["better"]],
        "unmeasured_on_one_side": unmeasured,
    }


def _verdict(delta: Dict[str, Any], remaining_blocking: Optional[int]) -> str:
    if delta["regressed"]:
        return "regressed"
    if delta["improved"]:
        return "improved"
    if (remaining_blocking or 0) > 0:
        return "unresolved"
    return "unchanged"


def compare_maps(baseline: Dict[str, Any], arm: Dict[str, Any]) -> List[Dict[str, Any]]:
    """compare per map, which is the unit a repair actually acts on"""

    left, right = _result_block(baseline) or {}, _result_block(arm) or {}
    before = {row["map"]: row for row in left.get("per_map") or []}
    after = {row["map"]: row for row in right.get("per_map") or []}

    rows: List[Dict[str, Any]] = []
    for name in sorted(set(before) | set(after)):
        if name not in after:
            rows.append({"map": name, "verdict": "map-missing", "side": "arm"})
            continue
        if name not in before:
            rows.append({"map": name, "verdict": "map-added", "side": "arm"})
            continue
        delta = _moves(before[name], after[name], PER_MAP_COMPARED)
        rows.append(
            {
                "map": name,
                "verdict": _verdict(delta, after[name].get("blocking_findings")),
                **delta,
                "remaining_blocking_findings": after[name].get("blocking_findings"),
                "remaining_failure_classes": after[name].get("failure_classes"),
            }
        )
    return rows


def compare(baseline: Dict[str, Any], arm: Dict[str, Any]) -> Dict[str, Any]:
    """compare per case verdict, the project roll-up and the per-map detail"""

    left, right = _result_block(baseline), _result_block(arm)
    if not left or not right:
        return {"verdict": "incomparable", "reason": "an arm produced no measurements"}

    delta = _moves(left, right, COMPARED)
    maps = compare_maps(baseline, arm)
    per_map_verdicts = collections.Counter(row["verdict"] for row in maps)

    if per_map_verdicts["regressed"] or per_map_verdicts["map-missing"]:
        verdict = "regressed"
    elif per_map_verdicts["improved"]:
        verdict = "improved"
    elif per_map_verdicts["unresolved"]:
        verdict = "unresolved"
    elif maps:
        verdict = "unchanged"
    else:
        verdict = _verdict(delta, right.get("blocking_findings"))

    unsatisfied = (right.get("cross_map") or {}).get("unsatisfied") or []
    if unsatisfied and verdict in ("improved", "unchanged"):
        verdict = "unresolved"

    return {
        "verdict": verdict,
        "cross_map_unsatisfied": unsatisfied,
        "verdict_basis": "per-map" if maps else "project-totals",
        "engine_partial": not (baseline.get("engine") and arm.get("engine")),
        "improved": delta["improved"],
        "regressed": delta["regressed"],
        "unmeasured_on_one_side": delta["unmeasured_on_one_side"],
        "remaining_blocking_findings": right.get("blocking_findings"),
        "remaining_failure_classes": right.get("failure_classes"),
        "map_verdicts": dict(sorted(per_map_verdicts.items())),
        "maps": [row for row in maps if row["verdict"] != "unchanged"],
    }


def _baseline_arm_for(arm: str, report: Dict[str, Any]) -> Optional[str]:
    """check which arm this one is compared against"""

    return report.get("baseline_arm") or DEFAULT_BASELINE_FOR.get(arm)


def build(
    reports: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    include_arms: Optional[set[str]] = None,
) -> Dict[str, Any]:
    arms = sorted(include_arms or {arm for _, arm in reports})
    projects = sorted({project for project, arm in reports if arm in arms})
    rows: List[Dict[str, Any]] = []
    for project in projects:
        for arm in arms:
            report = reports.get((project, arm))
            if report is None:
                continue
            baseline_arm = _baseline_arm_for(arm, report)
            if baseline_arm is None:
                continue  # a root: nothing above it to compare against
            baseline = reports.get((project, baseline_arm))
            if baseline is None:
                rows.append(
                    {
                        "project": project,
                        "arm": arm,
                        "baseline_arm": baseline_arm,
                        "verdict": "no-baseline",
                        "reason": f"no {baseline_arm} report for this project",
                    }
                )
                continue
            row: Dict[str, Any] = {
                "project": project,
                "arm": arm,
                "baseline_arm": baseline_arm,
            }
            row.update(compare(baseline, report))
            agent = report.get("agent")
            if agent:
                row["cost"] = {
                    "attempts": agent["attempts"],
                    "provider_calls": agent["provider_calls"],
                    "cache_hits": agent["cache_hits"],
                    "elapsed_s": agent["elapsed_s"],
                    "applied": agent["applied"],
                    "outcomes": agent["stats"]["outcomes"],
                }
            selection = report.get("mapping_selection")
            if selection:
                row["mapping_selection"] = {
                    "precision": selection["precision"],
                    "recall": selection["recall"],
                }
            rows.append(row)

    tally: Dict[str, Dict[str, int]] = {}
    map_tally: Dict[str, Dict[str, int]] = {}
    for row in rows:
        bucket = tally.setdefault(row["arm"], {})
        bucket[row["verdict"]] = bucket.get(row["verdict"], 0) + 1
        maps = map_tally.setdefault(row["arm"], {})
        for verdict, count in (row.get("map_verdicts") or {}).items():
            maps[verdict] = maps.get(verdict, 0) + count

    return {
        "experiment": "E18",
        "kind": "aggregate",
        "baseline_pairing": {
            arm: baseline
            for arm, baseline in DEFAULT_BASELINE_FOR.items()
            if arm in arms
        },
        "projects": projects,
        "arms": arms,
        # Both are reported: the project tally is what a reader scans, the map
        # tally is what the verdicts were actually derived from.
        "tally": tally,
        "map_tally": map_tally,
        "rows": rows,
    }


def render(payload: Dict[str, Any]) -> str:
    """create the actual report based on the given findings"""
    pairing = ", ".join(
        f"{arm} vs {base}" for arm, base in sorted(payload["baseline_pairing"].items())
    )
    lines = [
        f"E18 comparison ({pairing})",
        f"{len(payload['projects'])} project(s), arms present: "
        f"{', '.join(payload['arms']) or 'none'}",
        "",
    ]
    for arm, counts in sorted(payload["tally"].items()):
        ordered = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        lines.append(f"  {arm:14s} projects: {ordered}")
        maps = payload.get("map_tally", {}).get(arm) or {}
        if maps:
            ordered = ", ".join(f"{name}={count}" for name, count in sorted(maps.items()))
            lines.append(f"  {'':14s} maps:     {ordered}")

    regressed = [row for row in payload["rows"] if row["verdict"] == "regressed"]
    unresolved = [row for row in payload["rows"] if row["verdict"] == "unresolved"]

    lines.append("\nRegressions:" if regressed else "\nRegressions: none")
    for row in regressed:
        for entry in row.get("maps") or []:
            if entry["verdict"] not in ("regressed", "map-missing"):
                continue
            for move in entry.get("regressed") or [{"metric": entry["verdict"]}]:
                detail = (
                    f"{move['metric']}: {move['before']} -> {move['after']}"
                    if "before" in move
                    else move["metric"]
                )
                lines.append(
                    f"  {row['project']:30s} [{row['arm']}] {entry['map'][:44]:44s} {detail}"
                )

    lines.append("\nUnresolved:" if unresolved else "\nUnresolved: none")
    for row in unresolved:
        lines.append(
            f"  {row['project']:34s} [{row['arm']}] "
            f"{row['remaining_blocking_findings']} blocking finding(s), "
            f"{row.get('remaining_failure_classes')}"
        )

    partial = [row for row in payload["rows"] if row.get("engine_partial")]
    if partial:
        lines.append(
            f"\n{len(partial)} comparison(s) ran without engine evidence on at least "
            "one side; compiler acceptance alone cannot tell a map that runs from "
            "one that does not."
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="E18 cross-arm comparison from eval/reports/e18_*.json"
    )
    parser.add_argument("--json", action="store_true", help="print the payload instead")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("deterministic-automap", "llm-automap", "agent"),
        help="only aggregate these comparison arms (their baselines remain available)",
    )
    args = parser.parse_args(argv)

    reports = load_reports()
    if not reports:
        print("no e18_* arm reports in eval/reports/ — run eval.e18_agent_modes first")
        return 1
    payload = build(reports, include_arms=set(args.arms) if args.arms else None)
    path = _REPORTS / "e18_aggregate.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str) if args.json else render(payload))
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
