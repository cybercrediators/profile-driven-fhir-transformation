"""define and calculate coverage metrics/values"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from agent.validation.models import ActionOwner, Producer, Stage, ValidationFinding
from agent.validation.paths import _strip_root, normalize_target_path, target_path_spellings


def declared_target_structure(document: Mapping[str, Any]) -> Optional[str]:
    """The canonical of the target profile a map declares, version stripped."""

    for entry in document.get("structure") or []:
        if isinstance(entry, Mapping) and entry.get("mode") == "target":
            url = str(entry.get("url") or "").split("|", 1)[0]
            if url:
                return url
    return None


def declared_target_paths(
    mapping_table: Optional[Mapping[str, Any]],
    *,
    res_type: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> Dict[str, str]:
    """calculate target path: source field pairs for entries for the corresponding (one) map"""

    from mapping.rule_ir import mapping_target_path  # noqa: PLC0415

    roots = {root for root in (res_type, profile_id) if root}
    if not roots:
        return {}

    declared: Dict[str, str] = {}
    for source_field, value in (mapping_table or {}).items():
        target = mapping_target_path(value)
        if not target:
            continue
        path = normalize_target_path(str(target))
        root, _, remainder = path.partition(".")
        if roots:
            if root not in roots or not remainder:
                continue
            path = f"{res_type}.{remainder}" if res_type else path
        declared[path] = str(source_field)
    return declared


def prohibited_target_path(target_tree, path: str) -> Optional[str]:
    """check for prohibited element at/above path if profile has one"""

    if target_tree is None or not path:
        return None
    wanted = target_path_spellings(path)
    prohibited = sorted(
        (str(item) for item in getattr(target_tree, "_pruned_ids", set())),
        key=len,
    )
    for item in prohibited:
        for normalized in target_path_spellings(item):
            if any(
                spelling == normalized or spelling.startswith(normalized + ".")
                for spelling in wanted
            ):
                return item
    return None


def authored_selector_obligations(
    document: Mapping[str, Any],
) -> Set[Tuple[str, str]]:
    """pairing (target selector, source field) by map"""

    found: Set[Tuple[str, str]] = set()

    def link_ids(rules: Any, variable: str) -> Set[str]:
        values: Set[str] = set()
        for rule in rules or []:
            if not isinstance(rule, Mapping):
                continue
            for target in rule.get("target") or []:
                if not isinstance(target, Mapping):
                    continue
                if target.get("context") != variable or target.get("element") != "linkId":
                    continue
                for parameter in target.get("parameter") or []:
                    if isinstance(parameter, Mapping) and parameter.get("valueString"):
                        values.add(str(parameter["valueString"]))
        return values

    def descendant_sources(
        rules: Any, source_vars: Mapping[str, str]
    ) -> Set[str]:
        """resolved element-bearing sources anywhere below an item rule"""

        paths: Set[str] = set()
        for rule in rules or []:
            if not isinstance(rule, Mapping):
                continue
            local = dict(source_vars)
            for source in rule.get("source") or []:
                if not isinstance(source, Mapping):
                    continue
                parent = local.get(str(source.get("context") or ""))
                element = str(source.get("element") or "")
                path = f"{parent}.{element}" if parent and element else parent
                if path and element:
                    paths.add(path)
                if path and source.get("variable"):
                    local[str(source["variable"])] = path
            paths.update(descendant_sources(rule.get("rule"), local))
        return paths

    def walk(
        rules: Any,
        source_vars: Dict[str, str],
        target_vars: Dict[str, str],
    ) -> None:
        for rule in rules or []:
            if not isinstance(rule, Mapping):
                continue
            local_sources = dict(source_vars)
            local_targets = dict(target_vars)
            source_paths: Set[str] = set()
            item_targets: List[Tuple[str, str]] = []
            for source in rule.get("source") or []:
                if not isinstance(source, Mapping):
                    continue
                parent = local_sources.get(str(source.get("context") or ""))
                element = str(source.get("element") or "")
                path = f"{parent}.{element}" if parent and element else parent
                if path and element:
                    source_paths.add(path)
                if path and source.get("variable"):
                    local_sources[str(source["variable"])] = path
            for target in rule.get("target") or []:
                if not isinstance(target, Mapping):
                    continue
                parent = local_targets.get(str(target.get("context") or ""))
                element = str(target.get("element") or "")
                path = f"{parent}.{element}" if parent and element else parent
                variable = str(target.get("variable") or "")
                if path and variable:
                    local_targets[variable] = path
                if path and variable and element == "item":
                    item_targets.append((path, variable))

            for path, variable in item_targets:
                root = path.split(".", 1)[0]
                for link_id in link_ids(rule.get("rule"), variable):
                    selector = f"{root}.item[{link_id}]"
                    associated_sources = source_paths | descendant_sources(
                        rule.get("rule"), local_sources
                    )
                    for source_path in associated_sources:
                        found.add((selector, source_path))
                        found.add((selector, _strip_root(source_path)))
            walk(rule.get("rule"), local_sources, local_targets)

    for group in document.get("group") or []:
        if not isinstance(group, Mapping):
            continue
        sources: Dict[str, str] = {}
        targets: Dict[str, str] = {}
        for entry in group.get("input") or []:
            if not isinstance(entry, Mapping) or not entry.get("name"):
                continue
            if entry.get("mode") == "source":
                sources[str(entry["name"])] = str(entry.get("type") or "")
            elif entry.get("mode") == "target":
                targets[str(entry["name"])] = str(entry.get("type") or "")
        walk(group.get("rule"), sources, targets)
    return found


def required_gap_owner(entry: Mapping[str, Any], mapped_targets: Set[str]) -> ActionOwner:
    """check for required (provider) gap owner in single maps"""

    provider = str(entry.get("provider") or "")
    if provider and provider not in {"none", "unknown", ""}:
        return ActionOwner.MAP_FIXABLE
    path = normalize_target_path(str(entry.get("path") or ""))
    if path and any(
        target == path or target.startswith(path + ".") or path.startswith(target + ".")
        for target in mapped_targets
    ):
        return ActionOwner.MAP_FIXABLE
    return ActionOwner.MAPPING_INPUT_REQUIRED


def _coverage_findings(
    emitted: Set[str],
    *,
    target_tree,
    mapping_table: Optional[Mapping[str, Any]],
    map_url,
    map_id,
    profile_url,
    profile_id: Optional[str] = None,
    deferred: Optional[Set[str]] = None,
    emitted_partial: Optional[Set[str]] = None,
    selector_obligations: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[List[ValidationFinding], List[str], List[str]]:
    """check required-element coverage for single maps

    :return: (findings, covered_required_paths, satisfied_obligations)
    """

    findings: List[ValidationFinding] = []

    declared_targets = declared_target_paths(
        mapping_table,
        res_type=getattr(target_tree, "res_type", None) if target_tree else None,
        profile_id=profile_id,
    )

    def is_emitted(path: str) -> bool:
        if any(
            emitted_path == path or emitted_path.startswith(path + ".")
            for emitted_path in emitted
        ):
            return True
        if any(
            emitted_path.startswith(path)
            and len(emitted_path) > len(path)
            and emitted_path[len(path)].isupper()
            and "." not in emitted_path[len(path) :]
            for emitted_path in emitted
        ):
            return True
        return any(
            path.startswith(prefix + ".") for prefix in (emitted_partial or set())
        )

    covered_required: List[str] = []
    if target_tree is not None:
        for entry in target_tree.required_manifest():
            if not entry.get("active"):
                continue
            path = normalize_target_path(str(entry.get("path") or ""))
            if not path:
                continue
            if is_emitted(path):
                covered_required.append(path)
                continue
            if path in (deferred or set()):
                findings.append(
                    ValidationFinding.build(
                        Producer.COVERAGE,
                        Stage.COVERAGE,
                        "required-path-deferred",
                        f"Required element {path} is not written by this map; "
                        "the bundle assembler resolves it.",
                        map_url=map_url,
                        map_id=map_id,
                        profile_url=profile_url,
                        path=path,
                        evidence={
                            "min": entry.get("min"),
                            "element_id": entry.get("id"),
                            "resolved_by": "bundle-assembler",
                        },
                    )
                )
                continue
            owner = required_gap_owner(entry, set(declared_targets))
            findings.append(
                ValidationFinding.build(
                    Producer.COVERAGE,
                    Stage.COVERAGE,
                    "required-path-unmapped",
                    f"Required element {path} is not emitted by the map.",
                    owner=owner,
                    map_url=map_url,
                    map_id=map_id,
                    profile_url=profile_url,
                    path=path,
                    evidence={
                        "min": entry.get("min"),
                        "provider": entry.get("provider"),
                        "element_id": entry.get("id"),
                    },
                )
            )

    satisfied: List[str] = []
    for target, source_field in sorted(declared_targets.items()):
        selector_satisfied = any(
            authored_target == target
            and (
                authored_source == source_field
                or authored_source.split(".", 1)[-1]
                == source_field.split(".", 1)[-1]
            )
            for authored_target, authored_source in (selector_obligations or set())
        )
        if is_emitted(target) or selector_satisfied:
            satisfied.append(target)
            continue
        prohibited = prohibited_target_path(target_tree, target)
        if prohibited:
            findings.append(
                ValidationFinding.build(
                    Producer.COVERAGE,
                    Stage.COVERAGE,
                    "mapping-target-prohibited",
                    f"The mapping table routes {source_field!r} to {target}, but "
                    f"the target profile prohibits {prohibited} (max = 0).",
                    owner=ActionOwner.MAPPING_INPUT_REQUIRED,
                    map_url=map_url,
                    map_id=map_id,
                    profile_url=profile_url,
                    path=target,
                    evidence={
                        "source_field": source_field,
                        "prohibited_ancestor": prohibited,
                    },
                )
            )
            continue
        findings.append(
            ValidationFinding.build(
                Producer.COVERAGE,
                Stage.COVERAGE,
                "mapping-obligation-dropped",
                f"The mapping table routes {source_field!r} to {target}, but "
                "the map does not emit it.",
                map_url=map_url,
                map_id=map_id,
                profile_url=profile_url,
                path=target,
                evidence={"source_field": source_field},
            )
        )

    return findings, sorted(set(covered_required)), satisfied
