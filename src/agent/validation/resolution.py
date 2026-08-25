"""resolving target paths against the profile tree"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Tuple

from agent.validation.models import Producer, Stage, ValidationFinding
from agent.validation.paths import AddressedPath, _strip_root, normalize_target_path


logger = logging.getLogger(__name__)

RESOLUTION_SLICE_VARIANTS = "slice-variants"
RESOLUTION_CHOICE_TYPE_NOT_ALLOWED = "choice-type-not-allowed"
RESOLUTION_CHOICE_UNNARROWED = "choice-unnarrowed"

def _level_is_described(target_tree, path: str) -> bool:
    """check if profile tree actually expands the parent of path"""

    parent_path = path.rsplit(".", 1)[0] if "." in path else path
    parent = _resolve_target_path(target_tree, parent_path).node
    if parent is None:
        return False
    return any(":" not in key for key in parent.children)


def _slice_base(element_id: str) -> str:
    """an ElementDefinition id with every slice qualifier remove"""
    return ".".join(segment.split(":", 1)[0] for segment in element_id.split("."))


@dataclass(frozen=True)
class TargetResolution:
    """what the profile tree says about one map-authored target path"""
    path: str
    node: Optional[Any]
    status: str
    failed_segment: Optional[str] = None
    allowed_variants: Tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        if self.status == RESOLUTION_CHOICE_UNNARROWED:
            return True
        return self.node is not None and self.status != RESOLUTION_CHOICE_TYPE_NOT_ALLOWED


def _resolve_step(target_tree, path: str) -> Tuple[Optional[Any], str]:
    """resolve candidate path, treat slices of one element as resolved"""

    from mapping.target_tree import (  # noqa: PLC0415
        RESOLUTION_AMBIGUOUS_PATH,
        RESOLVED_STATUSES,
    )

    node, status = target_tree.resolve(path)
    if status in RESOLVED_STATUSES:
        return node, status
    if status == RESOLUTION_AMBIGUOUS_PATH:
        candidates = target_tree.matches(path)
        if candidates and len({_slice_base(item.eid) for item in candidates}) == 1:
            if len({tuple(sorted(item.types)) for item in candidates}) > 1:
                return candidates[0], RESOLUTION_CHOICE_UNNARROWED
            return candidates[0], RESOLUTION_SLICE_VARIANTS
    return None, status


def _variant_children(parent, segment: str) -> List[Any]:
    """children of parent that are concrete variants of the choice segment"""

    if parent is None:
        return []
    found = []
    for key, child in parent.children.items():
        name = key.split(":", 1)[0]
        if len(name) > len(segment) and name.startswith(segment) and name[len(segment)].isupper():
            found.append(child)
    return found


def _resolve_choice_segment(
    target_tree, prefix: str, parent, segment: str
) -> Optional[TargetResolution]:
    """reconcile one segment that names a choice element, or return None"""

    from mapping.fml_creator.fml_helper import fhir_type_suffix  # noqa: PLC0415

    base_path = f"{prefix}.{segment}[x]"
    node, status = _resolve_step(target_tree, base_path)
    if node is not None:
        if len([code for code in node.types if code]) > 1:
            return TargetResolution(base_path, node, RESOLUTION_CHOICE_UNNARROWED)
        return TargetResolution(base_path, node, status)

    variants = _variant_children(parent, segment)
    if len(variants) == 1:
        return TargetResolution(variants[0].path, variants[0], RESOLUTION_SLICE_VARIANTS)
    if variants:
        return TargetResolution(base_path, None, RESOLUTION_CHOICE_UNNARROWED)

    for index in range(1, len(segment)):
        if not segment[index].isupper():
            continue
        base, suffix = segment[:index], segment[index:]
        choice_path = f"{prefix}.{base}[x]"
        node, status = _resolve_step(target_tree, choice_path)
        if node is None:
            continue
        allowed = tuple(
            sorted({fhir_type_suffix(code) for code in node.types if code})
        )
        if suffix not in allowed:
            return TargetResolution(
                choice_path,
                node,
                RESOLUTION_CHOICE_TYPE_NOT_ALLOWED,
                failed_segment=segment,
                allowed_variants=tuple(f"{base}{item}" for item in allowed),
            )
        sliced = f"{choice_path}:{base}{suffix}"
        precise, precise_status = _resolve_step(target_tree, sliced)
        if precise is not None:
            return TargetResolution(sliced, precise, precise_status)
        return TargetResolution(choice_path, node, status)
    return None


def _resolve_target_path(target_tree, path: str) -> TargetResolution:
    """resolve a map-authored target path against a profile tree"""

    from mapping.target_tree import RESOLUTION_NOT_FOUND  # noqa: PLC0415

    segments = str(path).split(".")
    prefix = segments[0]
    node, status = _resolve_step(target_tree, prefix)
    if node is None:
        return TargetResolution(prefix, None, status, failed_segment=prefix)

    through_slice = False

    for segment in segments[1:]:
        candidate = f"{prefix}.{segment}"
        step, step_status = _resolve_step(target_tree, candidate)
        if step is not None:
            if step_status == RESOLUTION_CHOICE_UNNARROWED:
                return TargetResolution(candidate, step, step_status)
            prefix, node, status = candidate, step, step_status
            through_slice = through_slice or step_status == RESOLUTION_SLICE_VARIANTS
            continue

        unattributable = through_slice or _is_sliced(node)
        choice = _resolve_choice_segment(target_tree, prefix, node, segment)
        if choice is None:
            if unattributable:
                return TargetResolution(prefix, node, RESOLUTION_CHOICE_UNNARROWED)
            return TargetResolution(
                candidate, None, step_status or RESOLUTION_NOT_FOUND,
                failed_segment=segment,
            )
        if not choice.resolved and unattributable:
            return TargetResolution(prefix, node, RESOLUTION_CHOICE_UNNARROWED)
        if choice.status == RESOLUTION_CHOICE_UNNARROWED or not choice.resolved:
            return choice
        prefix, node, status = choice.path, choice.node, choice.status
        through_slice = through_slice or choice.status == RESOLUTION_SLICE_VARIANTS

    return TargetResolution(prefix, node, status)


def _is_sliced(node) -> bool:
    """check if the profile slices this element."""

    return node is not None and any(":" in key for key in node.children)


def _path_findings(
    targets: Sequence[AddressedPath],
    sources: Sequence[AddressedPath],
    *,
    target_tree,
    source_fields: Optional[Set[str]],
    map_url,
    map_id,
    profile_url,
) -> List[ValidationFinding]:
    """check if every addressed element exists where the map says it does"""

    from mapping.target_tree import RESOLUTION_AMBIGUOUS_PATH  # noqa: PLC0415

    findings: List[ValidationFinding] = []
    if target_tree is not None:
        seen: Set[str] = set()
        for entry in targets:
            key = normalize_target_path(entry.path)
            if key in seen:
                continue
            seen.add(key)
            resolution = _resolve_target_path(target_tree, entry.path)
            evidence = {"rule": entry.rule_name, "resolution": resolution.status}
            if resolution.resolved:
                if resolution.node is not None and resolution.node.prohibited:
                    findings.append(
                        ValidationFinding.build(
                            Producer.PATH_RESOLUTION,
                            Stage.PATHS,
                            "target-path-prohibited",
                            f"The map writes {entry.path}, which the profile "
                            "forbids (max = 0).",
                            map_url=map_url,
                            map_id=map_id,
                            profile_url=profile_url,
                            path=entry.path,
                            pointer=entry.pointer,
                            evidence={"rule": entry.rule_name},
                        )
                    )
                continue
            if target_tree.is_prohibited(entry.path):
                code, text = (
                    "target-path-prohibited",
                    f"The map writes {entry.path}, which the profile forbids "
                    "(max = 0).",
                )
            elif resolution.status == RESOLUTION_CHOICE_TYPE_NOT_ALLOWED:
                code, text = (
                    "target-choice-type-not-allowed",
                    f"The map writes {entry.path}, but the profile narrows that "
                    f"choice to {', '.join(resolution.allowed_variants) or 'no type'}.",
                )
                evidence["allowed_variants"] = list(resolution.allowed_variants)
            elif not _level_is_described(target_tree, entry.path):
                logger.debug(
                    "Skipping %s: the profile does not describe that level.",
                    entry.path,
                )
                continue
            elif resolution.status == RESOLUTION_AMBIGUOUS_PATH:
                code, text = (
                    "target-path-ambiguous",
                    f"{entry.path} matches several unrelated profile elements — "
                    "the map must address one by its ElementDefinition id.",
                )
            else:
                code, text = (
                    "target-path-not-found",
                    f"The profile has no element {entry.path}.",
                )
            findings.append(
                ValidationFinding.build(
                    Producer.PATH_RESOLUTION,
                    Stage.PATHS,
                    code,
                    text,
                    map_url=map_url,
                    map_id=map_id,
                    profile_url=profile_url,
                    path=entry.path,
                    pointer=entry.pointer,
                    evidence=evidence,
                )
            )

    if source_fields is not None:
        seen_sources: Set[str] = set()
        for entry in sources:
            leaf = _strip_root(entry.path)
            if not leaf or leaf in seen_sources or leaf in source_fields:
                continue
            seen_sources.add(leaf)
            findings.append(
                ValidationFinding.build(
                    Producer.PATH_RESOLUTION,
                    Stage.PATHS,
                    "source-path-not-found",
                    f"The map reads source field {leaf!r}, which the source "
                    "logical model does not declare.",
                    map_url=map_url,
                    map_id=map_id,
                    path=entry.path,
                    pointer=entry.pointer,
                    evidence={"rule": entry.rule_name},
                )
            )
    return findings
