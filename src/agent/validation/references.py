"""handle external/cross-map references"""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from agent.validation.models import ActionOwner, Producer, Stage, ValidationFinding
from agent.validation.paths import normalize_target_path, target_path_spellings


logger = logging.getLogger(__name__)


# set todo placeholder names for references (will be resolved when using bundle outputs)
DEFERRED_REFERENCE_PREFIX = "TODO-resolve-reference-"
REFERENCE_CONTRACT_PREFIX = "FHIRBRIDGE_REFERENCE:"


def _contract_path(rule: Mapping[str, Any]) -> Optional[str]:
    """The target path a deferred-reference rule declares, from its contract."""

    documentation = str(rule.get("documentation") or "")
    if not documentation.startswith(REFERENCE_CONTRACT_PREFIX):
        return None
    try:
        contract = json.loads(documentation[len(REFERENCE_CONTRACT_PREFIX) :])
    except ValueError:
        return None
    if not isinstance(contract, Mapping):
        return None
    root = str(contract.get("sourceType") or "")
    relative = str(contract.get("path") or "")
    if not relative:
        return None
    return f"{root}.{relative}" if root else relative


def deferred_reference_paths(document: Mapping[str, Any]) -> Set[str]:
    """target paths this map deliberately leaves to the bundle assembler"""

    found: Set[str] = set()

    def walk(rules) -> None:
        for rule in rules or []:
            if not isinstance(rule, Mapping):
                continue
            name = str(rule.get("name") or "")
            if name.startswith(DEFERRED_REFERENCE_PREFIX):
                path = _contract_path(rule)
                if path is None:
                    tail = name[len(DEFERRED_REFERENCE_PREFIX) :]
                    path = tail.replace("-", ".") if tail else None
                if path:
                    found.add(path)
            walk(rule.get("rule"))

    for group in document.get("group") or []:
        if isinstance(group, Mapping):
            walk(group.get("rule"))
    return found


def deferred_reference_contracts(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """structured bundle-assembler contracts embedded in deferred rules"""

    contracts: List[Dict[str, Any]] = []

    def walk(rules) -> None:
        for rule in rules or []:
            if not isinstance(rule, Mapping):
                continue
            name = str(rule.get("name") or "")
            documentation = str(rule.get("documentation") or "")
            if name.startswith(DEFERRED_REFERENCE_PREFIX) and documentation.startswith(
                REFERENCE_CONTRACT_PREFIX
            ):
                try:
                    raw = json.loads(documentation[len(REFERENCE_CONTRACT_PREFIX) :])
                except ValueError:
                    raw = None
                if isinstance(raw, Mapping):
                    contract = dict(raw)
                    path = _contract_path(rule)
                    if path:
                        contract["fullPath"] = path
                    contracts.append(contract)
            walk(rule.get("rule"))

    for group in document.get("group") or []:
        if isinstance(group, Mapping):
            walk(group.get("rule"))
    return contracts


def _element_target_profiles(profile_sd: Mapping[str, Any], path: str) -> List[str]:
    """targetProfile canonicals declared for *path* in a profile snapshot"""

    elements = (profile_sd.get("snapshot") or {}).get("element") or []
    wanted = target_path_spellings(path)
    canonicals: List[str] = []
    for element in elements:
        if not isinstance(element, Mapping):
            continue
        if not (target_path_spellings(str(element.get("path") or "")) & wanted):
            continue
        for entry in element.get("type") or []:
            if isinstance(entry, Mapping):
                canonicals.extend(str(item) for item in entry.get("targetProfile") or [])
    return sorted(set(canonicals))


def external_reference_paths(
    external_defaults: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """{normalized target path: declaration} for configured external refs"""

    resolved: Dict[str, Dict[str, Any]] = {}
    for entry in external_defaults or ():
        if not isinstance(entry, Mapping):
            continue
        raw_path = str(entry.get("path") or "").strip().strip(".")
        source_type = str(entry.get("source_type") or "").strip()
        if not raw_path:
            continue

        path = raw_path
        if "." in path:
            prefix, relative = path.split(".", 1)
            if source_type and prefix == source_type:
                path = relative
            elif source_type and prefix[:1].isupper():
                continue
            elif not source_type and prefix[:1].isupper():
                source_type, path = prefix, relative
        if not source_type:
            continue
        if path.endswith("[x]"):
            path = path[: -len("[x]")] + "Reference"
        resolved[normalize_target_path(f"{source_type}.{path}")] = dict(entry)
    return resolved


def _produced_profile_canonicals(profile: Optional[Mapping[str, Any]]) -> Set[str]:
    """canonicals a produced profile can satisfy as a Reference target"""

    if not profile:
        return set()
    produced = {
        str(profile.get("url") or "").split("|", 1)[0],
        str(profile.get("baseDefinition") or "").split("|", 1)[0],
    }
    resource_type = str(profile.get("type") or "")
    if resource_type:
        produced.add(f"http://hl7.org/fhir/StructureDefinition/{resource_type}")
    if str(profile.get("kind") or "") == "resource" or resource_type[:1].isupper():
        produced.add("http://hl7.org/fhir/StructureDefinition/Resource")
    produced.discard("")
    return produced


def recheck_cross_map_references(
    assembled: Sequence[Tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]],
    *,
    external_defaults: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[ValidationFinding]:
    """cross-map check over an assembled set of maps

    :param assembled: (structure map, its target profile or None) pairs.
    :param external_defaults: the projects external_reference_defaults
    :return: one finding per deferred reference nothing in the set satisfies.
    """

    supplied = external_reference_paths(external_defaults)
    produced: Set[str] = set()
    for document, profile in assembled:
        for entry in document.get("structure") or []:
            if isinstance(entry, Mapping) and entry.get("mode") == "target":
                produced.add(str(entry.get("url") or "").split("|", 1)[0])
        produced.update(_produced_profile_canonicals(profile))
    produced.discard("")

    findings: List[ValidationFinding] = []
    for document, profile in assembled:
        if profile is None:
            continue
        for path in sorted(deferred_reference_paths(document)):
            expected = _element_target_profiles(profile, path)
            if not expected:
                continue
            if any(canonical in produced for canonical in expected):
                continue
            declared = supplied.get(normalize_target_path(path))
            if declared is not None:
                logger.debug(
                    "%s is supplied by external_reference_defaults (%s).",
                    path,
                    declared.get("reference"),
                )
                continue
            findings.append(
                ValidationFinding.build(
                    Producer.COVERAGE,
                    Stage.COVERAGE,
                    "cross-map-reference-unsatisfied",
                    f"{path} is left to the bundle assembler, but no map in this "
                    "set produces "
                    + " or ".join(canonical.rsplit("/", 1)[-1] for canonical in expected)
                    + ".",
                    owner=ActionOwner.MAPPING_INPUT_REQUIRED,
                    map_url=document.get("url"),
                    map_id=document.get("id"),
                    profile_url=str(profile.get("url") or "") or None,
                    path=path,
                    evidence={
                        "expected_target_profiles": expected,
                        "produced_by_set": sorted(produced),
                    },
                )
            )
    return findings
