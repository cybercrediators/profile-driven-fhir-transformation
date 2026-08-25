"""handles the (limited) evidence one repair attempt is allowed to see"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

from agent.validation import (
    ActionOwner,
    ValidationFinding,
    addressed_paths,
    normalize_target_path,
)

logger = logging.getLogger(__name__)

DEFAULT_MAP_BUDGET_CHARS = 24000
DEFAULT_LIST_LIMIT = 40
MAX_FOCUS_PER_FINDING = 8


class RulePointer(BaseModel):
    """One rule, named and addressable."""

    model_config = ConfigDict(extra="forbid")

    pointer: str
    label: str
    name: str
    focused: bool = False
    source_variables: List[str] = Field(default_factory=list)
    target_variables: List[str] = Field(default_factory=list)
    members: List["PointerValue"] = Field(default_factory=list)


class PointerValue(BaseModel):
    """One addressable member beneath a focused rule, and what may be done to it"""

    model_config = ConfigDict(extra="forbid")

    pointer: str
    value: Any = None
    ops: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    json_type: Optional[str] = None


class InsertionPoint(BaseModel):
    """One deterministically authorized location for a new rule."""

    model_config = ConfigDict(extra="forbid")

    pointer: str
    label: str
    target_paths: List[str] = Field(default_factory=list)


class ContextListStatus(BaseModel):
    """Whether a bounded prompt list represents its complete source collection."""

    model_config = ConfigDict(extra="forbid")

    total: int
    shown: int
    truncated: bool


class FindingBrief(BaseModel):
    """A finding as the prompt states it. Never the raw internal record."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    code: str
    message: str
    producer: str
    stage: str
    path: Optional[str] = None
    pointer: Optional[str] = None
    rule_hint: Optional[str] = None
    fixture: Optional[str] = None


class RejectedAttempt(BaseModel):
    """What went wrong last time, in the terms the next attempt must avoid."""

    model_config = ConfigDict(extra="forbid")

    attempt: int
    rejections: List[Dict[str, Any]] = Field(default_factory=list)
    failed_invariants: List[Dict[str, Any]] = Field(default_factory=list)
    introduced: List[FindingBrief] = Field(default_factory=list)
    operations: List[Dict[str, Any]] = Field(default_factory=list)


class TargetElementBrief(BaseModel):
    """A profile element the map may write, with the constraints on it."""

    model_config = ConfigDict(extra="forbid")

    path: str
    types: List[str] = Field(default_factory=list)
    cardinality: str = ""
    required: bool = False
    fixed_value: Optional[Any] = None
    binding: Optional[str] = None


class AgentContext(BaseModel):
    """Everything one repair attempt is shown."""

    model_config = ConfigDict(extra="forbid")

    map_url: Optional[str] = None
    map_id: Optional[str] = None
    map_sha256: Optional[str] = None
    resource_type: Optional[str] = None
    source_type: Optional[str] = None
    profile_url: Optional[str] = None
    map_excerpt: Dict[str, Any] = Field(default_factory=dict)
    excerpt_pruned: bool = False
    pointers: List[RulePointer] = Field(default_factory=list)
    
    insertion_points: List[InsertionPoint] = Field(default_factory=list)
    findings: List[FindingBrief] = Field(default_factory=list)
    
    out_of_scope: List[Dict[str, str]] = Field(default_factory=list)
    source_fields: List[Dict[str, Any]] = Field(default_factory=list)
    mapping_rows: List[Dict[str, Any]] = Field(default_factory=list)
    
    concept_map_urls: List[str] = Field(default_factory=list)
    required_elements: List[TargetElementBrief] = Field(default_factory=list)
    prohibited_paths: List[str] = Field(default_factory=list)
    
    repair_hints: List[Dict[str, Any]] = Field(default_factory=list)
    list_status: Dict[str, ContextListStatus] = Field(default_factory=dict)
    attempts: List[RejectedAttempt] = Field(default_factory=list)

    def cache_context(self) -> Dict[str, Any]:
        """Identity of the evidence, digested into the LLM cache key"""

        return {
            "map": self.map_sha256,
            "findings": sorted(finding.finding_id for finding in self.findings),
            "attempts": [attempt.attempt for attempt in self.attempts],
            "pruned": self.excerpt_pruned,
        }


def _group_types(document: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """(source_type, resource_type) from the first group's declared inputs."""

    for group in document.get("group") or []:
        if not isinstance(group, Mapping):
            continue
        source_type = target_type = None
        for entry in group.get("input") or []:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("mode") == "source":
                source_type = entry.get("type")
            elif entry.get("mode") == "target":
                target_type = entry.get("type")
        if source_type or target_type:
            return (
                str(source_type) if source_type else None,
                str(target_type) if target_type else None,
            )
    return None, None


def finding_path(finding: ValidationFinding) -> str:
    """The target path a finding is about, however the producer expressed it"""

    return str(finding.path or finding.evidence.get("path_hint") or "")


_URL_PREDICATE = re.compile(r"\[url=['\"]([^'\"]+)['\"]\]")


def _url_predicate(path: str) -> Optional[str]:
    """The canonical an extension path pins itself to, if it carries one."""

    found = _URL_PREDICATE.search(path)
    return found.group(1) if found else None


def strip_predicates(path: str) -> str:
    """path without its [...] predicates."""

    return _URL_PREDICATE.sub("", path)


def _extension_url_owners(document: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Extension canonical -> the rule pointers that build that extension"""

    owners: Dict[str, List[str]] = {}

    def walk(rules, prefix: str) -> None:
        for index, rule in enumerate(rules or []):
            if not isinstance(rule, Mapping):
                continue
            pointer = f"{prefix}/{index}"
            for target in rule.get("target") or []:
                if not isinstance(target, Mapping) or target.get("element") != "url":
                    continue
                for parameter in target.get("parameter") or []:
                    value = (
                        parameter.get("valueString")
                        if isinstance(parameter, Mapping)
                        else None
                    )
                    if not value:
                        continue
                    seen = owners.setdefault(str(value), [])
                    seen.append(pointer)
                    container = _rule_prefix(pointer)
                    parent = container.rsplit("/rule/", 1)[0]
                    if parent and parent != container and "/rule/" in container:
                        seen.append(parent)
            walk(rule.get("rule"), f"{pointer}/rule")

    for group_index, group in enumerate(document.get("group") or []):
        if isinstance(group, Mapping):
            walk(group.get("rule"), f"/group/{group_index}/rule")
    return {url: sorted(dict.fromkeys(pointers)) for url, pointers in owners.items()}


def _focus_pointers(
    document: Mapping[str, Any],
    findings: Sequence[ValidationFinding],
) -> Set[str]:
    """Rule pointers implicated by findings"""

    _sources, targets = addressed_paths(document)
    by_path: Dict[str, List[str]] = {}
    for entry in targets:
        by_path.setdefault(entry.path, []).append(entry.pointer)

    names_wanted = {
        str(finding.evidence.get("rule_hint"))
        for finding in findings
        if finding.evidence.get("rule_hint")
    }
    by_name: Dict[str, List[str]] = {}
    for rule in describe_rules(document):
        by_name.setdefault(rule.name, []).append(rule.pointer)

    owners = _subtree_owners(document)
    extension_owners = _extension_url_owners(document)

    focus: Set[str] = set()
    for finding in findings:
        if finding.pointer:
            focus.add(_rule_prefix(finding.pointer))
        path = finding_path(finding)
        if not path:
            continue
        matched = (
            list(by_path.get(path, []))
            or _choice_matches(path, by_path)
            or _choice_base_matches(path, by_path)
        )
        canonical = _url_predicate(path)
        if not matched and canonical:
            matched = list(extension_owners.get(canonical, []))
            if not matched:
                bare = strip_predicates(path)
                matched = (
                    list(by_path.get(bare, []))
                    or _choice_matches(bare, by_path)
                    or _choice_base_matches(bare, by_path)
                )
        matched.extend(owners.get(path, []))
        matched.extend(
            pointer
            for variant, pointers in owners.items()
            if _is_choice_of(path, variant)
            for pointer in pointers
        )

        resolved: List[str] = []
        for pointer in matched:
            rule = _rule_prefix(pointer)
            resolved.append(rule)
            resolved.extend(_skeleton_ancestors(document, rule))
        distinct = sorted(dict.fromkeys(resolved))
        if len(distinct) > MAX_FOCUS_PER_FINDING:
            logger.debug(
                "%s is emitted by %d rules; too ambiguous to focus.",
                path,
                len(distinct),
            )
            continue
        focus.update(distinct)
    for name in names_wanted:
        for pointer in by_name.get(name, []):
            focus.add(pointer)
    return focus


def _skeleton_ancestors(document: Mapping[str, Any], pointer: str) -> List[str]:
    """enclosing rules that bind no target of their own"""

    found: List[str] = []
    tokens = pointer.split("/")
    while len(tokens) > 3:
        parent = "/".join(tokens[:-2])
        if not (parent.rsplit("/", 1)[-1].isdigit() and "/rule/" in parent):
            break
        rule = _rule_at_pointer(document, parent)
        if not isinstance(rule, Mapping) or (rule.get("target") or []):
            break
        found.append(parent)
        tokens = parent.split("/")
    return found


def _choice_base_matches(path: str, by_path: Mapping[str, List[str]]) -> List[str]:
    """pointers emitting the *base* of a concrete choice name"""

    leaf = path.rsplit(".", 1)[-1]
    parent = path[: -len(leaf)] if len(leaf) < len(path) else ""
    for index in range(len(leaf) - 1, 0, -1):
        if not leaf[index].isupper():
            continue
        candidate = f"{parent}{leaf[:index]}"
        if candidate in by_path:
            return list(by_path[candidate])
    return []


def _is_choice_of(base: str, candidate: str) -> bool:
    """Whether a candidate has a base and a choice type suffix."""

    if not candidate.startswith(base):
        return False
    suffix = candidate[len(base) :]
    return bool(suffix) and suffix[0].isupper() and "." not in suffix


def _subtree_owners(document: Mapping[str, Any]) -> Dict[str, List[str]]:
    """create a subtree {target path: pointers of the narrowest rules that produce it}"""

    _sources, targets = addressed_paths(document)
    by_pointer: Dict[str, Set[str]] = {}
    for entry in targets:
        rule = _rule_prefix(entry.pointer)
        tokens = rule.split("/")
        for index in range(len(tokens), 0, -1):
            prefix = "/".join(tokens[:index])
            if prefix.rsplit("/", 1)[-1].isdigit() and "/rule/" in prefix + "/":
                by_pointer.setdefault(prefix, set()).add(entry.path)

    owners: Dict[str, List[str]] = {}
    for pointer, paths in by_pointer.items():
        for path in paths:
            owners.setdefault(path, []).append(pointer)

    narrowed: Dict[str, List[str]] = {}
    for path, pointers in owners.items():
        keep = [
            pointer
            for pointer in pointers
            if not any(
                other != pointer and other.startswith(pointer + "/")
                for other in pointers
            )
        ]
        narrowed[path] = sorted(dict.fromkeys(keep))
    return narrowed


def _choice_matches(path: str, by_path: Mapping[str, List[str]]) -> List[str]:
    """Pointers emitting a typed form of a choice element"""

    found: List[str] = []
    for candidate, pointers in by_path.items():
        if not candidate.startswith(path):
            continue
        suffix = candidate[len(path) :]
        if suffix and suffix[0].isupper() and "." not in suffix:
            found.extend(pointers)
    return found




INSERTION_FINDING_CODES = frozenset(
    {
        "unmaterialized-nested-target",
        "mapping-obligation-dropped",
        "required-path-unmapped",
        "output-required-missing",
        "validate:structure",
    }
)


def _rule_at_pointer(
    document: Mapping[str, Any], pointer: str
) -> Optional[Mapping[str, Any]]:
    """retrieve a rule of a pointer from a given document"""
    node: Any = document
    for token in pointer.strip("/").split("/"):
        if not token:
            continue
        if isinstance(node, Mapping):
            node = node.get(token)
        elif isinstance(node, list) and token.isdigit():
            index = int(token)
            node = node[index] if index < len(node) else None
        else:
            return None
        if node is None:
            return None
    return node if isinstance(node, Mapping) else None


def _insertion_pointer(document: Mapping[str, Any], parent: str) -> str:
    """Return the exact RFC 6901 pointer that creates/appends a child rule."""

    node = _rule_at_pointer(document, parent)
    if isinstance(node, Mapping) and isinstance(node.get("rule"), list):
        return f"{parent}/rule/-"
    return f"{parent}/rule"


def _insertion_points(
    document: Mapping[str, Any], findings: Sequence[ValidationFinding]
) -> List[InsertionPoint]:
    """Resolve safe rule insertion locations for missing target obligations"""

    _sources, targets = addressed_paths(document)
    points: Dict[str, Dict[str, Any]] = {}

    for finding in findings:
        wanted = finding_path(finding)
        if finding.code not in _INSERTION_FINDING_CODES or not wanted:
            continue

        ancestors = [
            entry
            for entry in targets
            if wanted.startswith(entry.path.rstrip(".") + ".")
        ]
        ancestor = (
            max(ancestors, key=lambda entry: len(entry.path)) if ancestors else None
        )
        ancestor_target = (
            _rule_at_pointer(document, ancestor.pointer)
            if ancestor is not None
            else None
        )
        if ancestor is not None and ancestor_target and ancestor_target.get("variable"):
            parent = _rule_prefix(ancestor.pointer)
            pointer = _insertion_pointer(document, parent)
            label = f"Append a child rule beneath {ancestor.rule_name or ancestor.path}"
        else:
            root = wanted.split(".", 1)[0]
            pointer = ""
            label = ""
            for group_index, group in enumerate(document.get("group") or []):
                if not isinstance(group, Mapping):
                    continue
                target_types = {
                    str(entry.get("type") or "")
                    for entry in group.get("input") or []
                    if isinstance(entry, Mapping) and entry.get("mode") == "target"
                }
                if root not in target_types:
                    continue
                rules = group.get("rule")
                pointer = (
                    f"/group/{group_index}/rule/-"
                    if isinstance(rules, list)
                    else f"/group/{group_index}/rule"
                )
                label = f"Append a top-level rule to group {group.get('name') or group_index}"
                break
            if not pointer:
                continue

        entry = points.setdefault(pointer, {"label": label, "target_paths": []})
        entry["target_paths"].append(wanted)

    return [
        InsertionPoint(
            pointer=pointer,
            label=str(entry["label"]),
            target_paths=sorted(set(entry["target_paths"])),
        )
        for pointer, entry in points.items()
    ]


def _rule_prefix(pointer: str) -> str:
    """trim a pointer back to the rule it sits inside"""

    tokens = pointer.split("/")
    for index in range(len(tokens) - 1, 0, -1):
        if tokens[index - 1] == "rule" and tokens[index].isdigit():
            return "/".join(tokens[: index + 1])
    return pointer


def variables_in_scope(
    document: Mapping[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    """return a {rule pointer: {"source": [...], "target": [...]}}"""

    scopes: Dict[str, Dict[str, List[str]]] = {}

    def walk(rules: Any, pointer: str, sources: List[str], targets: List[str]) -> None:
        if not isinstance(rules, list):
            return
        for index, rule in enumerate(rules):
            if not isinstance(rule, Mapping):
                continue
            here = f"{pointer}/{index}"
            local_sources = list(sources)
            local_targets = list(targets)
            for entry in rule.get("source") or []:
                if isinstance(entry, Mapping) and entry.get("variable"):
                    local_sources.append(str(entry["variable"]))
            for entry in rule.get("target") or []:
                if isinstance(entry, Mapping) and entry.get("variable"):
                    local_targets.append(str(entry["variable"]))
            scopes[here] = {
                "source": sorted(dict.fromkeys(local_sources)),
                "target": sorted(dict.fromkeys(local_targets)),
            }
            walk(rule.get("rule"), f"{here}/rule", local_sources, local_targets)

    for group_index, group in enumerate(document.get("group") or []):
        if not isinstance(group, Mapping):
            continue
        sources = [
            str(entry["name"])
            for entry in group.get("input") or []
            if isinstance(entry, Mapping)
            and entry.get("name")
            and entry.get("mode") == "source"
        ]
        targets = [
            str(entry["name"])
            for entry in group.get("input") or []
            if isinstance(entry, Mapping)
            and entry.get("name")
            and entry.get("mode") == "target"
        ]
        walk(group.get("rule"), f"/group/{group_index}/rule", sources, targets)
    return scopes


MAX_MEMBERS_PER_RULE = 16
MAX_MEMBER_VALUE_CHARS = 80
MAX_MEMBERS_PER_CONTEXT = 150


def _json_type_name(value: Any) -> str:
    """The JSON type of value, in the words the schema error would use."""

    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return "string"


def _editable_members(rule: Mapping[str, Any], pointer: str) -> List[PointerValue]:
    """Leaf pointers beneath one rule, with their current values"""

    members: List[PointerValue] = []

    def walk(node: Any, path: str) -> None:
        if len(members) >= MAX_MEMBERS_PER_RULE:
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "rule":
                    continue
                walk(value, f"{path}/{key}")
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                if isinstance(value, Mapping):
                    members.append(
                        PointerValue(
                            pointer=f"{path}/{index}",
                            ops=["replace"],
                            note="the whole entry; replace it with an object",
                        )
                    )
                walk(value, f"{path}/{index}")
            if node and isinstance(node[0], Mapping):
                members.append(
                    PointerValue(
                        pointer=f"{path}/-",
                        ops=["add"],
                        note="append a new entry here",
                    )
                )
            return
        if isinstance(node, str) and len(node) > MAX_MEMBER_VALUE_CHARS:
            return
        members.append(
            PointerValue(
                pointer=path,
                value=node,
                ops=["replace", "remove"],
                json_type=_json_type_name(node),
            )
        )

    walk(rule, pointer)

    children = rule.get("rule")
    structural = [
        PointerValue(
            pointer=f"{pointer}/rule/-",
            ops=["add"],
            note="append a child rule here",
        )
        if isinstance(children, list)
        else PointerValue(
            pointer=f"{pointer}/rule",
            ops=["add"],
            note="this rule has no children yet; the value must be an array "
            "of rules",
        )
    ]
    return structural + members[: max(0, MAX_MEMBERS_PER_RULE - len(structural))]


def describe_rules(document: Mapping[str, Any]) -> List[RulePointer]:
    """Every rule in document including pointer, breadcrumb, and scope."""

    from agent.patch import describe_pointers  # noqa: PLC0415

    scopes = variables_in_scope(document)
    return [
        RulePointer(
            pointer=entry["pointer"],
            label=entry["label"],
            name=entry["name"],
            source_variables=scopes.get(entry["pointer"], {}).get("source", []),
            target_variables=scopes.get(entry["pointer"], {}).get("target", []),
        )
        for entry in describe_pointers(dict(document))
    ]


def _prune(document: Mapping[str, Any], focus: Set[str]) -> tuple[Dict[str, Any], bool]:
    """Keep only the map shape and the focused rules"""

    pruned = False

    def keep(pointer: str) -> bool:
        return any(
            target == pointer
            or target.startswith(pointer + "/")
            or pointer.startswith(target + "/")
            for target in focus
        )

    def walk(rules: Any, pointer: str) -> List[Any]:
        nonlocal pruned
        out: List[Any] = []
        for index, rule in enumerate(rules or []):
            if not isinstance(rule, Mapping):
                continue
            here = f"{pointer}/{index}"
            if not keep(here):
                pruned = True
                out.append(
                    {
                        "_elided": True,
                        "_pointer": here,
                        "name": rule.get("name"),
                    }
                )
                continue
            copy = dict(rule)
            if isinstance(rule.get("rule"), list):
                copy["rule"] = walk(rule["rule"], f"{here}/rule")
            out.append(copy)
        return out

    excerpt = {
        key: value for key, value in document.items() if key not in {"group", "text"}
    }
    groups = []
    for group_index, group in enumerate(document.get("group") or []):
        if not isinstance(group, Mapping):
            continue
        copy = dict(group)
        copy["rule"] = walk(group.get("rule"), f"/group/{group_index}/rule")
        groups.append(copy)
    excerpt["group"] = groups
    return excerpt, pruned


def _path_relevant(path: str, wanted: Set[str]) -> bool:
    return any(
        path == item
        or path.startswith(item.rstrip(".") + ".")
        or item.startswith(path.rstrip(".") + ".")
        for item in wanted
    )


_COPYABLE_TARGET_TYPES = frozenset({"string", "code", "id", "markdown", "uri", "url"})


def _choice_types(target_tree, path: str) -> List[str]:
    """Look up a normalized choice path using its real snapshot identity"""

    parent, separator, element = path.rpartition(".")
    choice_path = f"{parent}{separator}{element}[x]"
    return [str(item) for item in (target_tree.effective_types(choice_path) or [])]


def _choice_repair_hints(
    findings: Sequence[ValidationFinding],
    target_tree,
    mapping_table: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Concrete instructions for choice elements under repair."""

    if target_tree is None:
        return []

    from mapping.rule_ir import mapping_target_path  # noqa: PLC0415

    from agent.validation import target_path_spellings  # noqa: PLC0415

    routed: Dict[str, str] = {}
    for source_field, value in (mapping_table or {}).items():
        target = mapping_target_path(value)
        if not target:
            continue
        
        for spelling in target_path_spellings(str(target)):
            routed.setdefault(spelling, str(source_field).rsplit(".", 1)[-1])

    hints: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for finding in findings:
        path = normalize_target_path(finding_path(finding))
        if not path or path in seen:
            continue
        types = _choice_types(target_tree, path)
        
        if not types:
            continue
        seen.add(path)
        base = path.rsplit(".", 1)[-1]
        from mapping.fml_creator.fml_helper import is_primitive_type  # noqa: PLC0415

        options = []
        for name in types:
            if name in _COPYABLE_TARGET_TYPES:
                transform = "copy"
            elif is_primitive_type(name):
                transform = "cast"
            else:
                transform = "create"
            options.append({"element": base, "type": name, "transform": transform})
        hints.append(
            {
                "path": path,
                "source_field": routed.get(path),
                "options": options,
            }
        )
    return hints


def _required_elements(
    target_tree, findings: Sequence[ValidationFinding], limit: int
) -> tuple[List[TargetElementBrief], int]:
    """filter for required elements"""
    if target_tree is None:
        return [], 0
    briefs: List[TargetElementBrief] = []
    for entry in target_tree.required_manifest():
        if not entry.get("active"):
            continue
        node = target_tree.node(str(entry.get("id") or entry.get("path") or ""))
        binding = getattr(node, "binding", None) if node is not None else None
        briefs.append(
            TargetElementBrief(
                path=str(entry.get("path") or ""),
                types=list(getattr(node, "types", []) or [])
                if node is not None
                else [],
                cardinality=f"{entry.get('min', 0)}..{entry.get('max', '1')}",
                required=True,
                fixed_value=getattr(node, "fixed_value", None)
                if node is not None
                else None,
                binding=(
                    str(binding.get("valueSet"))
                    if isinstance(binding, Mapping) and binding.get("valueSet")
                    else None
                ),
            )
        )
    wanted = {finding_path(finding) for finding in findings} - {""}
    briefs.sort(key=lambda item: (not _path_relevant(item.path, wanted), item.path))
    return briefs[:limit], len(briefs)


def _mapping_rows(
    mapping_table: Optional[Mapping[str, Any]],
    findings: Sequence[ValidationFinding],
    limit: int,
) -> tuple[List[Dict[str, Any]], int]:
    """Mapping-table entries touching a path under repair, then the rest"""

    from mapping.rule_ir import mapping_target_path  # noqa: PLC0415

    wanted = {finding_path(finding) for finding in findings} - {""}
    relevant: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    for source_field, value in (mapping_table or {}).items():
        target = mapping_target_path(value)
        row: Dict[str, Any] = {"source": str(source_field), "target": str(target or "")}
        if isinstance(value, Mapping):
            for key in ("transform", "fixed_value", "concept_map_url", "target_system"):
                if key == "fixed_value" and key in value and value[key] is not None:
                    row[key] = value[key]
                elif key != "fixed_value" and value.get(key):
                    row[key] = value[key]
        bucket = (
            relevant
            if any(
                target and (path == target or str(target).startswith(str(path)))
                for path in wanted
            )
            else other
        )
        bucket.append(row)
    ordered = relevant + other
    return ordered[:limit], len(ordered)


def _source_field_briefs(
    source_fields: Iterable[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    limit: int,
) -> tuple[List[Dict[str, Any]], int]:
    """retrieve requested source rows from given source fields"""
    fields = [dict(spec) for spec in source_fields]
    wanted = {str(row.get("source")) for row in mapping_rows if row.get("source")}
    fields.sort(
        key=lambda field: (
            str(field.get("name") or field.get("id") or "") not in wanted,
        )
    )
    return fields[:limit], len(fields)


def _list_status(total: int, shown: int) -> ContextListStatus:
    return ContextListStatus(total=total, shown=shown, truncated=shown < total)


def build_context(
    document: Mapping[str, Any],
    findings: Sequence[ValidationFinding],
    *,
    all_findings: Sequence[ValidationFinding] = (),
    target_tree=None,
    mapping_table: Optional[Mapping[str, Any]] = None,
    source_fields: Iterable[Mapping[str, Any]] = (),
    profile_url: Optional[str] = None,
    attempts: Sequence[RejectedAttempt] = (),
    map_budget_chars: int = DEFAULT_MAP_BUDGET_CHARS,
    list_limit: int = DEFAULT_LIST_LIMIT,
) -> AgentContext:
    """Assemble the evidence for one repair attempt.

    :param document: the current StructureMap.
    :param findings: the map-fixable worklist this attempt may address.
        Anything else is filtered out here rather than trusted to a template.
    :param all_findings: the complete baseline set, so out-of-scope findings can
        be named without being offered.
    :param target_tree: profile tree, for required and prohibited elements.
    :param mapping_table: the project's source-to-target table.
    :param source_fields: field specs from the source logical model.
    :param attempts: what earlier attempts got wrong, most recent last.
    :param map_budget_chars: above this the excerpt is pruned to the focus.
    """

    from agent.patch import canonical_sha256  # noqa: PLC0415

    repairable = [
        finding
        for finding in findings
        if finding.action_owner is ActionOwner.MAP_FIXABLE
    ]
    if len(repairable) != len(findings):
        logger.debug(
            "Dropped %d non-map-fixable finding(s) from the prompt worklist.",
            len(findings) - len(repairable),
        )

    focus = _focus_pointers(document, repairable)
    insertion_points = _insertion_points(document, repairable)
    whole = json.dumps(document, separators=(",", ":"), default=str)
    if len(whole) <= map_budget_chars:
        excerpt, pruned = dict(document), False
    else:
        excerpt, pruned = _prune(document, focus)

    rules = describe_rules(document)
    
    remaining = MAX_MEMBERS_PER_CONTEXT
    for rule in rules:
        rule.focused = rule.pointer in focus
        if rule.focused and remaining > 0:
            node = _rule_at_pointer(document, rule.pointer)
            if isinstance(node, Mapping):
                rule.members = _editable_members(node, rule.pointer)[:remaining]
                remaining -= len(rule.members)
    selected_rules = [rule for rule in rules if rule.focused]
    selected_rules.extend(rule for rule in rules if not rule.focused)

    source_type, resource_type = _group_types(document)
    reported = {finding.finding_id for finding in repairable}
    all_out_of_scope = [
        {
            "code": finding.code,
            "owner": finding.action_owner.value,
            "gate": finding.gate_status.value,
            "message": finding.message,
        }
        for finding in all_findings
        if finding.finding_id not in reported
    ]
    out_of_scope = all_out_of_scope[:list_limit]

    mapping_rows, mapping_rows_total = _mapping_rows(
        mapping_table, repairable, list_limit
    )
    source_field_rows, source_fields_total = _source_field_briefs(
        source_fields, mapping_rows, list_limit * 2
    )
    required_elements, required_elements_total = _required_elements(
        target_tree, repairable, list_limit
    )
    repair_hints = _choice_repair_hints(repairable, target_tree, mapping_table)
    prohibited_all = (
        sorted(getattr(target_tree, "_pruned_ids", set()))
        if target_tree is not None
        else []
    )
    wanted_paths = {finding_path(finding) for finding in repairable} - {""}
    prohibited_all.sort(
        key=lambda path: (not _path_relevant(str(path), wanted_paths), str(path))
    )
    prohibited_paths = prohibited_all[:list_limit]
    selected_pointers = selected_rules[: list_limit * 4]
    return AgentContext(
        map_url=document.get("url"),
        map_id=document.get("id"),
        map_sha256=canonical_sha256(dict(document)),
        resource_type=resource_type,
        source_type=source_type,
        profile_url=profile_url,
        map_excerpt=excerpt,
        excerpt_pruned=pruned,
        pointers=selected_pointers,
        insertion_points=insertion_points,
        findings=[_brief(finding) for finding in repairable],
        out_of_scope=out_of_scope,
        source_fields=source_field_rows,
        mapping_rows=mapping_rows,
        concept_map_urls=sorted(
            {
                str(row["concept_map_url"])
                for row in mapping_rows
                if row.get("concept_map_url")
            }
        ),
        required_elements=required_elements,
        repair_hints=repair_hints,
        prohibited_paths=prohibited_paths,
        list_status={
            "pointers": _list_status(len(rules), len(selected_pointers)),
            "out_of_scope": _list_status(len(all_out_of_scope), len(out_of_scope)),
            "source_fields": _list_status(source_fields_total, len(source_field_rows)),
            "mapping_rows": _list_status(mapping_rows_total, len(mapping_rows)),
            "required_elements": _list_status(
                required_elements_total, len(required_elements)
            ),
            "prohibited_paths": _list_status(
                len(prohibited_all), len(prohibited_paths)
            ),
        },
        attempts=list(attempts),
    )


def _brief(finding: ValidationFinding) -> FindingBrief:
    return FindingBrief(
        finding_id=finding.finding_id,
        code=finding.code,
        message=finding.message,
        producer=finding.producer.value,
        stage=finding.stage.value,
        path=finding_path(finding) or None,
        pointer=finding.pointer,
        rule_hint=(
            str(finding.evidence.get("rule_hint"))
            if finding.evidence.get("rule_hint")
            else None
        ),
        fixture=finding.fixture_id,
    )
