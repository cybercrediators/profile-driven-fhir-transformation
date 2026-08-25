"""define target paths addressed by StructureMaps"""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from pydantic import BaseModel, ConfigDict


class AddressedPath(BaseModel):
    """One source or target element a map touches, with where it says so."""

    model_config = ConfigDict(extra="forbid")

    path: str
    # points at addressed src/target entry
    pointer: str
    rule_name: Optional[str] = None
    transform: Optional[str] = None

    structural: bool = False


def addressed_paths(document: Mapping[str, Any]) -> Tuple[List[AddressedPath], List[AddressedPath]]:
    """Resolve every source and target entry in document to a full path.

    :return: (sources, targets)
    """

    sources: List[AddressedPath] = []
    targets: List[AddressedPath] = []

    def walk(
        rules: Any,
        pointer: str,
        source_vars: Dict[str, str],
        target_vars: Dict[str, str],
    ) -> None:
        if not isinstance(rules, list):
            return
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            here = f"{pointer}/{index}"
            name = rule.get("name")
            local_sources = dict(source_vars)
            local_targets = dict(target_vars)

            for entry_index, entry in enumerate(rule.get("source") or []):
                if not isinstance(entry, dict):
                    continue
                parent = local_sources.get(str(entry.get("context") or ""))
                element = entry.get("element")
                if parent is None:
                    continue
                path = f"{parent}.{element}" if element else parent
                if element:
                    sources.append(
                        AddressedPath(
                            path=path,
                            pointer=f"{here}/source/{entry_index}",
                            rule_name=name,
                        )
                    )
                variable = entry.get("variable")
                if variable:
                    local_sources[str(variable)] = path

            for entry_index, entry in enumerate(rule.get("target") or []):
                if not isinstance(entry, dict):
                    continue
                parent = local_targets.get(str(entry.get("context") or ""))
                element = entry.get("element")
                if parent is None:
                    continue
                path = f"{parent}.{element}" if element else parent
                transform = entry.get("transform")
                if element:
                    targets.append(
                        AddressedPath(
                            path=path,
                            pointer=f"{here}/target/{entry_index}",
                            rule_name=name,
                            transform=transform,
                            structural=transform == "create",
                        )
                    )
                variable = entry.get("variable")
                if variable:
                    local_targets[str(variable)] = path

            walk(rule.get("rule"), f"{here}/rule", local_sources, local_targets)

    for group_index, group in enumerate(document.get("group") or []):
        if not isinstance(group, dict):
            continue
        source_vars: Dict[str, str] = {}
        target_vars: Dict[str, str] = {}
        for entry in group.get("input") or []:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            root = str(entry.get("type") or "")
            if entry.get("mode") == "source":
                source_vars[str(entry["name"])] = root
            elif entry.get("mode") == "target":
                target_vars[str(entry["name"])] = root
        walk(
            group.get("rule"),
            f"/group/{group_index}/rule",
            source_vars,
            target_vars,
        )

    return sources, targets


def _strip_root(path: str) -> str:
    """Drop the leading resource/logical-model segment from a dotted path."""

    return path.split(".", 1)[1] if "." in path else ""


def _split_outside_brackets(path: str, separator: str) -> List[str]:
    """Split on separator, ignoring occurrences inside []"""

    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for char in path:
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        if char == separator and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def normalize_target_path(path: str) -> str:
    """choice-normalized form used for set comparisons"""

    segments = []
    for segment in _split_outside_brackets(path, "."):
        segment = _split_outside_brackets(segment, ":")[0]
        if segment.endswith("[x]"):
            segment = segment[:-3]
        segments.append(segment)
    return ".".join(segments)


def resolve_tree_node(target_tree, path: str):
    """tree node for given path, whichever spelling the caller happens to hold"""

    if target_tree is None or not path:
        return None
    for candidate in _tree_lookup_spellings(path):
        node = target_tree.node(candidate)
        if node is not None:
            return node
    return None


def resolve_tree_types(target_tree, path: str) -> List[str]:
    """effective_types for given path (otherwise empty)"""

    if target_tree is None or not path:
        return []
    for candidate in _tree_lookup_spellings(path):
        types = target_tree.effective_types(candidate)
        if types:
            return [str(item) for item in types]
    return []


def _tree_lookup_spellings(path: str) -> List[str]:
    """spellings to try against a tree, most literal first"""

    normalized = normalize_target_path(path)
    parent, separator, leaf = normalized.rpartition(".")
    candidates = [path, normalized, f"{parent}{separator}{leaf}[x]"]
    # A concrete choice name reduced to its base: `valueQuantity` -> `value[x]`.
    for index in range(len(leaf) - 1, 0, -1):
        if leaf[index].isupper():
            candidates.append(f"{parent}{separator}{leaf[:index]}[x]")
    return list(dict.fromkeys(item for item in candidates if item))


def target_path_spellings(path: str) -> Set[str]:
    """equivalent ways to write one target path"""

    normalized = normalize_target_path(path)
    spellings = {normalized}

    concrete = []
    for segment in path.split("."):
        name, _, slice_name = segment.partition(":")
        if name.endswith("[x]") and slice_name:
            concrete.append(slice_name)
        else:
            concrete.append(normalize_target_path(segment))
    spellings.add(".".join(concrete))
    return {item for item in spellings if item}
