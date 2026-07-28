"""Profile-constrained target tree.

One structure behind N1/N4/F4 (and N3's type dispatch) instead of four emitter special
cases. Built from the profile snapshot, keyed by ``ElementDefinition.id`` so slice identity
and the parent chain survive, then overlaid with introspected datatype children for the
elements a profile does not spell out.

What the emitter and the coverage manifest both need from it:

* **N1** — a fixed/pattern value on a nested primitive leaf is a *provider*: the leaf must be
  emitted even though no source field maps to it (`category:obstetrics.coding.system`).
* **N2** — a node knows its parent chain, so a descendant is never attached to an ancestor's
  context (`Procedure.reasonReference.type` must not become `Procedure.type`).
* **N3** — a node carries the profile-narrowed type list, so a polymorphic element can be
  written with its concrete name instead of the engine guessing.
* **N4** — ``max = 0`` subtrees are pruned before rule generation, so a container expansion
  cannot emit a prohibited child (`Goal.description.coding`).
* **F4** — introspected children of a nested complex type are addressable by a mapping table.

Deliberately independent of ``app_state``/registry so it can be unit-tested against a bare
StructureDefinition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# providers, in precedence order — what will supply a value for a node
PROVIDER_SOURCE = "source"
PROVIDER_FIXED = "profile-fixed"
PROVIDER_CONFIG = "config"
PROVIDER_REQUIRED_PARENT = "required-parent"

_FIXED_PREFIXES = ("fixed", "pattern")
# keys on an ElementDefinition that are not fixed[x]/pattern[x] despite the prefix
_NOT_FIXED = {"fixedValueSet", "patternValueSet"}


@dataclass
class TargetNode:
    """One element of the profile-constrained target tree."""

    eid: str
    path: str
    min: int = 0
    max: str = "1"
    types: List[str] = dc_field(default_factory=list)
    slice_name: Optional[str] = None
    fixed_value: Any = None
    is_pattern: bool = False
    binding: Optional[dict] = None
    introspected: bool = False
    parent: Optional["TargetNode"] = None
    children: Dict[str, "TargetNode"] = dc_field(default_factory=dict)
    provider: Optional[str] = None

    @property
    def name(self) -> str:
        """Last path segment without the slice qualifier."""
        return self.path.split(".")[-1]

    @property
    def prohibited(self) -> bool:
        return str(self.max) == "0"

    @property
    def required(self) -> bool:
        return self.min > 0

    @property
    def is_choice(self) -> bool:
        return "[x]" in self.path

    def depth_below(self, ancestor: "TargetNode") -> int:
        """How many element levels separate this node from `ancestor` (0 = same node)."""
        steps, cur = 0, self
        while cur is not None and cur is not ancestor:
            cur, steps = cur.parent, steps + 1
        return steps if cur is ancestor else -1

    def descendants(self):
        for child in self.children.values():
            yield child
            yield from child.descendants()


def _split_eid(eid: str) -> List[str]:
    """Segments of an ElementDefinition.id, keeping ``name:slice`` segments intact."""
    return eid.split(".")


def _fixed_of(element: dict):
    """(value, is_pattern) for an ElementDefinition's fixed[x]/pattern[x], else (None, False)."""
    for key, value in element.items():
        if key in _NOT_FIXED or not key.startswith(_FIXED_PREFIXES):
            continue
        suffix = key[7:] if key.startswith("pattern") else key[5:]
        if suffix and suffix[:1].isupper():
            return value, key.startswith("pattern")
    return None, False


class TargetTree:
    """The element tree of one profile, pruned and annotated for rule generation."""

    def __init__(self, root: TargetNode, by_eid: Dict[str, TargetNode], res_type: str):
        self.root = root
        self._by_eid = by_eid
        self.res_type = res_type
        self._pruned_ids: set = set()

    # ------------------------------------------------------------------ build
    @classmethod
    def from_snapshot(cls, sd, introspect_children=None) -> Optional["TargetTree"]:
        """Build from a StructureDefinition (pydantic model or dict).

        ``introspect_children(type_code, parent_path)`` supplies datatype children for
        complex elements the profile leaves unexpanded; pass ``None`` to skip step 2.
        """
        data = sd if isinstance(sd, dict) else _model_dump(sd)
        if not data:
            return None
        elements = ((data.get("snapshot") or {}).get("element")) or []
        if not elements:
            return None
        res_type = data.get("type") or (elements[0].get("path") or "").split(".")[0]

        by_eid: Dict[str, TargetNode] = {}
        root: Optional[TargetNode] = None
        for element in elements:
            eid = element.get("id") or element.get("path")
            if not eid:
                continue
            node = TargetNode(
                eid=eid,
                path=element.get("path") or eid,
                min=int(element.get("min") or 0),
                max=str(element.get("max") if element.get("max") is not None else "1"),
                types=[t.get("code") for t in (element.get("type") or []) if t.get("code")],
                slice_name=element.get("sliceName"),
                binding=element.get("binding"),
            )
            node.fixed_value, node.is_pattern = _fixed_of(element)
            by_eid[eid] = node
            if root is None and "." not in eid:
                root = node

        if root is None:
            return None

        # 1. parent chain by ElementDefinition.id (slice identity intact)
        for eid, node in by_eid.items():
            segments = _split_eid(eid)
            if len(segments) < 2:
                continue
            last = segments[-1]
            # a slice's parent is the element it slices (`A.b:s` -> `A.b`), which by plain
            # id-prefix walking would look like a sibling
            if ":" in last:
                base = by_eid.get(".".join(segments[:-1] + [last.split(":", 1)[0]]))
                if base is not None and base is not node:
                    node.parent = base
                    base.children[last] = node
                    continue
            for cut in range(len(segments) - 1, 0, -1):
                parent = by_eid.get(".".join(segments[:cut]))
                if parent is not None:
                    node.parent = parent
                    parent.children[last] = node
                    break

        tree = cls(root, by_eid, res_type)
        # 3. prune prohibited subtrees before anything consumes the tree
        tree._prune_prohibited()
        # 2. overlay introspected datatype children where the profile is silent
        if introspect_children is not None:
            tree._overlay_introspected(introspect_children)
        return tree

    def _prune_prohibited(self) -> int:
        """Drop ``max=0`` nodes and everything beneath them (N4)."""
        dropped = 0
        for node in list(self._by_eid.values()):
            if not node.prohibited:
                continue
            if node.parent is not None:
                node.parent.children.pop(node.eid.split(".")[-1], None)
            for gone in [node, *node.descendants()]:
                self._pruned_ids.add(gone.eid)
                if self._by_eid.pop(gone.eid, None) is not None:
                    dropped += 1
        if dropped:
            logger.debug("Pruned %d prohibited (max=0) target nodes", dropped)
        return dropped

    def _overlay_introspected(self, introspect_children) -> int:
        """Add datatype children for complex nodes the profile does not expand (F4)."""
        added = 0
        for node in list(self._by_eid.values()):
            if node.children or node.prohibited or len(node.types) != 1:
                continue
            type_code = node.types[0]
            if not type_code or not type_code[:1].isupper():
                continue  # primitive
            try:
                fields = _datatype_children(introspect_children, type_code)
            except Exception as exc:  # introspection is best-effort
                logger.debug("introspection failed for %s (%s)", type_code, exc)
                continue
            for leaf, leaf_types, card in fields:
                sub_path = f"{node.path}.{leaf}"
                child = TargetNode(
                    eid=f"{node.eid}.{leaf}",
                    path=sub_path,
                    min=int((card or {}).get("min", 0) or 0),
                    max=str((card or {}).get("max", "1")),
                    types=list(leaf_types),
                    introspected=True,
                    parent=node,
                )
                node.children[leaf] = child
                self._by_eid.setdefault(child.eid, child)
                added += 1
        return added

    # ------------------------------------------------------------------ query
    def node(self, key: str) -> Optional[TargetNode]:
        """Look up by ElementDefinition.id, falling back to a unique path match."""
        hit = self._by_eid.get(key)
        if hit is not None:
            return hit
        matches = [n for n in self._by_eid.values() if n.path == key]
        return matches[0] if len(matches) == 1 else None

    def is_prohibited(self, key: str) -> bool:
        """True when the profile forbids this element or one of its ancestors (N4)."""
        if key in self._pruned_ids:
            return True
        return any(key.startswith(pruned + ".") for pruned in self._pruned_ids)

    def effective_types(self, key: str) -> List[str]:
        """Profile-narrowed candidate types of an element (N3 dispatch)."""
        node = self.node(key)
        return list(node.types) if node else []

    def fixed_descendants(self, key: str, max_depth: int = 3) -> List[TargetNode]:
        """Fixed/pattern leaves beneath a node that nothing else will supply (N1)."""
        node = self.node(key)
        if node is None:
            return []
        out = []
        for desc in node.descendants():
            if desc.fixed_value is None or desc.prohibited:
                continue
            if desc.depth_below(node) > max_depth:
                continue
            out.append(desc)
        return out

    def required_manifest(self) -> List[dict]:
        """Requirement manifest from the same tree the emitter uses (step 6)."""
        entries = []
        for node in self._by_eid.values():
            if not node.required or node is self.root:
                continue
            active = all(
                anc.required for anc in _ancestors(node) if anc is not self.root
            )
            entries.append(
                {
                    "id": node.eid,
                    "path": node.path,
                    "min": node.min,
                    "max": node.max,
                    "active": active,
                    "provider": node.provider,
                    "introspected": node.introspected,
                }
            )
        return sorted(entries, key=lambda e: e["id"])



_CHILD_CACHE: Dict[str, list] = {}


def _datatype_children(introspect_children, type_code: str) -> list:
    """Immediate children of a datatype, cached process-wide.

    Introspection costs 0.3-1 s per profile and the underlying helper is uncached, so a
    69-profile module would pay it 69 times. The shape depends only on the datatype, so
    introspect once against a placeholder path and re-path per node.
    """
    cached = _CHILD_CACHE.get(type_code)
    if cached is not None:
        return cached
    raw = introspect_children(type_code, "X") or []
    out = []
    for sub in raw:
        if not isinstance(sub, dict):
            continue
        path = sub.get("path") or ""
        if not path.startswith("X."):
            continue
        leaf = path[2:]
        if "." in leaf:
            continue  # immediate level only; deeper levels come on demand
        out.append((leaf, _types_of(sub), sub.get("cardinality") or {}))
    _CHILD_CACHE[type_code] = out
    return out


def _ancestors(node: TargetNode):
    cur = node.parent
    while cur is not None:
        yield cur
        cur = cur.parent


def _types_of(field: dict) -> List[str]:
    raw = field.get("type")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(t["code"]) for t in raw if isinstance(t, dict) and t.get("code")]
    return []


def _model_dump(sd) -> dict:
    for attr in ("model_dump", "dict"):
        fn = getattr(sd, attr, None)
        if callable(fn):
            try:
                dumped = fn(exclude_none=True)
            except TypeError:
                dumped = fn()
            if isinstance(dumped, dict):
                return dumped
    return {}
