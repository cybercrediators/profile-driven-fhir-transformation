"""Normalization spec for output FHIR

Public entry points:

  `normalize(resource_or_bundle, ...)`  canonical FHIR (volatile tokens
      remapped, dropped fields removed).
  `behavioral_equal(a, b)`              (bool, diff) on two outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

_DROP_META_KEYS = ("lastUpdated", "versionId", "source", "security", "tag")

_UUID_RE = re.compile(r"^urn:uuid:[0-9a-fA-F-]{36}$")


def _is_uuid(v: Any) -> bool:
    return isinstance(v, str) and bool(_UUID_RE.match(v))


def _iter_uuids(obj: Any):
    """Yield every urn:uuid value appearing anywhere in ``obj``."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_uuids(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _iter_uuids(x)
    elif _is_uuid(obj):
        yield obj


def _content_token_map(payload: Any) -> Dict[str, str]:
    """assign each ``urn:uuid`` a canonical token derived from the *content* of the resource it identifies"""
    # fullUrl uuid -> the resource it points at
    referent: Dict[str, Any] = {}
    for bundle in payload if isinstance(payload, list) else [payload]:
        if isinstance(bundle, dict):
            for entry in bundle.get("entry", []) or []:
                fu = entry.get("fullUrl")
                if _is_uuid(fu):
                    referent[fu] = entry.get("resource", {})

    cache: Dict[str, str] = {}

    def content_hash(u: str, stack: Tuple[str, ...]) -> str:
        if u in cache:
            return cache[u]
        if u in stack:
            return "↻"  # cycle guard
        res = referent.get(u)
        if res is None:
            return "DANGLING"  # referenced but not defined in payload

        def repl(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: repl(v)
                    for k, v in obj.items()
                    if not (k == "id")  # ids are volatile; exclude from key
                }
            if isinstance(obj, list):
                return [repl(x) for x in obj]
            if _is_uuid(obj):
                return content_hash(obj, stack + (u,))
            return obj

        key = json.dumps(_drop_meta(repl(res)), sort_keys=True, default=str)
        h = hashlib.sha1(key.encode()).hexdigest()
        cache[u] = h
        return h

    uuids = set(_iter_uuids(payload))
    return {u: f"urn:uuid:{content_hash(u, ())[:16]}" for u in uuids}


def _drop_meta(obj: Any) -> Any:
    """Strip server/runtime metadata (``meta.lastUpdated`` etc.) recursively."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "meta" and isinstance(v, dict):
                v = {mk: mv for mk, mv in v.items() if mk not in _DROP_META_KEYS}
                if not v:
                    continue
            out[k] = _drop_meta(v)
        return out
    if isinstance(obj, list):
        return [_drop_meta(x) for x in obj]
    return obj


def _strip(obj: Any, drop_ids: bool, tokens: Dict[str, str]) -> Any:
    """Drop/normalise volatile fields and apply the content-based uuid tokens."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "meta" and isinstance(v, dict):
                v = {mk: mv for mk, mv in v.items() if mk not in _DROP_META_KEYS}
                if not v:
                    continue
            if k == "id" and drop_ids:
                continue
            if k == "fullUrl":
                continue
            out[k] = _strip(v, drop_ids, tokens)
        return out
    if isinstance(obj, list):
        return [_strip(x, drop_ids, tokens) for x in obj]
    if _is_uuid(obj):
        return tokens.get(obj, obj)
    return obj


def _canonical_entry_key(entry: Dict[str, Any]) -> str:
    """stable key for a Bundle entry: its normalized resource content without the volatile fullUrl token"""
    res = entry.get("resource", {})
    return json.dumps(res, sort_keys=True, default=str)


def normalize(
    payload: Any,
    *,
    drop_resource_ids: bool = True,
    sort_bundle_entries: bool = True,
) -> Any:
    """return a canonical, comparison-ready copy of a FHIR resource/Bundle"""
    tokens = _content_token_map(payload)
    stripped = _strip(payload, drop_resource_ids, tokens)

    def fix_bundle(b: Any) -> Any:
        if isinstance(b, dict) and b.get("resourceType") == "Bundle":
            b = {k: v for k, v in b.items() if k != "id"}
            entries = b.get("entry", []) or []
            if sort_bundle_entries:
                entries = sorted(entries, key=_canonical_entry_key)
            b["entry"] = entries
        return b

    if isinstance(stripped, list):
        return [fix_bundle(x) for x in stripped]
    return fix_bundle(stripped)


def behavioral_equal(a: Any, b: Any) -> Tuple[bool, List[str]]:
    """compare two pipeline outputs for behavioral (semantic) equivalence"""
    na, nb = normalize(a), normalize(b)
    diffs: List[str] = []
    _deep_diff(na, nb, "$", diffs)
    return (not diffs), diffs


def _deep_diff(a: Any, b: Any, path: str, diffs: List[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                diffs.append(f"{path}.{k}: only in B")
            elif k not in b:
                diffs.append(f"{path}.{k}: only in A")
            else:
                _deep_diff(a[k], b[k], f"{path}.{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list len {len(a)} != {len(b)}")
        for i in range(min(len(a), len(b))):
            _deep_diff(a[i], b[i], f"{path}[{i}]", diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")
    elif type(a) is not type(b):
        diffs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
