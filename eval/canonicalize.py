"""canonicalization spec for JSON StructureMaps
(creates rule set to compare two maps by content regardless of naming/ordering)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

# define fixed expected keys
_VOLATILE_SM_KEYS = ("id", "name", "title", "description", "meta", "text", "date", "version")
_VOLATILE_RULE_KEYS = ("documentation", "name")
_VAR_REF_PARAM_KEYS = ("valueId",)


def _collect_variables(group: Dict[str, Any]) -> Dict[str, str]:
    """convert map each declared variable name to a canonical declaration-site token"""

    mapping: Dict[str, str] = {}
    used: Dict[str, int] = {}

    def assign(name: Optional[str], token: str) -> None:
        if not name or name in mapping:
            return
        n = used.get(token, 0)
        used[token] = n + 1
        mapping[name] = token if n == 0 else f"{token}#{n}"

    for inp in group.get("input", []) or []:
        nm = inp.get("name")
        assign(nm, f"in:{inp.get('mode', '')}:{nm}")

    def walk(rules: List[Dict[str, Any]], path: Tuple[str, ...]) -> None:
        for rule in rules or []:
            elems = [t.get("element") for t in (rule.get("target") or []) if t.get("element")]
            new_path = path + tuple(elems)
            for src in rule.get("source", []) or []:
                assign(src.get("variable"), f"s:{src.get('element') or '.'.join(path) or 'ctx'}")
            for tgt in rule.get("target", []) or []:
                assign(tgt.get("variable"), f"t:{'.'.join(new_path) or tgt.get('element') or 'ctx'}")
            walk(rule.get("rule", []) or [], new_path)

    walk(group.get("rule", []) or [], ())
    return mapping


def _rename(value: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    return mapping.get(value, value)


def _normalise_params(
    params: List[Dict[str, Any]], var_map: Dict[str, str]
) -> Tuple[Tuple[str, str], ...]:
    """normalize transform parameter list"""
    out: List[Tuple[str, str]] = []
    for p in params or []:
        for k, v in p.items():
            if k in _VAR_REF_PARAM_KEYS:
                out.append((k, str(_rename(str(v), var_map))))
            else:
                out.append((k, str(v)))
    return tuple(out)


def _rule_signature(
    rule: Dict[str, Any], target_path: Tuple[str, ...], var_map: Dict[str, str]
) -> Dict[str, Any]:
    """build the hashable signature for a single rule node"""
    sources = []
    for src in rule.get("source", []) or []:
        sources.append(
            (
                _rename(src.get("context"), var_map),
                src.get("element"),
                src.get("type"),
                src.get("condition"),
                src.get("check"),
            )
        )
    targets = []
    for tgt in rule.get("target", []) or []:
        targets.append(
            (
                _rename(tgt.get("context"), var_map),
                tgt.get("element"),
                tgt.get("transform"),
                tuple(tgt.get("listMode", []) or []),
                _normalise_params(tgt.get("parameter", []) or [], var_map),
            )
        )
    return {
        "target_path": target_path,
        "sources": tuple(sources),
        "targets": tuple(targets),
    }


def _freeze(obj: Any) -> Any:
    """recursively convert dict/list into hashable tuples for set membership"""
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return obj


def canonical_rule_signatures(sm: Dict[str, Any]) -> set:
    """create the set of canonical per-rule signatures for a StructureMap"""
    signatures = set()
    for group in sm.get("group", []) or []:
        var_map = _collect_variables(group)

        def walk(rules: List[Dict[str, Any]], path: Tuple[str, ...]) -> None:
            for rule in rules or []:
                elems = [t.get("element") for t in (rule.get("target") or []) if t.get("element")]
                new_path = path + tuple(elems)
                sig = _rule_signature(rule, new_path, var_map)
                signatures.add(_freeze(sig))
                walk(rule.get("rule", []) or [], new_path)

        walk(group.get("rule", []) or [], ())
    return signatures


def _canonicalise_rule_tree(rules: List[Dict[str, Any]], var_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """create the normalized form based on given map and rules"""
    out = []
    for rule in rules or []:
        r = {k: v for k, v in rule.items() if k not in _VOLATILE_RULE_KEYS}
        for src in r.get("source", []) or []:
            if "variable" in src:
                src["variable"] = _rename(src["variable"], var_map)
            if "context" in src:
                src["context"] = _rename(src["context"], var_map)
        for tgt in r.get("target", []) or []:
            if "variable" in tgt:
                tgt["variable"] = _rename(tgt["variable"], var_map)
            if "context" in tgt:
                tgt["context"] = _rename(tgt["context"], var_map)
            for p in tgt.get("parameter", []) or []:
                for k in _VAR_REF_PARAM_KEYS:
                    if k in p:
                        p[k] = _rename(str(p[k]), var_map)
        if r.get("rule"):
            r["rule"] = _canonicalise_rule_tree(r["rule"], var_map)
        out.append(r)
    out.sort(key=lambda x: _stable_key(x))
    return out


def _stable_key(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, default=str)


def canonical_map(sm: Dict[str, Any]) -> Dict[str, Any]:
    """return a stable, comparable copy of a StructureMap"""
    out = {k: v for k, v in sm.items() if k not in _VOLATILE_SM_KEYS}
    out["structure"] = [
        {k: v for k, v in s.items() if k != "alias"} for s in (sm.get("structure") or [])
    ]
    groups = []
    for group in copy.deepcopy(sm.get("group", []) or []):
        var_map = _collect_variables(group)
        g = dict(group)
        for inp in g.get("input", []) or []:
            if "name" in inp:
                inp["name"] = _rename(inp["name"], var_map)
        g["rule"] = _canonicalise_rule_tree(g.get("rule", []) or [], var_map)
        groups.append(g)
    out["group"] = groups
    return out


def rule_level_prf1(generated: Dict[str, Any], golden: Dict[str, Any]) -> Dict[str, Any]:
    """compute rule-level precision / recall / F1 of generated vs golden"""
    gen = canonical_rule_signatures(generated)
    gold = canonical_rule_signatures(golden)
    tp = gen & gold
    spurious = gen - gold
    missing = gold - gen
    p = len(tp) / (len(tp) + len(spurious)) if (tp or spurious) else 1.0
    r = len(tp) / (len(tp) + len(missing)) if (tp or missing) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "true_positive": len(tp),
        "spurious": len(spurious),
        "missing": len(missing),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "missing_signatures": sorted(map(str, missing)),
        "spurious_signatures": sorted(map(str, spurious)),
    }
