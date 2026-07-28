"""Backward-compatible rule-IR primitives used by mapping-table values.

The original contract is a JSON object of ``source-path -> target-path``
strings. Collection rules extend a value of that object instead of replacing
the format, so existing projects and automapping output remain valid.
"""

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Optional, Tuple


SOURCE_LIST_MODES = frozenset(
    {"first", "not_first", "last", "not_last", "only_one"}
)
TARGET_LIST_MODES = frozenset({"first", "share", "last", "collate"})
_FHIR_ID = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")
_COLLECTION_PROPERTIES = frozenset(
    {
        "target",
        "sourceListMode",
        "targetListMode",
        "listRuleId",
        "correlation",
    }
)


@dataclass(frozen=True)
class CollectionRuleSpec:
    """Normalized collection rule attached to a source/target parent pair."""

    source: str
    target: str
    source_list_mode: Optional[str] = None
    target_list_modes: Tuple[str, ...] = ()
    list_rule_id: Optional[str] = None
    source_key: Optional[str] = None
    target_key: Optional[str] = None
    invalid: bool = False


def mapping_target_path(value: Any) -> Optional[str]:
    """Return the target path from a legacy or typed mapping-table value."""

    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        target = value.get("target")
        return target if isinstance(target, str) and target else None
    return None


def _default_list_rule_id(target: str) -> str:
    """Create a deterministic FHIR ``id`` for a target share group."""

    readable = re.sub(r"[^A-Za-z0-9\-.]", "-", target).strip("-.")
    candidate = f"collection-{readable}" if readable else "collection"
    if len(candidate) <= 64:
        return candidate
    digest = hashlib.sha1(target.encode("utf-8")).hexdigest()[:10]
    return f"{candidate[:53]}-{digest}"


def parse_collection_rule(source: str, value: Any) -> Optional[CollectionRuleSpec]:
    """Parse one typed collection value.

    ``None`` means that ``value`` is a legacy mapping. Invalid typed rules raise
    ``ValueError`` so callers can diagnose them without inventing semantics.
    """

    if not isinstance(value, dict):
        return None

    unknown = sorted(set(value) - _COLLECTION_PROPERTIES)
    if unknown:
        raise ValueError(
            "unsupported collection rule properties: " + ", ".join(unknown)
        )

    target = mapping_target_path(value)
    if not target:
        raise ValueError("collection rule requires a non-empty string 'target'")

    source_mode = value.get("sourceListMode")
    if source_mode is not None and source_mode not in SOURCE_LIST_MODES:
        raise ValueError(
            f"sourceListMode must be one of {sorted(SOURCE_LIST_MODES)}, "
            f"got {source_mode!r}"
        )

    raw_target_modes = value.get("targetListMode")
    if raw_target_modes is None:
        target_modes = ()
    elif isinstance(raw_target_modes, str):
        target_modes = (raw_target_modes,)
    elif isinstance(raw_target_modes, list) and all(
        isinstance(mode, str) for mode in raw_target_modes
    ):
        target_modes = tuple(raw_target_modes)
    else:
        raise ValueError("targetListMode must be a string or a list of strings")

    invalid_modes = sorted(set(target_modes) - TARGET_LIST_MODES)
    if invalid_modes:
        raise ValueError(
            f"targetListMode must contain only {sorted(TARGET_LIST_MODES)}, "
            f"got {invalid_modes}"
        )
    if len(set(target_modes)) != len(target_modes):
        raise ValueError("targetListMode must not contain duplicate modes")

    list_rule_id = value.get("listRuleId")
    if list_rule_id is not None and (
        not isinstance(list_rule_id, str) or not _FHIR_ID.fullmatch(list_rule_id)
    ):
        raise ValueError("listRuleId must be a valid FHIR id (1..64 characters)")
    if list_rule_id and "share" not in target_modes:
        raise ValueError("listRuleId is only meaningful with targetListMode 'share'")
    if "share" in target_modes and not list_rule_id:
        list_rule_id = _default_list_rule_id(target)

    source_key = target_key = None
    correlation = value.get("correlation")
    if isinstance(correlation, str):
        source_key = target_key = correlation
    elif correlation is not None:
        if not isinstance(correlation, dict):
            raise ValueError("correlation must be a string or an object")
        unknown_correlation = sorted(
            set(correlation) - {"sourceKey", "targetKey"}
        )
        if unknown_correlation:
            raise ValueError(
                "unsupported correlation properties: "
                + ", ".join(unknown_correlation)
            )
        source_key = correlation.get("sourceKey")
        target_key = correlation.get("targetKey")
        if not isinstance(source_key, str) or not source_key:
            raise ValueError("correlation.sourceKey must be a non-empty string")
        if target_key is not None and (
            not isinstance(target_key, str) or not target_key
        ):
            raise ValueError(
                "correlation.targetKey must be a non-empty string when present"
            )

    return CollectionRuleSpec(
        source=source,
        target=target,
        source_list_mode=source_mode,
        target_list_modes=target_modes,
        list_rule_id=list_rule_id,
        source_key=source_key,
        target_key=target_key,
    )
