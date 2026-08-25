"""define admissibility rules (policy) for an agent patch proposal"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import json
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from agent.models import (
    GUARDED_OPS,
    MUTATING_OPS,
    PATCH_SCHEMA_VERSION,
    AgentPatch,
    PatchOp,
    PatchRejection,
    RejectionCode,
)

# set top level keys which should not be modified
PROTECTED_TOP_LEVEL: FrozenSet[str] = frozenset(
    {"resourceType", "id", "url", "name", "version"}
)

# define top level keys which can be modified
ALLOWED_TOP_LEVEL: FrozenSet[str] = frozenset(
    {
        "contact",
        "contained",
        "copyright",
        "date",
        "description",
        "experimental",
        "extension",
        "group",
        "identifier",
        "implicitRules",
        "import",
        "jurisdiction",
        "language",
        "meta",
        "modifierExtension",
        "publisher",
        "purpose",
        "status",
        "structure",
        "text",
        "title",
        "useContext",
    }
    | PROTECTED_TOP_LEVEL
)


@dataclass(frozen=True)
class PatchPolicy:
    """bounds and permissions for one agent run."""

    max_operations: int = 40
    max_payload_bytes: int = 64000
    allowed_ops: FrozenSet[PatchOp] = dc_field(
        default_factory=lambda: frozenset(PatchOp)
    )
    protected_top_level: FrozenSet[str] = PROTECTED_TOP_LEVEL
    allowed_top_level: FrozenSet[str] = ALLOWED_TOP_LEVEL
    require_guards: bool = True
    
    allowed_pointer_prefixes: Tuple[str, ...] = ()

    def as_report_dict(self) -> Dict[str, Any]:
        return {
            "max_operations": self.max_operations,
            "max_payload_bytes": self.max_payload_bytes,
            "allowed_ops": sorted(op.value for op in self.allowed_ops),
            "require_guards": self.require_guards,
            "allowed_pointer_prefixes": list(self.allowed_pointer_prefixes),
        }


def unescape_token(token: str) -> str:
    """decode RFC 6901 token (~1 to / and ~0 to ~)"""

    return token.replace("~1", "/").replace("~0", "~")


def pointer_tokens(pointer: str) -> List[str]:
    """split RFC 6901 pointer into decoded reference tokens

    :raises ValueError: when the pointer is not a valid RFC 6901 pointer.
    """

    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError("a JSON Pointer must be empty or start with '/'")
    return [unescape_token(token) for token in pointer[1:].split("/")]


def resolve_parent(document: Any, tokens: Sequence[str]) -> Optional[Any]:
    """return the container the final token addresses, or None if unreachable"""

    node = document
    for token in tokens[:-1]:
        if isinstance(node, dict):
            if token not in node:
                return None
            node = node[token]
        elif isinstance(node, list):
            try:
                node = node[int(token)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def _reject(code: RejectionCode, message: str, index=None, path=None) -> PatchRejection:
    return PatchRejection(
        code=code.value, message=message, operation_index=index, path=path
    )


def validate(
    proposal: AgentPatch,
    base: Dict[str, Any],
    base_sha256: str,
    policy: Optional[PatchPolicy] = None,
) -> List[PatchRejection]:
    """return every reason proposal is inadmissible against base"""

    policy = policy or PatchPolicy()
    rejections: List[PatchRejection] = []

    rejections.extend(_validate_envelope(proposal, base, base_sha256))

    operations = proposal.patch
    if not operations:
        rejections.append(
            _reject(RejectionCode.EMPTY_PATCH, "The patch contains no operations.")
        )
        return rejections

    if len(operations) > policy.max_operations:
        rejections.append(
            _reject(
                RejectionCode.TOO_MANY_OPERATIONS,
                f"{len(operations)} operations exceeds the limit of "
                f"{policy.max_operations}.",
            )
        )

    payload_size = len(
        json.dumps(proposal.as_rfc6902(), separators=(",", ":"), default=str)
    )
    if payload_size > policy.max_payload_bytes:
        rejections.append(
            _reject(
                RejectionCode.PAYLOAD_TOO_LARGE,
                f"Patch payload of {payload_size} bytes exceeds the limit of "
                f"{policy.max_payload_bytes}.",
            )
        )

    for index, op in enumerate(operations):
        rejections.extend(
            _validate_operation(index, op, operations, base, policy)
        )

    rejections.extend(_validate_sequencing(operations, base))

    if not any(PatchOp(op.op) in MUTATING_OPS for op in operations):
        rejections.append(
            _reject(
                RejectionCode.NO_EFFECT,
                "The patch only asserts; it changes nothing.",
            )
        )

    return rejections


def _validate_envelope(
    proposal: AgentPatch, base: Dict[str, Any], base_sha256: str
) -> List[PatchRejection]:
    """check that the proposal is for this map, at the current revision"""

    found: List[PatchRejection] = []

    if proposal.schema_version != PATCH_SCHEMA_VERSION:
        found.append(
            _reject(
                RejectionCode.MALFORMED_OP,
                f"Unsupported patch schema_version {proposal.schema_version}; "
                f"this build accepts {PATCH_SCHEMA_VERSION}.",
            )
        )

    if base.get("resourceType") != "StructureMap":
        found.append(
            _reject(
                RejectionCode.NOT_A_STRUCTURE_MAP,
                "The base document is not a StructureMap "
                f"(resourceType={base.get('resourceType')!r}).",
            )
        )

    if proposal.map_url != base.get("url"):
        found.append(
            _reject(
                RejectionCode.WRONG_MAP,
                f"The patch targets {proposal.map_url!r} but the base map is "
                f"{base.get('url')!r}.",
            )
        )
    if proposal.map_id != base.get("id"):
        found.append(
            _reject(
                RejectionCode.WRONG_MAP,
                f"The patch targets map id {proposal.map_id!r} but the base map "
                f"is {base.get('id')!r}.",
            )
        )

    if proposal.base_sha256 != base_sha256:
        found.append(
            _reject(
                RejectionCode.STALE_BASE,
                "The patch was written against a different revision of this map. "
                f"Expected base_sha256 {base_sha256}, got "
                f"{proposal.base_sha256 or '(none)'}. Copy the value from the "
                "prompt exactly, in full.",
            )
        )

    return found


def _validate_operation(
    index: int,
    op: Any,
    operations: Sequence[Any],
    base: Dict[str, Any],
    policy: PatchPolicy,
) -> List[PatchRejection]:
    """check given generated operations if they are valid to execute/have the correct format"""
    found: List[PatchRejection] = []
    path = op.path
    kind = PatchOp(op.op)

    if kind not in policy.allowed_ops:
        found.append(
            _reject(
                RejectionCode.UNSUPPORTED_OP,
                f"Operation '{kind.value}' is not permitted.",
                index,
                path,
            )
        )
        return found

    try:
        tokens = pointer_tokens(path)
    except ValueError as exc:
        found.append(_reject(RejectionCode.INVALID_POINTER, str(exc), index, path))
        return found

    if not tokens:
        if kind in MUTATING_OPS:
            found.append(
                _reject(
                    RejectionCode.ROOT_REPLACEMENT,
                    "The whole document may not be replaced; patch the specific "
                    "elements that are wrong.",
                    index,
                    path,
                )
            )
        return found

    first = tokens[0]
    if first not in policy.allowed_top_level:
        found.append(
            _reject(
                RejectionCode.OUT_OF_SCOPE_POINTER,
                f"'{first}' is not a StructureMap field.",
                index,
                path,
            )
        )
    elif kind in MUTATING_OPS and first in policy.protected_top_level:
        found.append(
            _reject(
                RejectionCode.PROTECTED_FIELD,
                f"'{first}' identifies the map and may not be changed by an agent.",
                index,
                path,
            )
        )

    if policy.allowed_pointer_prefixes and not any(
        path == prefix or path.startswith(prefix.rstrip("/") + "/")
        for prefix in policy.allowed_pointer_prefixes
    ):
        found.append(
            _reject(
                RejectionCode.OUT_OF_SCOPE_POINTER,
                "This pointer is outside the fragment the patch was asked about.",
                index,
                path,
            )
        )

    if policy.require_guards and _needs_guard(kind, tokens, base):
        previous = operations[index - 1] if index > 0 else None
        guarded = (
            previous is not None
            and PatchOp(previous.op) is PatchOp.TEST
            and previous.path == path
        )
        if not guarded:
            found.append(
                _reject(
                    RejectionCode.UNGUARDED_MUTATION,
                    f"'{kind.value}' overwrites existing content and must be "
                    "immediately preceded by a 'test' on the same path, so a "
                    "shifted index or a changed value fails the guard instead of "
                    "editing the wrong thing.",
                    index,
                    path,
                )
            )

    return found


def _needs_guard(kind: PatchOp, tokens: Sequence[str], base: Dict[str, Any]) -> bool:
    """check If operation destroys existing content and so must be guarded"""

    if kind in GUARDED_OPS:
        return True
    if kind is not PatchOp.ADD:
        return False

    parent = resolve_parent(base, tokens)
    return isinstance(parent, dict) and tokens[-1] in parent


def _array_shift(op: Any, base: Dict[str, Any]) -> Optional[str]:
    """the array pointer whose indices this operation invalidates, if any"""

    kind = PatchOp(op.op)
    if kind not in (PatchOp.ADD, PatchOp.REMOVE):
        return None
    try:
        tokens = pointer_tokens(op.path)
    except ValueError:
        return None
    if not tokens or tokens[-1] == "-":
        return None
    if not isinstance(resolve_parent(base, tokens), list):
        return None
    return op.path.rsplit("/", 1)[0]


def _validate_sequencing(
    operations: Sequence[Any], base: Dict[str, Any]
) -> List[PatchRejection]:
    """reject pointers that a preceding operation in the same patch invalidates"""

    found: List[PatchRejection] = []
    shifted: List[Tuple[int, str]] = []

    for index, op in enumerate(operations):
        for origin, array_pointer in shifted:
            inside = op.path.startswith(array_pointer + "/")
            if inside and op.path != f"{array_pointer}/-":
                found.append(
                    _reject(
                        RejectionCode.SHIFTED_POINTER,
                        f"Operation {origin} changes the length of "
                        f"'{array_pointer}', so this index no longer means what it "
                        "meant in the base document. Split the change into "
                        "separate attempts, or place the structural change last.",
                        index,
                        op.path,
                    )
                )
                break

        array_pointer = _array_shift(op, base)
        if array_pointer is not None:
            shifted.append((index, array_pointer))

    return found
