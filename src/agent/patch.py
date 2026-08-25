"""canonical hashing and guarded application of agent patches"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Union

from agent.models import (
    AgentPatch,
    PatchApplication,
    PatchRejection,
    RejectionCode,
)
from agent.policy import PatchPolicy, validate
from llm.errors import LLMDependencyError

logger = logging.getLogger(__name__)


def canonical_json(document: Any) -> str:
    """serialize document"""

    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def canonical_sha256(document: Any) -> str:
    """canonical serialization"""

    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def raw_sha256(content: Union[str, bytes]) -> str:
    """digest of exact bytes, recorded for provenance only"""

    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _load_jsonpatch():
    """Import the optional RFC 6902 implementation, or explain how to get it."""

    try:
        import jsonpatch  # noqa: PLC0415  (deliberately lazy)
    except ImportError as exc:
        raise LLMDependencyError("jsonpatch", feature="agent mode") from exc
    return jsonpatch


def inject_guards(proposal: AgentPatch, base: Dict[str, Any]) -> AgentPatch:
    """return proposal with a test in front of every destructive operation
        
    :param proposal: the envelope, whose own operations are never mutated.
    :param base: the document the patch will be applied to.
    :return: a new envelope. The input is untouched.
    """

    from agent.models import PatchOp, operation  # noqa: PLC0415
    from agent.policy import _needs_guard, pointer_tokens  # noqa: PLC0415

    operations: List[Any] = []
    for index, op in enumerate(proposal.patch):
        kind = PatchOp(op.op)
        previous = proposal.patch[index - 1] if index else None
        already_guarded = (
            previous is not None
            and PatchOp(previous.op) is PatchOp.TEST
            and previous.path == op.path
        )
        try:
            tokens = pointer_tokens(op.path)
            needs = (
                kind is not PatchOp.TEST
                and not already_guarded
                and _needs_guard(kind, tokens, base)
            )
        except (ValueError, TypeError, IndexError):
            needs, tokens = False, []
        if needs:
            current = _current_value(base, tokens)
            if current is not _MISSING_VALUE:
                operations.append(operation(PatchOp.TEST, op.path, current))
                logger.debug("Injected guard for %s %s.", kind.value, op.path)
        operations.append(op)

    if len(operations) == len(proposal.patch):
        return proposal
    return proposal.model_copy(update={"patch": operations})


_MISSING_VALUE = object()


def _inert_pointers(proposal: AgentPatch, base: Dict[str, Any]) -> List[str]:
    """pointers whose mutation writes the value already sitting there"""

    from agent.models import PatchOp as _PatchOp  # noqa: PLC0415
    from agent.policy import pointer_tokens as _tokens  # noqa: PLC0415

    inert: List[str] = []
    for op in proposal.patch:
        if _PatchOp(op.op) not in (_PatchOp.REPLACE, _PatchOp.ADD):
            continue
        try:
            current = _current_value(base, _tokens(op.path))
        except (ValueError, TypeError, IndexError):
            continue
        if current is not _MISSING_VALUE and current == getattr(op, "value", None):
            inert.append(op.path)
    return inert


def _current_value(base: Dict[str, Any], tokens: List[str]) -> Any:
    """The value a pointer addresses right now, or _MISSING_VALUE"""

    from agent.policy import resolve_parent  # noqa: PLC0415

    try:
        parent = resolve_parent(base, tokens)
    except (ValueError, TypeError, IndexError):
        return _MISSING_VALUE
    key = tokens[-1] if tokens else ""
    if isinstance(parent, dict) and key in parent:
        return parent[key]
    if isinstance(parent, list) and key.isdigit() and int(key) < len(parent):
        return parent[int(key)]
    return _MISSING_VALUE


def apply_patch(
    base: Dict[str, Any],
    proposal: AgentPatch,
    *,
    policy: Optional[PatchPolicy] = None,
) -> PatchApplication:
    """validate proposal against base and, if admissible, apply

    :param base: the StructureMap as a plain dict
    :param proposal: the patch envelope under evaluation.
    :param policy: bounds and permissions, the default policy when omitted.
    :return: the outcome, carrying either a candidate or structured rejections.
    """

    policy = policy or PatchPolicy()
    digest = canonical_sha256(base)
    proposal = inject_guards(proposal, base)
    normalized = proposal.as_rfc6902()

    rejections = validate(proposal, base, digest, policy)
    if rejections:
        logger.info(
            "Agent patch rejected before application (%s).",
            ", ".join(sorted({rejection.code for rejection in rejections})),
        )
        return PatchApplication(
            applied=False, normalized_patch=normalized, rejections=rejections
        )

    jsonpatch = _load_jsonpatch()
    working = copy.deepcopy(base)

    try:
        candidate = jsonpatch.apply_patch(working, normalized, in_place=False)
    except jsonpatch.JsonPatchTestFailed as exc:
        return PatchApplication(
            applied=False,
            normalized_patch=normalized,
            rejections=[
                PatchRejection(
                    code=RejectionCode.TEST_FAILED.value,
                    message=(
                        "A guard did not hold against the base document, so the "
                        f"patch addresses something that is not there: {exc}"
                    ),
                )
            ],
        )
    except jsonpatch.JsonPointerException as exc:
        return PatchApplication(
            applied=False,
            normalized_patch=normalized,
            rejections=[
                PatchRejection(
                    code=RejectionCode.INVALID_POINTER.value,
                    message=f"A pointer does not resolve against the map: {exc}",
                )
            ],
        )
    except jsonpatch.JsonPatchException as exc:
        return PatchApplication(
            applied=False,
            normalized_patch=normalized,
            rejections=[
                PatchRejection(
                    code=RejectionCode.APPLICATION_FAILED.value,
                    message=f"The patch could not be applied: {exc}",
                )
            ],
        )

    structural = _structural_rejection(candidate)
    if structural is not None:
        return PatchApplication(
            applied=False, normalized_patch=normalized, rejections=[structural]
        )

    candidate_digest = canonical_sha256(candidate)
    if candidate_digest == digest:
        inert = _inert_pointers(proposal, base)
        detail = (
            " Each of these already holds the value it was given: "
            + ", ".join(inert)
            + "."
            if inert
            else ""
        )
        return PatchApplication(
            applied=False,
            normalized_patch=normalized,
            rejections=[
                PatchRejection(
                    code=RejectionCode.NO_EFFECT.value,
                    message=(
                        "The patch applied cleanly but left the map unchanged."
                        + detail
                    ),
                    path=inert[0] if len(inert) == 1 else None,
                )
            ],
        )

    return PatchApplication(
        applied=True,
        candidate=candidate,
        candidate_sha256=candidate_digest,
        normalized_patch=normalized,
        rejections=[],
    )


def _structural_rejection(candidate: Dict[str, Any]) -> Optional[PatchRejection]:
    """reject a candidate that is no longer a well-formed StructureMap"""

    from fhir.resources.R4B.structuremap import StructureMap  # noqa: PLC0415

    try:
        StructureMap.model_validate(candidate)
    except Exception as exc:  # pydantic ValidationError, or anything it raises
        summary = " | ".join(str(exc).splitlines()[1:4]).strip()
        return PatchRejection(
            code=RejectionCode.INVALID_CANDIDATE.value,
            message=(
                "The patch applied but the result is not a valid StructureMap: "
                f"{summary or exc}"
            ),
        )
    return None


def describe_pointers(document: Dict[str, Any]) -> List[Dict[str, str]]:
    """map each rule to its JSON Pointer for the prompt context"""

    entries: List[Dict[str, str]] = []

    def walk(rules: Any, pointer: str, label: str) -> None:
        if not isinstance(rules, list):
            return
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            here = f"{pointer}/{index}"
            name = str(rule.get("name") or index)
            trail = f"{label} › {name}"
            entries.append({"pointer": here, "label": trail, "name": name})
            walk(rule.get("rule"), f"{here}/rule", trail)

    for group_index, group in enumerate(document.get("group") or []):
        if not isinstance(group, dict):
            continue
        group_name = str(group.get("name") or group_index)
        walk(group.get("rule"), f"/group/{group_index}/rule", group_name)

    return entries
