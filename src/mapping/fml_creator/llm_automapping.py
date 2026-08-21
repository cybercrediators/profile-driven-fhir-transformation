"""Handling optional llm automapping"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

from llm.errors import LLMConfigurationError, LLMResponseValidationError
from llm.models import LLMMessage, ModelSettings
from llm.prompts import AUTOMAPPING_SYSTEM, AUTOMAPPING_USER, render
from mapping.fml_creator.fml_automapper import AutomapCandidate, FMLAutomapper
from mapping.fml_creator.fml_helper import rebase_to_resource_identity
from mapping.target_tree import TargetTree

logger = logging.getLogger(__name__)

REPORT_VERSION = 1
FEATURE = "automapping"

#: The one non-candidate answer the model may give.
UNMAPPED = "unmapped"

DEFAULT_TOP_K = 5

DEFAULT_SECOND_PASS_TOP_K = 15


class CandidateSelection(BaseModel):
    """The model's answer. Deliberately the narrowest useful shape"""

    candidate_id: str = Field(
        description="The id of the chosen candidate, or 'unmapped'."
    )
    confidence: float = Field(
        description="Confidence as a decimal fraction between 0.0 and 1.0."
    )
    reason: Optional[str] = Field(
        default=None, description="One short sentence justifying the choice."
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _as_fraction(cls, value: Any) -> Any:
        """Accept a percentage, and never let this field fail a run"""

        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if number > 1.0:
            number = number / 100.0
        return min(1.0, max(0.0, number))

PooledCandidate = Tuple["AutomapCandidate", "ResourceTargets", str, Optional[str]]


@dataclass(frozen=True)
class ResourceTargets:
    """One profile's target surface, assembled by the generator."""

    res_type: str
    res_id: str
    target_fields: List[Dict[str, Any]]
    #: StructureDefinition (or dict) the target tree is built from.
    profile: Any


@dataclass(frozen=True)
class Offer:
    """One enumerated candidate as presented to the model."""

    offer_id: str
    candidate: AutomapCandidate
    #: The mapping-table target, already rebased onto the profile identity.
    emitted_target: str
    res_type: str
    res_id: str
    #: Element description carried over from the profile, when it has one.
    description: Optional[str] = None

    def as_report_dict(self) -> Dict[str, Any]:
        return {
            "offer_id": self.offer_id,
            "candidate_id": self.candidate.candidate_id,
            "target": self.emitted_target,
            "element_id": self.candidate.canonical_target_id,
            "resource": self.res_id or self.res_type,
            "types": list(self.candidate.types),
            "cardinality": f"{self.candidate.min}..{self.candidate.max}",
            "slice_name": self.candidate.slice_name,
            "score": round(self.candidate.scores.legacy, 6),
            "deterministic_rank_one": self.candidate.is_legacy_selection,
        }


@dataclass
class FieldAttempt:
    """One provider round for one source field."""

    pass_index: int
    offers: List[Offer]
    selection: Optional[CandidateSelection] = None
    cache_hit: bool = False
    rejected_reason: Optional[str] = None

    def as_report_dict(self) -> Dict[str, Any]:
        return {
            "pass": self.pass_index,
            "cache_hit": self.cache_hit,
            "offers": [offer.as_report_dict() for offer in self.offers],
            "selection": (
                {
                    "candidate_id": self.selection.candidate_id,
                    "confidence": self.selection.confidence,
                    "reason": self.selection.reason,
                }
                if self.selection is not None
                else None
            ),
            "rejected_reason": self.rejected_reason,
        }


@dataclass
class FieldOutcome:
    source_id: str
    attempts: List[FieldAttempt] = dc_field(default_factory=list)
    final_target: Optional[str] = None
    final_offer_id: Optional[str] = None
    rejected_reason: Optional[str] = None

    def as_report_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "final_target": self.final_target,
            "selected_offer_id": self.final_offer_id,
            "rejected_reason": self.rejected_reason,
            "attempts": [attempt.as_report_dict() for attempt in self.attempts],
        }


class LLMAutomappingStrategy:
    """Reranks deterministic candidates with a model, one source field at a time"""

    def __init__(
        self,
        automapper: FMLAutomapper,
        client: Any,
        model_settings: ModelSettings,
        *,
        top_k: int = DEFAULT_TOP_K,
        second_pass_top_k: Optional[int] = DEFAULT_SECOND_PASS_TOP_K,
    ) -> None:
        self.automapper = automapper
        self.client = client
        self.model_settings = model_settings
        self.top_k = _positive_width(top_k, "llm.automapping_top_k")
        self.second_pass_top_k = (
            _positive_width(second_pass_top_k, "llm.automapping_second_pass_top_k")
            if second_pass_top_k is not None
            else None
        )
        if self.second_pass_top_k is not None and self.second_pass_top_k <= self.top_k:
            raise LLMConfigurationError(
                "'llm.automapping_second_pass_top_k' must be greater than "
                "'llm.automapping_top_k', or null to disable the second pass."
            )
        self._outcomes: List[FieldOutcome] = []
        self._provider_calls = 0
        self._cache_hits = 0
        self._unusable_responses = 0
        self._skipped_profiles: List[Dict[str, str]] = []

    def build_mapping_table(
        self,
        resource_targets: Sequence[ResourceTargets],
        source_fields: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Return ``{source_field_id: target_path}`` for the whole project."""

        self._outcomes = []
        self._provider_calls = 0
        self._cache_hits = 0
        self._unusable_responses = 0
        self._skipped_profiles = []

        offers_by_source = self._collect_offers(resource_targets, source_fields)

        mapping_table: Dict[str, str] = {}
        reserved_targets = set()
        for source_field in source_fields:
            source_id = _source_id(source_field)
            if not source_id:
                continue
            available = offers_by_source.get(source_id, [])
            outcome = self._resolve_field(
                source_field, source_id, available, reserved_targets
            )
            self._outcomes.append(outcome)
            if outcome.final_target:
                mapping_table[source_id] = outcome.final_target
                reserved_targets.add(outcome.final_target)

        logger.info(
            "LLM automapping complete: %d/%d source fields mapped "
            "(%d provider calls, %d cache hits).",
            len(mapping_table),
            len(self._outcomes),
            self._provider_calls,
            self._cache_hits,
        )
        return mapping_table

    def report(
        self,
        mapping_table: Optional[Dict[str, str]] = None,
        *,
        compiled_mapping_table: Optional[Dict[str, str]] = None,
        compiler_diagnostics: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Auditable record of what was offered, selected, and compiled"""

        rejected = sum(
            1
            for outcome in self._outcomes
            for attempt in outcome.attempts
            if attempt.rejected_reason
        )
        allocation_rejections = sum(
            1 for outcome in self._outcomes if outcome.rejected_reason
        )
        proposed = dict(mapping_table or {})
        compiled = (
            dict(compiled_mapping_table)
            if compiled_mapping_table is not None
            else dict(proposed)
        )
        diagnostics = [dict(item) for item in (compiler_diagnostics or [])]
        compiler_rejections = [
            {
                "source_id": source_id,
                "target": target,
                "reason": "not-accepted-by-compiler",
                "diagnostics": [
                    diagnostic
                    for diagnostic in diagnostics
                    if diagnostic.get("source") == source_id
                    or diagnostic.get("previous_source") == source_id
                    or diagnostic.get("target") == target
                ],
            }
            for source_id, target in proposed.items()
            if source_id not in compiled
        ]
        mapped = len(compiled)
        return {
            "report_version": REPORT_VERSION,
            "mode": "llm",
            "provider": getattr(self.client, "provider", "unknown"),
            "model": self.model_settings.model,
            "structured_output": self.model_settings.structured_output.value,
            "top_k": self.top_k,
            "second_pass_top_k": self.second_pass_top_k,
            "summary": {
                "source_fields": len(self._outcomes),
                "mapped": mapped,
                "unmapped": len(self._outcomes) - mapped,
                "selected": len(proposed),
                "rejected_selections": rejected,
                "allocation_rejections": allocation_rejections,
                "compiler_rejections": len(compiler_rejections),
                "provider_calls": self._provider_calls,
                "cache_hits": self._cache_hits,
                "unusable_responses": self._unusable_responses,
            },
            "skipped_profiles": list(self._skipped_profiles),
            "fields": [outcome.as_report_dict() for outcome in self._outcomes],
            "proposed_mapping_table": proposed,
            "compiled_mapping_table": compiled,
            "mapping_table": compiled,
            "compiler_rejections": compiler_rejections,
            "compiler_diagnostics": diagnostics,
        }

    def _collect_offers(
        self,
        resource_targets: Sequence[ResourceTargets],
        source_fields: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[PooledCandidate]]:
        """Gather every offerable candidate for every source field, across profiles"""

        pooled: Dict[str, List[PooledCandidate]] = {}
        for targets in resource_targets:
            tree = _target_tree_for(targets)
            if tree is None:
                self._skipped_profiles.append(
                    {
                        "resource": targets.res_id or targets.res_type,
                        "reason": "no-target-tree",
                    }
                )
                continue

            # The third flatten keeps a candidate's description only for the entry
            # it copied; look it back up by element identity so every offer can
            # show the profile's own wording.
            descriptions = _descriptions_by_identity(targets.target_fields)

            for source_field in source_fields:
                source_id = _source_id(source_field)
                if not source_id:
                    continue
                candidates = self.automapper.find_llm_candidates(
                    source_field, targets.target_fields, tree, top_k=None
                )
                for candidate in candidates:
                    emitted = rebase_to_resource_identity(
                        candidate.mapping_target_path, targets.res_type, targets.res_id
                    )
                    if emitted:
                        identity = (
                            candidate.canonical_target_id
                            or candidate.element_id
                            or candidate.element_path
                            or ""
                        )
                        pooled.setdefault(source_id, []).append(
                            (candidate, targets, emitted, descriptions.get(identity))
                        )
        return pooled

    def _rank_offers(
        self,
        pooled: Sequence[PooledCandidate],
        limit: int,
        excluded_targets: Optional[set] = None,
    ) -> List[Offer]:
        """Order pooled candidates deterministically and enumerate the best limit"""

        def by_score(item: PooledCandidate):
            return (-item[0].scores.legacy, item[2])

        rank_one = [item for item in pooled if item[0].is_legacy_selection]
        rest = [item for item in pooled if not item[0].is_legacy_selection]
        ordered = sorted(rank_one, key=by_score) + sorted(rest, key=by_score)

        offers: List[Offer] = []
        seen_targets = set(excluded_targets or ())
        for candidate, targets, emitted, description in ordered:
            if emitted in seen_targets:
                continue
            seen_targets.add(emitted)
            offers.append(
                Offer(
                    offer_id=f"c{len(offers) + 1}",
                    candidate=candidate,
                    emitted_target=emitted,
                    res_type=targets.res_type,
                    res_id=targets.res_id,
                    description=description,
                )
            )
            if len(offers) >= limit:
                break
        return offers

    def _resolve_field(
        self,
        source_field: Dict[str, Any],
        source_id: str,
        pooled: Sequence[PooledCandidate],
        reserved_targets: set,
    ) -> FieldOutcome:
        outcome = FieldOutcome(source_id=source_id)
        if not pooled:
            return outcome

        if all(item[2] in reserved_targets for item in pooled):
            outcome.rejected_reason = "all-candidates-already-assigned"
            return outcome

        widths = [self.top_k]
        if self.second_pass_top_k and self.second_pass_top_k > self.top_k:
            widths.append(self.second_pass_top_k)

        previous_count = 0
        for pass_index, width in enumerate(widths, start=1):
            offers = self._rank_offers(pooled, width, excluded_targets=reserved_targets)
            if not offers:
                outcome.rejected_reason = "all-candidates-already-assigned"
                break
            if pass_index > 1 and len(offers) <= previous_count:
                # The wider pass found nothing new; a second identical question
                # would only cost a provider call.
                break
            previous_count = len(offers)

            attempt = self._ask(source_field, offers, pass_index)
            outcome.attempts.append(attempt)

            chosen = self._accepted_offer(attempt, offers)
            if chosen is not None:
                outcome.final_target = chosen.emitted_target
                outcome.final_offer_id = chosen.offer_id
                return outcome

        return outcome

    def _ask(
        self, source_field: Dict[str, Any], offers: Sequence[Offer], pass_index: int
    ) -> FieldAttempt:
        """One provider round. Provider failures are never swallowed."""

        attempt = FieldAttempt(pass_index=pass_index, offers=list(offers))
        messages = [
            LLMMessage.system(render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)),
            LLMMessage.user(
                render(
                    AUTOMAPPING_USER,
                    source=_source_context(source_field),
                    offers=list(offers),
                    unmapped_token=UNMAPPED,
                    pass_index=pass_index,
                    is_retry=pass_index > 1,
                )
            ),
        ]
        # Identity of the offered set, so a changed candidate list cannot reuse an
        # answer even if the rendered text happens to coincide.
        context = {
            "source_id": _source_id(source_field),
            "offers": [
                [
                    offer.offer_id,
                    offer.candidate.canonical_target_id,
                    offer.emitted_target,
                ]
                for offer in offers
            ],
        }

        try:
            response = self.client.complete(
                messages=messages,
                response_model=CandidateSelection,
                settings=self.model_settings,
                feature=FEATURE,
                context=context,
            )
        except LLMResponseValidationError as exc:
            self._provider_calls += 1
            self._unusable_responses += 1
            attempt.rejected_reason = f"unusable-response:{exc}"[:300]
            logger.warning(
                "LLM automapping discarded an unusable response for %s: %s",
                _source_id(source_field),
                exc,
            )
            return attempt

        if response.cache_hit:
            self._cache_hits += 1
        else:
            self._provider_calls += 1
        attempt.cache_hit = response.cache_hit
        attempt.selection = response.value
        return attempt

    def _accepted_offer(
        self, attempt: FieldAttempt, offers: Sequence[Offer]
    ) -> Optional[Offer]:
        """Validate the answer against the enumerated offers"""

        selection = attempt.selection
        if selection is None:
            return None

        answer = (selection.candidate_id or "").strip()
        if answer.lower() == UNMAPPED:
            return None

        by_id = {offer.offer_id: offer for offer in offers}
        chosen = by_id.get(answer)
        if chosen is None:
            attempt.rejected_reason = f"unknown-candidate-id:{answer[:64]}"
            logger.warning(
                "LLM automapping rejected an unknown candidate id %r for %s",
                answer[:64],
                _source_id_of_attempt(attempt),
            )
            return None
        return chosen

def _source_id(source_field: Dict[str, Any]) -> str:
    return source_field.get("id") or source_field.get("path", "")


def _positive_width(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LLMConfigurationError(
            f"'{name}' must be a positive integer, got {value!r}"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(
            f"'{name}' must be a positive integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise LLMConfigurationError(
            f"'{name}' must be a positive integer, got {value!r}"
        )
    return parsed


def _source_id_of_attempt(attempt: FieldAttempt) -> str:
    return attempt.offers[0].candidate.candidate_id if attempt.offers else "?"


def _target_tree_for(targets: ResourceTargets) -> Optional[TargetTree]:
    """Build the profile's target tree, or ``None`` when it cannot be built"""

    if targets.profile is None:
        return None
    try:
        from parser.resource_parser.fhir_type_introspection import (  # noqa: PLC0415
            get_complex_type_fields,
        )

        return TargetTree.from_snapshot(targets.profile, get_complex_type_fields)
    except Exception as exc:  # defensive: a malformed snapshot must not abort the run
        logger.debug(
            "target tree unavailable for %s (%s: %s)",
            targets.res_id or targets.res_type,
            type(exc).__name__,
            exc,
        )
        return None


def _source_context(source_field: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a source field into the values the prompt template renders"""

    source_type = source_field.get("type")
    minimum, maximum = _cardinality_of(source_field)
    return {
        "id": _source_id(source_field),
        "path": source_field.get("path"),
        "type": source_type if isinstance(source_type, str) else None,
        "description": source_field.get("description") or source_field.get("short"),
        "cardinality": (
            f"{minimum}..{maximum}"
            if minimum is not None or maximum is not None
            else None
        ),
    }


def _descriptions_by_identity(
    target_fields: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    """Map element id *and* path to the field's description, where it has one"""

    out: Dict[str, str] = {}

    def walk(fields):
        for field in fields or ():
            if not isinstance(field, dict):
                continue
            description = field.get("description") or field.get("short")
            if description:
                for key in (field.get("id"), field.get("path")):
                    if key:
                        out.setdefault(str(key), str(description))
            walk(field.get("children"))
            walk(field.get("type_structure"))

    walk(target_fields)
    return out


def _cardinality_of(field: Dict[str, Any]):
    card = field.get("cardinality")
    if isinstance(card, dict):
        return card.get("min"), card.get("max")
    return field.get("min"), field.get("max")
