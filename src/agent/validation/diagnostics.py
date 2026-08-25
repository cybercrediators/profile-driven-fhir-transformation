"""handling generator diagnostics and merge with validation findings"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from agent.validation.models import ActionOwner, GateStatus, Producer, Stage, ValidationFinding


_GENERATOR_GATE: Dict[str, GateStatus] = {
    "map-fixable": GateStatus.BLOCKING,
    "mapping-input-required": GateStatus.BLOCKING,
    "environment": GateStatus.BLOCKING,
    "advisory": GateStatus.NON_BLOCKING,
}


def _diagnostic_belongs_to(declared: Optional[str], wanted: Set[str]) -> bool:
    """decide if profile names are one of the wanted ones"""

    def leaf(value: str) -> str:
        return value.split("|", 1)[0].rsplit("/", 1)[-1]

    if not declared:
        return True
    return leaf(str(declared)) in {leaf(item) for item in wanted}


def findings_from_diagnostics(
    diagnostics: Iterable[Any],
    *,
    map_url: Optional[str] = None,
    map_id: Optional[str] = None,
    profile: Optional[Set[str]] = None,
) -> List[ValidationFinding]:
    """create shared view of map generation diagnostics and validation

    :param diagnostics: ``MappingDiagnostic`` models or raw diagnostic mappings.
    :param profile: identities of the map's own target profile — its ``id`` and
        canonical URL. A diagnostic naming a *different* profile is dropped.
    """

    from mapping.generation_result import MappingDiagnostic  # noqa: PLC0415

    wanted = {str(item) for item in (profile or set()) if item}
    findings: List[ValidationFinding] = []
    for raw in diagnostics or []:
        diagnostic = (
            raw
            if isinstance(raw, MappingDiagnostic)
            else MappingDiagnostic.from_raw(raw)
        )
        if wanted and not _diagnostic_belongs_to(diagnostic.profile, wanted):
            continue
        owner = ActionOwner(diagnostic.actionability.value)
        extra = diagnostic.model_dump(
            exclude={
                "code",
                "message",
                "severity",
                "diagnostic_id",
                "actionability",
                "profile",
            },
            exclude_none=True,
        )
        findings.append(
            ValidationFinding(
                finding_id=diagnostic.diagnostic_id,
                producer=Producer.GENERATOR,
                stage=Stage.GENERATION,
                code=diagnostic.code,
                message=diagnostic.message or diagnostic.code,
                severity=diagnostic.severity,
                map_url=map_url,
                map_id=map_id,
                profile_url=diagnostic.profile,
                path=extra.get("path") or extra.get("target"),
                action_owner=owner,
                gate_status=_GENERATOR_GATE.get(
                    diagnostic.actionability.value, GateStatus.BLOCKING
                ),
                evidence=extra,
            )
        )
    return findings
