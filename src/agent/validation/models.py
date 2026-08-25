"""definitions of findings/validations of agent-based fix problems"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

VALIDATION_CONTRACT_VERSION = 2


class Producer(str, Enum):
    """track component which observed a finding"""

    GENERATOR = "generator"
    FHIR_MODEL = "fhir-model"
    MAP_SEMANTICS = "map-semantics"
    PATH_RESOLUTION = "path-resolution"
    COVERAGE = "coverage"
    ENGINE = "engine"


class Stage(str, Enum):
    """where in the pipeline a finding was produced."""

    GENERATION = "generation"
    PARSE = "parse"
    SEMANTICS = "semantics"
    PATHS = "paths"
    COVERAGE = "coverage"
    BOOTSTRAP = "bootstrap"
    UPLOAD = "upload"
    TRANSFORM = "transform"
    VALIDATE = "validate"


class ActionOwner(str, Enum):
    """who has to act next"""

    MAP_FIXABLE = "map-fixable"
    MAPPING_INPUT_REQUIRED = "mapping-input-required"
    ENVIRONMENT = "environment"
    ADVISORY = "advisory"
    UNCLASSIFIED = "unclassified"


class GateStatus(str, Enum):
    """whether a finding participates in acceptance"""

    BLOCKING = "blocking"
    NON_BLOCKING = "non-blocking"


CLASSIFICATION: Dict[Tuple[Producer, str], Tuple[ActionOwner, GateStatus]] = {
    (Producer.FHIR_MODEL, "resource-invalid"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.FHIR_MODEL, "not-a-structure-map"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.MAP_SEMANTICS, "semantic-rule-invalid"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.PATH_RESOLUTION, "target-path-not-found"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.PATH_RESOLUTION, "target-path-ambiguous"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.PATH_RESOLUTION, "target-choice-type-not-allowed"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.PATH_RESOLUTION, "target-path-prohibited"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.PATH_RESOLUTION, "source-path-not-found"): (
        ActionOwner.MAPPING_INPUT_REQUIRED,
        GateStatus.BLOCKING,
    ),
    (Producer.COVERAGE, "required-path-unmapped"): (
        ActionOwner.MAPPING_INPUT_REQUIRED,
        GateStatus.BLOCKING,
    ),
    (Producer.COVERAGE, "mapping-obligation-dropped"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.COVERAGE, "required-path-deferred"): (
        ActionOwner.ADVISORY,
        GateStatus.NON_BLOCKING,
    ),
   (Producer.COVERAGE, "cross-map-reference-unsatisfied"): (
        ActionOwner.MAPPING_INPUT_REQUIRED,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "output-required-deferred"): (
        ActionOwner.ADVISORY,
        GateStatus.NON_BLOCKING,
    ),
    (Producer.ENGINE, "engine-unreachable"): (
        ActionOwner.ENVIRONMENT,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "dependency-missing"): (
        ActionOwner.ENVIRONMENT,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "upload:invalid"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "upload:structure"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "upload:processing"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "upload:not-supported"): (
        ActionOwner.ENVIRONMENT,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "transform:exception"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "transform:processing"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "transform:structure"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "transform:invalid"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "transform:not-found"): (
        ActionOwner.ENVIRONMENT,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "transform:no-output"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:required"): (
        ActionOwner.MAPPING_INPUT_REQUIRED,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:structure"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:value"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:invariant"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:code-invalid"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:business-rule"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:not-found"): (
        ActionOwner.ENVIRONMENT,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:invalid"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:processing"): (
        ActionOwner.MAP_FIXABLE,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate-unavailable"): (
        ActionOwner.ENVIRONMENT,
        GateStatus.BLOCKING,
    ),
    (Producer.ENGINE, "validate:informational"): (
        ActionOwner.ADVISORY,
        GateStatus.NON_BLOCKING,
    ),
}

UNKNOWN_CLASSIFICATION = (ActionOwner.UNCLASSIFIED, GateStatus.BLOCKING)


def classify(producer: Producer, code: str) -> Tuple[ActionOwner, GateStatus]:
    """Return the reviewed ``(owner, gate)`` for *code*, or the unknown default.

    :param producer: the component reporting the finding.
    :param code: the normalized, structured code — never a message.
    """

    return CLASSIFICATION.get((producer, code), UNKNOWN_CLASSIFICATION)


def finding_identity(
    producer: Producer,
    stage: Stage,
    code: str,
    *,
    path: Optional[str] = None,
    profile_url: Optional[str] = None,
    fixture_id: Optional[str] = None,
    discriminator: Optional[str] = None,
) -> str:
    """build stable IDS to link reports for comparison"""

    payload = json.dumps(
        {
            "producer": producer.value,
            "stage": stage.value,
            "code": code,
            "path": path,
            "profile": profile_url,
            "fixture": fixture_id,
            "discriminator": discriminator,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"vf-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


class ValidationFinding(BaseModel):
    """One piece of evidence about one map, from one producer."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    producer: Producer
    stage: Stage
    code: str
    message: str

    severity: Optional[str] = None
    issue_code: Optional[str] = None
    map_url: Optional[str] = None
    map_id: Optional[str] = None
    profile_url: Optional[str] = None
    path: Optional[str] = None
    
    pointer: Optional[str] = None
    fixture_id: Optional[str] = None
    
    discriminator: Optional[str] = None
    action_owner: ActionOwner
    gate_status: GateStatus
    
    demoted_by: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        producer: Producer,
        stage: Stage,
        code: str,
        message: str,
        *,
        owner: Optional[ActionOwner] = None,
        gate: Optional[GateStatus] = None,
        **fields: Any,
    ) -> "ValidationFinding":
        """create finding, classify unless producer already claims classification (see link/comparison)"""

        table_owner, table_gate = classify(producer, code)
        resolved_owner = owner or table_owner
        resolved_gate = gate or table_gate
        return cls(
            finding_id=finding_identity(
                producer,
                stage,
                code,
                path=fields.get("path"),
                profile_url=fields.get("profile_url"),
                fixture_id=fields.get("fixture_id"),
                discriminator=fields.get("discriminator"),
            ),
            producer=producer,
            stage=stage,
            code=code,
            message=message,
            action_owner=resolved_owner,
            gate_status=resolved_gate,
            **fields,
        )

    def demote(self, reason: str) -> "ValidationFinding":
        """return non-blocking copy, include reason"""

        return self.model_copy(
            update={"gate_status": GateStatus.NON_BLOCKING, "demoted_by": reason}
        )

    @property
    def blocking(self) -> bool:
        return self.gate_status is GateStatus.BLOCKING


class ValidationReport(BaseModel):
    """Everything known about one revision of one map."""

    model_config = ConfigDict(extra="forbid")

    contract_version: int = VALIDATION_CONTRACT_VERSION
    map_url: Optional[str] = None
    map_id: Optional[str] = None

    map_sha256: Optional[str] = None
    profile_url: Optional[str] = None
    findings: List[ValidationFinding] = Field(default_factory=list)
    
    covered_required_paths: List[str] = Field(default_factory=list)
    satisfied_obligations: List[str] = Field(default_factory=list)
    
    deferred_reference_paths: List[str] = Field(default_factory=list)
    
    executed_fixtures: List[str] = Field(default_factory=list)
    
    required_fixtures: List[str] = Field(default_factory=list)
    
    validated_fixtures: List[str] = Field(default_factory=list)
    
    validation_expected_fixtures: List[str] = Field(default_factory=list)
    
    evaluation_context_sha256: Optional[str] = None
    engine_available: bool = False
    engine_requested: bool = True

    @property
    def blocking_findings(self) -> List[ValidationFinding]:
        return [finding for finding in self.findings if finding.blocking]

    def by_owner(self, owner: ActionOwner) -> List[ValidationFinding]:
        return [finding for finding in self.findings if finding.action_owner is owner]

    def worklist(self) -> List[ValidationFinding]:
        """compile list of findings the agent is allowed to attemp (ordered by most constrained)"""

        return sorted(
            (
                finding
                for finding in self.findings
                if finding.action_owner is ActionOwner.MAP_FIXABLE
                and finding.blocking
            ),
            key=lambda finding: (finding.code, finding.path or ""),
        )

    def finding_ids(self) -> Set[str]:
        return {finding.finding_id for finding in self.findings}

    def blocking_ids(self) -> Set[str]:
        return {finding.finding_id for finding in self.blocking_findings}

    def as_report_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
