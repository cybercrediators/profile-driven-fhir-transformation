"""define typed contracts for the Structuremap diagnostics"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fhir.resources.R4B.structuremap import StructureMap
from pydantic import BaseModel, ConfigDict, Field

COVERAGE_REPORT_VERSION = 3


class DiagnosticActionability(str, Enum):
    """Deterministic owner of the next action for a generator diagnostic."""

    MAP_FIXABLE = "map-fixable"
    MAPPING_INPUT_REQUIRED = "mapping-input-required"
    ENVIRONMENT = "environment"
    ADVISORY = "advisory"


# explicit(!) registry of possible actionability items for possible agent fix inference
DIAGNOSTIC_ACTIONABILITY: Dict[str, DiagnosticActionability] = {
    # A generated map can contain the necessary source mapping while lacking the
    # target context needed to emit it.  This is the one current generator finding
    # for which a post-generation map correction can be appropriate.
    "unmaterialized-nested-target": DiagnosticActionability.MAP_FIXABLE,
    # The authored mapping/source contract must change or become more specific.
    "ambiguous-choice-type": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "collection-source-key-not-found": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "collection-source-not-repeating": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "collection-target-key-not-mapped": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "collection-target-not-repeating": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "contained-requires-typed-reference": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "derived-resource-ambiguous-target": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "derived-resource-not-a-resource": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "derived-resource-value-not-inferable": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "duplicate-target-assignment": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    # Two rules write one element with different values (typically a
    # profile-fixed code and a mapped source field) Which one is wanted is a
    # decision about the source contract, not one the generator may make.
    "duplicate-rule-name-conflict": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "invalid-collection-rule": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "invalid-typed-rule": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "mapping-source-path-ambiguous": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "mapping-source-path-not-found": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "mapping-target-choice-ambiguous": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "mapping-target-path-not-found": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "missing-discriminator-provider": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "modifier-extension-requires-slice": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "reference-representation-requires-source": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "required-extension-container-provider-missing": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "target-modifier-element": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "unresolved-map-placeholder": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "unsliced-provider-for-closed-slicing": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "unsupported-choice-type": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    "url-only-extension-pruned": DiagnosticActionability.MAPPING_INPUT_REQUIRED,
    # These require a profile, parser/specification, terminology, or canonical
    # setup change rather than a speculative StructureMap edit.
    "derived-resource-release-mismatch": DiagnosticActionability.ENVIRONMENT,
    "derived-resource-unresolvable": DiagnosticActionability.ENVIRONMENT,
    # A group or rule name past FHIR's 64-character id limit. It is derived from
    # a profile or element identifier the tool does not own, and shortening it
    # here would break any rule that references it by name.
    "map-token-too-long": DiagnosticActionability.ENVIRONMENT,
    "profile-canonical-conflict": DiagnosticActionability.ENVIRONMENT,
    "snapshot-slices-not-indexed": DiagnosticActionability.ENVIRONMENT,
    "unresolved-content-reference": DiagnosticActionability.ENVIRONMENT,
    "unsupported-slicing-rules": DiagnosticActionability.ENVIRONMENT,
    # Informational constraints and deliberate generator behavior.
    "deferred-descendant-slice": DiagnosticActionability.ADVISORY,
    "deferred-reference": DiagnosticActionability.ADVISORY,
    "misplaced-descendant-slice": DiagnosticActionability.ADVISORY,
    "redundant-parent-slice-suppressed": DiagnosticActionability.ADVISORY,
    "target-condition": DiagnosticActionability.ADVISORY,
    "target-default-value": DiagnosticActionability.ADVISORY,
    "target-fhirpath-constraint": DiagnosticActionability.ADVISORY,
    "target-max-length": DiagnosticActionability.ADVISORY,
    "target-max-value": DiagnosticActionability.ADVISORY,
    "target-min-value": DiagnosticActionability.ADVISORY,
    "xhtml-content-authored": DiagnosticActionability.ADVISORY,
}


def classify_diagnostic(code: str) -> DiagnosticActionability:
    """Return the reviewed actionability for *code* without heuristic inference."""

    return DIAGNOSTIC_ACTIONABILITY.get(code, DiagnosticActionability.ADVISORY)


def diagnostic_id(raw: Mapping[str, Any]) -> str:
    """Build a stable ID from the diagnostic code and its semantic location."""

    ignored = {"message", "severity", "diagnostic_id", "actionability"}
    identity = {
        key: value
        for key, value in raw.items()
        if key not in ignored and value is not None
    }
    identity["code"] = str(raw.get("code", "unknown"))
    if len(identity) == 1:
        # Some diagnostics have no structured locator.  The message is the best
        # available discriminator, but is intentionally only a fallback so routine
        # wording changes do not alter well-located IDs.
        identity["message"] = raw.get("message", "")
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"diag-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


class MappingDiagnostic(BaseModel):
    """Versioned public view of one raw generator diagnostic."""

    model_config = ConfigDict(extra="allow")

    code: str
    message: Optional[str] = None
    profile: Optional[str] = None
    severity: Optional[str] = None
    diagnostic_id: str
    actionability: DiagnosticActionability

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "MappingDiagnostic":
        value = dict(raw)
        code = str(value.get("code", "unknown"))
        # Recompute both fields rather than trusting data supplied by an external
        # report or, later, an LLM response.
        value["code"] = code
        value["diagnostic_id"] = diagnostic_id(value)
        value["actionability"] = classify_diagnostic(code)
        return cls.model_validate(value)


class CoverageSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    profiles: int
    required_total: int
    required_mapped: int
    required_unmapped: int
    static_required_total: int
    latent_required_total: int
    required_coverage_pct: float


class RequirementManifestEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    path: Optional[str] = None
    min: int = 0
    max: Optional[Any] = None
    active: bool
    provider: str
    status: str


class ProfileCoverage(BaseModel):
    model_config = ConfigDict(extra="allow")

    resource_type: str
    required_total: int
    required_mapped: int
    required_unmapped: int
    unmapped_required_paths: List[str] = Field(default_factory=list)
    static_required_total: int
    latent_required_total: int
    latent_required_paths: List[str] = Field(default_factory=list)
    requirement_manifest: List[RequirementManifestEntry] = Field(default_factory=list)


class CoverageReport(BaseModel):
    """The in-memory and persisted StructureMap coverage-report contract."""

    model_config = ConfigDict(extra="allow")

    report_version: int = COVERAGE_REPORT_VERSION
    map: Optional[str] = None
    note: str
    summary: CoverageSummary
    profiles: Dict[str, ProfileCoverage] = Field(default_factory=dict)
    mapping_diagnostics: List[MappingDiagnostic] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "CoverageReport":
        value = dict(raw)
        value["mapping_diagnostics"] = [
            MappingDiagnostic.from_raw(item)
            for item in value.get("mapping_diagnostics", [])
        ]
        return cls.model_validate(value)


@dataclass(frozen=True)
class GenerationArtifactPaths:
    """Filesystem artifacts written or loaded for one generation result."""

    structure_maps: tuple[Path, ...] = ()
    coverage_report: Optional[Path] = None


@dataclass(frozen=True)
class StructureMapGenerationResult:
    """Typed result returned alongside the legacy list-only API."""

    structure_maps: tuple[StructureMap, ...]
    coverage_report: Optional[CoverageReport]
    artifacts: GenerationArtifactPaths

    @property
    def mapping_diagnostics(self) -> tuple[MappingDiagnostic, ...]:
        if self.coverage_report is None:
            return ()
        return tuple(self.coverage_report.mapping_diagnostics)

    def maps(self) -> List[StructureMap]:
        """Return a mutable list for compatibility with existing callers."""

        return list(self.structure_maps)
