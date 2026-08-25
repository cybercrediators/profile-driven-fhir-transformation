"""acceptance for validation scores"""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from agent.validation.models import ActionOwner, Producer, Stage, ValidationFinding, ValidationReport
from agent.validation.paths import resolve_tree_node


logger = logging.getLogger(__name__)

def merge_engine_result(report: ValidationReport, result) -> ValidationReport:
    """merge given EngineResult into readable offline ValidationReport

    :param report: the result of :func:`validate_offline`. Modified in place.
    :param result: what the engine session returned for the same document.
    :return: the same report, for chaining.
    """

    report.findings.extend(result.findings)
    report.executed_fixtures = list(result.executed_fixtures)
    report.required_fixtures = list(result.required_fixtures)
    report.validated_fixtures = list(result.validated_fixtures)
    report.validation_expected_fixtures = list(
        result.validation_expected_fixtures
    )
    report.evaluation_context_sha256 = result.evaluation_context_sha256
    report.engine_requested = True
    report.engine_available = not any(
        finding.code == "engine-unreachable" for finding in result.findings
    )
    return report


class InvariantResult(BaseModel):
    """One acceptance invariant and what it saw."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    comparative: bool
    detail: str
    evidence: Dict[str, Any] = Field(default_factory=dict)


class AcceptanceDecision(BaseModel):
    """Whether a candidate may replace the baseline, and why."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    invariants: List[InvariantResult] = Field(default_factory=list)
    new_blocking_ids: List[str] = Field(default_factory=list)
    
    resolved_blocking_ids: List[str] = Field(default_factory=list)
    
    advisory_regressions: List[str] = Field(default_factory=list)

    @property
    def failures(self) -> List[InvariantResult]:
        return [item for item in self.invariants if not item.passed]

    @property
    def excused_evidence(self) -> List[str]:
        """defined invariants passed only because the environment was excused"""

        return [
            item.name
            for item in self.invariants
            if item.passed
            and (
                item.evidence.get("forgiven_environment_finding_ids")
                or item.evidence.get("excused_environment_fixtures")
            )
        ]

    @property
    def provisional(self) -> bool:
        """True when acceptance rested on excused, rather than obtained, evidence."""

        return self.accepted and bool(self.excused_evidence)

    def as_report_dict(self) -> Dict[str, Any]:
        return {
            **self.model_dump(mode="json"),
            "provisional": self.provisional,
            "excused_evidence": self.excused_evidence,
        }


def literal_target_values(
    document: Mapping[str, Any],
) -> Set[Tuple[str, str]]:
    """define (target path, JSON enc. value) for literals a map assigns"""

    found: Set[Tuple[str, str]] = set()

    def walk(rules: Any, target_vars: Dict[str, str]) -> None:
        for rule in rules or []:
            if not isinstance(rule, Mapping):
                continue
            local = dict(target_vars)
            for entry in rule.get("target") or []:
                if not isinstance(entry, Mapping):
                    continue
                parent = local.get(str(entry.get("context") or ""))
                element = entry.get("element")
                if parent is None or not element:
                    continue
                path = f"{parent}.{element}" if parent else str(element)
                parameters = [
                    p for p in (entry.get("parameter") or []) if isinstance(p, Mapping)
                ]
                if (
                    entry.get("transform") == "copy"
                    and parameters
                    and not any("valueId" in p for p in parameters)
                ):
                    value = next(iter(parameters[0].values()), None)
                    found.add((path, json.dumps(value, sort_keys=True, default=str)))
                if entry.get("variable"):
                    local[str(entry["variable"])] = path
            walk(rule.get("rule"), local)

    for group in document.get("group") or []:
        if not isinstance(group, Mapping):
            continue
        target_vars = {
            str(entry["name"]): str(entry.get("type") or "")
            for entry in group.get("input") or []
            if isinstance(entry, Mapping)
            and entry.get("name")
            and entry.get("mode") == "target"
        }
        walk(group.get("rule"), target_vars)
    return found


def decide_acceptance(
    baseline: ValidationReport,
    candidate: ValidationReport,
    *,
    application=None,
    require_engine: bool = True,
    baseline_document: Optional[Mapping[str, Any]] = None,
    candidate_document: Optional[Mapping[str, Any]] = None,
    target_tree=None,
) -> AcceptanceDecision:
    """Apply invariants to a candidate for deciding on possible patch acceptance

    :param baseline: report for the document the candidate would replace.
    :param candidate: report for the proposed document.
    :param application: the WP5 :class:`agent.models.PatchApplication`, when the
        candidate came from a patch. Its policy verdict is an invariant.
    :param baseline_document: the map as it stands. Together with
        *candidate_document* it enables the invented-value invariant, which no
        report can express because it compares what the two revisions *write*
        rather than what validating them found. Omitting either skips it.
    :param candidate_document: the proposed map.
    :param target_tree: the target profile's tree, so a constant the profile
        itself fixes is not counted as invented. Omitting it makes the
        invented-value invariant stricter, never looser.
    :param require_engine: when True, an unreachable engine fails acceptance.
        An explicitly offline diagnostic run may pass False — and must then not
        apply what it produced.
    """

    invariants: List[InvariantResult] = []

    def record(name, passed, comparative, detail, **evidence):
        invariants.append(
            InvariantResult(
                name=name,
                passed=bool(passed),
                comparative=comparative,
                detail=detail,
                evidence=evidence,
            )
        )

    parse_failures = [
        finding
        for finding in candidate.findings
        if finding.producer is Producer.FHIR_MODEL
    ]
    record(
        "candidate-parses",
        not parse_failures,
        False,
        "The candidate parses as an R4B StructureMap."
        if not parse_failures
        else f"The candidate does not parse ({len(parse_failures)} finding(s)).",
        finding_ids=[finding.finding_id for finding in parse_failures],
    )

    semantic_failures = [
        finding
        for finding in candidate.findings
        if finding.producer is Producer.MAP_SEMANTICS and finding.blocking
    ]
    record(
        "candidate-semantically-valid",
        not semantic_failures,
        False,
        "No blocking semantic errors."
        if not semantic_failures
        else f"{len(semantic_failures)} blocking semantic error(s).",
        finding_ids=[finding.finding_id for finding in semantic_failures],
    )

    if application is not None:
        record(
            "patch-policy-passed",
            getattr(application, "applied", False),
            False,
            "The patch passed every WP5 policy check."
            if getattr(application, "applied", False)
            else "The patch was rejected by WP5 policy.",
            rejections=[
                rejection.code for rejection in getattr(application, "rejections", [])
            ],
        )

    if require_engine:
        record(
            "engine-available",
            candidate.engine_requested and candidate.engine_available,
            False,
            "Matchbox executed the candidate."
            if candidate.engine_available
            else "Matchbox was not reachable, which is an environment failure "
            "and not evidence that the candidate is valid.",
            engine_requested=candidate.engine_requested,
        )
        def execution_failures_of(report: ValidationReport) -> List[ValidationFinding]:
            return [
                finding
                for finding in report.findings
                if finding.producer is Producer.ENGINE
                and finding.stage in (Stage.UPLOAD, Stage.TRANSFORM)
                and finding.blocking
            ]

        forgiven = {
            finding.finding_id
            for finding in execution_failures_of(baseline)
            if finding.action_owner is ActionOwner.ENVIRONMENT
        }
        execution_failures = execution_failures_of(candidate)
        new_execution_failures = [
            finding
            for finding in execution_failures
            if finding.finding_id not in forgiven
        ]
        newly_missing = sorted(
            (set(candidate.required_fixtures) - set(candidate.executed_fixtures))
            - (set(baseline.required_fixtures) - set(baseline.executed_fixtures))
        )
        record(
            "engine-executes-required-fixtures",
            candidate.engine_available
            and bool(candidate.required_fixtures)
            and not newly_missing
            and not new_execution_failures,
            True,
            "No required fixture stopped transforming."
            if candidate.engine_available
            and candidate.required_fixtures
            and not newly_missing
            and not new_execution_failures
            else (
                f"{len(new_execution_failures)} new blocking execution failure(s); "
                f"{len(newly_missing)} required fixture(s) stopped running."
            ),
            finding_ids=[finding.finding_id for finding in new_execution_failures],
            forgiven_environment_finding_ids=sorted(
                finding.finding_id
                for finding in execution_failures
                if finding.finding_id in forgiven
            ),
            missing_fixtures=newly_missing,
            required=list(candidate.required_fixtures),
        )
        record(
            "engine-evidence-present",
            bool(candidate.required_fixtures)
            and set(candidate.required_fixtures).issubset(
                candidate.executed_fixtures
            ),
            False,
            f"{len(candidate.executed_fixtures)} fixture(s) reached the engine."
            if candidate.executed_fixtures
            else "No fixture reached the engine, so there is no execution "
            "evidence for this candidate at all.",
            executed=list(candidate.executed_fixtures),
            required=list(candidate.required_fixtures),
        )

        if candidate.profile_url:
            excused_fixtures = {
                finding.fixture_id
                for finding in execution_failures_of(baseline)
                if finding.action_owner is ActionOwner.ENVIRONMENT and finding.fixture_id
            } & (
                set(baseline.validation_expected_fixtures)
                - set(baseline.validated_fixtures)
            )
            missing_validate_evidence = sorted(
                set(candidate.validation_expected_fixtures)
                - set(candidate.validated_fixtures)
            )
            newly_missing_evidence = sorted(
                set(missing_validate_evidence) - excused_fixtures
            )
            record(
                "validate-evidence-present",
                bool(candidate.validation_expected_fixtures)
                and not newly_missing_evidence,
                True,
                "Every required output completed $validate."
                if candidate.validation_expected_fixtures
                and not missing_validate_evidence
                else "No output lost $validate evidence the baseline had."
                if candidate.validation_expected_fixtures
                and not newly_missing_evidence
                else "A target profile is known, but required $validate evidence "
                "is missing.",
                expected=list(candidate.validation_expected_fixtures),
                validated=list(candidate.validated_fixtures),
                missing=newly_missing_evidence,
                excused_environment_fixtures=sorted(
                    set(missing_validate_evidence) & excused_fixtures
                ),
            )

    same_context = (not require_engine) or (
        bool(baseline.evaluation_context_sha256)
        and baseline.evaluation_context_sha256
        == candidate.evaluation_context_sha256
    )
    record(
        "engine-evaluation-context-parity",
        same_context,
        True,
        "Both revisions used the identical engine service and validation inputs."
        if same_context
        else "Baseline and candidate engine evidence came from different or "
        "unrecorded services, fixtures, profiles, or validation settings.",
        baseline=baseline.evaluation_context_sha256,
        candidate=candidate.evaluation_context_sha256,
    )

    required_parity = sorted(
        set(baseline.required_fixtures) ^ set(candidate.required_fixtures)
    )
    record(
        "required-fixture-parity",
        not required_parity,
        True,
        "Both revisions had the same gate-relevant fixtures."
        if not required_parity
        else "The gate-relevant fixture set differs between revisions.",
        fixtures=required_parity,
    )

    validation_expectation_parity = sorted(
        set(baseline.validation_expected_fixtures)
        ^ set(candidate.validation_expected_fixtures)
    )
    record(
        "validate-expectation-parity",
        not validation_expectation_parity,
        True,
        "Both revisions required conformance evidence for the same outputs."
        if not validation_expectation_parity
        else "The set of outputs expected to complete $validate differs.",
        fixtures=validation_expectation_parity,
    )

    missing_execution = sorted(
        set(baseline.executed_fixtures) ^ set(candidate.executed_fixtures)
    )
    record(
        "engine-fixture-parity",
        not missing_execution,
        True,
        "Both revisions were transformed with the same fixtures."
        if not missing_execution
        else f"{len(missing_execution)} fixture id(s) differ between the two runs.",
        fixtures=missing_execution,
    )
    missing_validation = sorted(
        set(baseline.validated_fixtures) - set(candidate.validated_fixtures)
    )
    record(
        "validate-coverage-parity",
        not missing_validation,
        True,
        "The candidate was validated on at least the outputs the baseline was."
        if not missing_validation
        else f"{len(missing_validation)} output(s) the baseline had validated "
        "were not validated for the candidate, so 'no new $validate error' "
        "would only mean the check did not run.",
        fixtures=missing_validation,
    )

    baseline_blocking = baseline.blocking_ids()
    candidate_blocking = candidate.blocking_ids()
    new_blocking = sorted(candidate_blocking - baseline_blocking)
    resolved = sorted(baseline_blocking - candidate_blocking)
    record(
        "no-new-blocking-findings",
        not new_blocking,
        True,
        "The candidate introduces no blocking finding."
        if not new_blocking
        else f"The candidate introduces {len(new_blocking)} blocking finding(s).",
        finding_ids=new_blocking,
    )

    lost_required = sorted(
        set(baseline.covered_required_paths) - set(candidate.covered_required_paths)
    )
    record(
        "no-required-coverage-loss",
        not lost_required,
        True,
        "Required-path coverage is preserved."
        if not lost_required
        else f"{len(lost_required)} required path(s) are no longer emitted.",
        paths=lost_required,
    )

    lost_obligations = sorted(
        set(baseline.satisfied_obligations) - set(candidate.satisfied_obligations)
    )
    record(
        "no-obligation-loss",
        not lost_obligations,
        True,
        "Every mapping-table obligation the baseline met is still met."
        if not lost_obligations
        else f"{len(lost_obligations)} mapping-table obligation(s) dropped.",
        paths=lost_obligations,
    )

    def validate_errors(report: ValidationReport) -> Set[str]:
        return {
            finding.finding_id
            for finding in report.findings
            if finding.producer is Producer.ENGINE
            and finding.stage is Stage.VALIDATE
            and finding.blocking
        }

    if baseline_document is not None and candidate_document is not None:
        added = (
            literal_target_values(candidate_document)
            - literal_target_values(baseline_document)
        )
        def profile_supplies(path: str) -> bool:
            if target_tree is None:
                return False
            node = resolve_tree_node(target_tree, path)
            return node is not None and getattr(node, "fixed_value", None) is not None

        invented = sorted(
            (path, value) for path, value in added if not profile_supplies(path)
        )
        record(
            "no-invented-target-value",
            not invented,
            True,
            "The candidate writes no constant the baseline did not."
            if not invented
            else (
                f"The candidate writes {len(invented)} constant(s) with no source "
                "field and no basis in the baseline: "
                + ", ".join(f"{path}={value}" for path, value in invented[:5])
            ),
            values=[{"path": path, "value": value} for path, value in invented],
        )

    new_validate = sorted(validate_errors(candidate) - validate_errors(baseline))
    record(
        "no-validate-regression",
        not new_validate,
        True,
        "No new target-profile $validate error."
        if not new_validate
        else f"{len(new_validate)} new $validate error(s).",
        finding_ids=new_validate,
    )

    advisory_regressions = sorted(
        {
            finding.finding_id
            for finding in candidate.findings
            if not finding.blocking
        }
        - {finding.finding_id for finding in baseline.findings}
    )

    decision = AcceptanceDecision(
        accepted=all(item.passed for item in invariants),
        invariants=invariants,
        new_blocking_ids=new_blocking,
        resolved_blocking_ids=resolved,
        advisory_regressions=advisory_regressions,
    )
    if not decision.accepted:
        logger.info(
            "Candidate rejected by %s.",
            ", ".join(item.name for item in decision.failures),
        )
    return decision
