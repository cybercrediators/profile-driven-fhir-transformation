"""define 'offline' validation"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from agent.validation.models import Producer, Stage, ValidationFinding, ValidationReport
from agent.validation.paths import addressed_paths, normalize_target_path
from agent.validation.resolution import RESOLUTION_CHOICE_UNNARROWED, _path_findings, _resolve_target_path
from agent.validation.references import deferred_reference_paths
from agent.validation.coverage import _coverage_findings, authored_selector_obligations



def _parse_structure_map(document: Mapping[str, Any]):
    """returns (model, findings), None when invalid"""

    from fhir.resources.R4B.structuremap import StructureMap  # noqa: PLC0415

    map_url = document.get("url")
    map_id = document.get("id")
    if document.get("resourceType") != "StructureMap":
        return None, [
            ValidationFinding.build(
                Producer.FHIR_MODEL,
                Stage.PARSE,
                "not-a-structure-map",
                "The document is not a StructureMap: resourceType is "
                f"{document.get('resourceType')!r}.",
                map_url=map_url,
                map_id=map_id,
            )
        ]
    try:
        return StructureMap.model_validate(dict(document)), []
    except Exception as exc:  # pydantic ValidationError and anything it raises
        return None, [
            ValidationFinding.build(
                Producer.FHIR_MODEL,
                Stage.PARSE,
                "resource-invalid",
                f"The map does not parse as an R4B StructureMap: {exc}",
                map_url=map_url,
                map_id=map_id,
                evidence={"error_type": type(exc).__name__},
            )
        ]


def _semantic_findings(structure_map, map_url, map_id) -> List[ValidationFinding]:
    """Variable scoping, transform arity, group signatures."""

    from mapping.rule_ir import validate_structure_map_semantics  # noqa: PLC0415

    findings = []
    for problem in validate_structure_map_semantics(structure_map):
        findings.append(
            ValidationFinding.build(
                Producer.MAP_SEMANTICS,
                Stage.SEMANTICS,
                "semantic-rule-invalid",
                problem["message"],
                map_url=map_url,
                map_id=map_id,
                pointer=problem["pointer"],
                path=problem.get("group"),
                evidence={"group": problem.get("group")},
            )
        )
    return findings


def build_target_tree(profile_sd, *, introspect: bool = True):
    """build tree, expanded into child datatypes

    :param profile_sd: the target profile ``StructureDefinition``.
    :param introspect: expand complex datatypes. Costs about a second per
        profile on first use and is cached per datatype thereafter; pass False
        only when that is genuinely too expensive.
    """

    from mapping.target_tree import TargetTree  # noqa: PLC0415

    if not introspect:
        return TargetTree.from_snapshot(profile_sd)
    from parser.resource_parser.fhir_type_introspection import (  # noqa: PLC0415
        get_complex_type_fields,
    )

    return TargetTree.from_snapshot(profile_sd, get_complex_type_fields)


def validate_offline(
    document: Mapping[str, Any],
    *,
    target_tree=None,
    mapping_table: Optional[Mapping[str, Any]] = None,
    source_fields: Optional[Iterable[str]] = None,
    profile_url: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> ValidationReport:
    """run defined validations against the given document

    :param document: the StructureMap as plain JSON.
    :param target_tree: :class:`mapping.target_tree.TargetTree` for the target
    :param mapping_table: the project's ``{source field: target path}`` table.
    :param source_fields: leaf names the source logical model declares.
    :param profile_url: canonical URL of the target profile, for the report.
    :param profile_id: the target profile's ``id``
    :return: the compiled report
    """

    from agent.patch import canonical_sha256  # noqa: PLC0415

    map_url = document.get("url")
    map_id = document.get("id")
    report = ValidationReport(
        map_url=map_url,
        map_id=map_id,
        map_sha256=canonical_sha256(dict(document)),
        profile_url=profile_url,
        engine_available=False,
        engine_requested=False,
    )

    structure_map, findings = _parse_structure_map(document)
    report.findings.extend(findings)
    if structure_map is None:
        return report

    report.findings.extend(_semantic_findings(structure_map, map_url, map_id))

    sources, targets = addressed_paths(document)
    report.findings.extend(
        _path_findings(
            targets,
            sources,
            target_tree=target_tree,
            source_fields=set(source_fields) if source_fields is not None else None,
            map_url=map_url,
            map_id=map_id,
            profile_url=profile_url,
        )
    )

    emitted = set()
    emitted_partial = set()
    for entry in targets:
        canonical = entry.path
        if target_tree is not None:
            resolution = _resolve_target_path(target_tree, entry.path)
            if resolution.resolved:
                canonical = resolution.path
                if resolution.status == RESOLUTION_CHOICE_UNNARROWED:
                    emitted_partial.add(normalize_target_path(canonical))
        emitted.add(normalize_target_path(canonical))
    deferred = deferred_reference_paths(document)
    if profile_id is None and profile_url:
        profile_id = str(profile_url).rstrip("/").split("/")[-1].split("|")[0]
    coverage_findings, covered, satisfied = _coverage_findings(
        emitted,
        target_tree=target_tree,
        mapping_table=mapping_table,
        map_url=map_url,
        map_id=map_id,
        profile_url=profile_url,
        profile_id=profile_id,
        deferred=deferred,
        emitted_partial=emitted_partial,
        selector_obligations=authored_selector_obligations(document),
    )
    report.findings.extend(coverage_findings)
    report.covered_required_paths = covered
    report.satisfied_obligations = satisfied
    report.deferred_reference_paths = sorted(deferred)
    return report
