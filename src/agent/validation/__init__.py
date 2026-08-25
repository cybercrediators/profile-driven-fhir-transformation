"""simple validation/accept gates for agent outputs"""

from __future__ import annotations

from agent.validation.models import (
    CLASSIFICATION,
    ActionOwner,
    GateStatus,
    Producer,
    Stage,
    UNKNOWN_CLASSIFICATION,
    VALIDATION_CONTRACT_VERSION,
    ValidationFinding,
    ValidationReport,
    classify,
    finding_identity,
    logger,
)
from agent.validation.paths import (
    AddressedPath,
    _split_outside_brackets,
    _strip_root,
    _tree_lookup_spellings,
    addressed_paths,
    normalize_target_path,
    resolve_tree_node,
    resolve_tree_types,
    target_path_spellings,
)
from agent.validation.diagnostics import (
    _diagnostic_belongs_to,
    findings_from_diagnostics,
)
from agent.validation.resolution import (
    RESOLUTION_CHOICE_TYPE_NOT_ALLOWED,
    RESOLUTION_CHOICE_UNNARROWED,
    RESOLUTION_SLICE_VARIANTS,
    TargetResolution,
    _is_sliced,
    _level_is_described,
    _path_findings,
    _resolve_choice_segment,
    _resolve_step,
    _resolve_target_path,
    _slice_base,
    _variant_children,
)
from agent.validation.references import (
    DEFERRED_REFERENCE_PREFIX,
    REFERENCE_CONTRACT_PREFIX,
    _contract_path,
    _element_target_profiles,
    _produced_profile_canonicals,
    deferred_reference_contracts,
    deferred_reference_paths,
    external_reference_paths,
    recheck_cross_map_references,
)
from agent.validation.coverage import (
    _coverage_findings,
    authored_selector_obligations,
    declared_target_paths,
    declared_target_structure,
    prohibited_target_path,
    required_gap_owner,
)
from agent.validation.offline import (
    _parse_structure_map,
    _semantic_findings,
    build_target_tree,
    validate_offline,
)
from agent.validation.acceptance import (
    AcceptanceDecision,
    InvariantResult,
    decide_acceptance,
    literal_target_values,
    merge_engine_result,
)

# define test cases
__all__ = [
    "AcceptanceDecision",
    "ActionOwner",
    "AddressedPath",
    "CLASSIFICATION",
    "DEFERRED_REFERENCE_PREFIX",
    "GateStatus",
    "InvariantResult",
    "Producer",
    "REFERENCE_CONTRACT_PREFIX",
    "RESOLUTION_CHOICE_TYPE_NOT_ALLOWED",
    "RESOLUTION_CHOICE_UNNARROWED",
    "RESOLUTION_SLICE_VARIANTS",
    "Stage",
    "TargetResolution",
    "UNKNOWN_CLASSIFICATION",
    "VALIDATION_CONTRACT_VERSION",
    "ValidationFinding",
    "ValidationReport",
    "addressed_paths",
    "authored_selector_obligations",
    "build_target_tree",
    "classify",
    "decide_acceptance",
    "declared_target_paths",
    "declared_target_structure",
    "deferred_reference_contracts",
    "deferred_reference_paths",
    "external_reference_paths",
    "finding_identity",
    "findings_from_diagnostics",
    "literal_target_values",
    "merge_engine_result",
    "normalize_target_path",
    "prohibited_target_path",
    "recheck_cross_map_references",
    "required_gap_owner",
    "resolve_tree_node",
    "resolve_tree_types",
    "target_path_spellings",
    "validate_offline",
]
