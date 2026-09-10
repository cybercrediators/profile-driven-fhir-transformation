"""
Create a template for the FHIR mapping language based on the resources and
identified fields from the parser
"""

from fhir.resources.R4B.structuremap import (
    StructureMap,
    StructureMapGroup,
    StructureMapGroupInput,
    StructureMapStructure,
)

from mapping.fml_creator.fml_factory import FMLRuleFactory
from mapping.fml_creator.fml_helper import (
    created_extension_variable,
    emits_only_url,
    flatten_profile_fields,
    has_fixed_value,
    rebase_to_resource_identity,
    rule_can_fire,
)
from mapping.fml_creator.fml_automapper import FMLAutomapper
from mapping import derived_resources
from mapping.generation_result import (
    COVERAGE_REPORT_VERSION,
    CoverageReport,
    GenerationArtifactPaths,
    MappingDiagnostic,
    StructureMapGenerationResult,
)
from helpers.utils import (
    FHIR_ID_MAX_LENGTH,
    FHIR_ID_PATTERN,
    get_value_from_element,
    resource_identity,
    fhir_id_token,
    fhir_map_token,
    fhir_name_token,
)

from typing import List, Any, Dict, Optional, Set, Tuple
import hashlib
import logging
from pathlib import Path
import re

from fhir.resources.R4B.structuredefinition import StructureDefinition
from mapping.fml_creator.fml_questionnaire import QuestionnaireMapCreator
from mapping.rule_ir import (
    CollectionRuleSpec,
    MAPPING_TRANSFORMS,
    SOURCE_LIST_MODES,
    TARGET_LIST_MODES,
    compile_rule_document,
    is_rule_document_key,
    mapping_target_path,
    parse_collection_rule,
)
from data_handling.app_state import AppState

_re_slice = re.compile(r":[^.]+")

logger = logging.getLogger(__name__)


def _is_questionnaire_instance(data: Any) -> bool:
    """True only for an actual Questionnaire resource.

    ``res_type`` is read from ``StructureDefinition.type``, so a *profile*
    constraining Questionnaire reports ``"Questionnaire"`` just as an instance
    does. Only an instance carries the fields the questionnaire creator reads
    (``subjectType``, ``item``); handing it a StructureDefinition raises
    ``AttributeError`` and aborts generation for the whole project.
    """
    if data is None:
        return False
    if type(data).__name__ == "StructureDefinition":
        return False
    return (getattr(data, "resourceType", None) or type(data).__name__) == "Questionnaire"


class StructureMapGenerator:
    def __init__(
        self,
        app_state: AppState,
        map_url: str,
        map_title: str,
        helper_map: StructureDefinition,
        map_name: Optional[str] = None,
        minimal: bool = False,
        status: str = "draft",
        create_references: bool = False,
        automapping: bool = False,
        automapper_instance: Optional[FMLAutomapper] = None,
        custom_mapping_table: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        plugins: Optional[List[Any]] = None,
        llm_automapping: Optional[Any] = None,
    ) -> None:

        self.app_state = app_state
        self.map_url = map_url
        self.map_name = map_name
        self.map_title = map_title
        self.helper_map = helper_map
        self.minimal = minimal
        self.status = status
        self.create_references = create_references
        self.automapping = automapping
        self.automapper_instance = automapper_instance
        self.custom_mapping_table = custom_mapping_table
        self.overwrite = overwrite
        self.plugins = plugins or []
        # Present only when `--auto-mapping-mode llm` was selected; injected rather
        # than constructed so this module never imports the optional LLM package.
        self.llm_automapping = llm_automapping
        self._llm_proposed_mapping_table = None
        self._llm_compiled_mapping_table = {}
        self._llm_diagnostics_start = 0
        self.current_profile_name = "unknown"
        self.mapping_diagnostics = []
        self._generated_structure_map_files = set()
        self.coverage_report: Optional[CoverageReport] = None
        self.coverage_report_path = None
        # {profile_id: {required_total, required_unmapped, unmapped_required_paths}}
        # per-profile required-element coverage report written alongside the maps.
        self._coverage = {}

        # create structuremap
        self.structure_maps = []

        # init automapper and rule factory
        self.automapper = self._init_automapper(
            self.automapping, self.automapper_instance, self.custom_mapping_table
        )
        self.factory = FMLRuleFactory(
            self.app_state, self.overwrite, map_url=self.map_url, plugins=self.plugins
        )
        self.factory.diagnostics = self.mapping_diagnostics

        # check if structure map(s) already exists
        if check_exist := self.app_state.dataIO.check_processed_structure_maps():
            if not self.overwrite:
                logger.info(
                    "Structure map(s) already exist! Loading previous structure maps..."
                )
                sms = self.app_state.dataIO.load_project_files(
                    self.app_state.dataIO.ProjectFolders.STRUCTURE_MAPS
                )
                for filename, sm in sms:
                    if (
                        isinstance(sm, dict)
                        and sm.get("resourceType") == "StructureMap"
                    ):
                        self.structure_maps.append(StructureMap(**sm))
                        self._generated_structure_map_files.add(filename)
            else:
                logger.info("Overwriting existing structure maps...")

        self.check_exist = check_exist

    def get_structure_maps(self) -> List[StructureMap]:
        return self.structure_maps

    @staticmethod
    def _folder_value(folder) -> str:
        return getattr(folder, "value", folder)

    def _artifact_path(self, folder, filename: str):
        project_dir = getattr(self.app_state.dataIO, "project_dir", None)
        if project_dir is None:
            return None
        return Path(project_dir) / self._folder_value(folder) / filename

    def get_generation_result(self) -> StructureMapGenerationResult:
        """Return generated maps and their structured diagnostic artifacts."""

        sm_folder = self.app_state.dataIO.ProjectFolders.STRUCTURE_MAPS
        structure_map_paths = tuple(
            path
            for filename in sorted(self._generated_structure_map_files)
            if (path := self._artifact_path(sm_folder, filename)) is not None
        )
        return StructureMapGenerationResult(
            structure_maps=tuple(self.structure_maps),
            coverage_report=self.coverage_report,
            artifacts=GenerationArtifactPaths(
                structure_maps=structure_map_paths,
                coverage_report=self.coverage_report_path,
            ),
        )

    def generate(self) -> List[StructureMap]:
        if self.check_exist and not self.overwrite and self.structure_maps:
            self._load_coverage_report()
            return self.structure_maps

        # proceed with generation
        resources, source_fields, active_types = self._prepare_resources_and_fields()

        if self.automapper:
            # Both strategies produce the same mapping-table contract, which is then
            # compiled and validated by the ordinary custom-table path below. The
            # model never sees or writes a StructureMap.
            if self.llm_automapping is not None:
                combined_table = self._build_llm_automapping(resources, source_fields)
            else:
                combined_table = self._build_combined_automapping(
                    resources, source_fields
                )
                self._save_automapping_table(combined_table)
            self.custom_mapping_table = combined_table
            self.automapper = None

        derived = derived_resources.discover(
            self.app_state,
            resources,
            self.custom_mapping_table,
            source_fields,
            self._append_diagnostic,
        )
        derived_by_url = {spec.target_profile: spec for spec in derived}
        if derived:
            resources = list(resources) + derived_resources.registry_entries(
                self.app_state, derived
            )
            for spec in derived:
                active_types.update(
                    {spec.target_type, spec.target_profile, spec.target_identity}
                )

        structure_maps = []
        for res_idx, res in enumerate(resources):
            url = next(iter(res))
            obj = self.app_state.registry.get_obj_by_name(url)
            # identity, not necessarily `id`: profiles without one are named by canonical
            profile_identity = (
                resource_identity(obj.data, url) if obj and obj.data else "unknown"
            )
            self.current_profile_name = profile_identity
            res_type = obj.res_type if not hasattr(obj.data, "type") else obj.data.type

            derived_spec = derived_by_url.get(url)
            mapping_table = (
                derived_resources.mapping_overlay(self.custom_mapping_table, derived_spec)
                if derived_spec
                else self.custom_mapping_table
            )

            automapped_mappings = self._process_resource_mapping(
                obj, res_type, source_fields, active_types, mapping_table=mapping_table
            )
            self._record_llm_compiled_mappings(
                res_type, profile_identity, automapped_mappings
            )
            self._record_profile_facet_diagnostics(
                obj, mapped_paths=set(automapped_mappings or {})
            )

            sm_url = f"{self.map_url}-{profile_identity}"
            sm_name = f"{res_idx + 1:03d}_{self.map_name}-{profile_identity}"
            sm_title = f"{self.map_title} - {profile_identity}"

            sm = self.factory.create_base_structure_map(
                sm_url, fhir_name_token(sm_name), sm_title, self.status
            )
            # Deterministic ID (like ConceptMaps) so repeated runs produce identical files
            url_hash = hashlib.md5(sm_url.encode("utf-8")).hexdigest()[:8]
            sm.id = f"sm-{fhir_id_token(profile_identity)}-{url_hash}"
            sm.description = (
                "Auto-generated StructureMap for given profile mappable fields "
                f"(Resource: {profile_identity}). Placeholders have to be set (or use auto-mapping)"
            )

            target_sd_url = (
                "http://hl7.org/fhir/StructureDefinition/QuestionnaireResponse"
                if res_type == "Questionnaire"
                else obj.data.url
            )
            sm.structure = [
                StructureMapStructure.model_construct(
                    url=self.helper_map.url, mode="source", alias="Source"
                ),
                StructureMapStructure.model_construct(
                    url=target_sd_url, mode="target", alias=f"{profile_identity}"
                ),
            ]

            typed_document = compile_rule_document(
                mapping_table,
                profile_id=profile_identity,
                profile_url=getattr(obj.data, "url", "") or "",
                resource_type=res_type,
            )
            for diagnostic in typed_document.diagnostics:
                self._append_diagnostic(**diagnostic)
            if typed_document.imports:
                sm.import_fhir = typed_document.imports
            if typed_document.structures:
                known_structures = {
                    (structure.url, structure.mode, structure.alias)
                    for structure in sm.structure
                }
                for structure in typed_document.structures:
                    identity = (structure.url, structure.mode, structure.alias)
                    if identity not in known_structures:
                        sm.structure.append(structure)
                        known_structures.add(identity)

            res_group = self.create_res_group(
                res_type,
                obj,
                self.create_references,
                source_obj=self.helper_map,
                automapped_mappings=automapped_mappings,
            )
            if typed_document.reference_paths:
                res_group.rule = self._remove_deferred_reference_rules(
                    res_group.rule, res_type, typed_document.reference_paths
                )
            if typed_document.primary_rules:
                res_group.rule = list(res_group.rule or []) + list(
                    typed_document.primary_rules
                )

            groups = [res_group, *typed_document.groups]
            sm.group = groups
            self._sanitize_todo_rules(sm)
            self._disambiguate_shadowed_variables(sm)
            self._resolve_duplicate_rule_names(sm)
            self._report_overlong_map_tokens(sm)
            self._normalize_and_validate_structure_map(sm)

            self.app_state.dataIO.store_project_file(
                self.app_state.dataIO.ProjectFolders.STRUCTURE_MAPS,
                f"{sm_name}.json",
                sm.model_dump_json(indent=2),
                mode="STR",
                overwrite=self.overwrite,
            )
            self._generated_structure_map_files.add(f"{sm_name}.json")
            structure_maps.append(sm)

        self.structure_maps = structure_maps

        self._finalize_llm_automapping_report()

        self._save_coverage_report()

        self._cleanup_stale_concept_maps(structure_maps)
        self._cleanup_stale_structure_maps(structure_maps)

        return structure_maps

    @staticmethod
    def _remove_deferred_reference_rules(rules, res_type, reference_paths):
        """Remove default TODO wiring superseded by an explicit reference policy."""
        kept = []
        for rule in rules or []:
            nested = StructureMapGenerator._remove_deferred_reference_rules(
                getattr(rule, "rule", None), res_type, reference_paths
            )
            if getattr(rule, "rule", None) is not None:
                rule.rule = nested or None
            name = getattr(rule, "name", "") or ""
            documentation = getattr(rule, "documentation", "") or ""
            if name.startswith("TODO-resolve-reference"):
                match = re.match(
                    rf"Reference<{re.escape(res_type)}\.([^>]+)>",
                    documentation,
                )
                if match:
                    path = match.group(1)
                    if path.endswith("[x]"):
                        path = path[: -len("[x]")] + "Reference"
                    if path in reference_paths:
                        continue
            kept.append(rule)
        return kept

    def _cleanup_stale_structure_maps(self, structure_maps):
        """delete generated-named SM files in structure_maps/ that this run did not produce"""

        folder = self.app_state.dataIO.ProjectFolders.STRUCTURE_MAPS
        sm_dir = self.app_state.dataIO.project_dir / folder.value
        if not sm_dir.is_dir():
            return
        current = getattr(self, "_generated_structure_map_files", None) or {
            f"{sm.name}.json" for sm in structure_maps
        }
        generated_name = re.compile(rf"^\d{{3}}_{re.escape(self.map_name)}-.+\.json$")
        for f in sm_dir.iterdir():
            if f.name in current or not generated_name.match(f.name):
                continue
            logger.info(f"Removing stale StructureMap: {f.name}")
            self.app_state.dataIO.delete_file(folder.value + "/" + f.name)

    def _disambiguate_shadowed_variables(self, structure_map) -> None:
        """Rename variables that shadow one already in scope"""

        used: Set[str] = set()

        def collect(rules) -> None:
            for rule in rules or []:
                for source in getattr(rule, "source", None) or []:
                    if getattr(source, "variable", None):
                        used.add(source.variable)
                for target in getattr(rule, "target", None) or []:
                    if getattr(target, "variable", None):
                        used.add(target.variable)
                collect(getattr(rule, "rule", None))

        def fresh(name: str) -> str:
            suffix = 2
            while f"{name}-{suffix}" in used:
                suffix += 1
            allocated = f"{name}-{suffix}"
            used.add(allocated)
            return allocated

        def declare(holder, scope: Dict[str, str], group_name: str) -> None:
            name = getattr(holder, "variable", None)
            if not name:
                return
            if name in scope:
                renamed = fresh(name)
                logger.debug(
                    "Renaming shadowed variable %r to %r in group %s.",
                    name,
                    renamed,
                    group_name,
                )
                holder.variable = renamed
                scope[name] = renamed
            else:
                scope[name] = name

        def walk(rules, scope: Dict[str, str], group_name: str) -> None:
            for rule in rules or []:
                local = dict(scope)
                for source in getattr(rule, "source", None) or []:
                    context = getattr(source, "context", None)
                    if context:
                        source.context = local.get(context, context)
                    declare(source, local, group_name)
                for target in getattr(rule, "target", None) or []:
                    context = getattr(target, "context", None)
                    if context:
                        target.context = local.get(context, context)
                    for parameter in getattr(target, "parameter", None) or []:
                        value_id = getattr(parameter, "valueId", None)
                        if value_id and value_id in local:
                            parameter.valueId = local[value_id]
                    declare(target, local, group_name)
                for dependent in getattr(rule, "dependent", None) or []:
                    supplied = getattr(dependent, "variable", None)
                    if supplied:
                        dependent.variable = [
                            local.get(item, item) for item in supplied
                        ]
                walk(getattr(rule, "rule", None), local, group_name)

        for group in getattr(structure_map, "group", None) or []:
            used.clear()
            collect(getattr(group, "rule", None))
            scope = {
                item.name: item.name
                for item in (getattr(group, "input", None) or [])
                if getattr(item, "name", None)
            }
            used.update(scope)
            walk(
                getattr(group, "rule", None),
                scope,
                getattr(group, "name", "") or "",
            )

    def _resolve_duplicate_rule_names(self, structure_map) -> None:
        """Make sibling rule names unique, distinguishing redundancy from conflict"""

        def rule_form(rule) -> str:
            try:
                return rule.model_dump_json(exclude_none=True)
            except Exception:
                return repr(rule)

        def walk(rules, group_name: str) -> List[Any]:
            if not rules:
                return rules
            kept: List[Any] = []
            seen_forms: Dict[str, str] = {}
            used_names: Set[str] = set()
            for rule in rules:
                name = getattr(rule, "name", None)
                form = rule_form(rule)
                if name and seen_forms.get(name) == form:
                    logger.debug(
                        "Dropping a duplicate emission of rule %r in %s.",
                        name,
                        group_name,
                    )
                    continue
                if name and name in used_names:
                    suffix = 2
                    while f"{name}-{suffix}" in used_names:
                        suffix += 1
                    self._append_diagnostic(
                        "duplicate-rule-name-conflict",
                        f"Rules {name!r} and {name}-{suffix} in group "
                        f"{group_name!r} write the same target element with "
                        "different values — typically a profile-fixed value and "
                        "a mapped source field. Both are emitted; remove the "
                        "mapping-table entry if the profile's value is the one "
                        "you want.",
                        group=group_name,
                        rule=name,
                    )
                    rule.name = f"{name}-{suffix}"
                    name = rule.name
                if name:
                    used_names.add(name)
                    seen_forms[name] = form
                rule.rule = walk(getattr(rule, "rule", None), group_name)
                kept.append(rule)
            return kept

        for group in getattr(structure_map, "group", None) or []:
            group.rule = walk(
                getattr(group, "rule", None), getattr(group, "name", "") or ""
            )

    def _report_overlong_map_tokens(self, structure_map) -> None:
        """Report names past FHIR's 64-character ``id`` limit"""

        overlong: List[str] = []

        def check(where: str, value) -> None:
            if value and len(str(value)) > FHIR_ID_MAX_LENGTH:
                overlong.append(f"{where} ({len(str(value))} chars): {value}")

        def walk_rules(rules) -> None:
            for rule in rules or []:
                check("rule.name", getattr(rule, "name", None))
                walk_rules(getattr(rule, "rule", None))

        for group in getattr(structure_map, "group", None) or []:
            check("group.name", getattr(group, "name", None))
            walk_rules(getattr(group, "rule", None))

        for entry in overlong:
            self._append_diagnostic(
                "map-token-too-long",
                f"{entry} exceeds FHIR's {FHIR_ID_MAX_LENGTH}-character limit for "
                "an id. Shorten the profile or element identifier it is derived "
                "from; truncating it here could break a rule that references it "
                "by name.",
            )

    @staticmethod
    def _normalize_and_validate_structure_map(structure_map):
        """Populate required target context types and reject malformed targets."""
        name = getattr(structure_map, "name", "") or ""
        if not re.fullmatch(r"[A-Z][A-Za-z0-9_]{0,254}", name):
            raise ValueError(f"StructureMap.name is not invariant-safe: {name!r}")

        def _walk(rules):
            for rule in rules or []:
                for source in getattr(rule, "source", None) or []:
                    source_mode = getattr(source, "listMode", None)
                    if source_mode and source_mode not in SOURCE_LIST_MODES:
                        raise ValueError(
                            f"StructureMap rule {getattr(rule, 'name', '<unnamed>')!r} "
                            f"has unsupported source list mode {source_mode!r}"
                        )
                    if source_mode and not getattr(source, "element", None):
                        logger.debug(
                            "Dropping source list mode %r on rule %r: the source "
                            "reads its context, not a repeating element.",
                            source_mode,
                            getattr(rule, "name", "<unnamed>"),
                        )
                        source.listMode = None
                for target in getattr(rule, "target", None) or []:
                    context = getattr(target, "context", None)
                    element = getattr(target, "element", None)
                    if element and not context:
                        raise ValueError(
                            f"StructureMap rule {getattr(rule, 'name', '<unnamed>')!r} "
                            f"targets element {element!r} without a context"
                        )
                    if context and not getattr(target, "contextType", None):
                        target.contextType = "variable"
                    transform = getattr(target, "transform", None)
                    if transform and transform not in MAPPING_TRANSFORMS:
                        raise ValueError(
                            f"StructureMap rule {getattr(rule, 'name', '<unnamed>')!r} "
                            f"has unsupported transform {transform!r}"
                        )
                    target_modes = list(getattr(target, "listMode", None) or [])
                    invalid_modes = sorted(set(target_modes) - TARGET_LIST_MODES)
                    if invalid_modes:
                        raise ValueError(
                            f"StructureMap rule {getattr(rule, 'name', '<unnamed>')!r} "
                            f"has unsupported target list modes {invalid_modes!r}"
                        )
                    list_rule_id = getattr(target, "listRuleId", None)
                    if list_rule_id and "share" not in target_modes:
                        raise ValueError(
                            f"StructureMap rule {getattr(rule, 'name', '<unnamed>')!r} "
                            "has listRuleId without target list mode 'share'"
                        )
                _walk(getattr(rule, "rule", None))

        groups = getattr(structure_map, "group", None) or []
        group_names = [
            getattr(group, "name", None)
            for group in groups
            if getattr(group, "name", None)
        ]
        if len(group_names) != len(set(group_names)):
            raise ValueError("StructureMap group names must be unique")
        for group in groups:
            _walk(getattr(group, "rule", None))
        StructureMapGenerator._reject_invalid_map_tokens(structure_map)
        return structure_map

    @staticmethod
    def _reject_invalid_map_tokens(structure_map) -> None:
        """Fail on a name FHIR types as ``id`` that is not a valid id"""

        offenders: List[str] = []

        def check(where: str, value) -> None:
            if value and not FHIR_ID_PATTERN.match(str(value)):
                offenders.append(f"{where}={value!r}")

        def walk_rules(rules, trail: str) -> None:
            for index, rule in enumerate(rules or []):
                where = f"{trail}/rule[{index}]"
                check(f"{where}.name", getattr(rule, "name", None))
                for source in getattr(rule, "source", None) or []:
                    check(f"{where}.source.context", getattr(source, "context", None))
                    check(f"{where}.source.variable", getattr(source, "variable", None))
                for target in getattr(rule, "target", None) or []:
                    check(f"{where}.target.context", getattr(target, "context", None))
                    check(f"{where}.target.variable", getattr(target, "variable", None))
                for dependent in getattr(rule, "dependent", None) or []:
                    check(f"{where}.dependent.name", getattr(dependent, "name", None))
                walk_rules(getattr(rule, "rule", None), where)

        for index, group in enumerate(getattr(structure_map, "group", None) or []):
            where = f"group[{index}]"
            check(f"{where}.name", getattr(group, "name", None))
            check(f"{where}.extends", getattr(group, "extends", None))
            for group_input in getattr(group, "input", None) or []:
                check(f"{where}.input.name", getattr(group_input, "name", None))
            walk_rules(getattr(group, "rule", None), where)

        if offenders:
            raise ValueError(
                "StructureMap contains names FHIR requires to match "
                f"{FHIR_ID_PATTERN.pattern}: " + ", ".join(offenders[:10])
            )

    def _append_diagnostic(self, code, message, **details):
        def _plain(value):
            if hasattr(value, "model_dump"):
                return _plain(value.model_dump(exclude_none=True))
            if isinstance(value, dict):
                return {key: _plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_plain(item) for item in value]
            return value

        diagnostic = _plain(
            {
                "code": code,
                "message": message,
                "profile": getattr(self, "current_profile_name", "unknown"),
                **details,
            }
        )
        if not hasattr(self, "mapping_diagnostics"):
            self.mapping_diagnostics = []
        if diagnostic not in self.mapping_diagnostics:
            self.mapping_diagnostics.append(diagnostic)
        return diagnostic

    def _record_profile_facet_diagnostics(self, obj, mapped_paths=None):
        """Surface profile facets that constrain output but do not invent source values."""

        mapped_paths = {str(path) for path in (mapped_paths or set()) if path}

        def _has_explicit_provider(path):
            return any(
                candidate == path
                or candidate.startswith(path + ".")
                or candidate.startswith(path + ":")
                for candidate in mapped_paths
            )

        def _walk(fields):
            for field in fields or []:
                if not isinstance(field, dict):
                    continue
                path = field.get("id") or field.get("path")
                if field.get("content_reference") and not field.get(
                    "content_reference_resolved"
                ):
                    self._append_diagnostic(
                        "unresolved-content-reference",
                        f"Could not resolve {field['content_reference']} for {path}.",
                        path=path,
                        severity="error",
                    )
                for constraint in field.get("constraints") or []:
                    self._append_diagnostic(
                        "target-fhirpath-constraint",
                        f"Target constraint {constraint.get('key', '<unnamed>')} "
                        f"must be validated for {path}.",
                        path=path,
                        severity=constraint.get("severity", "error"),
                        expression=constraint.get("expression"),
                    )
                if field.get("conditions"):
                    self._append_diagnostic(
                        "target-condition",
                        f"Target conditions apply to {path}.",
                        path=path,
                        conditions=list(field["conditions"]),
                        severity="information",
                    )
                for facet in ("default_value", "min_value", "max_value", "max_length"):
                    if facet in field:
                        message = (
                            f"{facet} constrains {path}; it remains "
                            "validator-enforced."
                        )
                        if facet in ("min_value", "max_value", "max_length"):
                            message += (
                                " Direct scalar mappings also receive a "
                                "source-side check when the constraint is safely "
                                "comparable."
                            )
                        self._append_diagnostic(
                            f"target-{facet.replace('_', '-')}",
                            message,
                            path=path,
                            value=field[facet],
                            severity="information",
                        )
                if field.get("is_modifier") and not _has_explicit_provider(path):
                    self._append_diagnostic(
                        "target-modifier-element",
                        f"{path} is a modifier element and requires explicit mapping intent.",
                        path=path,
                        severity="warning",
                    )
                _walk(field.get("children"))
                _walk(field.get("slices"))

        _walk(getattr(obj, "mappable_fields", None))

    def _sanitize_todo_rules(self, structure_map):
        """Remove unsafe TODOs while retaining intentional source scaffolding.

        A ``TODO_*`` source element is executable but safe: FHIR Mapping Language
        treats a missing source element as a non-match, so the rule remains a
        human-fillable skeleton without making ``$transform`` fail.  TODOs in
        target assignments (or in other source expressions such as ``check`` and
        ``condition``) cannot be interpreted safely and are moved to diagnostics.
        Deferred reference rules are instructions for ``BundleService`` and must
        remain machine-readable in the map.
        """

        def _contains_todo(value):
            if isinstance(value, str):
                return "TODO" in value
            if isinstance(value, dict):
                return any(_contains_todo(v) for v in value.values())
            if isinstance(value, list):
                return any(_contains_todo(v) for v in value)
            if hasattr(value, "model_dump"):
                return _contains_todo(value.model_dump(exclude_none=True))
            if hasattr(value, "__dict__"):
                return _contains_todo(vars(value))
            return False

        def _sanitize(rules, group_name):
            kept = []
            for rule in rules or []:
                name = getattr(rule, "name", "") or ""
                if name.startswith("TODO-resolve-reference"):
                    self._append_diagnostic(
                        "deferred-reference",
                        getattr(rule, "documentation", None)
                        or f"Reference rule {name} requires BundleService resolution.",
                        group=group_name,
                        rule=name,
                        severity="warning",
                    )
                    kept.append(rule)
                    continue

                source_expressions = []
                for source in getattr(rule, "source", None) or []:
                    if hasattr(source, "model_dump"):
                        source_data = source.model_dump(exclude_none=True)
                    else:
                        source_data = dict(vars(source))
                    # An unresolved source element is the supported sparse-map
                    # scaffold. All other source properties affect execution and
                    # must be fully specified.
                    source_data.pop("element", None)
                    source_data.pop("element__ext", None)
                    source_expressions.append(source_data)

                executable = {
                    "source_expressions": source_expressions,
                    "target": getattr(rule, "target", None) or [],
                    "dependent": getattr(rule, "dependent", None) or [],
                }
                if _contains_todo(executable):
                    self._append_diagnostic(
                        "unresolved-map-placeholder",
                        f"Rule {name} was omitted because it contains an "
                        "unresolved target or source-expression TODO placeholder.",
                        group=group_name,
                        rule=name,
                        severity="warning",
                    )
                    continue
                surviving = _sanitize(getattr(rule, "rule", None), group_name)
                ext_var = created_extension_variable(rule)
                if (
                    ext_var
                    and emits_only_url(surviving, ext_var)
                    and all(rule_can_fire(s) for s in getattr(rule, "source", None) or [])
                ):
                    # Pruning a child can leave its parent extension holding nothing
                    # but a url, which violates ext-1
                    # (`extension.exists() != value.exists()`) and fails the whole
                    # instance — so the parent has to go too. Bottom-up recursion makes
                    # this cascade: a dropped sub-extension can empty its own parent,
                    # which is checked in turn. Rules gated behind a TODO source are
                    # left alone: they never fire, so they ship nothing invalid.
                    self._append_diagnostic(
                        "url-only-extension-pruned",
                        f"Rule {name} was omitted because sanitizing its children left "
                        "it writing only the extension url, which violates ext-1.",
                        group=group_name,
                        rule=name,
                        severity="warning",
                    )
                    continue
                rule.rule = surviving or None
                kept.append(rule)
            return kept

        for group in getattr(structure_map, "group", None) or []:
            group.rule = _sanitize(
                getattr(group, "rule", None), getattr(group, "name", "<unnamed>")
            )
        return structure_map

    def _cleanup_stale_concept_maps(self, structure_maps):
        """Delete cm-*.json files in source_data/concept_maps that no current map references"""
        sm_folder = self.app_state.dataIO.ProjectFolders.CONCEPT_MAPS
        sm_dir = self.app_state.dataIO.project_dir / sm_folder.value

        # Collect all CM IDs referenced in the current maps
        referenced = set()

        def _find_refs(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    _find_refs(v)
            elif isinstance(obj, list):
                for v in obj:
                    _find_refs(v)
            elif isinstance(obj, str) and "/ConceptMap/cm-" in obj:
                referenced.add(obj.split("/ConceptMap/")[-1] + ".json")

        for sm in structure_maps:
            _find_refs(sm.model_dump())

        # Find and remove stale CM files
        for f in sm_dir.iterdir():
            if (
                f.name.startswith("cm-")
                and f.name.endswith(".json")
                and f.name not in referenced
            ):
                logger.info(f"Removing stale ConceptMap: {f.name}")
                self.app_state.dataIO.delete_file(sm_folder.value + "/" + f.name)

    def _prepare_resources_and_fields(self):
        resources = self.app_state.registry.get_all_mappable_fields_of_roots()
        source_fields = self._extract_source_fields(self.helper_map)

        active_types = set()
        for res_dict in resources:
            for url in res_dict:
                obj = self.app_state.registry.get_obj_by_name(url)
                if obj and obj.data:
                    # Try to get type from StructureDefinition.type or resourceType
                    t = getattr(obj.data, "type", None) or getattr(
                        obj.data, "resourceType", None
                    )
                    if t:
                        active_types.add(t)
                    # Also add the canonical URL and id
                    if hasattr(obj.data, "url") and obj.data.url:
                        active_types.add(obj.data.url)
                    active_types.add(resource_identity(obj.data))

        if self.custom_mapping_table:
            for v in self.custom_mapping_table.values():
                target_path = mapping_target_path(v)
                if target_path and "." in target_path:
                    active_types.add(target_path.split(".")[0])

        source_field_types = self._build_source_field_types(source_fields)
        self.factory.source_field_types = source_field_types
        self.factory.source_field_max = {
            field["path"].split(".")[-1]: field.get("max", "1")
            for field in source_fields
            if field.get("path") and "." in field["path"]
        }

        return resources, source_fields, active_types

    def _flatten_fields_recursively(
        self, fields: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Recursively flatten a hierarchical list of fields into a single list.
        """
        result = []
        for field in fields:
            result.append(field)
            if field.get("children"):
                result.extend(self._flatten_fields_recursively(field["children"]))
            if field.get("type_structure"):
                result.extend(self._flatten_fields_recursively(field["type_structure"]))
        return result

    def _process_resource_mapping(
        self,
        obj,
        res_type,
        source_fields,
        active_types: Set[str] = None,
        mapping_table=None,
    ):
        # Defaults to the project's table; a caller passes one only to add bindings
        # that apply to this profile alone (see derived_resources.mapping_overlay).
        if mapping_table is None:
            mapping_table = self.custom_mapping_table
        special_targets = set()
        if isinstance(mapping_table, dict):
            profile_id = resource_identity(obj.data)
            for source_key, value in mapping_table.items():
                if is_rule_document_key(source_key):
                    continue
                target = mapping_target_path(value)
                if not target or "." not in target:
                    continue
                prefix, relative = target.split(".", 1)
                if prefix in (res_type, profile_id):
                    special_targets.add(f"{res_type}.{relative}")
        flat_fields = flatten_profile_fields(
            self.app_state,
            res_type,
            obj.mappable_fields,
            include_special_paths=special_targets,
        )

        automap_source_list = self._flatten_fields_recursively(flat_fields)

        automapped_paths, automapped_mappings = self._get_automapped_mappings(
            mapping_table,
            self.automapper,
            source_fields,
            automap_source_list,
            res_type,
            resource_identity(obj.data),
        )

        self._record_coverage(obj, res_type, automapped_paths)

        if self.minimal:
            logger.info(
                f"Filtering fields for minimal mode (Resource: {obj.data.id})..."
            )
            required_paths = set()
            self._collect_required_paths_from_fields(
                obj.mappable_fields, required_paths, active_types
            )
            for auto_path in automapped_paths:
                self._add_required_path(auto_path, required_paths)

            filtered_fields = [
                field
                for field in obj.mappable_fields
                if self._should_keep_path(field.get("path"), required_paths)
            ]

            for field in filtered_fields:
                self._prune_nested_components(
                    field, required_paths, mapped_paths=automapped_paths
                )
            logger.info(f"Minimal mode: Kept {len(filtered_fields)} root fields.")
            obj.mappable_fields = filtered_fields

        return automapped_mappings

    def _record_coverage(self, obj, res_type, automapped_paths):
        """Build a slice-aware minimum-cardinality requirement manifest"""
        mapped_paths = {p for p in (automapped_paths or set()) if p}

        def _norm(path):
            return (path or "").replace("[x]", "")

        def _at_or_below(candidate, parent):
            candidate, parent = _norm(candidate), _norm(parent)
            return bool(
                candidate
                and parent
                and (
                    candidate == parent
                    or candidate.startswith(parent + ".")
                    or candidate.startswith(parent + ":")
                )
            )

        def _provider(paths, requirement_id):
            return any(_at_or_below(path, requirement_id) for path in paths)

        def _ancestor_ids(element_id):
            """Yield structural ancestors, including the unsliced base of slice roots."""
            current = element_id
            seen = set()
            while "." in current:
                current = current.rsplit(".", 1)[0]
                base = (
                    current.rsplit(":", 1)[0]
                    if ":" in current.rsplit(".", 1)[-1]
                    else None
                )
                for candidate in (current, base):
                    if candidate and candidate not in seen:
                        seen.add(candidate)
                        yield candidate

        data = getattr(obj, "data", None)
        snapshot = getattr(data, "snapshot", None)
        elements = list(getattr(snapshot, "element", None) or [])

        if elements:
            by_id = {}
            fixed_ids = set()
            mandatory_slice_ids = set()
            requirements = []

            def _fixed_provider_paths(element_id, value):
                """Expand a fixed/pattern datatype into the element IDs it supplies"""
                if hasattr(value, "model_dump"):
                    value = value.model_dump(exclude_none=True)
                elif hasattr(value, "dict"):
                    value = value.dict(exclude_none=True)

                supplied = set()

                def _walk(current, current_id):
                    if isinstance(current, dict):
                        if not current:
                            supplied.add(current_id)
                        for key, child in current.items():
                            _walk(child, f"{current_id}.{key}")
                    elif isinstance(current, (list, tuple)):
                        if not current:
                            supplied.add(current_id)
                        for child in current:
                            _walk(child, current_id)
                    else:
                        supplied.add(current_id)

                _walk(value, element_id)
                return supplied

            for elem in elements:
                element_id = getattr(elem, "id", None) or getattr(elem, "path", None)
                path = getattr(elem, "path", None) or element_id
                if not element_id:
                    continue
                by_id[element_id] = elem
                _, fixed_value = get_value_from_element(elem, ("fixed", "pattern"))
                if fixed_value is not None:
                    fixed_ids.update(_fixed_provider_paths(element_id, fixed_value))
                min_cardinality = getattr(elem, "min", None) or 0
                if getattr(elem, "sliceName", None) and min_cardinality >= 1:
                    mandatory_slice_ids.add(element_id)
                if (
                    "." in element_id
                    and min_cardinality >= 1
                    and str(getattr(elem, "max", None)) != "0"
                ):
                    requirements.append((element_id, path, elem))

            activation_paths = mapped_paths | mandatory_slice_ids
            manifest = []
            for element_id, path, elem in requirements:
                inactive_gates = []
                for ancestor_id in _ancestor_ids(element_id):
                    if "." not in ancestor_id:
                        continue
                    ancestor = by_id.get(ancestor_id)
                    if ancestor is None:
                        continue
                    ancestor_min = getattr(ancestor, "min", None) or 0
                    if ancestor_min < 1 and not _provider(activation_paths, ancestor_id):
                        inactive_gates.append(ancestor_id)

                active = not inactive_gates
                if _provider(fixed_ids, element_id):
                    provider = "profile-fixed"
                elif _provider(mapped_paths, element_id):
                    provider = "mapped"
                else:
                    provider = "missing"
                status = (
                    "latent"
                    if not active
                    else ("covered" if provider != "missing" else "unmapped")
                )
                entry = {
                    "id": element_id,
                    "path": path,
                    "min": getattr(elem, "min", None),
                    "max": getattr(elem, "max", None),
                    "active": active,
                    "provider": provider,
                    "status": status,
                }
                slice_name = getattr(elem, "sliceName", None)
                if slice_name:
                    entry["slice_name"] = slice_name
                if inactive_gates:
                    entry["inactive_optional_ancestors"] = inactive_gates
                manifest.append(entry)
        else:
            manifest = []

            def _has_fixed(field):
                value = field.get("fixed_value")
                if has_fixed_value(value):
                    return True
                return any(
                    isinstance(child, dict) and _has_fixed(child)
                    for key in ("children", "type_structure", "slices")
                    for child in (field.get(key) or [])
                )

            def _walk(fields):
                for field in fields or []:
                    if not isinstance(field, dict):
                        continue
                    element_id = field.get("id") or field.get("path")
                    is_required = bool(field.get("is_required")) or (
                        field.get("cardinality", {}) or {}
                    ).get("min", 0) >= 1
                    if is_required and element_id:
                        fixed = _has_fixed(field)
                        mapped = _provider(mapped_paths, element_id)
                        provider = (
                            "profile-fixed"
                            if fixed
                            else ("mapped" if mapped else "missing")
                        )
                        manifest.append(
                            {
                                "id": element_id,
                                "path": field.get("path") or element_id,
                                "min": (field.get("cardinality", {}) or {}).get("min"),
                                "max": (field.get("cardinality", {}) or {}).get("max"),
                                "active": True,
                                "provider": provider,
                                "status": (
                                    "covered" if provider != "missing" else "unmapped"
                                ),
                            }
                        )
                    for key in ("children", "type_structure", "slices"):
                        _walk(field.get(key))

            _walk(getattr(obj, "mappable_fields", None))

        # Keep the first occurrence of an ID; snapshot IDs, unlike paths, retain slice names.
        manifest = list({entry["id"]: entry for entry in manifest}.values())
        active = [entry for entry in manifest if entry["active"]]
        latent = [entry for entry in manifest if not entry["active"]]
        unmapped = [entry for entry in active if entry["provider"] == "missing"]
        profile_coverage = {
            "resource_type": res_type,
            # Backwards-compatible names now deliberately describe the active cohort.
            "required_total": len(active),
            "required_mapped": len(active) - len(unmapped),
            "required_unmapped": len(unmapped),
            "unmapped_required_paths": [entry["id"] for entry in unmapped],
            "static_required_total": len(manifest),
            "latent_required_total": len(latent),
            "latent_required_paths": [entry["id"] for entry in latent],
            "requirement_manifest": manifest,
        }
        self._coverage[resource_identity(obj.data)] = profile_coverage

    def _coverage_report_filename(self) -> str:
        return f"{self.map_name}_coverage.json"

    def _build_coverage_report(self) -> Optional[CoverageReport]:
        """Build the public coverage contract from current generator state."""

        if not self._coverage and not self.mapping_diagnostics:
            return None
        total_req = sum(p["required_total"] for p in self._coverage.values())
        total_unmapped = sum(p["required_unmapped"] for p in self._coverage.values())
        static_req = sum(
            p.get("static_required_total", p["required_total"])
            for p in self._coverage.values()
        )
        latent_req = sum(
            p.get("latent_required_total", 0) for p in self._coverage.values()
        )
        covered = total_req - total_unmapped
        report = {
            "report_version": COVERAGE_REPORT_VERSION,
            "map": self.map_name,
            "note": (
                "Requirements are enumerated independently from the StructureDefinition "
                "snapshot and keyed by ElementDefinition.id so slice identity is preserved. "
                "required_total is the active cohort for this draft; constraints below absent "
                "optional parents are listed separately as latent. Coverage describes declared "
                "providers, not successful FHIR validation."
            ),
            "summary": {
                "profiles": len(self._coverage),
                "required_total": total_req,
                "required_mapped": covered,
                "required_unmapped": total_unmapped,
                "static_required_total": static_req,
                "latent_required_total": latent_req,
                "required_coverage_pct": (
                    round(100.0 * covered / total_req, 1) if total_req else 100.0
                ),
            },
            "profiles": self._coverage,
            "mapping_diagnostics": [
                MappingDiagnostic.from_raw(diagnostic)
                for diagnostic in self.mapping_diagnostics
            ],
        }
        return CoverageReport.model_validate(report)

    def _load_coverage_report(self) -> Optional[CoverageReport]:
        """Load an existing v2+ report and enrich its diagnostics in memory."""

        loader = getattr(self.app_state.dataIO, "load_project_file", None)
        if loader is None:
            return None
        filename = self._coverage_report_filename()
        try:
            raw = loader(
                self.app_state.dataIO.ProjectFolders.SOURCE_DATA,
                filename,
            )
            if not isinstance(raw, dict):
                return None
            self.coverage_report = CoverageReport.from_raw(raw)
            self.coverage_report_path = self._artifact_path(
                self.app_state.dataIO.ProjectFolders.SOURCE_DATA,
                filename,
            )
            return self.coverage_report
        except Exception as exc:
            logger.warning("Could not load coverage report: %s", exc)
            return None

    def _save_coverage_report(self) -> Optional[CoverageReport]:
        """Persist and return the aggregated required-element coverage report."""

        report = self._build_coverage_report()
        self.coverage_report = report
        if report is None:
            self.coverage_report_path = None
            return None

        filename = self._coverage_report_filename()
        try:
            self.app_state.dataIO.store_project_file(
                self.app_state.dataIO.ProjectFolders.SOURCE_DATA,
                filename,
                report.model_dump_json(indent=2, exclude_none=True),
                mode="STR",
                overwrite=True,
            )
            self.coverage_report_path = self._artifact_path(
                self.app_state.dataIO.ProjectFolders.SOURCE_DATA,
                filename,
            )
            summary = report.summary
            logger.info(
                "Required-element coverage: %d/%d mapped across %d profiles (%s unmapped).",
                summary.required_mapped,
                summary.required_total,
                summary.profiles,
                summary.required_unmapped,
            )
        except Exception as exc:
            self.coverage_report_path = None
            logger.warning("Could not write coverage report: %s", exc)
        return report

    def _iter_mappable_resources(self, resources: list):
        """Yield ``(obj, res_type, target_fields)`` for every resolvable resource"""

        for res_dict in resources:
            url = next(iter(res_dict))
            obj = self.app_state.registry.get_obj_by_name(url)
            if not obj or not obj.data:
                continue
            res_type = obj.res_type if not hasattr(obj.data, "type") else obj.data.type
            flat_fields = flatten_profile_fields(
                self.app_state, res_type, obj.mappable_fields
            )
            yield obj, res_type, self._flatten_fields_recursively(flat_fields)

    def _build_llm_automapping(
        self, resources: list, source_fields: list
    ) -> Dict[str, str]:
        """Delegate target selection to the configured LLM strategy"""

        from mapping.fml_creator.llm_automapping import ResourceTargets

        resource_targets = [
            ResourceTargets(
                res_type=res_type,
                res_id=resource_identity(obj.data),
                target_fields=target_fields,
                profile=obj.data,
            )
            for obj, res_type, target_fields in self._iter_mappable_resources(resources)
        ]

        mapping_table = self.llm_automapping.build_mapping_table(
            resource_targets, source_fields
        )
        self._llm_proposed_mapping_table = dict(mapping_table)
        self._llm_compiled_mapping_table = {}
        self._llm_diagnostics_start = len(self.mapping_diagnostics)
        logger.info(
            f"LLM automapping complete: {len(mapping_table)} source fields mapped."
        )
        return mapping_table

    def _record_llm_compiled_mappings(
        self,
        res_type: str,
        res_id: str,
        automapped_mappings: Dict[str, str],
    ) -> None:
        """Record model selections the ordinary compiler actually accepted."""

        if self._llm_proposed_mapping_table is None:
            return

        for source_id, target_value in self._llm_proposed_mapping_table.items():
            target_text = mapping_target_path(target_value)
            if not target_text:
                continue
            prefix, separator, remainder = target_text.partition(".")
            if prefix not in {res_type, res_id}:
                continue
            normalized_target = f"{res_type}.{remainder}" if separator else res_type
            if automapped_mappings.get(normalized_target) == source_id:
                self._llm_compiled_mapping_table[source_id] = target_text

    def _finalize_llm_automapping_report(self) -> None:
        """Persist the audit report only after normal compilation succeeds."""

        if self._llm_proposed_mapping_table is None:
            return

        diagnostics = self.mapping_diagnostics[self._llm_diagnostics_start :]
        report = self.llm_automapping.report(
            self._llm_proposed_mapping_table,
            compiled_mapping_table=self._llm_compiled_mapping_table,
            compiler_diagnostics=diagnostics,
        )
        self._save_automapping_table(self._llm_compiled_mapping_table)
        self._save_llm_automapping_report(report)

    def _save_llm_automapping_report(self, report: Dict[str, Any]) -> None:
        """Persist the LLM automapping audit trail beside the mapping table."""

        import json

        filename = f"{self.map_name}_llm_automapping_report.json"
        self.app_state.dataIO.store_project_file(
            self.app_state.dataIO.ProjectFolders.SOURCE_DATA,
            filename,
            json.dumps(report, indent=2, default=str),
            mode="STR",
            overwrite=True,
        )
        logger.info(f"Saved LLM automapping report to source_data/{filename}")

    def _build_combined_automapping(
        self, resources: list, source_fields: list
    ) -> Dict[str, str]:
        """automapping for every source field against every resource's target fields"""
        # {source_field_id: {"target": path, "score": float, "resource": res_type}}
        best: Dict[str, dict] = {}

        for obj, res_type, target_fields in self._iter_mappable_resources(resources):
            logger.info(
                f"Automapping pass for resource {obj.data.id} ({len(source_fields)} source fields)..."
            )
            for source_field in source_fields:
                match, score = self.automapper.find_mapping(source_field, target_fields)
                if not match or score <= 0:
                    continue
                src_id = source_field.get("id") or source_field.get("path", "")
                if src_id not in best or score > best[src_id]["score"]:
                    target_path = match.get("path") or match.get("_virtual_path")
                    res_id = resource_identity(obj.data)
                    target_path = rebase_to_resource_identity(
                        target_path, res_type, res_id
                    )
                    best[src_id] = {
                        "target": target_path,
                        "score": score,
                        "resource": res_type,
                    }
                    logger.info(
                        f"  Best so far: {src_id} → {target_path} ({res_id}, score={score:.3f})"
                    )

        mapping_table = {src_id: entry["target"] for src_id, entry in best.items()}
        logger.info(
            f"Combined automapping complete: {len(mapping_table)} source fields mapped."
        )
        return mapping_table

    def _save_automapping_table(self, mapping_table: Dict[str, str]) -> None:
        """Persist the combined automapping result in the source_data folder."""
        import json

        filename = f"{self.map_name}_automapping.json"
        try:
            self.app_state.dataIO.store_project_file(
                self.app_state.dataIO.ProjectFolders.SOURCE_DATA,
                filename,
                json.dumps(mapping_table, indent=2),
                mode="STR",
                overwrite=True,
            )
            logger.info(f"Saved combined automapping table to source_data/{filename}")
        except Exception as e:
            logger.warning(f"Could not save automapping table: {e}")

    def _init_automapper(
        self,
        automapping: bool,
        automapper_instance: FMLAutomapper = None,
        custom_mapping_table: dict = None,
    ) -> Optional[FMLAutomapper]:
        automapper = None
        if custom_mapping_table:
            logger.info("Custom mapping table provided. Automapping will be skipped.")
        elif automapping:
            if automapper_instance:
                logger.info("Using provided automapper instance...")
                automapper = automapper_instance
            else:
                logger.info("Initializing automapper...")
                automapper = FMLAutomapper(use_word2vec=True)
        return automapper

    def _extract_source_fields(
        self, helper_map: StructureDefinition
    ) -> List[Dict[str, Any]]:
        source_fields = []
        if helper_map.snapshot and helper_map.snapshot.element:
            for elem in helper_map.snapshot.element:
                # Include the element type so the factory can narrow value[x] selections
                elem_type = "string"
                if elem.type:
                    t = elem.type[0]
                    elem_type = (
                        t.code
                        if isinstance(t, str)
                        else (getattr(t, "code", None) or "string")
                    )
                source_fields.append(
                    {
                        "id": elem.id,
                        "path": elem.path,
                        "description": elem.short or elem.definition or "",
                        "type": elem_type,
                        "max": elem.max,
                        "extension": list(elem.extension or []),
                    }
                )
        return source_fields

    def _build_source_field_types(
        self, source_fields: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Return {field_local_name: fhir_type_code} from the source field list."""
        return {
            f["path"].split(".")[-1]: f.get("type", "string")
            for f in source_fields
            if f.get("path") and "." in f["path"]
        }

    def _get_automapped_mappings(
        self,
        custom_mapping_table: dict = None,
        automapper: FMLAutomapper = None,
        source_fields: List[Dict[str, Any]] = None,
        target_fields: List[Dict[str, Any]] = None,
        res_type: str = "",
        res_id: str = "",
    ) -> Tuple[Set[str], Dict[str, str]]:
        automapped_paths = set()
        automapped_mappings = {}
        self.factory.explicit_mapping_targets = set()
        self.factory.coded_code_leaf_sources = {}
        if custom_mapping_table:
            automapped_paths, automapped_mappings = self._apply_custom_mapping_table(
                custom_mapping_table,
                source_fields,
                target_fields,
                res_type,
                res_id,
            )
        elif automapper:
            logger.info(f"Running automapping for resource {res_id}...")
            for source_field in source_fields:
                match, score = automapper.find_mapping(source_field, target_fields)
                if match:
                    match_path = match.get("_virtual_path", match.get("path"))
                    target_path = match.get("path") or match_path
                    logger.info(
                        f"Automapped {source_field['id']} -> {target_path} (score: {score:.2f})"
                    )
                    automapped_paths.add(target_path)
                    automapped_mappings[target_path] = source_field["id"]
                    self.factory.explicit_mapping_targets.add(target_path)
                    if match_path and match_path != target_path:
                        automapped_paths.add(match_path)
                        automapped_mappings[match_path] = source_field["id"]
                        self.factory.explicit_mapping_targets.add(match_path)
        return automapped_paths, automapped_mappings

    @staticmethod
    def _type_structure_has(field, type_code, tail):
        """Whether `tail` names an element of `type_code`'s expanded structure."""
        if not tail:
            return True
        wanted = tail.lstrip(".")
        for type_ref in field.get("type") or []:
            if not isinstance(type_ref, dict) or type_ref.get("code") != type_code:
                continue
            for child in type_ref.get("type_structure") or []:
                if not isinstance(child, dict):
                    continue
                path = str(child.get("path") or "")
                if path.split("[x]", 1)[-1].lstrip(".") == wanted:
                    return True
        return False

    def _resolve_sliced_choice_target(self, target, target_index, field_by_identity):
        """Resolve `*<slice>.value<Type>.<rest>` when no exact index entry exists"""
        if ":" not in target or "[x]" in target:
            return None
        segments = target.split(".")
        # Longest slice-qualified ancestor that the index already knows.
        for cut in range(len(segments) - 1, 0, -1):
            prefix = ".".join(segments[:cut])
            if ":" not in prefix or prefix not in field_by_identity:
                continue
            remainder = segments[cut:]
            head, tail = remainder[0], "".join(f".{s}" for s in remainder[1:])
            owner = field_by_identity[prefix]
            candidates = []
            for child in owner.get("children") or []:
                if not isinstance(child, dict):
                    continue
                name = str(child.get("path") or "").rsplit(".", 1)[-1]
                if not name.endswith("[x]"):
                    continue
                base = name[: -len("[x]")]
                if not head.startswith(base) or head == base:
                    continue
                concrete = head[len(base):]
                for type_ref in child.get("type") or []:
                    code = (
                        type_ref.get("code") if isinstance(type_ref, dict) else type_ref
                    )
                    if not code or self.factory._choice_suffix(code) != concrete:
                        continue
                    if not self._type_structure_has(child, code, tail):
                        continue
                    candidates.append(f"{prefix}.{base}[x]:{head}{tail}")
            unique = sorted(set(candidates))
            if len(unique) == 1:
                return unique[0]
            if unique:
                self._append_diagnostic(
                    "mapping-target-choice-ambiguous",
                    "Custom mapping target names a choice that more than one element "
                    "of the slice could satisfy; qualify it explicitly.",
                    target=target,
                    candidates=unique,
                )
                return None
        return None

    def _apply_custom_mapping_table(
        self,
        mapping_table: Dict[str, Any],
        source_fields: List[Dict[str, Any]],
        target_fields: List[Dict[str, Any]],
        res_type: str,
        res_id: str = "",
    ) -> Tuple[Set[str], Dict[str, str]]:
        automapped_paths = set()
        automapped_mappings = {}

        if not isinstance(mapping_table, dict):
            logger.warning("Custom mapping table is not a dict. Skipping.")
            return automapped_paths, automapped_mappings

        if not hasattr(self.factory, "explicit_mapping_targets"):
            self.factory.explicit_mapping_targets = set()
        if not hasattr(self.factory, "coded_code_leaf_sources"):
            self.factory.coded_code_leaf_sources = {}
        if not hasattr(self, "mapping_diagnostics"):
            self.mapping_diagnostics = []
        # Collection declarations are scoped to the target profile currently
        # being generated.
        self.factory.collection_rules = {}

        source_index = {}
        relative_source_candidates = {}
        for field in source_fields or []:
            field_id = field.get("id")
            field_path = field.get("path")
            resolved_id = field_id or field_path
            if field_id:
                source_index[field_id] = field_id
            if field_path:
                source_index[field_path] = resolved_id
            # Mapping tables historically address source elements relative to the
            # generated logical model root (`givenName`, `contact.value`, ...).
            # Retain that documented shorthand, but only when it resolves
            # uniquely; exact full paths always take precedence.
            for candidate in {field_id, field_path} - {None}:
                if "." not in candidate:
                    continue
                relative = candidate.split(".", 1)[1]
                relative_source_candidates.setdefault(relative, set()).add(resolved_id)

        ambiguous_source_paths = set()
        for relative, candidates in relative_source_candidates.items():
            if relative in source_index:
                continue
            if len(candidates) == 1:
                source_index[relative] = next(iter(candidates))
            else:
                ambiguous_source_paths.add(relative)

        target_index = {}
        target_cardinality = {}

        def _flatten_targets(fields, parent_virtual_path=None):
            flat = []
            for f in fields or []:
                field_path = f.get("path")
                if not field_path:
                    continue
                field_name = field_path.split(".")[-1]
                virtual_path = (
                    f"{parent_virtual_path}.{field_name}"
                    if parent_virtual_path
                    else field_path
                )
                flat.append(
                    {
                        "path": field_path,
                        "virtual_path": virtual_path,
                        "field": f,
                    }
                )
                if "[x]" in field_path:
                    base = field_path.replace("[x]", "")
                    cts = list(f.get("choice_types") or [])
                    if not cts:
                        t = f.get("type")
                        if isinstance(t, list) and t:
                            cts = [
                                item.get("code") if isinstance(item, dict) else item
                                for item in t
                                if (item.get("code") if isinstance(item, dict) else item)
                            ]
                        elif isinstance(t, str):
                            cts = [t]
                    for ct in cts:
                        suffix = self.factory._choice_suffix(ct)
                        concrete_path = f"{base}{suffix}"
                        concrete_head = concrete_path.rsplit(".", 1)[-1]
                        qualified_path = f"{field_path}:{concrete_head}"
                        # Keep the author's concrete choice selection.  Mapping it back to
                        # the raw ``value[x]`` path discards the only type information a
                        # primitive ``copy`` transform has (N3).
                        target_index[concrete_path] = qualified_path
                        # The documented explicit form keeps the snapshot's `[x]`
                        # marker and appends the concrete choice after `:`:
                        # `effective[x]:effectiveDateTime`.
                        target_index[qualified_path] = qualified_path
                    # children of each candidate type, so a mapping table can address
                    # inside an unsliced multi-type choice (`value[x].coding.code`).
                    # Registered under both the `[x]` form (the parser's own child
                    # paths) and the concrete form (`valueCodeableConcept.coding.code`).
                    # `choice_structures` is only populated for top-level fields (conv_mappable);
                    # a choice nested in a backbone (`Claim.diagnosis.diagnosis[x]`) still has
                    # each candidate's children under `type[i].type_structure`.
                    structures = f.get("choice_structures") or {
                        t.get("code"): t.get("type_structure")
                        for t in (f.get("type") or [])
                        if isinstance(t, dict) and t.get("code") and t.get("type_structure")
                    }
                    for ct, structure in structures.items():
                        child_entries = _flatten_targets(structure, virtual_path)
                        flat.extend(child_entries)
                        suffix = self.factory._choice_suffix(ct)
                        if not suffix:
                            continue
                        for child in child_entries:
                            child_path = child["path"]
                            if child_path.startswith(f"{field_path}."):
                                tail = child_path[len(field_path) :]
                                concrete_head = f"{base}{suffix}".rsplit(".", 1)[-1]
                                qualified_child = (
                                    f"{field_path}:{concrete_head}{tail}"
                                )
                                target_index[f"{base}{suffix}{tail}"] = qualified_child
                                target_index[qualified_child] = qualified_child
                if f.get("children"):
                    flat.extend(_flatten_targets(f.get("children"), virtual_path))
                if f.get("type_structure"):
                    flat.extend(_flatten_targets(f.get("type_structure"), virtual_path))
                else:
                    # F4: a complex element nested in a backbone keeps its datatype children
                    # under `type[0].type_structure` — only top-level fields get them lifted
                    # to field level by conv_mappable. The snapshot's own children are a
                    # *partial* view (evo13's `participant.type` lists extension and text but
                    # not coding), so merge rather than choose: without this,
                    # `reaction.substance.coding.code` has no target to bind to while its
                    # sibling `reaction.manifestation.coding.code` resolves fine.
                    raw_types = f.get("type")
                    if isinstance(raw_types, list) and len(raw_types) == 1:
                        nested = raw_types[0].get("type_structure") if isinstance(raw_types[0], dict) else None
                        if nested:
                            flat.extend(_flatten_targets(nested, virtual_path))
                for sl in f.get("slices") or []:
                    if not isinstance(sl, dict):
                        continue
                    slice_name = sl.get("sliceName", "")
                    if not slice_name:
                        continue
                    slice_identity = sl.get("slice_identity") or sl.get("id")
                    if slice_identity and ":" in slice_identity:
                        slice_path = slice_identity
                    elif "." in field_path:
                        prefix, last_seg = field_path.rsplit(".", 1)
                        slice_path = f"{prefix}.{last_seg}:{slice_name}"
                    else:
                        slice_path = f"{field_path}:{slice_name}"
                    sl_copy = dict(sl)
                    sl_copy["path"] = slice_path
                    # `slice_path` already contains the sliced element. Reusing
                    # `virtual_path` here prepends that element a second time
                    # (`Medication.code.coding.coding:slice.code`) and makes the
                    # exact snapshot identity unreachable.
                    flat.extend(_flatten_targets([sl_copy], parent_virtual_path))
            return flat

        # Keeps the field record reachable by its slice-carrying identity. The
        # on-demand fallback below has to inspect the declared types of a slice's
        # `value[x]`, which the flat index alone cannot answer.
        field_by_identity = {}

        for entry in _flatten_targets(target_fields or []):
            field_path = entry["path"]
            virtual_path = entry["virtual_path"]
            cardinality = (entry.get("field") or {}).get("cardinality") or {}
            if entry.get("field") is not None:
                field_by_identity.setdefault(virtual_path, entry["field"])
                field_by_identity.setdefault(field_path, entry["field"])

            target_index[field_path] = field_path
            target_cardinality[field_path] = cardinality
            target_index[virtual_path] = (
                virtual_path if ":" in virtual_path else field_path
            )
            target_cardinality[virtual_path] = cardinality

            if field_path.startswith(f"{res_type}."):
                clean_path = field_path[len(f"{res_type}.") :]
                target_index[clean_path] = field_path
            if virtual_path.startswith(f"{res_type}."):
                clean_virtual = virtual_path[len(f"{res_type}.") :]
                target_index[clean_virtual] = (
                    virtual_path if ":" in clean_virtual else field_path
                )

        collection_candidates = {}
        # Process typed collection parents before their legacy descendants so
        # inferred ancestor providers cannot overwrite the intended source
        # iteration context. Legacy-only tables retain their insertion order.
        ordered_entries = sorted(
            mapping_table.items(),
            key=lambda entry: 0 if isinstance(entry[1], dict) else 1,
        )

        for source_key, target_value in ordered_entries:
            if is_rule_document_key(source_key):
                continue
            target_text = mapping_target_path(target_value)
            if not target_text:
                if isinstance(target_value, dict):
                    self._append_diagnostic(
                        "invalid-collection-rule",
                        "collection rule requires a non-empty string 'target'",
                        source=source_key,
                    )
                continue

            prefix = target_text.split(".")[0] if "." in target_text else target_text
            if prefix != res_type and prefix != res_id:
                continue
            if prefix == res_id and prefix != res_type:
                normalized_target = f"{res_type}.{target_text[len(prefix)+1:]}"
            else:
                normalized_target = target_text

            source_id = source_index.get(source_key)
            source_is_todo = isinstance(source_key, str) and source_key.startswith(
                ("TODO_", "TODO-")
            )
            source_valid = source_id is not None or source_is_todo
            if source_is_todo:
                # Sparse mapping tables deliberately use non-matching TODO source
                # elements as human-fillable scaffolds.
                source_id = source_key
            elif source_key in ambiguous_source_paths:
                self._append_diagnostic(
                    "mapping-source-path-ambiguous",
                    "Relative custom mapping source matches multiple elements in "
                    "the representative source structure. Use the full source path.",
                    source=source_key,
                    target=target_text,
                )
                logger.warning(
                    "Relative custom mapping source is ambiguous; use a full path: %s",
                    source_key,
                )
            elif source_id is None:
                self._append_diagnostic(
                    "mapping-source-path-not-found",
                    "Custom mapping source is absent from the representative "
                    "source structure.",
                    source=source_key,
                    target=target_text,
                )
                logger.warning(
                    "Custom mapping source not found in representative source: %s",
                    source_key,
                )

            target_path = target_index.get(normalized_target)
            if not target_path and not normalized_target.startswith(f"{res_type}."):
                target_path = target_index.get(f"{res_type}.{normalized_target}")

            if not target_path:
                # Exact lookups first; only a target that matched nothing is worth
                # re-interpreting, so nothing that already resolves can be redirected.
                target_path = self._resolve_sliced_choice_target(
                    normalized_target, target_index, field_by_identity
                )

            if not target_path:
                self._append_diagnostic(
                    "mapping-target-path-not-found",
                    "Custom mapping target is absent from the target profile snapshot. "
                    "Use an exact element, choice variant, or slice path.",
                    source=source_key,
                    target=target_text,
                )
                logger.warning(
                    "Custom mapping target not found in %s (id=%s): %s",
                    res_type,
                    res_id,
                    target_text,
                )

            if not source_valid or not target_path:
                continue

            if isinstance(target_value, dict):
                try:
                    spec = parse_collection_rule(source_key, target_value)
                except ValueError as exc:
                    self._append_diagnostic(
                        "invalid-collection-rule",
                        str(exc),
                        source=source_key,
                        target=target_text,
                    )
                    spec = CollectionRuleSpec(
                        source=source_key,
                        target=target_text,
                        invalid=True,
                    )
                if spec:
                    collection_candidates[target_path] = CollectionRuleSpec(
                        source=source_id,
                        target=target_path,
                        source_list_mode=spec.source_list_mode,
                        target_list_modes=spec.target_list_modes,
                        list_rule_id=spec.list_rule_id,
                        source_key=spec.source_key,
                        target_key=spec.target_key,
                        invalid=spec.invalid,
                    )

            automapped_paths.add(target_path)
            previous_source = automapped_mappings.get(target_path)
            if previous_source and previous_source != source_id:
                diagnostic = {
                    "code": "duplicate-target-assignment",
                    "target": target_path,
                    "previous_source": previous_source,
                    "source": source_id,
                }
                if not hasattr(self, "mapping_diagnostics"):
                    self.mapping_diagnostics = []
                self.mapping_diagnostics.append(diagnostic)
                logger.warning(
                    "Duplicate custom mapping target %s: %s is overwritten by %s.",
                    target_path,
                    previous_source,
                    source_id,
                )
            automapped_mappings[target_path] = source_id
            self.factory.explicit_mapping_targets.add(target_path)
            ancestor = target_path
            skip_add = False
            while True:
                last_seg = ancestor.split(".")[-1]
                if ":" in last_seg:
                    ancestor = ancestor.rsplit(":", 1)[0]
                    skip_add = True
                    continue
                elif "." in ancestor:
                    # Strip last dot-segment
                    ancestor = ancestor.rsplit(".", 1)[0]
                    if ":" not in ancestor and not ancestor.endswith("extension"):
                        break
                else:
                    break
                skip_add = False
                if target_path.rsplit(".", 1)[-1] == "code":
                    store = self.factory.coded_code_leaf_sources
                    store[ancestor] = source_id
                    plain = _re_slice.sub("", ancestor)
                    if plain != ancestor:
                        if plain not in store:
                            store[plain] = source_id
                        elif store[plain] != source_id:
                            store[plain] = None
                if not skip_add and ancestor not in automapped_mappings:
                    automapped_mappings[ancestor] = source_id
                    automapped_paths.add(ancestor)

        for target_path, spec in collection_candidates.items():
            invalid = spec.invalid
            source_field = next(
                (
                    field
                    for field in source_fields or []
                    if spec.source in (field.get("id"), field.get("path"))
                ),
                None,
            )
            source_base = (
                source_field.get("path") if source_field is not None else spec.source
            )

            def _known_non_repeating(maximum):
                if maximum is None:
                    return False
                if maximum in ("*", "n"):
                    return False
                try:
                    return int(maximum) <= 1
                except (TypeError, ValueError):
                    return False

            source_max = source_field.get("max") if source_field else None
            if _known_non_repeating(source_max):
                invalid = True
                self._append_diagnostic(
                    "collection-source-not-repeating",
                    "A collection rule requires a repeating source element in "
                    "the representative source snapshot.",
                    source=spec.source,
                    source_max=source_max,
                    target=target_path,
                )
            target_max = (target_cardinality.get(target_path) or {}).get("max")
            if _known_non_repeating(target_max):
                invalid = True
                self._append_diagnostic(
                    "collection-target-not-repeating",
                    "A collection rule requires a repeating target element.",
                    source=spec.source,
                    target=target_path,
                    target_max=target_max,
                )
            if spec.source_key:
                source_key_path = f"{source_base}.{spec.source_key}"
                expected_source = source_index.get(source_key_path)
                if expected_source is None:
                    invalid = True
                    self._append_diagnostic(
                        "collection-source-key-not-found",
                        "The declared correlation source key is absent from the "
                        "representative source snapshot.",
                        source=spec.source,
                        source_key=spec.source_key,
                        target=target_path,
                    )
                if spec.target_key:
                    target_key_path = f"{target_path}.{spec.target_key}"
                    mapped_source = automapped_mappings.get(target_key_path)
                    if expected_source is not None and mapped_source != expected_source:
                        invalid = True
                        self._append_diagnostic(
                            "collection-target-key-not-mapped",
                            "The declared source correlation key is not mapped to "
                            "the declared target key in the same collection.",
                            source_key=source_key_path,
                            target_key=target_key_path,
                            mapped_source=mapped_source,
                        )
            if invalid:
                spec = CollectionRuleSpec(
                    source=spec.source,
                    target=spec.target,
                    source_list_mode=spec.source_list_mode,
                    target_list_modes=spec.target_list_modes,
                    list_rule_id=spec.list_rule_id,
                    source_key=spec.source_key,
                    target_key=spec.target_key,
                    invalid=True,
                )
            self.factory.collection_rules[target_path] = spec

        return automapped_paths, automapped_mappings

    def create_res_group(
        self,
        res_type: str,
        res_obj: Any,
        create_references: bool,
        source_obj: Any = None,
        automapped_mappings: Dict[str, str] = None,
    ) -> StructureMapGroup:
        """
        Create a resource group for a structuremap
        Input:
            res_type: (underlying) type of the given resource
            res_obj: (actual) resource object
            create_references: if references should be created
            automapped_mappings: dict of target_path -> source_field_id
        """
        if res_type == "Questionnaire" and _is_questionnaire_instance(res_obj.data):
            logger.info(
                f"Delegating StructureMap generation for Questionnaire {res_obj.data.id}"
            )
            creator = QuestionnaireMapCreator(res_obj, factory=self.factory)
            return creator.generate_group(
                source_alias="Source",
                target_alias=resource_identity(res_obj.data),
                mapping_table=self.custom_mapping_table,
            )

        logger.info("Creating group for resource %s", res_obj.data.id)
        source_type = source_obj.type if source_obj else "SourceData"
        group = StructureMapGroup.model_construct(
            name=fhir_map_token(f"Transform-{resource_identity(res_obj.data)}"),
            typeMode="none",
        )
        group.input = [
            StructureMapGroupInput.model_construct(
                name="source", mode="source", type=source_type
            ),
            StructureMapGroupInput.model_construct(
                name="target", mode="target", type=res_type
            ),
        ]
        logger.info(
            "Creating rules for mappable fields of resource %s", res_obj.data.id
        )
        # `id` is 0..1 and published IGs do ship StructureDefinitions without one
        # (fhir.bfarm.de is one). Keying the mapping-table lookup on a missing `id`
        # makes every row unbindable, so each rule silently falls back to its
        # `TODO_MAP_*` placeholder and the transform emits a well-formed but empty
        # resource, reporting no error. `resource_identity` is the same fallback the
        # map name, the file name and the target alias already use.
        self.factory._current_profile_id = resource_identity(res_obj.data)
        self.factory._current_profile_sd = res_obj.data
        self.factory.record_unindexed_snapshot_slices(
            res_obj.data, res_obj.mappable_fields
        )
        field_rules = self.factory.create_field_rules(
            res_type,
            res_obj.mappable_fields,
            parent_source_context="source",
            parent_target_context="target",
            create_references=create_references,
            automapped_mappings=automapped_mappings,
        )
        explicit_meta_profile = any(
            path == f"{res_type}.meta.profile"
            for path in (automapped_mappings or {})
        )
        if self._target_declares_meta(res_obj, res_type) and not explicit_meta_profile:
            # Precedence: an authored mapping (handled above) > a canonical the
            # snapshot pins on meta.profile > the StructureDefinition's own url.
            pinned = self._snapshot_fixed_meta_profile(res_obj, res_type)
            declared = res_obj.data.url
            if pinned and pinned != declared:
                self._append_diagnostic(
                    "profile-canonical-conflict",
                    f"{res_type}.meta.profile is pinned to {pinned} but the "
                    f"StructureDefinition's canonical is {declared}. Stamping the "
                    "pinned value so the instance satisfies its own profile; note "
                    "that the pinned canonical may not resolve on the server.",
                    path=f"{res_type}.meta.profile",
                    severity="warning",
                )
            meta_rule = self.factory.create_meta_profile_rule(
                pinned or declared,
                source_context="source",
                target_context="target",
            )
            group.rule = [meta_rule] + field_rules
        else:
            # matchbox resolves $transform target element names against the profile snapshot
            # addin meta.profile on a profile whose
            # (minimal/incomplete) snapshot omits `meta` makes the transform
            # fail with "Unrecognised name meta on <Type>". Skip it there.
            logger.info(
                "Skipping meta.profile stamping for %s: snapshot does not "
                "declare %s.meta (incomplete snapshot).",
                res_obj.data.id,
                res_type,
            )
            group.rule = field_rules
        return group

    def _snapshot_fixed_meta_profile(self, res_obj, res_type) -> Optional[str]:
        """A ``fixedCanonical``/``patternCanonical`` pinned on ``<res_type>.meta.profile``.

        A profile may pin the canonical its instances must declare, and that value
        is not always the StructureDefinition's own ``url`` — evo13 pins
        ``http://…/E44_EVO13_PR_CorrectionRequest`` on an SD whose url is the
        ``https://`` form. Stamping the url then fails the fixed value. The pinned
        value wins; the caller reports the disagreement rather than hiding it.
        """
        data = getattr(res_obj, "data", None)
        snapshot = getattr(data, "snapshot", None)
        elements = getattr(snapshot, "element", None) if snapshot else None
        target = f"{res_type}.meta.profile"
        for element in elements or []:
            if getattr(element, "path", None) != target:
                continue
            for attribute in ("fixedCanonical", "patternCanonical",
                              "fixedUri", "patternUri"):
                value = getattr(element, attribute, None)
                if isinstance(value, str) and value:
                    return value
        return None

    def _target_declares_meta(self, res_obj, res_type) -> bool:
        """check if the target profile's snapshot declares ``<res_type>.meta``"""
        data = getattr(res_obj, "data", None)
        snapshot = getattr(data, "snapshot", None)
        elements = getattr(snapshot, "element", None) if snapshot else None
        if not elements:
            return True
        meta_path = f"{res_type}.meta"
        return any(getattr(e, "path", None) == meta_path for e in elements)

    def _add_required_path(self, path: Optional[str], collector: Set[str]) -> None:
        """
        Add the given path and all its ancestors to the collector set
        """
        if not path:
            return
        segments = path.split(".")
        prefix = []
        for segment in segments:
            if not segment:
                continue
            prefix.append(segment)
            collector.add(".".join(prefix))

    def _collect_required_paths_from_fields(
        self,
        fields: List[Dict[str, Any]],
        collector: Set[str],
        active_types: Set[str] = None,
    ) -> bool:
        """
        Recursively collect required field paths from a field tree. Returns True if any field was kept.
        """
        any_kept = False
        for field in fields or []:
            if not isinstance(field, dict):
                continue
            path = field.get("path")
            is_req = field.get("is_required")
            min_c = field.get("cardinality", {}).get("min", 0)
            must_support = bool(field.get("must_support"))

            should_keep = (
                is_req
                or min_c >= 1
                or has_fixed_value(field.get("fixed_value"))
                or must_support
            )

            if not should_keep and self.create_references:
                is_ref = field.get("type") == "Reference" or field.get(
                    "reference_target"
                )
                if is_ref:
                    target = field.get("reference_target")
                    if (
                        target
                        and active_types
                        and target in active_types
                        and field.get("reference_target_in_registry")
                    ):
                        should_keep = True

            # Check children/nested structures and propagate kept status up
            children_kept = False
            if field.get("children"):
                if self._collect_required_paths_from_fields(
                    field["children"], collector, active_types
                ):
                    children_kept = True

            if field.get("type_structure"):
                self._collect_required_paths_from_fields(
                    field["type_structure"], collector, active_types
                )

            field_types = field.get("type")
            if isinstance(field_types, list):
                for type_info in field_types:
                    if not isinstance(type_info, dict) or not type_info.get(
                        "type_structure"
                    ):
                        continue
                    self._collect_required_paths_from_fields(
                        type_info["type_structure"], collector, active_types
                    )

            if field.get("slices"):
                self._collect_required_paths_from_fields(
                    field["slices"], collector, active_types
                )

            # Recurse into profile fields if present
            field_type = field.get("type")
            if isinstance(field_type, list):
                for t in field_type:
                    if isinstance(t, dict) and t.get("profile"):
                        for p in t["profile"]:
                            if isinstance(p, list):  # List of fields
                                if self._collect_required_paths_from_fields(
                                    p, collector, active_types
                                ):
                                    children_kept = True

            if children_kept:
                field_type_val = field.get("type")
                is_backbone = field_type_val == "BackboneElement" or (
                    isinstance(field_type_val, list)
                    and any(
                        (isinstance(t, dict) and t.get("code") == "BackboneElement")
                        or t == "BackboneElement"
                        for t in field_type_val
                    )
                )
                if not is_backbone or (is_req or min_c >= 1):
                    should_keep = True

            if should_keep:
                self._add_required_path(path, collector)
                any_kept = True

        return any_kept

    def _should_keep_path(self, path: Optional[str], allowed_paths: Set[str]) -> bool:
        return bool(path and path in allowed_paths)

    def _prune_nested_components(
        self,
        field: Dict[str, Any],
        allowed_paths: Set[str],
        mapped_paths: Set[str] = None,
    ) -> None:
        """
        remove nested child/type_structure entries that are not required
        """
        for key in ("children", "type_structure", "slices"):
            nested_entries = field.get(key) or []
            if not nested_entries:
                continue

            def _norm(p):
                return ".".join(
                    s.split(":")[0].replace("[x]", "") for s in (p or "").split(".")
                )

            pruned_entries = []
            for entry in nested_entries:
                entry_path = entry.get("path")
                keep = self._should_keep_path(entry_path, allowed_paths)
                if not keep and has_fixed_value(entry.get("fixed_value")):
                    keep = True
                if not keep and entry_path:
                    ep = _norm(entry_path)
                    for ap in (
                        mapped_paths if mapped_paths is not None else allowed_paths
                    ):
                        apn = _norm(ap)
                        # entry is the mapped leaf, or an ancestor backbone of a mapped path
                        if apn == ep or apn.startswith(ep + "."):
                            keep = True
                            break
                if keep:
                    self._prune_nested_components(
                        entry, allowed_paths, mapped_paths=mapped_paths
                    )
                    pruned_entries.append(entry)
            field[key] = pruned_entries

        field_types = field.get("type")
        if isinstance(field_types, list):
            for type_info in field_types:
                if isinstance(type_info, dict) and type_info.get("type_structure"):
                    self._prune_nested_components(
                        type_info, allowed_paths, mapped_paths=mapped_paths
                    )
