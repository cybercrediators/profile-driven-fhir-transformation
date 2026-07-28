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
from mapping.fml_creator.fml_helper import flatten_profile_fields
from mapping.fml_creator.fml_automapper import FMLAutomapper
from helpers.utils import get_value_from_element, resource_identity

from typing import List, Any, Dict, Optional, Set, Tuple
import logging
import re
logger = logging.getLogger(__name__)
import hashlib

from fhir.resources.R4B.structuredefinition import StructureDefinition
from mapping.fml_creator.fml_questionnaire import QuestionnaireMapCreator
from data_handling.app_state import AppState


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
        self.current_profile_name = "unknown"
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

        # check if structure map(s) already exists
        if check_exist := self.app_state.dataIO.check_processed_structure_maps():
            if not self.overwrite:
                logger.info(
                    "Structure map(s) already exist! Loading previous structure maps..."
                )
                sms = self.app_state.dataIO.load_project_files(
                    self.app_state.dataIO.ProjectFolders.STRUCTURE_MAPS
                )
                for _, sm in sms:
                    if (
                        isinstance(sm, dict)
                        and sm.get("resourceType") == "StructureMap"
                    ):
                        self.structure_maps.append(StructureMap(**sm))
            else:
                logger.info("Overwriting existing structure maps...")

        self.check_exist = check_exist

    def get_structure_maps(self) -> List[StructureMap]:
        return self.structure_maps

    def generate(self) -> List[StructureMap]:
        if self.check_exist and not self.overwrite and self.structure_maps:
            return self.structure_maps

        # proceed with generation
        resources, source_fields, active_types = self._prepare_resources_and_fields()

        if self.automapper:
            combined_table = self._build_combined_automapping(resources, source_fields)
            self._save_automapping_table(combined_table)
            self.custom_mapping_table = combined_table
            self.automapper = None

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

            automapped_mappings = self._process_resource_mapping(
                obj, res_type, source_fields, active_types
            )

            sm_url = f"{self.map_url}-{profile_identity}"
            sm_name = f"{res_idx + 1:03d}_{self.map_name}-{profile_identity}"
            sm_title = f"{self.map_title} - {profile_identity}"

            sm = self.factory.create_base_structure_map(
                sm_url, sm_name, sm_title, self.status
            )
            # Deterministic ID (like ConceptMaps) so repeated runs produce identical files
            url_hash = hashlib.md5(sm_url.encode("utf-8")).hexdigest()[:8]
            sm.id = f"sm-{profile_identity[:50]}-{url_hash}"
            sm.description = (
                "Auto-generated StructureMap for given profile mappable fields "
                f"(Resource: {profile_identity}). Placeholders have to be set (or use auto-mapping)"
            )

            # For Questionnaire resources the transform target is QuestionnaireResponse,
            # so point to the QR base SD rather than the Questionnaire canonical URL.
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

            res_group = self.create_res_group(
                res_type,
                obj,
                self.create_references,
                source_obj=self.helper_map,
                automapped_mappings=automapped_mappings,
            )

            groups = [res_group]
            sm.group = groups

            self.app_state.dataIO.store_project_file(
                self.app_state.dataIO.ProjectFolders.STRUCTURE_MAPS,
                f"{sm_name}.json",
                sm.model_dump_json(indent=2),
                mode="STR",
                overwrite=self.overwrite,
            )
            structure_maps.append(sm)

        self.structure_maps = structure_maps

        self._save_coverage_report()

        self._cleanup_stale_concept_maps(structure_maps)
        self._cleanup_stale_structure_maps(structure_maps)

        return structure_maps

    def _cleanup_stale_structure_maps(self, structure_maps):
        """delete generated-named SM files in structure_maps/ that this run did not produce"""

        folder = self.app_state.dataIO.ProjectFolders.STRUCTURE_MAPS
        sm_dir = self.app_state.dataIO.project_dir / folder.value
        if not sm_dir.is_dir():
            return
        current = {f"{sm.name}.json" for sm in structure_maps}
        generated_name = re.compile(rf"^\d{{3}}_{re.escape(self.map_name)}-.+\.json$")
        for f in sm_dir.iterdir():
            if f.name in current or not generated_name.match(f.name):
                continue
            logger.info(f"Removing stale StructureMap: {f.name}")
            self.app_state.dataIO.delete_file(folder.value + "/" + f.name)

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
                if isinstance(v, str) and "." in v:
                    active_types.add(v.split(".")[0])

        source_field_types = self._build_source_field_types(source_fields)
        self.factory.source_field_types = source_field_types

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
        self, obj, res_type, source_fields, active_types: Set[str] = None
    ):
        flat_fields = flatten_profile_fields(
            self.app_state, res_type, obj.mappable_fields
        )

        automap_source_list = self._flatten_fields_recursively(flat_fields)

        automapped_paths, automapped_mappings = self._get_automapped_mappings(
            self.custom_mapping_table,
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
                if value is not None and value != []:
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

    def _save_coverage_report(self):
        """persist the aggregated required-element coverage report to the project's source_data folder"""

        if not self._coverage:
            return
        import json

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
            "report_version": 2,
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
        }
        try:
            self.app_state.dataIO.store_project_file(
                self.app_state.dataIO.ProjectFolders.SOURCE_DATA,
                f"{self.map_name}_coverage.json",
                json.dumps(report, indent=2),
                mode="STR",
                overwrite=True,
            )
            logger.info(
                "Required-element coverage: %d/%d mapped across %d profiles (%s unmapped).",
                covered,
                total_req,
                len(self._coverage),
                total_unmapped,
            )
        except Exception as e:
            logger.warning("Could not write coverage report: %s", e)

    def _build_combined_automapping(
        self, resources: list, source_fields: list
    ) -> Dict[str, str]:
        """automapping for every source field against every resource's target fields"""
        # {source_field_id: {"target": path, "score": float, "resource": res_type}}
        best: Dict[str, dict] = {}

        for res_dict in resources:
            url = next(iter(res_dict))
            obj = self.app_state.registry.get_obj_by_name(url)
            if not obj or not obj.data:
                continue
            res_type = obj.res_type if not hasattr(obj.data, "type") else obj.data.type
            flat_fields = flatten_profile_fields(
                self.app_state, res_type, obj.mappable_fields
            )
            target_fields = self._flatten_fields_recursively(flat_fields)

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
                    if (
                        target_path
                        and res_id
                        and res_id != res_type
                        and target_path.startswith(f"{res_type}.")
                    ):
                        target_path = f"{res_id}.{target_path[len(res_type)+1:]}"
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
                    if match_path and match_path != target_path:
                        automapped_paths.add(match_path)
                        automapped_mappings[match_path] = source_field["id"]
        return automapped_paths, automapped_mappings

    def _apply_custom_mapping_table(
        self,
        mapping_table: Dict[str, str],
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

        source_index = {}
        for field in source_fields or []:
            field_id = field.get("id")
            field_path = field.get("path")
            if field_id:
                source_index[field_id] = field_id
            if field_path:
                source_index[field_path] = field_id or field_path

        target_index = {}

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
                flat.append({"path": field_path, "virtual_path": virtual_path})
                if "[x]" in field_path:
                    base = field_path.replace("[x]", "")
                    cts = list(f.get("choice_types") or [])
                    if not cts:
                        t = f.get("type")
                        if isinstance(t, list) and t:
                            tc = t[0].get("code") if isinstance(t[0], dict) else t[0]
                            if tc:
                                cts = [tc]
                        elif isinstance(t, str):
                            cts = [t]
                    for ct in cts:
                        suffix = self.factory._choice_suffix(ct)
                        concrete_path = f"{base}{suffix}"
                        target_index[concrete_path] = field_path
                    # children of each candidate type, so a mapping table can address
                    # inside an unsliced multi-type choice (`value[x].coding.code`).
                    # Registered under both the `[x]` form (the parser's own child
                    # paths) and the concrete form (`valueCodeableConcept.coding.code`).
                    for ct, structure in (f.get("choice_structures") or {}).items():
                        child_entries = _flatten_targets(structure, virtual_path)
                        flat.extend(child_entries)
                        suffix = self.factory._choice_suffix(ct)
                        if not suffix:
                            continue
                        for child in child_entries:
                            child_path = child["path"]
                            if child_path.startswith(f"{field_path}."):
                                tail = child_path[len(field_path) :]
                                target_index[f"{base}{suffix}{tail}"] = child_path
                if f.get("children"):
                    flat.extend(_flatten_targets(f.get("children"), virtual_path))
                if f.get("type_structure"):
                    flat.extend(_flatten_targets(f.get("type_structure"), virtual_path))
                for sl in f.get("slices") or []:
                    if not isinstance(sl, dict):
                        continue
                    slice_name = sl.get("sliceName", "")
                    if not slice_name:
                        continue
                    if "." in field_path:
                        prefix, last_seg = field_path.rsplit(".", 1)
                        slice_path = f"{prefix}.{last_seg}:{slice_name}"
                    else:
                        slice_path = f"{field_path}:{slice_name}"
                    sl_copy = dict(sl)
                    sl_copy["path"] = slice_path
                    flat.extend(_flatten_targets([sl_copy], virtual_path))
            return flat

        for entry in _flatten_targets(target_fields or []):
            field_path = entry["path"]
            virtual_path = entry["virtual_path"]

            target_index[field_path] = field_path
            target_index[virtual_path] = (
                virtual_path if ":" in virtual_path else field_path
            )

            if field_path.startswith(f"{res_type}."):
                clean_path = field_path[len(f"{res_type}.") :]
                target_index[clean_path] = field_path
            if virtual_path.startswith(f"{res_type}."):
                clean_virtual = virtual_path[len(f"{res_type}.") :]
                target_index[clean_virtual] = (
                    virtual_path if ":" in clean_virtual else field_path
                )

        for source_key, target_value in mapping_table.items():
            if not isinstance(target_value, str) or not target_value:
                continue

            prefix = target_value.split(".")[0] if "." in target_value else target_value
            if prefix != res_type and prefix != res_id:
                continue
            if prefix == res_id and prefix != res_type:
                normalized_target = f"{res_type}.{target_value[len(prefix)+1:]}"
            else:
                normalized_target = target_value

            source_id = source_index.get(source_key, source_key)

            target_path = target_index.get(normalized_target)
            if not target_path and not normalized_target.startswith(f"{res_type}."):
                target_path = target_index.get(f"{res_type}.{normalized_target}")

            if not target_path:
                base_candidate = (
                    normalized_target.split(":")[0]
                    if ":" in normalized_target
                    else normalized_target.rsplit(".", 1)[0]
                )
                if base_candidate in target_index:
                    # If the base resolves to a choice element (value[x]) keyed under its
                    # concrete name (valueQuantity), rewrite the whole target to the value[x]
                    # form so a sub-leaf mapping (valueQuantity.value) matches the actual field
                    # path (value[x].value) — otherwise the leaf's source never resolves.
                    resolved_base = target_index[base_candidate]
                    sub = normalized_target[len(base_candidate):]
                    if "[x]" in str(resolved_base) and resolved_base != base_candidate:
                        target_path = f"{resolved_base}{sub}"
                    else:
                        target_path = normalized_target

            if not target_path:
                logger.warning(
                    "Custom mapping target not found in %s (id=%s): %s",
                    res_type,
                    res_id,
                    target_value,
                )
                continue

            automapped_paths.add(target_path)
            automapped_mappings[target_path] = source_id
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
                if not skip_add and ancestor not in automapped_mappings:
                    automapped_mappings[ancestor] = source_id
                    automapped_paths.add(ancestor)

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
        if res_type == "Questionnaire":
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
            name=f"Transform-{resource_identity(res_obj.data)}", typeMode="none"
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
        if res_type == "Questionnaire":
            # group.rule = create_qr_response_rules(res_obj, parent_source_context="source", parent_target_context="target")
            logger.warning("Questionnaire currently not supported!")
        else:
            self.factory._current_profile_id = getattr(res_obj.data, "id", None)
            self.factory._current_profile_sd = res_obj.data
            field_rules = self.factory.create_field_rules(
                res_type,
                res_obj.mappable_fields,
                parent_source_context="source",
                parent_target_context="target",
                create_references=create_references,
                automapped_mappings=automapped_mappings,
            )
            if self._target_declares_meta(res_obj, res_type):
                meta_rule = self.factory.create_meta_profile_rule(
                    res_obj.data.url, source_context="source", target_context="target"
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
                is_req or min_c >= 1 or bool(field.get("fixed_value")) or must_support
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
                if not keep and entry.get("fixed_value"):
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
