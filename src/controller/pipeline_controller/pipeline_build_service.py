from dataclasses import dataclass
from pathlib import Path

import logging
logger = logging.getLogger(__name__)
from helpers import utils
from parser import resource_processing
from mapping import fml_structure
from mapping.fml_map import StructureMapGenerator
from fhir.resources.R4B.structuredefinition import StructureDefinition

from data_handling.app_state import AppState
from controller.pipeline_controller.pipeline_state import PipelineState


@dataclass
class BuildOptions:
    """build configuration (mostly CLI/overwrite flags) for the build service."""

    project_name: str = ""
    input_source_example: str = ""
    force_overwrite: bool = False
    reprocess_profile: bool = False
    reprocess_helper_definition: bool = False
    reprocess_structure_map: bool = False
    create_references: bool = False
    minimal_mode: bool = False
    automapping: bool = False
    modular_structure_maps: bool = False


class PipelineBuildService:
    """process profiles, generate the source helper SD and StructureMap(s)."""

    def __init__(
        self,
        app_state: AppState,
        state: PipelineState,
        options: BuildOptions,
        *,
        plugins=None,
        automapper=None,
    ):
        self.app_state = app_state
        self.state = state
        self.options = options
        self.plugins = plugins or []
        self.automapper = automapper

    def initial_processing(self):
        """Run all pipeline steps in sequence (parse → source-def → static-gen-sm)."""
        logger.info("Starting initial processing...")
        if not self.run_process():
            return None
        if not self.run_source_def():
            return None
        if not self.run_static_gen_sm():
            return None
        logger.info(
            "Initial processing completed. Adjust StructureMaps to be used for data transformation as needed!"
        )

    def run_process(self) -> bool:
        """Step 1: Process profile files and populate registry."""
        logger.info("Running process step...")
        file_names = self.app_state.dataIO.get_json_profile_files(
            self.app_state.conf["profile_path"]
        )
        self.process_files(
            file_names,
            self.app_state,
            self.options.force_overwrite or self.options.reprocess_profile,
        )
        logger.info("Profile processing complete.")
        return True

    def _ensure_processed(self) -> bool:
        """Run process step as a dependency without propagating force_overwrite."""
        file_names = self.app_state.dataIO.get_json_profile_files(
            self.app_state.conf["profile_path"]
        )
        self.process_files(file_names, self.app_state, self.options.reprocess_profile)
        return True

    def run_source_def(self) -> bool:
        """Step 2: Generate source helper StructureDefinition from source data example."""
        logger.info("Running source-def step...")
        if not self._ensure_processed():
            return False
        sd_name, sd_id, sd_url, *_ = self._naming_scheme()
        helper_map = self.create_source_map(
            Path(self.options.input_source_example),
            sd_id,
            sd_url,
            sd_name,
            overwrite=self.options.force_overwrite
            or self.options.reprocess_helper_definition,
        )
        if not helper_map:
            logger.error("Failed to create source helper map.")
            return False
        if sd_url not in self.state.source_helper_urls:
            self.state.source_helper_urls.append(sd_url)
        logger.info("Source helper StructureDefinition complete.")
        return True

    def run_static_gen_sm(self) -> bool:
        """Step 3: Generate StructureMap(s) statically from profile + source helper SD."""
        logger.info("Running static-gen-sm step...")
        if not self.options.automapping and not self.state.custom_mapping_table:
            logger.error(
                "No field mapping source provided for StructureMap generation. "
                "Pass --auto-mapping (-am) to enable automapping, or provide a mapping table via --mapping-table-path (-mt)."
            )
            return False
        if self.state.custom_mapping_table:
            logger.info(
                "Using custom mapping table for StructureMap generation (automapping skipped)."
            )

        # Hook 2: let plugins enrich the automapping table before SM generation
        if self.plugins and self.state.custom_mapping_table:
            for plugin in self.plugins:
                try:
                    self.state.custom_mapping_table = plugin.enrich_automapping(
                        self.state.custom_mapping_table
                    )
                except Exception as e:
                    logger.warning(
                        "Plugin '%s' enrich_automapping failed: %s", plugin.plugin_id, e
                    )

        if not self._ensure_processed():
            return False
        sd_name, sd_id, sd_url, sm_name, sm_url, sm_title = self._naming_scheme()
        helper_map = self._load_helper_map(sd_url)
        if not helper_map:
            logger.error(
                "Source helper map not found in cache or disk. Run 'pipeline source-def' first."
            )
            return False
        sms = self.generate_structure_maps(
            sm_url,
            sm_name,
            sm_title,
            helper_map,
            minimal=self.options.minimal_mode,
            create_references=self.options.create_references,
            automapping=self.options.automapping,
            modular=self.options.modular_structure_maps,
            overwrite=self.options.force_overwrite
            or self.options.reprocess_structure_map,
        )
        if not sms:
            logger.error("Failed to generate StructureMap(s).")
            return False
        if len(sms) == 1:
            self.state.structure_map_urls.append(sms[0].url)
        else:
            self.state.structure_map_urls.extend([s.url for s in sms if s])
        logger.info(
            "StructureMap generation complete. Adjust as needed before uploading to matchbox."
        )
        return True

    def _naming_scheme(self) -> tuple:
        """Return (sd_name, sd_id, sd_url, sm_name, sm_url, sm_title) for this project."""
        base_url = self.app_state.conf.get(
            "base_profile_url", "http://example.org"
        ).rstrip("/")
        project_name = self.options.project_name
        sd_id = "source-definition-" + self._clean_name(project_name)
        return (
            "SourceDefinition_" + project_name,
            sd_id,
            f"{base_url}/StructureDefinition/{sd_id}",
            "structure_map_" + project_name.replace("-", "_").lower(),
            f"{base_url}/StructureMap/{sd_id}",
            "Structure Map for " + project_name,
        )

    def _clean_name(self, name: str) -> str:
        """clean a given name based on naming constraints to be used in URLs and IDs."""
        return name.replace(" ", "-").replace("_", "-").replace("/", "-").lower()

    def process_files(self, files, app_state: AppState, overwrite=False):
        """process profile files and populate the registry."""
        resource_processing.process_files(files, app_state, overwrite)

    def _load_helper_map(self, sd_url: str):
        """load helper StructureDefinition from cache, then disk."""
        cached = self.app_state.cache.get_resource_from_cache(sd_url)
        if cached:
            return StructureDefinition(**cached) if isinstance(cached, dict) else cached
        for f in self.app_state.dataIO.get_source_helper_map_files():
            data = utils.get_json(f)
            if isinstance(data, dict) and data.get("url") == sd_url:
                return StructureDefinition(**data)
        return None

    def create_source_map(
        self, source_data_path, model_id, model_url, model_name, overwrite=False
    ):
        """create a source helper StructureDefinition from the input source example and cache it."""
        try:
            helper_map = fml_structure.generate_helper_map(
                source_data_path,
                self.app_state,
                model_id,
                model_url,
                model_name,
                overwrite=overwrite,
            )
        except Exception as e:
            logger.error(f"Error generating source map: {e}")
            return None

        if self.plugins:
            helper_map = self._apply_source_def_plugins(helper_map, model_id)

        self.app_state.cache.add_resource_to_cache(helper_map.model_dump())
        return helper_map

    def _apply_source_def_plugins(self, helper_map, model_id):
        """run post_source_def hooks, persist the enriched SD, return the (possibly) updated map."""
        sd_dict = helper_map.model_dump()
        source_fields = [
            {"id": el.id, "path": el.path}
            for el in (helper_map.snapshot.element if helper_map.snapshot else [])
        ]

        for plugin in self.plugins:
            try:
                sd_dict = plugin.post_source_def(sd_dict, source_fields)
            except Exception as e:
                logger.warning(
                    "Plugin '%s' post_source_def failed: %s",
                    plugin.plugin_id,
                    e,
                )

        try:
            enriched = StructureDefinition.model_validate(sd_dict)
            self.app_state.dataIO.store_project_file(
                self.app_state.dataIO.ProjectFolders.SOURCE_MAPS,
                f"{model_id}.json",
                enriched.model_dump_json(indent=2),
                mode="STR",
                overwrite=True,
            )
            return enriched
        except Exception as e:
            logger.warning(
                "Plugin post_source_def rebuild failed, using original: %s", e
            )
            return helper_map

    def generate_structure_maps(
        self,
        map_url,
        map_name,
        map_title,
        helper_map,
        minimal=False,
        status="draft",
        create_references=False,
        automapping=False,
        modular=False,
        overwrite=False,
    ):
        """structuremap generation call"""
        smg = StructureMapGenerator(
            app_state=self.app_state,
            map_url=map_url,
            map_title=map_title,
            helper_map=helper_map,
            map_name=map_name,
            minimal=minimal,
            status=status,
            create_references=create_references,
            automapping=automapping,
            automapper_instance=self.automapper,
            custom_mapping_table=self.state.custom_mapping_table,
            overwrite=overwrite,
            plugins=self.plugins,
        )
        smg.generate()
        sms = smg.get_structure_maps()
        for sm in sms:
            self.app_state.cache.add_resource_to_cache(sm.model_dump())
        return sms
