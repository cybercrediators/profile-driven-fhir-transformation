import logging
logger = logging.getLogger(__name__)
from controller.external_services.matchbox_controller import MatchboxController
from data_handling.registry.registry import Registry
from data_handling.caching.cache_connectors import CacheConnectors
from data_handling.data_io import DataIO
from data_handling.app_state import AppState
from helpers import config, utils
from pathlib import Path
from mapping.fml_creator.fml_automapper import FMLAutomapper
from plugins.registry import load_plugins
from controller.pipeline_controller.pipeline_state import PipelineState
from controller.pipeline_controller.pipeline_build_service import (
    PipelineBuildService,
    BuildOptions,
)
from controller.pipeline_controller.pipeline_matchbox_sync import PipelineMatchboxSync
from controller.pipeline_controller.pipeline_transform_service import (
    PipelineTransformService,
)
from controller.pipeline_controller.pipeline_validation_service import (
    PipelineValidationService,
)
from controller.pipeline_controller.pipeline_instance_validator import (
    PipelineInstanceValidator,
)

import sys
import json


class PipelineController:
    """controller managing pipeline functions"""

    def __init__(
        self,
        conf: dict,
        force_overwrite: bool = False,
        reprocess_profile: bool = False,
        reprocess_helper_definition: bool = False,
        reprocess_structure_map: bool = False,
        create_references: bool = False,
        minimal_mode: bool = False,
        automapping: bool = False,
        tarred_profile: str = None,
    ):
        self.conf = conf

        project_path = self.conf.get("project_path")
        if project_path is None:
            logger.error("No project path was provided!")
            sys.exit(1)
        project_path = Path(project_path)

        # init infrastructure (registry, cache, project IO) and the shared app state
        registry = Registry()
        cache = self.init_cache()
        dataIO = DataIO(project_path)
        self.app_state = AppState(self.conf, registry, cache, dataIO)

        # shared runtime state flowing between the pipeline services
        self.state = PipelineState()
        self.state.custom_mapping_table = self._load_custom_mapping_table()

        # check if tarred profile was provided, if not create one
        if not dataIO.find_tarred_profile() or force_overwrite:
            logger.info(
                "No tarred profile package found, creating package from profile files..."
            )
            dataIO.create_tar_package(overwrite=force_overwrite)

        # add the profile from zipped input if provided
        if tarred_profile:
            dataIO.add_tarred_profile(tarred_profile, overwrite=force_overwrite)

        # init automapper (background); skipped when a custom mapping table is present
        self.automapper = None
        if self.state.custom_mapping_table:
            logger.info("Custom mapping table provided. Automapper will be skipped.")
            automapping = False
        elif automapping:
            logger.info("Initializing automapper in background...")
            self.automapper = FMLAutomapper(use_word2vec=True)

        # input data sample path
        input_source_example = self.conf.get("input_source_example", "")
        if not input_source_example:
            logger.warning("No input source example path provided in config!")

        # project name for auto-config naming (kept if configs already exist)
        project_name = (
            project_path.parent.name if not project_path.is_dir() else project_path.name
        )

        # load pipeline plugins
        self.plugins = load_plugins(self.conf.get("plugins", []))

        mb_conf = self.conf.get("matchbox_connection", {})
        if not mb_conf:
            raise ValueError("No matchbox connection configuration provided!")

        # create matchbox controller
        try:
            self.matchbox_controller = MatchboxController(mb_conf)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize MatchboxController: {e}")

        # matchbox sync service (upload / verify / change-detection)
        self.matchbox_sync = PipelineMatchboxSync(
            self.app_state, self.state, self.matchbox_controller
        )

        # transform service (source data -> FHIR resources via matchbox)
        self.transform_service = PipelineTransformService(
            self.app_state, self.state, self.matchbox_controller
        )

        # validation service (resource/profile + local/server setup checks)
        self.validation_service = PipelineValidationService(
            self.app_state, self.state, self.matchbox_controller, self.matchbox_sync
        )

        # instance validator (example-instance direct + round-trip validation)
        self.instance_validator = PipelineInstanceValidator(
            self.app_state,
            self.state,
            self.matchbox_controller,
            self.validation_service,
            self.transform_service,
        )

        # build service (parse → source-def → static-gen-sm)
        build_options = BuildOptions(
            project_name=project_name,
            input_source_example=input_source_example,
            force_overwrite=force_overwrite,
            reprocess_profile=reprocess_profile,
            reprocess_helper_definition=reprocess_helper_definition,
            reprocess_structure_map=reprocess_structure_map,
            create_references=create_references,
            minimal_mode=minimal_mode,
            automapping=automapping,
            modular_structure_maps=bool(
                self.conf.get("modular_structure_maps", False)
            ),
        )
        self.build_service = PipelineBuildService(
            self.app_state,
            self.state,
            build_options,
            plugins=self.plugins,
            automapper=self.automapper,
        )

    def _load_custom_mapping_table(self):
        """Load an optional custom source->target mapping table from config (object or path)."""
        mapping_table_value = self.conf.get("mapping_table") or self.conf.get(
            "custom_mapping_table"
        )
        mapping_table_path = self.conf.get("mapping_table_path") or self.conf.get(
            "custom_mapping_table_path"
        )
        if isinstance(mapping_table_value, dict):
            logger.info("Using custom mapping table from config object.")
            return mapping_table_value
        if isinstance(mapping_table_path, str) and mapping_table_path.strip():
            try:
                table = utils.get_json(mapping_table_path)
                logger.info("Loaded custom mapping table from %s", mapping_table_path)
                return table
            except Exception as e:
                logger.error(f"Failed to load custom mapping table: {e}")
        return None

    @property
    def source_helper_urls(self):
        return self.state.source_helper_urls

    @source_helper_urls.setter
    def source_helper_urls(self, value):
        self.state.source_helper_urls = value

    @property
    def structure_map_urls(self):
        return self.state.structure_map_urls

    @structure_map_urls.setter
    def structure_map_urls(self, value):
        self.state.structure_map_urls = value

    @property
    def custom_mapping_table(self):
        return self.state.custom_mapping_table

    @custom_mapping_table.setter
    def custom_mapping_table(self, value):
        self.state.custom_mapping_table = value

    @property
    def input_source_example(self):
        return self.build_service.options.input_source_example

    @input_source_example.setter
    def input_source_example(self, value):
        self.build_service.options.input_source_example = value

    def initial_processing(self):
        """initial processing: parse profiles, create source helper SD, generate static-gen SM, populate registry"""
        return self.build_service.initial_processing()

    def run_process(self) -> bool:
        """run pipeline processing: initial processing + matchbox upload + validation"""
        return self.build_service.run_process()

    def run_source_def(self) -> bool:
        """run source helper SD creation"""
        return self.build_service.run_source_def()

    def run_static_gen_sm(self) -> bool:
        """generate static SMs"""
        return self.build_service.run_static_gen_sm()

    def _ensure_processed(self) -> bool:
        """check if certain processing steps are done"""
        return self.build_service._ensure_processed()

    def prepare_matchbox_setup(self, force_upload=False, read_only=False):
        """check if the matchbox server is set up to support the given config"""
        return self.matchbox_sync.prepare_matchbox_setup(
            force_upload=force_upload, read_only=read_only
        )

    def sync_changed_resources(self) -> bool:
        """synchronize changed resources with matchbox"""
        return self.matchbox_sync.sync_changed_resources()

    def check_matchbox_connection(self):
        """just check if matchbox answers with the capability statement for health check"""
        return self.matchbox_sync.check_matchbox_connection()

    def transform_data(
        self,
        input_data: dict,
        structure_map_url: str = None,
        map_outputs: dict = None,
        map_errors: dict = None,
    ):
        """transform a single given data object"""
        return self.transform_service.transform_data(
            input_data, structure_map_url, map_outputs=map_outputs,
            map_errors=map_errors,
        )

    def transform_data_batch(self, input_data_list: list, structure_map_url: str = None):
        """transform a list of data objects"""
        return self.transform_service.transform_data_batch(
            input_data_list, structure_map_url
        )

    def transform_data_from_disk(
        self, input_data_path, output_data_path, structure_map_url: str = None
    ):
        """transform data object loading from a given fs path"""
        return self.transform_service.transform_data_from_disk(
            input_data_path, output_data_path, structure_map_url
        )

    def validate_data(self, resource_obj, profile_url):
        """validate a single data object"""
        return self.validation_service.validate_data(resource_obj, profile_url)

    def validate_data_from_disk(self, input_data_path, profile_url):
        """validate data loaded from a given fs path"""
        return self.validation_service.validate_data_from_disk(
            input_data_path, profile_url
        )

    def validate_local_files(self):
        """check if all needed files are present and valid locally to run the pipeline"""
        return self.validation_service.validate_local_files()

    def validate_setup(self, read_only=True):
        """check if matchbox has all needed resources loaded"""
        return self.validation_service.validate_setup(read_only=read_only)

    def run_validate(self, input_file: str, profile_url: str = None):
        """use target profile and a transformed resource to validate the results"""
        return self.validation_service.run_validate(input_file, profile_url)

    def run_instance_validation(
        self,
        examples_dir: str = None,
        direct_only: bool = False,
        from_element_examples: bool = False,
        report_path: str = None,
    ):
        """validate example instances PROVIDED by the profile (extracted) and validate them"""
        return self.instance_validator.run_instance_validation(
            examples_dir=examples_dir,
            direct_only=direct_only,
            from_element_examples=from_element_examples,
            report_path=report_path,
        )
    def run_reverse_map(
        self,
        input_file: str,
        output_file: str = None,
        source_data: str = None,
        report_path: str = None,
    ):
        """reverse-map transformed FHIR output back to flat source-shaped records (PoC)
        """
        import data_handling.reverse_mapping as rm

        mapping_table = self.state.custom_mapping_table
        if not mapping_table:
            logger.error(
                "No mapping table available (-mt / config) — reverse mapping needs one."
            )
            return None

        payload = utils.get_json(input_file)
        project_dir = self.app_state.dataIO.project_dir

        # profile-id -> base-type context for profile-rooted mapping paths
        input_profile_dir = (
            project_dir / self.app_state.dataIO.ProjectFolders.INPUT_PROFILE.value
        )
        structure_definitions = []
        if input_profile_dir.is_dir():
            for f in sorted(input_profile_dir.glob("*.json")):
                data = utils.get_json(f)
                if (
                    isinstance(data, dict)
                    and data.get("resourceType") == "StructureDefinition"
                ):
                    structure_definitions.append(data)
        profile_ids_map = rm.profile_ids_by_type(structure_definitions)

        inverse_cms = rm.load_inverse_concept_maps(
            project_dir / self.app_state.dataIO.ProjectFolders.CONCEPT_MAPS.value
        )

        records = rm.reverse_map_payload(
            payload,
            mapping_table,
            profile_ids_map=profile_ids_map,
            inverse_cms=inverse_cms,
        )
        text = json.dumps(records, indent=2, default=str)
        if output_file:
            Path(output_file).write_text(text)
            logger.info("Reverse-mapped %d record(s) written to %s", len(records), output_file)
        else:
            print(text)

        if source_data:
            sources = utils.get_json(source_data)
            report = rm.roundtrip_report(records, sources, mapping_table)
            if not report_path:
                report_dir = project_dir / "converted_data"
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = str(report_dir / "reverse_mapping_report.json")
            Path(report_path).write_text(json.dumps(report, indent=2, default=str))
            logger.info(
                "Round-trip report (%s) written to %s",
                ", ".join(f"{k}={v}" for k, v in report["summary"].items()),
                report_path,
            )
        return records

    def init_cache(self):
        """Initialize the cache connector"""
        try:
            cache = CacheConnectors[self.conf.get("external_cache_service")].value(
                **self.conf.get("cache_args", {})
            )
        except KeyError as e:
            logger.error(
                "Cache %s does not have an implementation ready: %s",
                self.conf.get("external_cache_service"),
                e,
            )
            raise NotImplementedError(
                f"Cache named: {self.conf.get('external_cache_service')} is not implemented!"
            )
        return cache

    def display_config(self):
        """Display the current configuration"""
        config.show_config(self.app_state.conf)

    def run_export_fields(
        self, output_file: str = None, roots_only: bool = False
    ) -> bool:
        """Export all mappable fields from the processed registry as flat JSON."""
        self._ensure_processed()
        if roots_only:
            raw = self.app_state.registry.get_all_mappable_fields_of_roots()
        else:
            raw = self.app_state.registry.get_all_mappable_fields()

        # Serialize MappableField dataclasses / dicts uniformly
        def _to_dict(field):
            if hasattr(field, "__dataclass_fields__"):
                return {k: getattr(field, k) for k in field.__dataclass_fields__}
            return field

        result = {
            url: [_to_dict(f) for f in fields]
            for entry in raw
            for url, fields in entry.items()
        }
        text = json.dumps(result, indent=2, default=str)

        if output_file:
            Path(output_file).write_text(text)
            logger.info(f"Mappable fields written to {output_file}")
        else:
            print(text)
        return True

