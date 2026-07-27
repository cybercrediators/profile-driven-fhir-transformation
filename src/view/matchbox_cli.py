import sys
import json
from pathlib import Path

from helpers import utils
import logging
logger = logging.getLogger(__name__)
from controller.external_services.matchbox_controller import MatchboxController
from controller.external_services.matchbox_transform_service import (
    MatchboxTransformService,
)


class MatchboxCLI:
    def __init__(self, conf):
        self.conf = conf

    def handle_matchbox_command(
        self,
        matchbox_action: str,
        package_name: str = None,
        package_version: str = None,
        package_url: str = None,
        package_path: str = None,
        ig_url: str = None,
        ig_id: str = None,
        sd_path: str = None,
        sm_path: str = None,
        cm_path: str = None,
        input_file: str = None,
        structure_map_url: str = None,
        bundle: bool = False,
        batch: bool = False,
        resource_type: str = None,
        resource_id: str = None,
        resource_url: str = None,
        profile_url: str = None,
        output_file: str = None,
    ):
        """validate the matchbox connection, then dispatch to the action handlers"""
        if not self.conf.get("matchbox_connection"):
            logger.error("No matchbox connection defined in config!")
            sys.exit(1)

        mc = MatchboxController(self.conf.get("matchbox_connection"))
        self._ping(mc)

        handlers = {
            "install-package": lambda: self._install_package(
                mc, package_name, package_version, package_url, package_path
            ),
            "check-installed-ig": lambda: self._check_installed_ig(mc, ig_url, ig_id),
            "upload-sd": lambda: self._upload_resource(
                mc.upload_structure_definition, sd_path, "StructureDefinition"
            ),
            "upload-sm": lambda: self._upload_resource(
                mc.upload_structure_map, sm_path, "StructureMap"
            ),
            "upload-cm": lambda: self._upload_resource(
                mc.upload_concept_map, cm_path, "ConceptMap"
            ),
            "transform-data": lambda: self._transform_data(
                mc, input_file, structure_map_url, bundle, batch, output_file
            ),
            "get-resource": lambda: self._get_resource(
                mc, resource_type, resource_id, resource_url
            ),
            "validate-data": lambda: self._validate_data(mc, input_file, profile_url),
        }

        handler = handlers.get(matchbox_action)
        if handler is None:
            logger.error(f"Unknown or missing matchbox action: {matchbox_action!r}")
            sys.exit(1)
        handler()

    def _ping(self, mc: MatchboxController):
        """Abort unless the matchbox server answers with a capability statement."""
        logger.info("Pinging matchbox server...")
        if mc.get_capability_statement():
            logger.info("Matchbox server is reachable. Capability statement received.")
        else:
            logger.error("Failed to reach matchbox server.")
            sys.exit(1)

    def _read_json_or_exit(self, path: str, label: str) -> dict:
        """Read a JSON file, exiting with an error if the path is missing/invalid."""
        if not path or not Path(path).is_file():
            logger.error(f"No valid {label} file path provided!")
            sys.exit(1)
        return utils.get_json(path)

    def _emit(self, payload, output_file: str):
        """Write JSON to a file when requested, otherwise print it."""
        text = json.dumps(payload, indent=2)
        if output_file:
            Path(output_file).write_text(text)
            logger.info(f"Output written to {output_file}")
        else:
            print(text)

    def _install_package(
        self, mc, package_name, package_version, package_url, package_path
    ):
        response = mc.install_npm_package(
            package_name=package_name,
            package_version=package_version,
            package_url=package_url,
            package_path=package_path,
        )
        if response:
            logger.info("Package installation request sent successfully.")
            print(json.dumps(response, indent=2))
        else:
            logger.error("Failed to send package installation request.")
            sys.exit(1)

    def _check_installed_ig(self, mc, ig_url, ig_id):
        if mc.check_implementation_guide_installed(ig_url=ig_url, ig_id=ig_id):
            logger.info("Implementation Guide is installed on matchbox.")
        else:
            logger.info("Implementation Guide is NOT installed on matchbox.")
        sys.exit(0)

    def _upload_resource(self, upload_fn, path: str, label: str):
        """upload an SD/SM/CM resource (dev mode must be enabled on matchbox!)"""
        resource_json = self._read_json_or_exit(path, label)
        if upload_fn(resource_json):
            logger.info(f"{label} uploaded successfully.")
        else:
            logger.error(f"Failed to upload {label}.")
            sys.exit(1)

    def _transform_data(
        self, mc, input_file, structure_map_url, bundle, batch, output_file
    ):
        """transform data through matchbox, use provided structuremap and input data"""
        input_data = self._read_json_or_exit(input_file, "input data")
        from plugins.registry import load_plugins

        service = MatchboxTransformService(
            mc,
            self.conf.get("project_path"),
            plugins=load_plugins(self.conf.get("plugins", [])),
            external_reference_defaults=self.conf.get(
                "external_reference_defaults", []
            ),
        )
        output = service.transform(
            input_data, structure_map_url, bundle=bundle, batch=batch
        )
        if output is None:
            sys.exit(1)
        logger.info("Data transformation successful.")
        self._emit(output, output_file)

    def _get_resource(self, mc, resource_type, resource_id, resource_url):
        """retrieve a resource from matchbox by ID or URL"""
        if resource_type is None:
            logger.error("No resource type provided!")
            sys.exit(1)
        if resource_id:
            resource = mc.get_resource_by_id(
                resource_type=resource_type, resource_id=resource_id
            )
        elif resource_url:
            resource = mc.get_resource_by_url(
                resource_type=resource_type, resource_url=resource_url
            )
        else:
            logger.error("No resource identifier (ID or URL) provided!")
            sys.exit(1)
        if resource:
            logger.info("Resource retrieved successfully.")
            print(json.dumps(resource, indent=2))

    def _validate_data(self, mc, input_file, profile_url):
        """validate FHIR resources against a configured/given profile"""
        if not input_file or not Path(input_file).is_file():
            logger.error("No valid input data file path provided!")
            sys.exit(1)
        if not profile_url:
            logger.error("No profile URL provided for validation!")
            sys.exit(1)
        input_data = utils.get_json(input_file)
        validation_response = mc.validate_fhir_resources(
            res_obj=input_data, profile_url=profile_url
        )
        if validation_response:
            logger.info("Data validation response received.")
            print(json.dumps(validation_response, indent=2))
        else:
            logger.error("Failed to execute data validation!")
            sys.exit(1)
