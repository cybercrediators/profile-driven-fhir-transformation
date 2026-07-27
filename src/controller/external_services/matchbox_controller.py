import json
from pathlib import Path
from ..connector.matchbox_connector import MatchboxConnector
from helpers import utils


import logging
logger = logging.getLogger(__name__)

class MatchboxController:
    def __init__(self, matchbox_con: dict):
        self.mc = MatchboxConnector(matchbox_con)

    def get_capability_statement(self):
        """Get the matchbox server capability statement"""
        logger.info("Fetching capability statement from matchbox server...")
        return self.mc.send_request("metadata", "GET")

    def install_npm_package(
        self,
        package_name: str,
        package_version: str,
        package_url: str = None,
        package_path: Path = None,
    ):
        if not package_name and not package_version:
            logger.warning("No package name or version to install!")
            return None
        params = {"name": package_name, "version": package_version}
        if package_path:
            logger.info("Uploading package from path %s", package_path)
            headers = {
                "accept": "application/fhir+json",
                "content-type": "application/octet-stream",
            }
            files = Path(package_path).read_bytes()
            return self.mc.send_request(
                "$install-npm-package",
                "POST",
                params=params,
                headers=headers,
                data=files,
            )
        if package_url:
            params["url"] = package_url
        headers = {"accept": "application/fhir+json"}
        return self.mc.send_request(
            "$install-npm-package", "POST", params=params, headers=headers
        )

    def get_resource_by_id(self, resource_type: str, resource_id: str):
        """
        Get a given resource by type and id
        """
        endpoint = f"{resource_type}/{resource_id}"
        logger.info("Fetching resource from %s", endpoint)
        return self.mc.send_request(endpoint, "GET")

    def get_resource_by_url(self, resource_type: str, resource_url: str):
        """Retrieve a given resource by type and url"""
        params = {"url": resource_url}
        logger.info(
            "Fetching resource of type %s with url %s", resource_type, resource_url
        )
        response = self.mc.send_request(resource_type, "GET", params=params)
        if response and "entry" in response and len(response["entry"]) > 0:
            # logger.info("Resource found: %s", response['entry'])
            logger.info("Resource found! ")  # %s", response['entry'])
            return response["entry"][0]["resource"]
        logger.info(
            "Resource of type %s with url %s not found", resource_type, resource_url
        )
        return None

    def upload_resource_string(self, resource_body, resource_type=None):
        """
        Create a given resource (won't check for existing resources)
        """
        if resource_type is None:
            resource_type = resource_body.get("resourceType")
        return self.mc.send_request(resource_type, "POST", data=resource_body)

    def check_implementation_guide_installed(
        self, ig_url: str = None, ig_id: str = None
    ):
        """
        Check if an implementation guide is installed on the matchbox server
        """
        if ig_url is None and ig_id is None:
            logger.warning("No implementation guide URL or ID provided to check!")
            return False
        querystring = {}
        if ig_url:
            logger.info(
                "Checking if implementation guide %s is installed on matchbox", ig_url
            )
            querystring["url"] = ig_url
        elif ig_id:
            logger.info(
                "Checking if implementation guide with ID %s is installed on matchbox",
                ig_id,
            )
            querystring["id"] = ig_id
        response = self.mc.send_request(
            "ImplementationGuide", "GET", params=querystring
        )
        if response and "entry" in response and len(response["entry"]) > 0:
            # Matchbox may return all IGs even when filtered by id, so verify the match.
            if ig_id:
                matched = any(
                    e.get("resource", {}).get("packageId") == ig_id
                    for e in response["entry"]
                )
                if not matched:
                    logger.info("Implementation guide %s not found", ig_id)
                    return False
            logger.info("Implementation guide found! -> IG installed on the server")
            return True
        logger.info("Implementation guide %s not found", ig_url or ig_id)
        return False

    def upload_structure_map(self, structure_map):
        """
        Upload all existing structuremaps to Matchbox
        """
        return self.upload_resource_string(
            json.dumps(structure_map), resource_type="StructureMap"
        )

    def upload_concept_map(self, concept_map):
        """
        Upload concept maps to matchbox
        """
        return self.upload_resource_string(
            json.dumps(concept_map), resource_type="ConceptMap"
        )

    def upload_structure_definition(self, sd):
        """
        Upload helper maps to matchbox
        """
        if isinstance(sd, dict) and sd.get("derivation") == "constraint":
            tail = str(sd.get("url", "")).rsplit("/", 1)[-1]
            if utils.get_model_class(tail) is not None:
                logger.warning(
                    "Uploading profile %s whose canonical tail '%s' is a core resource type — "
                    "matchbox will store it under id '%s', shadowing the core StructureDefinition "
                    "on a persisted server.",
                    sd.get("url"),
                    tail,
                    tail,
                )
        return self.upload_resource_string(
            json.dumps(sd), resource_type="StructureDefinition"
        )

    def transform_data(self, source_obj, structure_map_url):
        """
        Transform data using matchbox ($transform)
        """
        # structure map and profile must be installed previously!
        response = self.mc.send_request(
            "StructureMap/$transform",
            "POST",
            params={"source": structure_map_url},
            headers={
                "Content-Type": "application/fhir+json",
                "Accept": "application/fhir+json",
            },
            data=json.dumps(source_obj),
        )
        return response

    def validate_fhir_resources(self, res_obj, profile_url):
        """
        Validate FHIR resources using matchbox ($validate)
        """
        response = self.mc.send_request(
            "$validate",
            "POST",
            params={"profile": profile_url},
            data=json.dumps(res_obj),
        )
        return response
