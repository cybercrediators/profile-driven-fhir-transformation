from pathlib import Path

import logging
logger = logging.getLogger(__name__)
from helpers import utils

from data_handling.app_state import AppState
from controller.pipeline_controller.pipeline_state import PipelineState


class PipelineValidationService:
    """validate FHIR resources against profiles via matchbox and check local/server setup"""

    def __init__(
        self,
        app_state: AppState,
        state: PipelineState,
        matchbox_controller,
        matchbox_sync,
    ):
        self.app_state = app_state
        self.state = state
        self.matchbox_controller = matchbox_controller
        self.matchbox_sync = matchbox_sync

    def validate_data(self, resource_obj, profile_url):
        """validate a FHIR resource against a profile using matchbox"""
        validation_result = self.matchbox_controller.validate_fhir_resources(
            resource_obj, profile_url
        )
        logger.info(f"Validation result: {validation_result}")
        return validation_result

    def validate_data_from_disk(self, input_data_path: Path, profile_url: str):
        """validate FHIR resource read from disk against a profile using matchbox"""
        input_data = utils.get_json(input_data_path)
        return self.validate_data(input_data, profile_url)

    def validate_local_files(self):
        """check if local files required for the pipeline are available"""
        logger.info("Validating local project files...")
        if (
            self.app_state.dataIO.check_processed_resources()
            and self.app_state.dataIO.check_processed_helper_maps()
            and self.app_state.dataIO.check_processed_structure_maps()
        ):
            logger.info("All required local files are available.")
            return True
        return False

    def validate_setup(self, read_only=True):
        """check that local files and matchbox resources are all in place.

        read_only=True (default): no uploads; returns False if anything is missing
        read_only=False: uploads any missing resources (equivalent to prepare_matchbox_setup)
        """
        if self.validate_local_files() and self.matchbox_sync.prepare_matchbox_setup(
            read_only=read_only
        ):
            return True
        return False

    def run_validate(self, input_file: str, profile_url: str = None):
        """validate an already-transformed FHIR resource against its target profile"""
        resource = utils.get_json(input_file)
        if not resource:
            logger.error(f"Could not load resource from: {input_file}")
            return

        if not profile_url:
            resource_type = resource.get("resourceType")
            profile_url = self.find_target_profile_for_type(resource_type)
            if not profile_url:
                logger.error(
                    f"No profile URL provided and none found in loaded SMs for resourceType '{resource_type}'. "
                    "Use -p to specify a profile URL."
                )
                return
            logger.info(f"Derived profile URL from SMs: {profile_url}")

        logger.info(f"Validating {input_file} against {profile_url}...")
        outcome = self.matchbox_controller.validate_fhir_resources(
            resource, profile_url
        )
        result = self.summarize_outcome(outcome)

        errors, warnings = result["errors"], result["warnings"]
        if errors:
            logger.warning(f"FAIL — {len(errors)} error(s), {len(warnings)} warning(s)")
            for e in errors:
                logger.warning(f"  ERROR: {e}")
            for w in warnings:
                logger.info(f"  WARN:  {w}")
        else:
            logger.info(f"PASS — {len(warnings)} warning(s)")
            for w in warnings:
                logger.info(f"  WARN:  {w}")

        return result

    def summarize_outcome(self, outcome) -> dict:
        """parse a matchbox $validate OperationOutcome into {status, errors, warnings}."""
        errors, warnings = [], []
        if (
            isinstance(outcome, dict)
            and outcome.get("resourceType") == "OperationOutcome"
        ):
            for issue in outcome.get("issue", []):
                severity = issue.get("severity", "")
                msg = issue.get("diagnostics") or issue.get("details", {}).get(
                    "text", ""
                )
                if severity in ("error", "fatal"):
                    errors.append(msg)
                elif severity == "warning":
                    warnings.append(msg)
        return {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "warnings": warnings,
        }

    def find_target_profile_for_type(self, resource_type: str) -> str:
        """look through loaded SMs and return the best target profile URL for resource_type"""
        candidates = []
        for f in self.app_state.dataIO.get_structure_map_files() or []:
            data = utils.get_json(f)
            if not isinstance(data, dict) or data.get("resourceType") != "StructureMap":
                continue
            fhir_type = next(
                (
                    inp.get("type")
                    for g in data.get("group", [])
                    for inp in g.get("input", [])
                    if inp.get("mode") == "target"
                ),
                None,
            )
            if fhir_type != resource_type:
                continue
            for s in data.get("structure", []):
                if s.get("mode") == "target":
                    url = s.get("url", "")
                    is_base = "hl7.org/fhir/StructureDefinition" in url
                    candidates.append((url, is_base))
        for url, is_base in candidates:
            if not is_base:
                return url
        return candidates[0][0] if candidates else None
