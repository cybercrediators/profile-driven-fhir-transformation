from pathlib import Path

import logging
logger = logging.getLogger(__name__)
from helpers import utils

from data_handling.app_state import AppState
from controller.pipeline_controller.pipeline_state import PipelineState
import data_handling.instance_validation as iv


class PipelineInstanceValidator:
    """
    extract example instances from the given/configured profile to validate them against the profile 
    and check the resulting coverage of the created mappings
    """

    def __init__(
        self,
        app_state: AppState,
        state: PipelineState,
        matchbox_controller,
        validation_service,
        transform_service,
    ):
        self.app_state = app_state
        self.state = state
        self.matchbox_controller = matchbox_controller
        self.validation_service = validation_service
        self.transform_service = transform_service

    def run_instance_validation(
        self,
        examples_dir: str = None,
        direct_only: bool = False,
        from_element_examples: bool = False,
        report_path: str = None,
    ) -> dict:
        """
        each instance:
            - directly $validate it against its profile
            - unless direct_only, reverse-extract flat source obj via mapping table, then run it through SM
            - compare output to original instance + validation
        requires:
            - matchbox connection + profile/SM setup (prepare_matchbox)
        """

        # read example instances from the examples/ and input_profile/ folders + given dirs
        loaded = []
        seen_files = set()

        def _load_dir(path: Path):
            if not path or not path.is_dir():
                return
            for f in sorted(path.glob("*.json")):
                if str(f) in seen_files:
                    continue
                seen_files.add(str(f))
                data = utils.get_json(f)
                if isinstance(data, dict):
                    loaded.append((str(f), data))

        _load_dir(
            self.app_state.dataIO.project_dir
            / self.app_state.dataIO.ProjectFolders.EXAMPLES.value
        )
        _load_dir(
            self.app_state.dataIO.project_dir
            / self.app_state.dataIO.ProjectFolders.INPUT_PROFILE.value
        )
        if examples_dir:
            _load_dir(Path(examples_dir))

        ig_resource = next(
            (d for _, d in loaded if d.get("resourceType") == "ImplementationGuide"),
            None,
        )
        structure_definitions = [
            d for _, d in loaded if d.get("resourceType") == "StructureDefinition"
        ]

        sd_id_by_url = {
            iv.strip_version(sd["url"]): sd["id"]
            for sd in structure_definitions
            if sd.get("url") and sd.get("id")
        }

        records = iv.discover_examples(
            loaded,
            ig_resource=ig_resource,
            structure_definitions=structure_definitions,
            include_element_examples=from_element_examples,
        )
        if not records:
            logger.warning(
                "No example instances discovered (examples/, input_profile/, --examples-dir)."
            )
            return {"summary": {"n": 0}, "instances": []}
        logger.info("Discovered %d example instance(s) for validation.", len(records))

        # ensure SMs are loaded (mirrors transform_data's disk-load)
        if not self.state.structure_map_urls:
            for f in sorted(
                self.app_state.dataIO.get_structure_map_files() or [], key=str
            ):
                data = utils.get_json(f)
                if (
                    isinstance(data, dict)
                    and data.get("resourceType") == "StructureMap"
                    and data.get("url")
                ):
                    self.app_state.cache.add_resource_to_cache(data)
                    self.state.structure_map_urls.append(data["url"])

        # load mapping table (required for full check)
        mapping_table = self.state.custom_mapping_table
        if not mapping_table and not direct_only:
            logger.warning(
                "No mapping table available (-mt / config) — falling back to direct validation only."
            )
            direct_only = True

        instances = []
        for record in records:
            resource = record["resource"]
            resource_type = resource.get("resourceType")
            profile_url = record.get(
                "profile_url"
            ) or self.validation_service.find_target_profile_for_type(resource_type)
            profile_ids = set()
            if profile_url:
                stripped = iv.strip_version(profile_url)
                if stripped in sd_id_by_url:
                    profile_ids.add(sd_id_by_url[stripped])
                profile_ids.add(stripped.rsplit("/", 1)[-1])
            entry = {
                "file": record.get("file"),
                "resourceType": resource_type,
                "id": resource.get("id"),
                "profile": profile_url,
                "source": record.get("source"),
                "direct_validate": None,
                "roundtrip_validate": None,
                "coverage": None,
                "diffs": [],
            }

            # direct validation of the raw example
            if profile_url:
                entry["direct_validate"] = self.validation_service.summarize_outcome(
                    self.matchbox_controller.validate_fhir_resources(
                        resource, profile_url
                    )
                )["status"]
            else:
                logger.warning(
                    "No profile resolved for %s/%s — skipping direct validation.",
                    resource_type,
                    resource.get("id"),
                )

            # transform, compare + validate.
            if not direct_only and profile_url:
                sm_url = self._find_structure_map_for_profile(
                    profile_url
                ) or self._find_structure_map_for_type(resource_type)
                if not sm_url:
                    logger.info(
                        "No StructureMap targets %s skipping round-trip for this example.",
                        resource_type,
                    )
                else:
                    flat_source = iv.reverse_extract(
                        resource, mapping_table, profile_ids=profile_ids
                    )
                    if not flat_source:
                        logger.info(
                            "Reverse-extraction produced no source fields for %s skipping round-trip.",
                            resource_type,
                        )
                    else:
                        produced = self.transform_service.transform_data(
                            flat_source, structure_map_url=sm_url
                        )
                        output = self._first_of_type(produced, resource_type)
                        if output is None:
                            entry["roundtrip_validate"] = "NO_OUTPUT"
                        else:
                            comparison = iv.compare_instance(
                                resource, output, mapping_table, profile_ids=profile_ids
                            )
                            entry["coverage"] = comparison["coverage"]
                            entry["diffs"] = comparison["diffs"]
                            entry["roundtrip_validate"] = (
                                self.validation_service.summarize_outcome(
                                    self.matchbox_controller.validate_fhir_resources(
                                        output, profile_url
                                    )
                                )["status"]
                            )

            instances.append(entry)

        report = {
            "summary": self._summarize_instance_report(instances),
            "instances": instances,
        }

        # generate summry
        s = report["summary"]
        logger.info(
            "Instance validation: %d example(s) | direct PASS %d/%d | round-trip PASS %d/%d | mean coverage %s",
            s["n"],
            s["direct_pass"],
            s["direct_total"],
            s["roundtrip_pass"],
            s["roundtrip_total"],
            f"{s['mean_coverage']:.2%}" if s["mean_coverage"] is not None else "n/a",
        )

        # generate JSON report
        out_path = (
            Path(report_path)
            if report_path
            else (
                self.app_state.dataIO.project_dir
                / "converted_data"
                / "instance_validation_report.json"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        utils.store_json(report, out_path)
        logger.info("Instance validation report written to %s", out_path)
        return report

    def _find_structure_map_for_type(self, resource_type: str) -> str:
        """Return the StructureMap URL whose target group input type == resource_type."""
        for f in self.app_state.dataIO.get_structure_map_files() or []:
            data = utils.get_json(f)
            if not isinstance(data, dict) or data.get("resourceType") != "StructureMap":
                continue
            target_type = next(
                (
                    inp.get("type")
                    for g in data.get("group", [])
                    for inp in g.get("input", [])
                    if inp.get("mode") == "target"
                ),
                None,
            )
            if target_type == resource_type and data.get("url"):
                return data["url"]
        return None

    def _find_structure_map_for_profile(self, profile_url: str) -> str:
        """return the StructureMap URL whose target structure.url matches profile_url"""
        if not profile_url:
            return None

        target = iv.strip_version(profile_url)
        for f in self.app_state.dataIO.get_structure_map_files() or []:
            data = utils.get_json(f)
            if not isinstance(data, dict) or data.get("resourceType") != "StructureMap":
                continue
            for s in data.get("structure", []):
                if (
                    s.get("mode") == "target"
                    and iv.strip_version(s.get("url", "")) == target
                ):
                    return data.get("url")
        return None

    def _first_of_type(self, produced, resource_type: str):
        """pick the first resource of resource_type from a transform_data result)"""
        if not produced:
            return None
        items = produced if isinstance(produced, list) else [produced]
        for item in items:
            if isinstance(item, dict) and item.get("resourceType") == resource_type:
                return item
        return items[0] if isinstance(items[0], dict) else None

    def _summarize_instance_report(self, instances: list) -> dict:
        """aggregate per-instance results into summary counters"""
        direct = [
            i["direct_validate"] for i in instances if i["direct_validate"] is not None
        ]
        roundtrip = [
            i["roundtrip_validate"]
            for i in instances
            if i["roundtrip_validate"] is not None
        ]
        coverages = [i["coverage"] for i in instances if i["coverage"] is not None]
        return {
            "n": len(instances),
            "direct_total": len(direct),
            "direct_pass": sum(1 for s in direct if s == "PASS"),
            "roundtrip_total": len(roundtrip),
            "roundtrip_pass": sum(1 for s in roundtrip if s == "PASS"),
            "mean_coverage": (sum(coverages) / len(coverages)) if coverages else None,
        }
