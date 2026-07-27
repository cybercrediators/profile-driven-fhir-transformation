from pathlib import Path

import logging
logger = logging.getLogger(__name__)
from helpers import utils

from data_handling.app_state import AppState
from controller.pipeline_controller.pipeline_state import PipelineState


class PipelineTransformService:
    """transform source data into FHIR resources via matchbox StructureMaps"""

    def __init__(self, app_state: AppState, state: PipelineState, matchbox_controller):
        self.app_state = app_state
        self.state = state
        self.matchbox_controller = matchbox_controller

    def transform_data(
        self,
        input_data: dict,
        structure_map_url: str = None,
        map_outputs: dict = None,
        map_errors: dict = None,
    ) -> dict:
        """transform input data through matchbox using single transform (structure_map_url) or use all loaded structure maps
        """

        input_data = self._prune_empty_values(input_data)

        if not input_data.get("resourceType"):
            inferred = self._infer_source_resource_type()
            if inferred:
                input_data = dict(input_data)
                input_data["resourceType"] = inferred
                logger.info(
                    f"Auto-inferred resourceType from source definition: {inferred}"
                )
            else:
                logger.error(
                    "Input data has no resourceType and none could be inferred from the source definition!"
                )
                return None

        if structure_map_url:
            result = self._single_transform(input_data, structure_map_url)
            if not result:
                logger.error(
                    "Transform FAILED for map %s — no output (see matchbox error above).",
                    structure_map_url,
                )
                if map_errors is not None:
                    map_errors[structure_map_url] = (
                        "transform returned no output (engine error)"
                    )
            if map_outputs is not None and isinstance(result, dict):
                if result.get("resourceType") == "Bundle" and "entry" in result:
                    produced = [
                        e["resource"] for e in result["entry"] if "resource" in e
                    ]
                else:
                    produced = [result]
                map_outputs.setdefault(structure_map_url, []).extend(produced)
            return result

        if not self.state.structure_map_urls:
            seen_urls = set()
            for f in sorted(
                self.app_state.dataIO.get_structure_map_files() or [], key=str
            ):
                data = utils.get_json(f)
                if (
                    isinstance(data, dict)
                    and data.get("resourceType") == "StructureMap"
                ):
                    url = data.get("url")
                    if url:
                        self.app_state.cache.add_resource_to_cache(data)
                        if url not in seen_urls:
                            seen_urls.add(url)
                            self.state.structure_map_urls.append(url)
        if not self.state.structure_map_urls:
            logger.error("No structure map URL provided and no loaded maps found!")
            return None

        aggregated_resources = []

        trigger_map = self._build_trigger_map(self.state.structure_map_urls)
        composition = self._load_composition()

        for map_url in self.state.structure_map_urls:
            triggers = trigger_map.get(map_url)
            for sub_input in self._inputs_for_map(map_url, input_data, composition):
                if triggers and not (triggers & self._record_leaf_keys(sub_input)):
                    logger.info(
                        "Gating: skip %s — none of its source fields %s present",
                        map_url.split("/")[-1],
                        sorted(triggers),
                    )
                    continue

                result = self._single_transform(sub_input, map_url)
                if not result:
                    # The map was NOT gated — its source fields are present, so
                    # no output means the engine failed. Never let this look
                    # like a deliberate skip.
                    logger.error(
                        "Transform FAILED for map %s — no output "
                        "(see matchbox error above).",
                        map_url,
                    )
                    if map_errors is not None:
                        map_errors[map_url] = (
                            "transform returned no output (engine error)"
                        )
                    continue

                # Flatten results (handle Bundles vs single resources)
                if result.get("resourceType") == "Bundle" and "entry" in result:
                    produced = [
                        e["resource"] for e in result["entry"] if "resource" in e
                    ]
                else:
                    produced = [result]
                aggregated_resources.extend(produced)
                if map_outputs is not None:
                    map_outputs.setdefault(map_url, []).extend(produced)

        if not aggregated_resources:
            logger.warning("No resources produced from transformation.")
            return None

        return aggregated_resources

    def _load_composition(self) -> dict:
        """Load the optional per-project record-explosion descriptor.

        """
        cached = getattr(self.state, "composition_descriptor", None)
        if cached is not None:
            return cached
        descriptor = {}
        try:
            loaded = self.app_state.dataIO.load_project_file(
                self.app_state.dataIO.ProjectFolders.SOURCE_DATA, "composition.json"
            )
            if isinstance(loaded, dict):
                descriptor = loaded
        except Exception as e:
            logger.warning(
                "Could not read composition.json (%s: %s) — record explosion DISABLED for this run.",
                type(e).__name__, e,
            )
        self.state.composition_descriptor = descriptor
        return descriptor

    def _inputs_for_map(self, map_url: str, record: dict, composition: dict) -> list:
        """Return the source object(s) a map should run over.

        """
        entry = None
        for profile_id, e in composition.items():
            if map_url.endswith(profile_id):
                entry = e
                break

        if isinstance(entry, dict):
            paths = entry.get("paths")
            when = entry.get("when") or {}
        else:
            paths = entry
            when = {}

        def matches(obj):
            return all(obj.get(k) == v for k, v in when.items())

        rtype = record.get("resourceType")

        def stamp(obj):
            sub = dict(obj)
            if rtype and "resourceType" not in sub:
                sub["resourceType"] = rtype
            return sub

        if not paths:
            return [stamp(record)] if matches(record) else []

        out = []
        for path in paths:
            node = record if path in ("", ".") else self._get_path(record, path)
            if isinstance(node, dict) and node and matches(node):
                out.append(stamp(node))
        return out

    @staticmethod
    def _get_path(obj, dotted: str):
        """Traverse a dotted path into a nested dict; None if any segment is missing."""
        for seg in dotted.split("."):
            if not isinstance(obj, dict):
                return None
            obj = obj.get(seg)
        return obj

    @staticmethod
    def _extract_trigger_fields(sm: dict) -> set:
        """Source-record leaf fields a map reads (its data 'triggers').

        """
        fields = set()

        def walk(rule):
            for s in rule.get("source", []) or []:
                el = s.get("element")
                if el and not str(el).startswith("TODO"):
                    fields.add(el)
            for r in rule.get("rule", []) or []:
                walk(r)

        for g in sm.get("group", []) or []:
            for r in g.get("rule", []) or []:
                walk(r)
        return fields

    def _build_trigger_map(self, urls) -> dict:
        """Map each loaded StructureMap url → the set of source fields it reads."""
        out = {}
        for url in urls:
            sm = self.app_state.cache.get_resource_from_cache(url)
            if isinstance(sm, dict):
                out[url] = self._extract_trigger_fields(sm)
        return out

    @classmethod
    def _prune_empty_values(cls, obj):
        """Drop empty leaves (None, "", and containers left empty by pruning) from
        a source record. Same emptiness notion as _record_leaf_keys"""
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                pruned = cls._prune_empty_values(v)
                if pruned in (None, "", [], {}) and k != "resourceType":
                    continue
                out[k] = pruned
            return out
        if isinstance(obj, list):
            return [p for v in obj if (p := cls._prune_empty_values(v)) not in (None, "", [], {})]
        return obj

    @staticmethod
    def _record_leaf_keys(record: dict) -> set:
        """Flatten a (possibly nested) source record to the set of leaf field names that
        carry a non-empty value — what a map's triggers are matched against."""
        leaves = set()

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "resourceType":
                        continue
                    if isinstance(v, (dict, list)):
                        walk(v)
                    elif v not in (None, "", []):
                        leaves.add(k)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(record)
        return leaves

    def transform_data_batch(
        self, input_data_list: list, structure_map_url: str = None
    ) -> list:
        """transform list of input data objects"""
        results = []
        for input_data in input_data_list:
            resources = self.transform_data(input_data, structure_map_url)
            if resources is not None:
                results.append(resources)
        return results

    def transform_data_from_disk(
        self,
        input_data_path: Path,
        output_data_path: Path,
        structure_map_url: str = None,
    ):
        """transform input data read from disk and store the result"""
        input_data = utils.get_json(input_data_path)
        transformed_data = self.transform_data(input_data, structure_map_url)
        if transformed_data is None:
            logger.error("Data transformation failed!")
            return False
        utils.store_json(transformed_data, output_data_path)
        logger.info(f"Transformed data stored to {output_data_path}")
        return True

    def _single_transform(self, input_data: dict, structure_map_url: str) -> dict:
        """transform data using a specific structure map url"""
        if input_data.get("resourceType") is None:
            logger.error("Input data must have a resourceType defined!")
            return None

        logger.info(f"Transforming data using map: {structure_map_url}")
        return self.matchbox_controller.transform_data(input_data, structure_map_url)

    def _infer_source_resource_type(self) -> str:
        """return the FHIR type from the loaded source helper StructureDefinition, or None if unavailable"""
        for url in self.state.source_helper_urls:
            sd = self.app_state.cache.get_resource_from_cache(url)
            if sd and sd.get("type"):
                return sd["type"]
        for f in self.app_state.dataIO.get_source_helper_map_files() or []:
            sd = utils.get_json(f)
            if isinstance(sd, dict) and sd.get("type"):
                return sd["type"]
        return None
