import json

from helpers import utils
from controller.bundle_service import BundleService
from data_handling.data_io import DataIO

import logging
logger = logging.getLogger(__name__)


class MatchboxTransformService:
    """handle transforms through matchbox"""

    def __init__(
        self,
        matchbox_controller,
        project_path: str = None,
        plugins: list = None,
        external_reference_defaults: list = None,
    ):
        self.matchbox_controller = matchbox_controller
        self.project_path = project_path
        self.plugins = plugins or []
        self.external_reference_defaults = external_reference_defaults or []
        self._dataIO = DataIO(project_path) if project_path else None

    def transform(
        self,
        input_data,
        structure_map_url: str = None,
        *,
        bundle: bool = False,
        batch: bool = False,
    ):
        """transform input data based on the provided structure map(s)"""
        sm_urls = self._resolve_sm_urls(structure_map_url)
        if not sm_urls:
            return None

        sm_data = self._load_structure_maps()
        registry = self._load_registry()
        input_data = self._inject_resource_type(input_data)

        if batch:
            items = input_data if isinstance(input_data, list) else [input_data]
            all_results = []
            for item in items:
                map_outputs = {}
                resources = self._transform_one(item, sm_urls, map_outputs)
                if resources:
                    all_results.append(
                        self._assemble(
                            resources, bundle, sm_data, registry, item, map_outputs
                        )
                    )
            if not all_results:
                logger.error("Batch transformation produced no resources!")
                return None
            return all_results

        map_outputs = {}
        resources = self._transform_one(input_data, sm_urls, map_outputs)
        if not resources:
            logger.error("Data transformation produced no resources!")
            return None
        return self._assemble(
            resources, bundle, sm_data, registry, input_data, map_outputs
        )

    def _resolve_sm_urls(self, structure_map_url: str) -> list:
        """return the StructureMap URLs to use, or [] (with a logged error)"""
        if structure_map_url:
            return [structure_map_url]
        if not self.project_path:
            logger.error(
                "No --structure-map-url provided and no project_path in config!"
            )
            return []
        urls = self._load_structure_map_urls()
        if not urls:
            logger.error("No StructureMap URLs found in project structure_maps folder!")
            return []
        logger.info(f"Using {len(urls)} project StructureMap(s): {urls}")
        return urls

    def _load_structure_map_urls(self) -> list:
        """return sorted StructureMap URLs from the project's structure_maps folder"""
        return [
            sm.get("url")
            for sm in self._load_structure_maps()
            if sm.get("url")
        ]

    def _load_structure_maps(self) -> list:
        """return StructureMap dicts from the project's structure_maps folder"""
        if not self._dataIO:
            return []
        maps = []
        for f in sorted(self._dataIO.get_structure_map_files() or []):
            data = utils.get_json(f)
            if isinstance(data, dict) and data.get("resourceType") == "StructureMap":
                maps.append(data)
        return maps

    def _inject_resource_type(self, input_data):
        """try to infer resourcetype from structuredefintion and add it to the input data"""
        needs_rt = (
            any(
                isinstance(it, dict) and not it.get("resourceType")
                for it in input_data
            )
            if isinstance(input_data, list)
            else not input_data.get("resourceType")
        )
        if not (needs_rt and self.project_path):
            return input_data

        inferred = self._infer_source_resource_type()
        if not inferred:
            return input_data
        logger.info(f"Auto-inferred resourceType: {inferred}")

        def _with_rt(obj):
            if isinstance(obj, dict) and not obj.get("resourceType"):
                obj = dict(obj)
                obj["resourceType"] = inferred
            return obj

        return (
            [_with_rt(o) for o in input_data]
            if isinstance(input_data, list)
            else _with_rt(input_data)
        )

    def _infer_source_resource_type(self) -> str:
        """return the FHIR type from the project's source definitions, or None"""
        if not self._dataIO:
            return None
        for f in self._dataIO.get_source_helper_map_files() or []:
            sd = utils.get_json(f)
            if isinstance(sd, dict) and sd.get("type"):
                return sd["type"]
        return None

    def _transform_one(self, item: dict, sm_urls: list, map_outputs: dict = None) -> list:
        """run all StructureMap transforms for a single source object"""
        resources = []
        for url in sm_urls:
            result = self.matchbox_controller.transform_data(item, url)
            if not result:
                logger.warning(f"No result from transform with map: {url}")
                continue
            if result.get("resourceType") == "Bundle" and "entry" in result:
                produced = [
                    e["resource"] for e in result["entry"] if "resource" in e
                ]
            else:
                produced = [result]
            resources.extend(produced)
            if map_outputs is not None:
                map_outputs.setdefault(url, []).extend(produced)
        return resources

    def _assemble(
        self,
        resources: list,
        bundle: bool,
        sm_data: list,
        registry,
        source_record=None,
        map_outputs=None,
    ):
        """wrap transformed resources in a Bundle or normalise them to a list"""
        if bundle:
            return BundleService.create_bundle(
                resources,
                structure_maps=sm_data,
                registry=registry,
                plugins=self.plugins,
                source_record=source_record if isinstance(source_record, dict) else None,
                map_outputs=map_outputs,
                external_reference_defaults=self.external_reference_defaults,
            )
        return [BundleService.normalize_list_cardinality(r) for r in resources]

    def _load_registry(self):
        """reconstruct a registry from the jsonpickled processed_resources"""
        if not self._dataIO:
            return None
        try:
            import jsonpickle
            from data_handling.registry.registry import Registry
        except Exception as e:
            logger.debug("Registry rebuild unavailable (%s: %s)", type(e).__name__, e)
            return None
        registry = Registry()
        for f in sorted(self._dataIO.get_processed_resources() or []):
            if not f.is_file():
                continue
            try:
                obj = jsonpickle.decode(json.loads(f.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(
                    "Skipping unreadable processed resource %s (%s: %s)",
                    f.name, type(e).__name__, e,
                )
                continue
            if not hasattr(obj, "data"):
                continue
            url = getattr(getattr(obj, "data", None), "url", None) or f.name
            registry.registry_objects[url] = obj
        return registry if registry.registry_objects else None
