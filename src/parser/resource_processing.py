import jsonpickle

from helpers import utils
from data_handling.app_state import AppState
from fhir_spec.context import ensure_fhir_spec_context
from parser.resource_parser import resource_obj_parser

import logging
logger = logging.getLogger(__name__)

# restructure -> this should maybe be a controller?


def _is_mappable_root(obj) -> bool:
    """A root map target is a resource-level StructureDefinition profile or a Questionnaire.

    Excludes non-SD conformance resources (ValueSet/CodeSystem/ImplementationGuide/
    CapabilityStatement) and non-resource SDs (Extension / logical models / complex-types),
    which otherwise get rooted when a full IG package is ingested and produce bogus maps.
    """
    res_type = obj.res_type or ""
    if "Questionnaire" in res_type:
        return True
    if "StructureDefinition" in res_type:
        kind = getattr(obj.data, "kind", "") or ""
        sd_type = getattr(obj.data, "type", "") or ""
        return "resource" in kind and sd_type != "Extension"
    return False


def process_files(files, app_state: AppState, overwrite=False):
    # for mod in fhir_modules:
    #    for name, cls in inspect.getmembers(importlib.import_module(f"fhir.resources.R4B.{mod}"), inspect.isclass):
    #        print(name, cls)
    spec_context = ensure_fhir_spec_context(app_state)
    for file in files:
        obj, res_type = utils.json_file_to_obj(file, spec_context=spec_context)
        try:
            app_state.registry.add_fhir_object(obj, res_type)
        except ReferenceError as e:
            logger.warning(f"{file} {e} ... Skipping")

    if check_res := check_results(app_state):
        if not overwrite:
            logger.info("Previous results already exist! Loading previous results...")
            load_results(app_state)

    # print(registry.get_all_obj_by_type("StructureDefinition"))

    if overwrite or not check_res:
        # TODO: enable configured root resources
        # root_res = app_state.conf.get("root_resources")
        for res_name in app_state.registry.registry_objects:
            res = app_state.registry.get_obj_by_name(res_name)
            if res is None:
                logger.error("Could not find resource %s", res_name)
                break
            # Process only StructureDefinitions
            if "StructureDefinition" in res.res_type:
                if "resource" not in res.data.kind:
                    continue
                resource_obj_parser.parse_resource(res, app_state)
                res.set_root()
            # process Questionnaires
            if "Questionnaire" in res.res_type:
                resource_obj_parser.parse_resource(res, app_state)
        # what if we have "unprocessed" resources
        # print("UNPROCESSED:")
        # print(app_state.registry.get_all_unprocessed_objects())

        # identify "root" resources only
        # -> only resource-level StructureDefinition profiles + Questionnaires are map targets
        # -> NOT ValueSet/CodeSystem/ImplementationGuide/CapabilityStatement (non-SD), and
        #    NOT Extension/logical-model SDs (kind != resource). Ingesting a full IG package
        #    pulls all of these in; rooting them produced bogus maps targeting ValueSets/IGs.
        for x in app_state.registry.registry_objects:
            a = app_state.registry.get_obj_by_name(x)
            if a and not a.used_by and _is_mappable_root(a):
                a.set_root()
                # print({a.data.url: a.used_by})

        # store results
        store_results(app_state)

        # if not root_res:
        # Process StructureDefinitions first
        # sds = app_state.registry.get_all_obj_by_type("StructureDefinition")
        # for sd in sds:
        #     resource_obj_parser.parse_structure_definition(sd, app_state)

        # TODO: create fhir resource graph in case there are no root files given
        # if not root_res:
        #     fhir_graph = create_resource_graph(app_state)
        #     show_graph(fhir_graph)
        #     root_res = get_graph_root(fhir_graph)
        #     # print(fhir_graph.number_of_nodes())
        #     # print(fhir_graph.number_of_edges())

        #     # pprint(get_graph_root(fhir_graph))
    # for url, val in app_state.registry.registry_objects.items():
    #     print(url)
    #     if val.is_root:
    #         print(val.mappable_fields)

    # TODO: process other resources (in case there are other mappable fields to keep in mind)


def _result_filename(url: str, val) -> str:
    """on-disk name for a processed resource (avoid using potentially optional id field)"""
    return utils.resource_identity(val.data, url)


def store_results(app_state: AppState):
    """Store results from the analyzation process on-disk in the corresponding project folder(s)"""
    for url, val in app_state.registry.registry_objects.items():
        # TODO: remove this after testing
        # res_dir = app_state.dataIO.project_dir / 'processed_resources' / val.data.id
        res_json = jsonpickle.encode(val)
        # utils.store_json(res_json, res_dir)
        app_state.dataIO.store_project_file(
            app_state.dataIO.ProjectFolders.PROCESSED_RESOURCES,
            _result_filename(url, val),
            res_json,
            overwrite=True,
        )


def check_results(app_state: AppState):
    """Check if results already exist"""
    return app_state.dataIO.check_processed_resources()


def load_results(app_state: AppState):
    """Load existing results from registry/previous run if already eixsting"""

    missing = []
    for url, val in app_state.registry.registry_objects.items():
        filename = _result_filename(url, val)
        res_json = app_state.dataIO.load_project_file(
            app_state.dataIO.ProjectFolders.PROCESSED_RESOURCES, filename
        )
        if res_json is None:
            missing.append(filename)
            continue
        app_state.registry.registry_objects[url] = jsonpickle.decode(res_json)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} processed resource(s) are missing, so this project's "
            "processed_resources/ is older than its input_profile/: "
            + ", ".join(sorted(missing)[:5])
            + (" …" if len(missing) > 5 else "")
            + ". Re-run `pipeline process -f` to regenerate them."
        )
