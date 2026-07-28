import logging
logger = logging.getLogger(__name__)
from typing import Any, List
from fhir.resources.R4B.structuredefinition import (
    StructureDefinition,
    StructureDefinitionSnapshot,
)
from fhir.resources.R4B.elementdefinition import (
    ElementDefinition,
    ElementDefinitionType,
)

from pathlib import Path
from data_handling.app_state import AppState
from helpers import utils


def generate_helper_map(
    input_json: Path,
    app_state: AppState,
    model_id: str,
    model_url: str,
    model_name: str,
    overwrite=False,
):
    """
    Create a dummy/placeholder StructureDefinition for a given json input
    """
    # TODO: add helper map to cache/mapping folder persistency

    # check if helper maps exist
    if _check_exist := app_state.dataIO.check_processed_helper_maps():
        if not overwrite:
            logger.info("Helper map already exist! Loading previous helper maps...")
            helper_map = app_state.dataIO.load_project_file(
                app_state.dataIO.ProjectFolders.SOURCE_MAPS, model_id + ".json"
            )
            if helper_map:
                return StructureDefinition(**helper_map)
            logger.info("Helper map with the given id not found! Re-generating...")
        else:
            logger.info("Overwriting existing helper maps...")

    # read input_json
    if not input_json.exists():
        # check if input json file can be found in project folder
        if not (
            input_json := app_state.dataIO.project_dir
            / app_state.dataIO.ProjectFolders.SOURCE_DATA.value
            / input_json.name
        ).exists():
            raise FileNotFoundError(f"Input JSON file not found: {input_json}")

    data = utils.get_json(input_json)

    # construct base StructureDefinition
    helper_map = StructureDefinition.model_construct(
        id=model_id,
        name=model_name,
        url=model_url,
        status="draft",
        abstract=False,
        kind="logical",
        baseDefinition="http://hl7.org/fhir/StructureDefinition/Element",
        derivation="specialization",
    )
    type_name = model_name.capitalize()
    helper_map.type = type_name

    # construct ElementDefinitions
    element_definitions = []
    # pprint(data)

    if isinstance(data, list):
        if not data:
            raise ValueError(
                "Input JSON is an empty top-level array; a representative object "
                "containing every source field is required."
            )
        if len(data) > 1:
            logger.warning(
                "Source input contains %d top-level examples; only the first "
                "representative object is inspected.",
                len(data),
            )
        # create the structure for only one/the first element
        data = data[0]

    if not isinstance(data, dict):
        raise TypeError("Source should be a JSON object or a list of JSON objects!")

    # Add root element
    root_elem = construct_element_definition(
        type_name,
        type_name,
        ElementDefinitionType.model_construct(code="Element"),
        e_min=0,
        e_max="*",
    )
    element_definitions.append(root_elem)

    for key, value in data.items():
        path = f"{type_name}.{key}"
        create_json_element_definition(path, value, element_definitions)

    helper_map.snapshot = StructureDefinitionSnapshot.model_construct(
        element=element_definitions
    )

    # pprint(helper_map.model_dump_json())
    # TODO: Store map
    # utils.store_json(helper_map.model_dump_json(), app_state.dataIO.project_dir / "structure_maps" / "helper_map.json")
    # store helper map in project folder (if not exists or overwrite)
    # print(helper_map.model_dump(mode='json'))
    app_state.dataIO.store_project_file(
        app_state.dataIO.ProjectFolders.SOURCE_MAPS,
        model_id + ".json",
        helper_map.model_dump_json(indent=2),
        mode="STR",
        overwrite=overwrite,
    )

    return helper_map


def construct_element_definition(e_id, e_path, e_type, e_min=0, e_max="1"):
    return ElementDefinition.model_construct(
        id=e_id,
        path=e_path,
        min=e_min,
        max=e_max,
        type=[e_type],
    )


def map_fhir_type(value: Any) -> str:
    """
    Return a FHIR compatible primitive type
    """
    # TODO: add *all* json data types
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "decimal"
    # default to string
    return "string"


def create_json_element_definition(
    path: str, data: Any, element_definitions: List[ElementDefinition]
):
    """
    Recursively create ElementDefinitions from a given JSON file
    """
    e_id = path

    if data is None:
        raise ValueError(
            f"Cannot infer a source type for {path}: null carries no type. "
            "Provide a representative non-null value."
        )

    # check if nested object
    if isinstance(data, dict):
        parent_elem = construct_element_definition(
            e_id, path, ElementDefinitionType.model_construct(code="Element")
        )
        element_definitions.append(parent_elem)
        for key, value in data.items():
            child_path = f"{path}.{key}"
            create_json_element_definition(child_path, value, element_definitions)

    # check if array -> max should be '*'
    elif isinstance(data, list):
        if not data:
            raise ValueError(
                f"Cannot infer an item type for {path}: the representative array "
                "is empty. Include one representative item."
            )
        sample = data[0]
        array_type = (
            "Element"
            if isinstance(sample, dict)
            else map_fhir_type(sample)
        )
        parent_elem = construct_element_definition(
            e_id,
            path,
            ElementDefinitionType.model_construct(code=array_type),
            e_min=0,
            e_max="*",
        )
        element_definitions.append(parent_elem)
        if isinstance(sample, dict):
            for key, value in sample.items():
                child_path = f"{path}.{key}"
                create_json_element_definition(
                    child_path, value, element_definitions
                )

    # just a value
    else:
        e_type = ElementDefinitionType.model_construct(code=map_fhir_type(data))
        elem = construct_element_definition(e_id, path, e_type)
        element_definitions.append(elem)
