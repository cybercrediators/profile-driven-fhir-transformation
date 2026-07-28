import uuid

from data_handling.app_state import AppState
from parser.resource_parser import element_definition_parser
import logging
logger = logging.getLogger(__name__)

from fhir.resources.R4B import get_fhir_model_class
from fhir.resources.R4B.meta import Meta


def parse_structure_definition(sd, app_state: AppState, p_prefix=None):
    """
    process and resolve structure definitions

    use snapshot of underlying resource for parsing, through fhir-core abstracts
    """
    # check if object was already processed
    if sd is None or sd.is_processed():
        logger.info("Object was already processed OR is None! Returning original object")
        return sd, app_state
    logger.info(f"Processing: {sd.data.name}")

    snapshot = getattr(sd.data, "snapshot", None)
    if not snapshot or not getattr(snapshot, "element", None):
        identity = getattr(sd.data, "url", None) or getattr(sd.data, "name", "unknown")
        raise ValueError(
            f"StructureDefinition {identity} has no usable snapshot. "
            "Compile/expand snapshots before StructureMap generation; "
            "differential expansion is intentionally out of scope."
        )

    # set structuredefinition to processed
    sd.set_processed()

    # Underlying Resource
    res_type = sd.data.type
    print(res_type)
    if not res_type:
        logger.error(
            "Error: StructureDefinition %s does not have a 'type' defined", sd.data.name
        )
        return sd, app_state

    # underlying baseDefinition for mappable identification
    base_definition = getattr(sd.data, "baseDefinition", None)
    if base_definition is None:
        logger.warning(
            "StructureDefinition %s does not have a baseDefinition. Continuing...",
            sd.data.name,
        )
    else:
        app_state.registry.add_to_used_by(base_definition, sd.data.url)

    _res_class = get_fhir_class(res_type)
    if not _res_class:
        return sd, app_state
    # sd.template = _res_class()

    res_dict = {}

    # NOTE: generate random placeholder template id?
    tmp_template_id = str(uuid.uuid4())
    _cur_prefix = ""
    if not p_prefix:
        _cur_prefix = res_type
    else:
        _cur_prefix = f"{p_prefix}.contained[{res_type}:{tmp_template_id}]"

    logger.info("Created underlying base class: %s", res_type)

    # add meta
    res_dict["meta"] = Meta.model_construct(profile=[sd.data.url])
    logger.info("Setting meta for template class!")
    # if hasattr(sd.template, 'meta'):
    #     sd.template.meta = Meta.construct(profile=[sd.data.url])
    #     logger.info("Setting meta for template class!")

    # processed needed for slicing
    elem_list = list(
        sorted(sd.data.snapshot.element, key=lambda e: (len(e.path.split(".")), e.path))
    )
    sliced_elems = extract_slices(elem_list)
    slice_descendants = extract_slice_descendants(elem_list)

    path_map = {}

    # parse ElementDefinition(s)
    for elem in elem_list:

        # skip slice roots (handled in the slicing block) and slice descendants
        # (attached as the slice's children there) — else descendants would bind to
        # the unsliced element by their colliding path.
        if elem in sliced_elems or elem in slice_descendants:
            logger.info("This is a slice/slice-descendant, handled with its slice...")
            continue

        field_info = None

        # handle slicing first
        if elem.slicing:
            logger.info("This element is a slice, starting to expand slices...")
            print("DISCRIMINATORS Follow: ")
            field_info = element_definition_parser.parse_element_definition(
                res_dict, elem, sd, app_state
            )
            if not field_info:
                logger.warning("No information given for element, skipping!")
                continue
            field_info["slices"] = []
            # TODO: check if different slicing types needed, validation will be done later anyways
            for sl in elem.slicing.discriminator:
                print(sl)
                slices = get_slices(sliced_elems, elem.path)
                field_info["slicing_type"] = sl.type
                field_info["discriminator"] = {"path": sl.path, "type": sl.type}
                # value_disc = ['value', 'pattern']
                # if any(x in sl.type for x in value_disc):
                # print("IS A VALUE DISCRIMINATOR")
                for s in slices:
                    slice_field_info = (
                        element_definition_parser.parse_element_definition(
                            res_dict, s, sd, app_state
                        )
                    )
                    # parse_element_definition returns None for skipped/meta elements — don't
                    # append None into the slices list (downstream consumers iterate + .get on it).
                    if slice_field_info is not None:
                        _attach_slice_children(
                            slice_field_info, s, slice_descendants, res_dict, sd, app_state
                        )
                        field_info["slices"].append(slice_field_info)
                # if 'exists' in sl.type:
                #     print("IS AN EXISTS DISCRIMINATOR")
                # if 'type' in sl.type:
                #     print("IS AN TYPE DISCRIMINATOR")
                # if 'position' in sl.type:
                #     print("IS A POSITION DISCRIMINATOR")
            # pprint(field_info)
        else:
            field_info = element_definition_parser.parse_element_definition(
                res_dict, elem, sd, app_state
            )

        if field_info is None:
            continue

        # Build hierarchy
        field_info.setdefault("children", [])
        path_map[field_info["path"]] = field_info

        path_parts = field_info["path"].split(".")
        parent_path = ".".join(path_parts[:-1])

        if parent_path in path_map:
            path_map[parent_path]["children"].append(field_info)
        else:
            sd.mappable_fields.append(field_info)

    # print(sd.template.dict())
    # pprint(res_dict)
    # pprint(app_state.registry.get_all_mappable_fields())

    return sd, app_state


def extract_slices(elem_list):
    """Get all slice elements from the given list (those who have a sliceName)"""
    return [x for x in elem_list if (x.sliceName is not None)]


def extract_slice_descendants(elem_list):
    """Sub-elements *under* a slice (e.g. ``...property:weight.type``, ``...:weight.valueQuantity``)"""
    out = []
    for x in elem_list:
        eid = getattr(x, "id", None)
        if eid and ":" in eid and "." in eid.rsplit(":", 1)[-1]:
            out.append(x)
    return out


def _attach_slice_children(slice_info, slice_elem, descendants, res_dict, sd, app_state):
    """Parse the descendant ElementDefinitions of one slice and nest them under ``slice_info``
    by ``id`` (the only key that distinguishes a slice's sub-tree from the unsliced element's)."""
    slice_id = getattr(slice_elem, "id", None)
    if not slice_id:
        return
    slice_info.setdefault("children", [])
    local_map = {slice_id: slice_info}
    kids = [d for d in descendants if (d.id or "").startswith(slice_id + ".")]
    # shallow-to-deep so a parent is always present before its child is attached
    for d in sorted(kids, key=lambda e: (e.id or "").count(".")):
        child_info = element_definition_parser.parse_element_definition(
            res_dict, d, sd, app_state
        )
        if child_info is None:
            continue
        child_info.setdefault("children", [])
        local_map[d.id] = child_info
        parent_id = d.id.rsplit(".", 1)[0]
        parent = local_map.get(parent_id, slice_info)
        parent.setdefault("children", []).append(child_info)


def get_slices(slices, root_path):
    """Return slices from slice list based on the given root path"""
    return [x for x in slices if (root_path in x.path) and (x.sliceName is not None)]


def get_fhir_class(type_name: str):
    """access fhir model class"""
    try:
        return get_fhir_model_class(type_name)
    except (KeyError, ValueError) as e:
        logger.error(
            "Model class for %s not found in fhir.resources! Error: %s", type_name, e
        )
        return None
