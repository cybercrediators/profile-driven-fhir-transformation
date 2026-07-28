import uuid
from copy import deepcopy

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
    id_map = {}

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
            field_info = element_definition_parser.parse_element_definition(
                res_dict, elem, sd, app_state
            )
            if not field_info:
                logger.warning("No information given for element, skipping!")
                continue
            discriminators = [
                {"path": sl.path, "type": sl.type}
                for sl in (elem.slicing.discriminator or [])
            ]
            field_info["slicing"] = {
                "discriminators": discriminators,
                "ordered": bool(elem.slicing.ordered),
                "rules": elem.slicing.rules,
            }
            # Legacy single-discriminator keys remain for existing emitters.
            if discriminators:
                field_info["slicing_type"] = discriminators[0]["type"]
                field_info["discriminator_path"] = discriminators[0]["path"]
                field_info["discriminator"] = discriminators[0]
            field_info["slices"] = []
            root_id = elem.id or elem.path
            for slice_elem in get_slices(
                sliced_elems, elem.path, root_id=root_id, direct_only=True
            ):
                slice_info = _parse_slice_tree(
                    slice_elem,
                    root_id,
                    sliced_elems,
                    slice_descendants,
                    res_dict,
                    sd,
                    app_state,
                    field_info["slicing"],
                )
                if slice_info is not None:
                    field_info["slices"].append(slice_info)
        else:
            field_info = element_definition_parser.parse_element_definition(
                res_dict, elem, sd, app_state
            )

        if field_info is None:
            continue

        # Build hierarchy
        field_info.setdefault("children", [])
        path_map[field_info["path"]] = field_info
        field_id = field_info.get("id") or field_info["path"]
        id_map[field_id] = field_info

        parent_id = field_id.rsplit(".", 1)[0] if "." in field_id else ""
        parent_path = (
            field_info["path"].rsplit(".", 1)[0]
            if "." in field_info["path"]
            else ""
        )

        if parent_id in id_map:
            id_map[parent_id]["children"].append(field_info)
        elif parent_path in path_map:
            path_map[parent_path]["children"].append(field_info)
        else:
            sd.mappable_fields.append(field_info)

    _resolve_content_references(sd.mappable_fields)

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
        if child_info is None and getattr(d, "max", None) == "0":
            # A prohibited descendant is not mappable, but an `exists` discriminator
            # needs its exact slice-local cardinality in the compiler IR.
            child_info = {
                "path": d.path,
                "id": d.id,
                "type": [
                    {"code": getattr(t, "code", None) or "N/A"}
                    for t in (getattr(d, "type", None) or [])
                ]
                or [{"code": "N/A"}],
                "cardinality": {"min": d.min, "max": d.max},
                "description": d.short or d.definition,
                "is_required": False,
                "is_prohibited": True,
                "fixed_value": [],
                "options": [],
            }
        if child_info is None:
            continue
        child_info.setdefault("children", [])
        local_map[d.id] = child_info
        parent_id = d.id.rsplit(".", 1)[0]
        parent = local_map.get(parent_id, slice_info)
        parent.setdefault("children", []).append(child_info)


def _slice_selector(element_id, root_id):
    prefix = f"{root_id}:"
    if not element_id or not element_id.startswith(prefix):
        return None
    selector = element_id[len(prefix) :]
    return selector if "." not in selector else None


def get_slices(
    slices,
    root_path,
    root_id=None,
    direct_only=False,
    parent_selector=None,
):
    """Return slice roots for one exact path and ElementDefinition identity."""
    root_id = root_id or root_path
    out = []
    for element in slices:
        if element.sliceName is None or element.path != root_path:
            continue
        selector = _slice_selector(getattr(element, "id", None), root_id)
        if selector is None and getattr(element, "id", None) == root_id:
            selector = element.sliceName
        if selector is None:
            continue
        selector_parent = selector.rsplit("/", 1)[0] if "/" in selector else None
        if direct_only and selector_parent is not None:
            continue
        if parent_selector is not None and selector_parent != parent_selector:
            continue
        out.append(element)
    return out


def _parse_slice_tree(
    slice_elem,
    root_id,
    all_slices,
    descendants,
    res_dict,
    sd,
    app_state,
    slicing,
    parent_slice_info=None,
):
    info = element_definition_parser.parse_element_definition(
        res_dict, slice_elem, sd, app_state
    )
    if info is None:
        return None
    selector = _slice_selector(slice_elem.id, root_id) or slice_elem.sliceName
    info["slice_identity"] = slice_elem.id
    info["slice_selector"] = selector
    info["slicing"] = deepcopy(slicing)
    _attach_slice_children(
        info, slice_elem, descendants, res_dict, sd, app_state
    )
    if parent_slice_info is not None:
        _inherit_reslice_constraints(parent_slice_info, info)
    reslices = get_slices(
        all_slices,
        slice_elem.path,
        root_id=root_id,
        parent_selector=selector,
    )
    if reslices:
        parsed_reslices = []
        for child in reslices:
            parsed = _parse_slice_tree(
                child,
                root_id,
                all_slices,
                descendants,
                res_dict,
                sd,
                app_state,
                slicing,
                parent_slice_info=info,
            )
            if parsed is not None:
                parsed_reslices.append(parsed)
        info["slices"] = parsed_reslices
    return info


def _child_identity(field):
    identity = field.get("id") or field.get("path") or ""
    return identity.rsplit(".", 1)[-1]


def _merge_inherited_field(inherited, declared):
    merged = deepcopy(inherited)
    for key, value in declared.items():
        if key == "children":
            continue
        if key == "fixed_value" and not value:
            continue
        merged[key] = deepcopy(value)

    children = [deepcopy(child) for child in (inherited.get("children") or [])]
    positions = {
        _child_identity(child): index for index, child in enumerate(children)
    }
    for child in declared.get("children") or []:
        key = _child_identity(child)
        if key in positions:
            children[positions[key]] = _merge_inherited_field(
                children[positions[key]], child
            )
        else:
            children.append(deepcopy(child))
    merged["children"] = children
    return merged


def _inherit_reslice_constraints(parent_slice, reslice):
    """Apply parent-slice constraints to a reslice and rebase inherited identities."""
    parent_id = parent_slice.get("slice_identity") or parent_slice.get("id")
    child_id = reslice.get("slice_identity") or reslice.get("id")
    inherited_children = [
        _rebased_copy(
            child,
            parent_slice.get("path"),
            reslice.get("path"),
            parent_id,
            child_id,
        )
        for child in (parent_slice.get("children") or [])
    ]
    inherited = {"children": inherited_children}
    merged = _merge_inherited_field(inherited, reslice)
    reslice["children"] = merged["children"]

    if not reslice.get("fixed_value") and parent_slice.get("fixed_value"):
        reslice["fixed_value"] = deepcopy(parent_slice["fixed_value"])
        for key in ("fixed_kind", "is_pattern", "pattern_value"):
            if key in parent_slice:
                reslice[key] = deepcopy(parent_slice[key])


def _walk_fields(fields):
    for field in fields or []:
        if not isinstance(field, dict):
            continue
        yield field
        yield from _walk_fields(field.get("children"))
        yield from _walk_fields(field.get("slices"))


def _rebased_copy(value, old_path, new_path, old_id, new_id):
    copied = deepcopy(value)
    for field in _walk_fields([copied]):
        path = field.get("path")
        if path and old_path and (path == old_path or path.startswith(old_path + ".")):
            field["path"] = new_path + path[len(old_path) :]
        eid = field.get("id")
        if eid and old_id and (eid == old_id or eid.startswith(old_id + ".")):
            field["id"] = new_id + eid[len(old_id) :]
    return copied


def _type_codes_for_resolution(raw_types):
    return [
        item.get("code") if isinstance(item, dict) else getattr(item, "code", item)
        for item in (raw_types or [])
    ]


def _resolve_content_references(fields):
    """Resolve local ``contentReference`` links into a rebased generation view."""
    all_fields = list(_walk_fields(fields))
    by_id = {
        field.get("id"): field
        for field in all_fields
        if field.get("id")
    }
    for _ in range(3):
        changed = False
        for field in all_fields:
            reference = field.get("content_reference")
            if not reference or field.get("content_reference_resolved"):
                continue
            target_id = str(reference).lstrip("#")
            target = by_id.get(target_id)
            if target is None or target is field:
                field["content_reference_resolved"] = False
                continue
            target_types = target.get("type") or []
            own_types = field.get("type") or []
            if not own_types or _type_codes_for_resolution(own_types) == ["N/A"]:
                field["type"] = deepcopy(target_types)
            if not field.get("children") and target.get("children"):
                old_path = target.get("path") or target_id
                new_path = field.get("path") or field.get("id")
                field["children"] = [
                    _rebased_copy(
                        child,
                        old_path,
                        new_path,
                        target_id,
                        field.get("id") or new_path,
                    )
                    for child in target["children"]
                ]
            field["content_reference_target"] = target_id
            field["content_reference_resolved"] = True
            changed = True
        if not changed:
            break


def get_fhir_class(type_name: str):
    """access fhir model class"""
    try:
        return get_fhir_model_class(type_name)
    except (KeyError, ValueError) as e:
        logger.error(
            "Model class for %s not found in fhir.resources! Error: %s", type_name, e
        )
        return None
