import logging
logger = logging.getLogger(__name__)

from parser.resource_parser import structure_definition_parser
from parser.resource_parser.value_expander import expand_valueset
from parser.resource_parser.fhir_type_introspection import (
    get_complex_type_fields,
    is_primitive,
)
from parser.resource_parser.template_writer import set_value_in_temp_dict
from helpers.utils import get_value_from_element


def parse_element_definition(res_dict, elem, sd, app_state):
    """parse element definitions into flattened field_info dict:

    cardinality, description, fixed/pattern/default value, bound-ValueSet options, the
    extension URL (for Extension slices), a reference target, and the mustSupport / comment /
    min-maxValue / maxLength / contentReference facets.

    Constraints and other non-generative facets are retained for compiler diagnostics;
    they are not treated as source values.
    """
    logger.info("Processing element: %s", elem.id)
    if len(elem.path.split(".")) == 1 or elem.max == "0":
        if elem.max == "0" and getattr(sd, "prohibited_paths", None) is not None:
            parts = elem.path.split(".")
            if len(parts) > 1:
                sd.prohibited_paths.add(".".join(parts[1:]))
        logger.info("%s not required -> skipping", elem.path)
        return None

    # handle element type
    elem_type = []
    # canonical URL of the extension SD, if this is an Extension slice
    extension_profile_url = None
    if elem.type:
        for t in elem.type:
            tmp = {"code": t.code if t else "N/A"}
            if t.targetProfile:
                tmp["targetProfile"] = t.targetProfile
            if t.profile:
                tmp["profile_canonical"] = list(t.profile)
                tmp["profile"] = []
                for pf in t.profile:
                    profile_sd = app_state.registry.get_obj_by_name(pf)
                    # TODO: change this to parse_resource when its done
                    profile_sd, app_state = (
                        structure_definition_parser.parse_structure_definition(
                            profile_sd, app_state
                        )
                    )
                    app_state.registry.add_to_used_by(pf, sd.data.url)
                    if profile_sd is None:
                        tmp["profile"].append(pf)
                        continue
                    tmp["profile"].append(profile_sd.mappable_fields)
                    if t.code == "Extension" and extension_profile_url is None:
                        extension_profile_url = pf

            # expand complex types via fhir.resources
            if (
                t.code
                and not is_primitive(t.code)
                and t.code not in ["BackboneElement", "Extension", "N/A"]
            ):
                complex_fields = get_complex_type_fields(t.code, parent_path=elem.path)
                if complex_fields:
                    tmp["type_structure"] = complex_fields
                    logger.info(
                        f"Expanded complex type {t.code} with {len(complex_fields)} fields using fhir.resources"
                    )
                else:
                    logger.warning(
                        f"Could not expand complex type {t.code} using fhir.resources"
                    )

            elem_type.append(tmp)
    else:
        elem_type.append({"code": "N/A"})

    def _attach_binding_facets(field_info, expand=False):
        """Record the binding constraint on the field model"""
        if elem.binding and elem.binding.valueSet:
            field_info["binding_strength"] = elem.binding.strength
            field_info["valueSetUrl"] = elem.binding.valueSet
            if expand:
                field_info["options"] = expand_valueset(
                    elem.binding.valueSet, [], app_state
                )
            # add used_by to value set
            app_state.registry.add_to_used_by(elem.binding.valueSet, sd.data.url)
        return field_info

    attr, value = get_value_from_element(elem, ("fixed", "pattern"))
    if value is not None and not (
        isinstance(value, (list, dict, str)) and len(value) == 0
    ):
        is_pattern = bool(attr) and attr.startswith("pattern")
        logger.info(
            "%s has a %s value. Setting pre-defined value...",
            elem.path,
            "pattern" if is_pattern else "fixed",
        )
        set_value_in_temp_dict(res_dict, elem.path.split(".")[1:], value, sd)
        # fixed_value stays populated for BOTH fixed[x] and pattern[x]
        field_info = create_field_info(elem, elem_type, value)
        if is_pattern:
            field_info["is_pattern"] = True
            field_info["pattern_value"] = value
            field_info["fixed_kind"] = "pattern"
        else:
            field_info["fixed_kind"] = "fixed"
        return _attach_binding_facets(field_info)

    child_url = get_ref_profile(elem)
    if child_url:
        ref = {"reference": child_url}
        set_value_in_temp_dict(res_dict, elem.path.split(".")[1:], ref, sd)
        field_info = create_field_info(elem, elem_type)
        field_info["reference_target"] = child_url
        return _attach_binding_facets(field_info)

    field_info = create_field_info(elem, elem_type)

    if extension_profile_url:
        field_info["extension_url"] = extension_profile_url
    return _attach_binding_facets(field_info, expand=True)


def create_field_info(elem, elem_type, fixed_value=[]):
    """
    Create a simple field_info dict for a given ElementDefinition element
    """
    base_info = {
        "path": elem.path,
        "id": elem.id,
        "type": elem_type,
        "cardinality": {"min": elem.min, "max": elem.max},
        "description": elem.short or elem.definition,
        "is_required": elem.min is not None and elem.min > 0,
        "fixed_value": fixed_value,
        "options": [],
    }
    if elem.sliceName:
        base_info["sliceName"] = elem.sliceName
    if elem.mustSupport:
        base_info["must_support"] = True
    if elem.comment:
        base_info["comment"] = elem.comment
    if elem.maxLength is not None:
        base_info["max_length"] = elem.maxLength
    if elem.contentReference:
        base_info["content_reference"] = elem.contentReference
    if elem.condition:
        base_info["conditions"] = list(elem.condition)
    if elem.constraint:
        base_info["constraints"] = [
            c.model_dump(exclude_none=True) if hasattr(c, "model_dump") else c
            for c in elem.constraint
        ]
    if elem.mapping:
        base_info["mappings"] = [
            m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
            for m in elem.mapping
        ]
    if elem.isModifier is not None:
        base_info["is_modifier"] = bool(elem.isModifier)
    if elem.isSummary is not None:
        base_info["is_summary"] = bool(elem.isSummary)
    if elem.requirements:
        base_info["requirements"] = elem.requirements
    if elem.alias:
        base_info["aliases"] = list(elem.alias)
    if elem.meaningWhenMissing:
        base_info["meaning_when_missing"] = elem.meaningWhenMissing
    if elem.orderMeaning:
        base_info["order_meaning"] = elem.orderMeaning

    default_attr, default_val = get_value_from_element(elem, ("defaultValue",))
    if default_attr and "__" not in default_attr and default_val is not None:
        base_info["default_value"] = {
            "type": default_attr[len("defaultValue"):],
            "value": default_val,
        }

    min_attr, min_val = get_value_from_element(elem, ("minValue",))
    if min_attr and "__" not in min_attr and min_val is not None:
        base_info["min_value"] = {"type": min_attr[len("minValue"):], "value": min_val}
    max_attr, max_val = get_value_from_element(elem, ("maxValue",))
    if max_attr and "__" not in max_attr and max_val is not None:
        base_info["max_value"] = {"type": max_attr[len("maxValue"):], "value": max_val}

    return base_info


def get_ref_profile(elem_def):
    """Return the single targetProfile URL of a Reference element, or None."""
    if elem_def.type:
        elem_type = elem_def.type[0]
        if (
            elem_type.code == "Reference"
            and elem_type.targetProfile
            and len(elem_type.targetProfile) == 1
        ):
            return elem_type.targetProfile[0]
    return None
