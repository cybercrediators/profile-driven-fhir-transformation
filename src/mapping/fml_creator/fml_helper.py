from fhir.resources.R4B.structuremap import StructureMapGroupRuleTargetParameter
import re
from typing import Optional

import logging
logger = logging.getLogger(__name__)

from data_handling.url_resolver.fhir_url_resolver import resolve_url
from parser.resource_parser.fhir_type_introspection import PRIMITIVES

BASE_META_SUFFIXES = (
    ".id",
    ".meta",
    ".implicitRules",
    ".language",
    ".text",
    ".contained",
    ".modifierExtension",
)


def is_meaningful_modifier_extension(field):
    """check if a field is a profile-defined modifierExtension slice (not the empty base element)"""
    if not isinstance(field, dict):
        return False
    if not field.get("path", "").endswith(".modifierExtension"):
        return False
    return bool(
        field.get("sliceName") or field.get("slices") or field.get("extension_url")
    )


def conv_mappable(app_state, field):
    """convert a parsed mappable field to a normalized dict for rule generation"""

    def get_ref_target_type(ref_obj):
        if isinstance(ref_obj, dict):
            return ref_obj.get("type") or ref_obj.get("resourceType")
        try:
            ref_type = getattr(ref_obj, "type", None)
            if isinstance(ref_type, list):
                ref_type = ref_type[0] if ref_type else None
            if ref_type:
                return ref_type
            return getattr(ref_obj, "resource_type", None) or getattr(
                ref_obj, "resourceType", None
            )
        except Exception:
            return None

    conv = dict(field)
    type_list = field.get("type", [])

    if isinstance(type_list, list) and len(type_list) > 1:
        conv["type"] = "choice"
        conv["choice_types"] = [t.get("code") for t in type_list]
        conv["is_type_choice"] = True
    elif isinstance(type_list, list) and type_list:
        type_info = type_list[0]
        conv["type"] = type_info.get("code", "string")
        if conv["type"] == "Reference" and "targetProfile" in type_info.keys():
            target_profiles = type_info["targetProfile"]
            if target_profiles:
                target_url = target_profiles[0]
                candidates = []
                for url in target_profiles:
                    ref_obj = resolve_url(url, app_state)
                    base = (
                        get_ref_target_type(ref_obj) if ref_obj else None
                    ) or url.split("/")[-1]
                    if base and base != "TYPE-NOT-FOUND" and base not in candidates:
                        candidates.append(base)
                target_type = "|".join(candidates) if candidates else "TYPE-NOT-FOUND"

                conv["reference_target"] = target_type
                conv["reference_target_profile"] = target_url
                conv["reference_target_in_registry"] = bool(
                    any(app_state.registry.get_obj_by_name(u) for u in target_profiles)
                )
                # print("Reference target type:", target_type, "for profile/resource", target_url)
        if type_info.get("profile"):
            conv["profile_structure"] = type_info["profile"]
        if type_info.get("type_structure"):
            conv["type_structure"] = type_info["type_structure"]
    elif isinstance(type_list, str):
        conv["type"] = type_list
    else:
        conv["type"] = "string"

    return conv


def _expanded_profile_fields(profile_structure):
    """Return resolved profile fields, or ``None`` for a canonical-only reference.

    """
    if (
        isinstance(profile_structure, list)
        and profile_structure
        and isinstance(profile_structure[0], list)
    ):
        return profile_structure[0]
    return None


def flatten_profile_fields(app_state, res_type, fields):
    """get a flattened view of the mappable fields for the automapper, hoisting slices to top level
    and enriching them with derived keys (choice_types, reference_target, profile-defined children, extension_url, …)"""
    logger.info("Preprocessing mappable fields for further tasks")
    results = []
    for field in fields:
        conv = conv_mappable(app_state, field)
        # pprint(conv)
        path = conv["path"]
        # skip meta paths (but keep profile-defined modifierExtension slices)
        if path.endswith(BASE_META_SUFFIXES) and not is_meaningful_modifier_extension(
            conv
        ):
            continue

        profile_structure = conv.get("profile_structure")
        if profile_structure and ".extension" in path:
            if isinstance(profile_structure, list):
                nested_fields = profile_structure[0] if profile_structure else []
                if isinstance(nested_fields, list):
                    for nested in nested_fields:
                        if (
                            isinstance(nested, dict)
                            and nested.get("path", "").endswith(".url")
                            and nested.get("fixed_value")
                        ):
                            conv["extension_url"] = nested["fixed_value"]
                            break
        results.append(conv)

        nested_fields = _expanded_profile_fields(profile_structure)
        if nested_fields is not None:
            conv["children"] = []
            for nested_field in nested_fields:
                if not isinstance(nested_field, dict):
                    continue
                nested_path = nested_field.get("path", "")
                if nested_path.startswith("Extension."):
                    nested_path = nested_path.replace("Extension.", f"{path}.", 1)
                elif "." not in nested_path:
                    nested_path = f"{path}.{nested_path}"
                nested_copy = nested_field.copy()
                nested_copy["path"] = nested_path

                nested_conv = conv_mappable(app_state, nested_copy)
                # Append to children instead of results list to preserve hierarchy for context
                conv["children"].append(nested_conv)
                # results.append(nested_conv)

        if slices := field.get("slices"):
            for sl in slices:
                slice_fields = process_slice(app_state, field, sl)
                results.extend(slice_fields)

    return results


# def process_profile_structure(app_state, profile_structure, conv, path):
#     res = []
#     if profile_structure and isinstance(profile_structure, list):
#         nested_fields = profile_structure[0] if profile_structure else []
#         # print(nested_fields)
#         for nested in nested_fields:
#             nested_path = nested.get("path", "")
#             if nested_path.endswith(".url") and nested.get("fixed_value"):
#                 conv["extension_url"] = nested["fixed_value"]
#                 break
#             if nested_path.startswith("Extension."):
#                 # print(nested_path)
#                 # print(type(nested_path))
#                 nested_path = nested_path.replace("Extension.", f"{path}.", 1)
#             elif "." not in nested:
#                 nested_path = f"{path}.{nested_path}"
#             nested_copy = nested.copy()
#             nested_copy["path"] = nested_path
#             nested_conv = conv_mappable(app_state, nested_copy)
#             res.append(nested_conv)
#         return res
#     return None


def process_slice(app_state, field, slice):
    """
    Create (flat) fields for slices
    """
    if not isinstance(slice, dict):
        return []
    slice_name = slice.get("sliceName", "unknown")
    base_path = field["path"]
    field_name = base_path.split(".")[-1]
    sliced_path = base_path.replace(field_name, f"{field_name}:{slice_name}")

    inline_profile = None
    for t in slice.get("type") or []:
        if isinstance(t, dict) and isinstance(t.get("profile"), list) and t["profile"]:
            inline_profile = t["profile"]
            break

    conv_slice = conv_mappable(app_state, slice)
    conv_slice["path"] = sliced_path

    if not conv_slice.get("profile_structure") and inline_profile:
        conv_slice["profile_structure"] = inline_profile

    result = []
    # if profile_structure := conv_slice.get("profile_structure"):
    #     if res_conv := process_profile_structure(app_state, profile_structure, conv_slice, sliced_path):
    #         result.extend(res_conv)
    if field.get("slicing_type") or field.get("discriminator_path"):
        conv_slice["slicing"] = {
            "discriminator": [
                {
                    "path": field.get("discriminator_path", "url"),
                    "type": field.get("slicing_type", "value"),
                }
            ]
        }
        # Store discriminator info at field level too
        conv_slice["discriminator_path"] = field.get("discriminator_path", "url")
        conv_slice["slicing_type"] = field.get("slicing_type", "value")

    # Extract extension URL from slice profile structure
    if conv_slice.get("profile_structure"):
        profile_struct = conv_slice["profile_structure"]
        nested_fields = _expanded_profile_fields(profile_struct)
        if nested_fields is not None:
            for nested in nested_fields:
                if not isinstance(nested, dict):
                    continue
                if nested.get("path", "").endswith(".url") and nested.get(
                    "fixed_value"
                ):
                    conv_slice["extension_url"] = nested["fixed_value"]
                    break

    result = [conv_slice]

    # Process nested structure in slice
    if conv_slice.get("profile_structure"):
        profile_struct = conv_slice["profile_structure"]
        nested_fields = _expanded_profile_fields(profile_struct)
        if nested_fields is not None:
            conv_slice["children"] = []

            for nested_field in nested_fields:
                if not isinstance(nested_field, dict):
                    continue
                nested_path = nested_field.get("path", "")

                # Adjust path to be under the slice (use colon notation)
                if nested_path.startswith("Extension."):
                    nested_path = nested_path.replace(
                        "Extension.", f"{sliced_path}.", 1
                    )
                elif "." not in nested_path:
                    nested_path = f"{sliced_path}.{nested_path}"

                nested_copy = nested_field.copy()
                nested_copy["path"] = nested_path

                nested_normalized = conv_mappable(app_state, nested_copy)

                if nested_normalized.get("slices"):
                    updated_sub_slices = []
                    for sub_sl in nested_normalized["slices"]:
                        if not isinstance(sub_sl, dict) or not sub_sl.get("sliceName"):
                            updated_sub_slices.append(sub_sl)
                            continue
                        sub_sl_copy = dict(sub_sl)
                        sub_name = sub_sl["sliceName"]
                        if "." in nested_path:
                            prefix, last = nested_path.rsplit(".", 1)
                            sub_sl_copy["path"] = f"{prefix}.{last}:{sub_name}"
                        else:
                            sub_sl_copy["path"] = f"{nested_path}:{sub_name}"
                        # Normalize type so create_mappable_field_rule sees "Extension" string
                        sub_sl_copy = conv_mappable(app_state, sub_sl_copy)
                        updated_sub_slices.append(sub_sl_copy)
                    nested_normalized["slices"] = updated_sub_slices

                conv_slice["children"].append(nested_normalized)
    return result


def find_reference_fields(fields):
    """
    Find all references and target types (recursive)
    """
    refs = set()
    for field in fields:
        if field.get("type") == "Reference":
            reference_target = field.get("reference_target")
            # print(field)
            if reference_target:
                refs.add(reference_target)

        if field.get("children"):
            refs.update(find_reference_fields(field["children"]))

        if field.get("slices"):
            refs.update(find_reference_fields(field["slices"]))
    return refs


PRIMITIVE_TYPES = {p.lower() for p in PRIMITIVES}
_CANONICAL_PRIMITIVE = {p.lower(): p for p in PRIMITIVES}


def attr(obj, name, default=None):
    """Read an attribute/key uniformly from a dict OR a fhir.resources model object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def canonical_primitive(type_code: str) -> Optional[str]:
    """return the canonical PascalCase name for a FHIR primitive type code, or None if not a primitive"""
    if not type_code:
        return None
    return _CANONICAL_PRIMITIVE.get(type_code.lower())


def fhir_type_suffix(type_code: str) -> str:
    """get the value[x] suffix for a FHIR type code, in canonical PascalCase (e.g. "dateTime" -> "DateTime")"""
    if not type_code:
        return ""
    name = _CANONICAL_PRIMITIVE.get(type_code.lower(), type_code)
    return name[0].upper() + name[1:]


def infer_extension_value_type(field_type, field):
    """retrieve the base value type for an extension field, or "string" if unknown. For a polymorphic extension, the first type is used."""
    if "value_type" in field:
        base_type = field["value_type"]

    elif field_type != "Extension":
        base_type = field_type
    else:
        base_type = "string"

    # base_type may arrive as a list (multi-type extension value / raw type list) or a
    # type dict ({"code": ...}) — normalize to a single type-code string.
    if isinstance(base_type, list):
        base_type = base_type[0] if base_type else "string"
    if isinstance(base_type, dict):
        base_type = base_type.get("code", "string")
    if not isinstance(base_type, str) or not base_type:
        base_type = "string"

    if base_type.lower() in PRIMITIVE_TYPES:
        return f"value{fhir_type_suffix(base_type)}"
    return f"value{base_type}"


def is_primitive_type(type_name):
    t = (type_name or "").lower()
    if "fhirpath/system." in t or t.startswith("system."):
        return True
    return t in PRIMITIVE_TYPES


def parse_slice_info(field_name, field):
    """
    Get slice information
    """
    has_slice_name = field.get("sliceName") is not None
    is_slice = ":" in field_name or has_slice_name
    slice_info = {}

    if is_slice:
        if ":" in field_name:
            parts = field_name.split(":")
            base_name = parts[0]
            slice_name = parts[1] if len(parts) > 1 else "unknown"
        else:
            base_name = field_name
            slice_name = field.get("sliceName", "unknown")

        slice_info = {"base_info": base_name, "slice_name": slice_name}

        if field.get("slicing"):
            slice_info["discriminator"] = {
                "path": field["slicing"]["discriminator"][0].get("path", "url"),
                "type": field["slicing"]["discriminator"][0].get("type", "value"),
                "value": field.get("fixed_value"),
            }

    return is_slice, slice_info


COPYABLE_PRIMITIVES = {
    "boolean",
    "string",
    "uri",
    "url",
    "oid",
    "id",
    "code",
    "canonical",
    "uuid",
    "markdown",
    "base64binary",
}

CASTABLE_PRIMITIVES = {
    "date",
    "datetime",
    "time",
    "instant",
    "integer",
    "integer64",
    "unsignedint",
    "positiveint",
    "decimal",
}

assert COPYABLE_PRIMITIVES.isdisjoint(
    CASTABLE_PRIMITIVES
), "copy/cast primitive sets overlap"
assert (
    COPYABLE_PRIMITIVES | CASTABLE_PRIMITIVES
) <= PRIMITIVE_TYPES, "copy/cast sets contain a non-primitive type"
assert PRIMITIVE_TYPES - (COPYABLE_PRIMITIVES | CASTABLE_PRIMITIVES) == {
    "xhtml"
}, "unexpected primitive missing from copy/cast partition"


def get_transform_for_type(field_type, source_variable=None):
    """Return appropriate StructureMap transform metadata for a primitive field."""
    normalized = field_type or "string"
    key = normalized.lower()

    def _value_id_param():
        if not source_variable:
            return None
        return StructureMapGroupRuleTargetParameter.model_construct(
            valueId=source_variable
        )

    if key in COPYABLE_PRIMITIVES:
        params = []
        src_param = _value_id_param()
        if src_param:
            params.append(src_param)
        return {"transform": "copy", "parameters": params}

    if key in CASTABLE_PRIMITIVES:
        params = []
        src_param = _value_id_param()
        if src_param:
            params.append(src_param)
        params.append(
            StructureMapGroupRuleTargetParameter.model_construct(valueString=normalized)
        )
        return {"transform": "cast", "parameters": params}

    # For complex types: default to create transform referencing target type
    return {
        "transform": "create",
        "parameters": [
            StructureMapGroupRuleTargetParameter.model_construct(valueString=normalized)
        ],
    }


def clean_field_name(field_name):
    """Keep a valid field name for StructureMap rules"""
    field_name = field_name.replace("_", "-")
    return re.sub(r"\W+", "-", field_name).strip("-")


def fixed_scalar_literal(value) -> str:
    """FHIR lexical form of a fixed scalar constant. Python booleans stringify as
    'True'/'False', which is not valid FHIR boolean lexical form — lowercase them."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def local_element_name(field_id: str) -> str:
    """get the local element name from a FHIR path or element id, stripping any profile/extension prefixes"""
    if field_id and "." in field_id and not field_id.startswith("TODO"):
        return field_id.split(".")[-1]
    return field_id


def type_codes(field_type) -> list:
    """return a list of type codes for a field type, which may be a string, dict, or list of dicts"""
    if field_type is None:
        return []
    if isinstance(field_type, str):
        return [field_type]
    if isinstance(field_type, dict):
        code = field_type.get("code")
        return [code] if code else []
    if isinstance(field_type, list):
        codes = []
        for t in field_type:
            if isinstance(t, dict):
                if t.get("code"):
                    codes.append(t["code"])
            elif isinstance(t, str):
                codes.append(t)
        return codes
    return []


def is_extension_type(field_type) -> bool:
    """True if the field type is (or includes) Extension."""
    return "Extension" in type_codes(field_type)


def is_reference_type(field_type) -> bool:
    """True if the field type is (or includes) Reference."""
    return "Reference" in type_codes(field_type)
