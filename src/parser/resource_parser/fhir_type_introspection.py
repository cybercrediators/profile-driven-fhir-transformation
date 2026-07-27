import importlib
import types
import datetime
import decimal
import uuid as _uuid
from typing import List, get_origin, get_args, Union, Annotated

import logging
logger = logging.getLogger(__name__)

from fhir.resources.R4B import fhirtypes as _fhirtypes

# primitives for spec introspection and primitive registry
_BARE_PRIMITIVE_CARRIERS = (
    bool,
    int,
    float,
    str,
    bytes,
    datetime.date,
    datetime.datetime,
    datetime.time,
    decimal.Decimal,
    _uuid.UUID,
)


def _build_primitive_registry():
    """derive the FHIR primitive type tables directly from fhir.resources library"""
    names = set()
    marker_to_name = {}
    carrier_names = {}  # carrier python type -> [primitive names sharing it]
    for attr in dir(_fhirtypes):
        if not attr.endswith("Type"):
            continue
        alias = getattr(_fhirtypes, attr)
        fhir_name = attr[:-4]
        fhir_name = fhir_name[0].lower() + fhir_name[1:]
        carrier = None
        if type(alias).__name__ == "_AnnotatedAlias":
            carrier = get_args(alias)[0]
            marker = get_args(alias)[1]
            names.add(fhir_name)
            marker_to_name[type(marker).__name__] = fhir_name
        elif isinstance(alias, type) and issubclass(alias, _BARE_PRIMITIVE_CARRIERS):
            carrier = alias
            names.add(fhir_name)
        if isinstance(carrier, type) and issubclass(carrier, _BARE_PRIMITIVE_CARRIERS):
            carrier_names.setdefault(carrier, []).append(fhir_name)
    carrier_to_name = {c: n[0] for c, n in carrier_names.items() if len(n) == 1}
    return names, marker_to_name, carrier_to_name


PRIMITIVES, _PRIMITIVE_MARKER_TO_NAME, _CARRIER_TO_PRIMITIVE = _build_primitive_registry()


def is_primitive(type_code):
    return type_code in PRIMITIVES

def _is_standalone_fhir_type(name, _cache={}):
    """if name is top-level fhir.resources type (module)"""
    if not isinstance(name, str) or not name:
        return False
    if name in _cache:
        return _cache[name]
    try:
        importlib.import_module(f"fhir.resources.R4B.{name.lower()}")
        ok = True
    except Exception:
        ok = False
    _cache[name] = ok
    return ok


def get_complex_type_fields(
    type_code, parent_path="", recursion_depth=0, containing_module=None
):
    """get fields of a complex type using fhir.resources"""
    if recursion_depth > 5:
        logger.warning(f"Max recursion depth reached for {type_code}")
        return []

    try:
        module = importlib.import_module(f"fhir.resources.R4B.{type_code.lower()}")
        cls = getattr(module, type_code)
    except (ImportError, AttributeError):
        if containing_module is not None and hasattr(containing_module, type_code):
            module = containing_module
            cls = getattr(containing_module, type_code)
        elif type_code.endswith("Type"):
            try:
                short_code = type_code[:-4]
                module = importlib.import_module(
                    f"fhir.resources.R4B.{short_code.lower()}"
                )
                cls = getattr(module, short_code)
            except (ImportError, AttributeError):
                logger.warning(
                    f"Could not load fhir.resources class for type: {type_code}"
                )
                return []
        else:
            logger.warning(f"Could not load fhir.resources class for type: {type_code}")
            return []

    fields = []
    # handle pydantic v2/v1 backward compatibility
    model_fields = getattr(cls, "model_fields", getattr(cls, "__fields__", {}))

    for fhir_name, field_info in model_fields.items():
        if hasattr(field_info, "alias") and field_info.alias:
            fhir_name = field_info.alias
        elif (
            hasattr(field_info, "serialization_alias")
            and field_info.serialization_alias
        ):
            fhir_name = field_info.serialization_alias

        # skip internal fields
        if (
            fhir_name.startswith("_")
            or fhir_name == "fhir_comments"
            or fhir_name == "extension"
            or fhir_name == "id"
        ):
            continue

        # determine type and cardinality
        annotation = getattr(
            field_info, "annotation", getattr(field_info, "outer_type_", None)
        )
        inner_type, is_list, is_optional, metadata = extract_inner_type(annotation)

        field_type_code = get_fhir_type_name(inner_type)

        prim = _fhir_primitive_from_metadata(metadata)
        if prim:
            field_type_code = prim

        # check and recursively expand if it's a complex type
        sub_fields = []
        is_backbone_subtype = False
        if (
            not is_primitive(field_type_code)
            and field_type_code != "N/A"
            and field_type_code != "Element"
        ):
            # avoid infinite recursion for self-referencing types if any
            if field_type_code != type_code:
                sub_fields = get_complex_type_fields(
                    field_type_code,
                    f"{parent_path}.{fhir_name}",
                    recursion_depth + 1,
                    containing_module=module,
                )
            # anonymous backbone sub-types have no named (certain types are not supported by matchbox, so only emit as generic)
            is_backbone_subtype = not _is_standalone_fhir_type(field_type_code)

        emit_type = "BackboneElement" if is_backbone_subtype else field_type_code
        field_dict = {
            "path": f"{parent_path}.{fhir_name}",
            "id": f"{type_code}.{fhir_name}",
            "type": emit_type,
            "description": getattr(field_info, "description", ""),
            "is_required": not is_optional,
            "is_list": is_list,
        }

        if sub_fields:
            field_dict["type_structure"] = sub_fields

        fields.append(field_dict)

    return fields


def extract_inner_type(annotation):
    """extracts the inner type, is_list flag, is_optional flag, and metadata from a type annotation"""
    is_list = False
    is_optional = False
    metadata = []

    origin = get_origin(annotation)
    args = get_args(annotation)

    if (
        origin is Union
        or (origin is not None and str(origin) == "typing.Union")
        or origin is types.UnionType
    ):
        if type(None) in args:
            is_optional = True
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                annotation = non_none_args[0]
                origin = get_origin(annotation)
                args = get_args(annotation)
            else:
                annotation = non_none_args[0]
                origin = get_origin(annotation)
                args = get_args(annotation)

    if origin is list or origin is List:
        is_list = True
        if args:
            annotation = args[0]
            origin = get_origin(annotation)
            args = get_args(annotation)

            # List[Optional[Type]] which is common in fhir.resources
            if (
                origin is Union
                or (origin is not None and str(origin) == "typing.Union")
                or origin is types.UnionType
            ):
                if type(None) in args:
                    non_none_args = [a for a in args if a is not type(None)]
                    if len(non_none_args) == 1:
                        annotation = non_none_args[0]

    # Annotated
    if hasattr(annotation, "__origin__") and annotation.__origin__ is Annotated:
        metadata = get_args(annotation)[1:]
        annotation = get_args(annotation)[0]
    elif str(annotation).startswith("typing.Annotated"):
        args = get_args(annotation)
        if args:
            annotation = args[0]
            metadata = args[1:]

    return annotation, is_list, is_optional, metadata


def _fhir_primitive_from_metadata(metadata):
    """return the FHIR primitive type carried by a fhir.resources field's Annotated
    metadata, or None, via the spec-derived marker map (see _build_primitive_registry).
    """
    for meta in metadata or []:
        name = getattr(getattr(meta, "__class__", None), "__name__", "")
        fhir_name = _PRIMITIVE_MARKER_TO_NAME.get(name)
        if fhir_name:
            return fhir_name
    return None


def get_fhir_type_name(py_type):
    """map a python type / fhir.resources class to a FHIR type name, with translation"""
    origin = get_origin(py_type)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(py_type) if a is not type(None)]
        if non_none:
            py_type = non_none[0]

    if isinstance(py_type, type):
        derived = _CARRIER_TO_PRIMITIVE.get(py_type)
        if derived:
            return derived

    if py_type is str:
        return "string"
    if py_type is int:
        return "integer"
    if py_type is datetime.datetime:
        return "dateTime"
    if py_type is float:
        return "decimal"

    if hasattr(py_type, "__name__"):
        name = py_type.__name__
        if name.endswith("Type"):
            name = name[:-4]
        return name

    return str(py_type)
