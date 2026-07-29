"""Shared utility helpers: JSON I/O and fhir.resources model conversion."""

import json
import re
import pkgutil
import importlib
from functools import lru_cache

import fhir.resources.R4B
import logging
logger = logging.getLogger(__name__)


def get_json(fname):
    with open(fname, encoding="utf-8") as j:
        return json.load(j)


def store_json(data, fname):
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def store_json_str(data_str, fname):
    with open(fname, "w", encoding="utf-8") as f:
        f.write(data_str)


def clean_obj(obj):
    """Remove None fields from dict/obj"""
    f = {k: v for k, v in obj.items() if v is not None and k != "text"}
    obj.clear()
    obj.update(f)
    return obj


@lru_cache(maxsize=1)
def get_predef_modules():
    return list(
        [name for _, name, _ in pkgutil.iter_modules(fhir.resources.R4B.__path__)]
    )


def get_model_class(res_type):
    """Return the fhir.resources.R4B model class for a resource type, or None if unknown."""
    if res_type is None:
        return None
    if res_type.lower() not in get_predef_modules():
        return None
    return getattr(
        importlib.import_module(f"fhir.resources.R4B.{res_type.lower()}"), res_type
    )


def json_file_to_obj(file, spec_context=None):
    content = get_json(file)
    res_type = content.get("resourceType") if isinstance(content, dict) else None
    if spec_context is not None and isinstance(content, dict):
        spec_context.assert_profile_compatible(content)
    return json_to_obj(content, res_type), res_type


def json_to_obj(content, res_type):
    """Convert a FHIR resource json string to a fhir.resource object"""
    cls_ = get_model_class(res_type)
    if cls_ is not None:
        try:
            obj = cls_(**content)
            return obj
        except Exception as e:
            try:
                cleaned = _strip_primitive_extensions(content)
                obj = cls_(**cleaned)
                logger.debug(
                    f"Validated {res_type} after sanitizing R4B-incompatible fields ({e})."
                )
                return obj
            except Exception as e2:
                logger.debug(
                    f"{res_type} does not validate against fhir.resources R4B; using raw dict ({e2})."
                )
                return content
    return None


def _strip_primitive_extensions(obj, parent_key=None):
    """Recursively drop content the pinned fhir.resources **R4B** models reject but that
    is irrelevant to profile flattening / map generation, so a fetched resource can still
    validate into a typed object instead of falling back to a raw dict"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            if k == "additional" and parent_key == "binding":
                continue
            out[k] = _strip_primitive_extensions(v, k)
        return out
    if isinstance(obj, list):
        return [_strip_primitive_extensions(x, parent_key) for x in obj]
    return obj


def get_resource_type(fname):
    content = get_json(fname)
    # if isinstance(content, list):
    # print(f"READ {fname}")
    # print(len(content))
    # print(content)
    if isinstance(content, dict):
        return content.get("resourceType")
    return None


def resource_identity(data, registry_url: str = "") -> str:
    """Stable identity for a conformance resource.

    ``id`` is optional in FHIR — the canonical url is the identity — and published IGs
    do ship StructureDefinitions without one. Fall back to the canonical's last segment
    so such a profile still gets an on-disk name, a StructureMap name and, above all, a
    string a mapping table can target when a sibling profile shares its resource type.
    """
    res_id = getattr(data, "id", None)
    if res_id:
        return res_id
    canonical = getattr(data, "url", None) or registry_url or ""
    return canonical.rstrip("/").split("/")[-1].split("|")[0] or "unnamed_resource"


def fhir_id_token(value: str, max_len: int = 50) -> str:
    """Make a string usable as (part of) a FHIR resource id.

    ``Resource.id`` is restricted to ``[A-Za-z0-9-.]``; identities derived from a
    canonical url may carry other characters (Capable.repository's profiles are named
    ``Communication_Profile``), which a FHIR server rejects. Only ids are sanitized —
    map names and mapping-table prefixes keep the profile's real name.
    """
    return re.sub(r"[^A-Za-z0-9.-]", "-", value or "")[:max_len]


def fhir_name_token(value: str, max_len: int = 255) -> str:
    """Return an invariant-safe FHIR ``name`` value.

    ``name`` is stricter than ``id``: it starts with an uppercase letter and then
    contains only letters, digits, and underscores.
    """
    token = re.sub(r"[^A-Za-z0-9_]", "_", value or "")
    if not token or not token[0].isalpha():
        token = f"Map_{token}"
    token = token[0].upper() + token[1:]
    return token[:max_len]


def get_value_from_element(element, attr_names):
    """
    Extracts value[x] from a given element
    """
    for attr_name in dir(element):
        if attr_name.startswith(attr_names):
            val = getattr(element, attr_name)
            if val is not None:
                return attr_name, val
    return None, None
