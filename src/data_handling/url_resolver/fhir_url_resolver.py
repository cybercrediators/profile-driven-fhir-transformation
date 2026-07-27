import logging
logger = logging.getLogger(__name__)
from helpers import utils
from data_handling.url_resolver import url_resolve_adapter
from data_handling.app_state import AppState
import json
import fnmatch
import os


def resolve_url(
    canonical_url: str,
    app_state: AppState,
    resolver: url_resolve_adapter.URLResolveAdapterBase = None,
):
    """
    Resolve a canonical FHIR URL from the registry, cache, local packages or the network.

    Returns a ``fhir.resources`` object (or the raw dict when validation fails — see
    :func:`helpers.utils.json_to_obj`), or ``None`` if the resource cannot be found.
    """
    # Check the registry
    logger.info("Checking the registry")
    if app_state.registry is None:
        logger.info("No registry given")
    elif canonical_url in app_state.registry.registry_objects:
        return app_state.registry.registry_objects[canonical_url].data
    # Check the cache
    logger.info("Not found in Registry...Checking Cache")
    obj = app_state.cache.get_resource_from_cache(canonical_url)
    if obj:
        return utils.json_to_obj(obj, obj.get("resourceType"))
    # Check local packages
    logger.info("Not found in Cache...Checking local packages")
    obj = get_resource_from_local_package(canonical_url, app_state.conf)
    if obj:
        return utils.json_to_obj(obj, obj.get("resourceType"))

    # Try resolving the URL/use the given resolver
    logger.info("Not found in local packages...Try to resolve the URL")
    if resolver is None:
        for r in url_resolve_adapter.Resolvers:
            resolver = r.value(app_state.conf)
            obj = resolver.query(url=canonical_url)
            if obj:
                break
    else:
        obj = resolver.query(canonical_url)
    if obj is None:
        logger.warning("WARNING - No resource found!")
        return None
    # Cache the raw dict, but return a fhir object so all paths share one contract.
    app_state.cache.add_resource_to_cache(obj)
    return utils.json_to_obj(obj, obj.get("resourceType"))

def resolve_add_to_registry(
    canonical_url: str,
    app_state: AppState,
    resolver: url_resolve_adapter.URLResolveAdapterBase = None,
):
    """
    Try to resolve a resource, then add it to the registry:
    WARNIng: DO NOT USE for non-profile resources...
    """
    obj = app_state.registry.get_obj_by_name(canonical_url)
    if not obj:
        resolved = resolve_url(canonical_url, app_state, resolver)
        if resolved is None:
            return None
        # resolve_url returns a fhir object (or a dict fallback) — handle both.
        res_type = (
            resolved.get("resourceType")
            if isinstance(resolved, dict)
            else type(resolved).__name__
        )
        obj = app_state.registry.add_fhir_object(resolved, res_type)
    return obj

def get_resource_from_local_package(canonical_url: str, conf: dict):
    """Get resources from a locally installed simplifier packages"""
    resource_path = conf.get("resource_cache_path")
    logger.info(f"Trying to retrieve url from local resource cache {resource_path}...")
    # look for `.index.json` files (simplifier packages)
    index_files = find_simplifier_index_files(resource_path)
    # logger.info(f"Found index files in: {index_files}")
    found_resource = None
    # TODO: alternative to provided index files only -> use add script to generate index manually
    for idx_file in index_files:
        res = lookup_in_index(idx_file, canonical_url)
        if res:
            fr_fname = os.path.dirname(idx_file) + "/" + res["filename"]
            found_resource = utils.get_json(fr_fname)
            logger.info(f"Found resource in local registry: {fr_fname}")
            break
    return found_resource

def find_simplifier_index_files(dir_name):
    """Look for index json files containing the local simplifier canonical urls and filenames"""
    files = []
    for root, _, fnames in os.walk(dir_name):
        for fname in fnmatch.filter(fnames, ".index.json"):
            files.append(os.path.join(root, fname))
    return files

def lookup_in_index(index_file, url):
    """Lookup a canonical url in an index file and return the information"""
    base_url = url.split("|")[0]
    with open(index_file, encoding="UTF-8") as f:
        contents = json.load(f)
    # .index.json entries are not guaranteed to carry a "url"!
    return next(
        (x for x in contents.get("files", []) if x.get("url") == base_url), None
    )
