from controller.connector.terminology_connector import TerminologyConnector
import logging
logger = logging.getLogger(__name__)
from data_handling.url_resolver.fhir_url_resolver import resolve_url

from fhir.resources.R4B.codesystem import CodeSystem
from fhir.resources.R4B.valueset import ValueSet


def _concept_options(concepts, system, version=None):
    """Flatten a CodeSystem concept hierarchy while retaining system/version."""
    options = []
    for concept in concepts or []:
        option = {
            "code": concept.code,
            "display": concept.display,
            "system": system,
        }
        if version:
            option["version"] = version
        options.append(option)
        options.extend(
            _concept_options(
                getattr(concept, "concept", None), system, version=version
            )
        )
    return options


def _option_key(option):
    return (
        option.get("system"),
        option.get("version"),
        option.get("code"),
    )


def _dedupe_options(options):
    result = []
    seen = set()
    for option in options:
        key = _option_key(option)
        if key in seen:
            continue
        seen.add(key)
        result.append(option)
    return result


def expand_valueset(vs_url, processed_vs_urls, app_state):
    logger.info(f"Processing {vs_url}")
    if tmp_vs := app_state.registry.get_obj_by_name(vs_url):
        # TODO: check if already processed and return result + set used_by accordingly
        tmp_vs.processed = 1
        if tmp_vs.expanded_values:
            return tmp_vs.expanded_values

    if vs_url in processed_vs_urls:
        return []

    processed_vs_urls.append(vs_url)
    # print(processed_vs_urls)

    # vs = resolve_add_to_registry(vs_url, app_state)
    vs = resolve_url(vs_url, app_state)

    if not isinstance(vs, ValueSet) or not vs.compose:
        logger.info("Instance is no valueset!")
        processed_vs_urls.pop()
        return []

    # Expansion via terminology server
    try:
        term_server = TerminologyConnector(app_state)
        response = term_server.query_terminology_server(vs_url)
        if response is not None:
            vs_values = term_server.extract_values(response)
            if tmp_vs is not None:
                tmp_vs.expanded_values = vs_values
            processed_vs_urls.pop()
            return vs_values
        else:
            logger.info(
                f"No expansion via external terminology server possible for {vs_url}, continuing manually..."
            )

    except Exception as e:
        logger.warning(
            f"Error while expanding {vs_url} via server ({type(e).__name__}: {e}), trying manually..."
        )

    options = []

    if vs.compose.include:
        for include in vs.compose.include:
            if include.system and not include.concept and not include.filter:
                # cs = resolve_add_to_registry(include.system, app_state)
                cs = resolve_url(include.system, app_state)
                tmp_options = []
                if (
                    isinstance(cs, CodeSystem)
                    and cs.content == "complete"
                    and cs.concept
                ):
                    tmp_options.extend(
                        _concept_options(
                            cs.concept,
                            cs.url,
                            version=include.version or cs.version,
                        )
                    )
                else:
                    placeholder = {
                        "code": "FROM_CS",
                        "display": f"from system: {include.system}",
                        "system": include.system,
                    }
                    if include.version:
                        placeholder["version"] = include.version
                    tmp_options.append(placeholder)
                if cs:
                    cs_url = cs.url if isinstance(cs, CodeSystem) else cs.get("url")
                    if cs_url and (
                        tmp_cs := app_state.registry.get_obj_by_name(cs_url)
                    ):
                        tmp_cs.processed = 1
                app_state.registry.add_to_used_by(include.system, vs_url)
                options.extend(tmp_options)
            if include.concept:
                logger.info("ValueSet includes a Concept")
                for concept in include.concept:
                    option = {
                        "code": concept.code,
                        "display": concept.display,
                        "system": include.system,
                    }
                    if include.version:
                        option["version"] = include.version
                    options.append(option)
                app_state.registry.add_to_used_by(include.system, vs_url)
            if include.valueSet:
                logger.info(f"ValueSet {vs_url} includes another ValueSet(s)")
                print(include.valueSet)
                for incl_vs_url in include.valueSet:
                    app_state.registry.add_to_used_by(incl_vs_url, vs_url)

                    options.extend(
                        expand_valueset(incl_vs_url, processed_vs_urls, app_state)
                    )
            if include.filter:
                logger.info(f"ValueSet {vs_url} includes a filter")
                for f in include.filter:
                    marker = {
                        "code": (
                            f"FILTER: property='{f.property}', "
                            f"op='{f.op}', value='{f.value}'"
                        ),
                        "display": (
                            "Code must be selected from this filter "
                            "(terminology server)"
                        ),
                        "system": include.system,
                    }
                    if include.version:
                        marker["version"] = include.version
                    options.append(marker)
                app_state.registry.add_to_used_by(include.system, vs_url)

    excluded_keys = set()
    excluded_systems = set()
    for exclude in (vs.compose.exclude or []):
        if exclude.system and not exclude.concept and not exclude.filter and not exclude.valueSet:
            excluded_systems.add((exclude.system, exclude.version))
        for concept in exclude.concept or []:
            excluded_keys.add((exclude.system, exclude.version, concept.code))
        for excluded_vs_url in exclude.valueSet or []:
            app_state.registry.add_to_used_by(excluded_vs_url, vs_url)
            for option in expand_valueset(
                excluded_vs_url, processed_vs_urls, app_state
            ):
                excluded_keys.add(_option_key(option))
        for f in exclude.filter or []:
            logger.warning(
                "ValueSet %s has an exclusion filter (%s %s %s) that requires "
                "terminology-server evaluation; local compose expansion leaves "
                "the affected codes unresolved.",
                vs_url,
                f.property,
                f.op,
                f.value,
            )

    def _excluded(option):
        key = _option_key(option)
        unversioned_key = (option.get("system"), None, option.get("code"))
        if key in excluded_keys or unversioned_key in excluded_keys:
            return True
        return any(
            option.get("system") == system
            and (version is None or option.get("version") == version)
            for system, version in excluded_systems
        )

    options = _dedupe_options([option for option in options if not _excluded(option)])
    if tmp_vs:
        tmp_vs.expanded_values = options
    # print(options)
    processed_vs_urls.pop()
    return options
