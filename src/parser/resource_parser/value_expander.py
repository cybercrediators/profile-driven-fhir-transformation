from controller.connector.terminology_connector import TerminologyConnector
import logging
logger = logging.getLogger(__name__)
from data_handling.url_resolver.fhir_url_resolver import resolve_url

from fhir.resources.R4B.codesystem import CodeSystem
from fhir.resources.R4B.valueset import ValueSet


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
                    for concept in cs.concept:
                        tmp_options.append(
                            {
                                "code": concept.code,
                                "display": concept.display,
                                "system": cs.url,
                            }
                        )
                else:
                    tmp_options.append(
                        {
                            "code": "FROM_CS",
                            "display": f"from system: {include.system}",
                            "system": include.system,
                        }
                    )
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
                    options.append(
                        {
                            "code": concept.code,
                            "display": concept.display,
                            "system": include.system,
                        }
                    )
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
                    options.append(
                        {
                            "code": f"FILTER: op='{f.op}', value='{f.value}'",
                            "display": "Code must be selected from this filter (terminology server)",
                            "system": include.system,
                        }
                    )
                app_state.registry.add_to_used_by(include.system, vs_url)
        if tmp_vs:
            tmp_vs.expanded_values = options
    # print(options)
    processed_vs_urls.pop()
    return options
