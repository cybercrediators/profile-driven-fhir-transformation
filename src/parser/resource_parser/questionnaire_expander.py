import logging
logger = logging.getLogger(__name__)
from data_handling.app_state import AppState
from parser.resource_parser.value_expander import expand_valueset


def parse_questionnaire(questionnaire, app_state: AppState):
    """
    wrapper to pre-process a Questionnaire for mapping (map generator walks object, information already retrieved at processing)
    """
    if questionnaire is None or questionnaire.is_processed():
        logger.info("Questionnaire is empty or already processed!")
        return questionnaire, app_state

    if not questionnaire.data.item:
        logger.error("Questionnaire %s does not have any items!", questionnaire.data.url)
        return questionnaire, app_state

    questionnaire.set_processed()
    _prewarm_item_value_sets(questionnaire.data.item, app_state, questionnaire.data.url)
    return questionnaire, app_state


def _prewarm_item_value_sets(items, app_state, questionnaire_url):
    """recursively expand (and register the dependency for) every item's answerValueSet"""
    for item in items:
        if item.answerValueSet:
            expand_valueset(item.answerValueSet, [], app_state)
            app_state.registry.add_to_used_by(item.answerValueSet, questionnaire_url)
        if item.item:
            _prewarm_item_value_sets(item.item, app_state, questionnaire_url)
