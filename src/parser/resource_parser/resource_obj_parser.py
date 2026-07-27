from data_handling.app_state import AppState
from parser.resource_parser import questionnaire_expander

from parser.resource_parser import structure_definition_parser


def parse_resource(resource, app_state: AppState):
    """
    - Read standard resource (no StructureDefinition!)
    - Parse resource contents (fhir.resources model)
    - Identify mappable fields
    - Return usable resource object
    - Set processed and used_in
    """
    if resource.res_type == "StructureDefinition":
        return structure_definition_parser.parse_structure_definition(
            resource, app_state
        )
    if resource.res_type == "Questionnaire":
        return questionnaire_expander.parse_questionnaire(resource, app_state)
    # TODO: enable standard resources too?
