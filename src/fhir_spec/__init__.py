"""Release-aware FHIR specification metadata used by the generator."""

from fhir_spec.context import (
    FHIRSpecContext,
    StructureMapCapabilities,
    activate_fhir_spec_context,
    clear_fhir_spec_context,
    ensure_fhir_spec_context,
    get_fhir_spec_context,
)

__all__ = [
    "FHIRSpecContext",
    "StructureMapCapabilities",
    "activate_fhir_spec_context",
    "clear_fhir_spec_context",
    "ensure_fhir_spec_context",
    "get_fhir_spec_context",
]
