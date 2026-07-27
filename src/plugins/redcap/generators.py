"""
ConceptMap generation from REDCap codebook choices.
"""

import json
from typing import Optional
import logging
logger = logging.getLogger(__name__)


def _resolve_explicit_mapping(
    choices: list[dict],
    explicit_mapping: dict[str, str],
    field_name: str,
) -> list[dict]:
    """
    Build ConceptMap elements from an explicit mapping table, matching each
    REDCap choice by CODE first, then by LABEL if no code match is found.

    This supports both KDS-style label→FHIR definitions
      e.g.  {"Melanom": "363346000", "Leukämie": "87163000"}
    and numeric-code definitions
      e.g.  {"1": "Y", "2": "N", "3": "U"}.

    The REDCap code (not the label) is always used as the ConceptMap source
    code, since that is what appears in the exported data.
    """
    elements = []
    # Build a normalised label → fhir_code index for label matching
    label_index = {k.strip().lower(): v for k, v in explicit_mapping.items()}

    for choice in choices:
        redcap_code = choice["code"]
        redcap_label = choice["label"].strip()

        # Try exact code match first
        fhir_code = explicit_mapping.get(redcap_code)
        if fhir_code is None:
            # Fall back to label match (case-insensitive)
            fhir_code = label_index.get(redcap_label.lower())
        if fhir_code is None:
            logger.debug(
                "Field '%s': no FHIR code mapping found for code='%s' label='%s' — "
                "falling back to label as target code.",
                field_name,
                redcap_code,
                redcap_label,
            )
            fhir_code = redcap_label  # fallback: use label text

        elements.append(_make_element(redcap_code, fhir_code, redcap_label))

    return elements


def _all_codes_non_numeric(choices: list[dict]) -> bool:
    """True if every code in the choice list is non-numeric (letters/words)."""
    return all(not c["code"].lstrip("-").isdigit() for c in choices)


def _cm_url(base_url: str, field_name: str) -> str:
    return f"{base_url}/ConceptMap/redcap-{field_name}"


def _make_element(source_code: str, target_code: str, target_display: str = "") -> dict:
    elem: dict = {
        "code": source_code,
        "target": [{"code": target_code, "equivalence": "equivalent"}],
    }
    if target_display:
        elem["target"][0]["display"] = target_display
    return elem


def build_concept_map(
    field_name: str,
    choices: list[dict],
    base_url: str,
    fhir_annotation: Optional[dict] = None,
    existing_cm: Optional[dict] = None,
    explicit_mapping: Optional[dict] = None,
) -> Optional[dict]:
    """
    Build a FHIR ConceptMap for the given field's choices.

    :param field_name: REDCap field name.
    :param choices: Parsed list of {code, label} dicts from the codebook.
    :param base_url: The project's base URL (e.g. http://example.org).
    :param fhir_annotation: Parsed @FHIR-MAPPING annotation, if any.
    :param existing_cm: Existing ConceptMap from the static generator. New
                        entries are merged into its first group.
    :param explicit_mapping: {redcap_code: fhir_code} dict from the user's
                             FHIR code mapping table. Takes priority over all
                             other strategies when present.
    :returns: FHIR ConceptMap dict, or None if no CM is needed.
    """
    if not choices:
        return None

    cm_url = _cm_url(base_url, field_name)

    # The mapping may key on REDCap codes OR REDCap labels — resolve both.
    if explicit_mapping:
        logger.debug("Field '%s': using explicit FHIR code mapping.", field_name)
        elements = _resolve_explicit_mapping(choices, explicit_mapping, field_name)
        source_system = "SourceData"
        target_system = (
            fhir_annotation.get("primaryElementSystem", "SourceData")
            if fhir_annotation
            else "SourceData"
        )
        return _assemble_cm(
            cm_url, field_name, source_system, target_system, elements, existing_cm
        )

    if _all_codes_non_numeric(choices):
        logger.debug("Field '%s': non-numeric codes → identity ConceptMap.", field_name)
        target_system = (
            fhir_annotation.get("primaryElementSystem", "SourceData")
            if fhir_annotation
            else "SourceData"
        )
        elements = [_make_element(c["code"], c["code"], c["label"]) for c in choices]
        return _assemble_cm(
            cm_url, field_name, target_system, target_system, elements, existing_cm
        )

    # Adds human-readable labels; FHIR code alignment requires explicit mapping.
    logger.debug(
        "Field '%s': numeric codes without explicit mapping → label-based ConceptMap. "
        "Provide a FHIR code mapping table for full FHIR alignment.",
        field_name,
    )
    elements = [_make_element(c["code"], c["label"]) for c in choices]
    return _assemble_cm(
        cm_url, field_name, "SourceData", "SourceData", elements, existing_cm
    )


def _assemble_cm(
    cm_url: str,
    field_name: str,
    source_system: str,
    target_system: str,
    elements: list[dict],
    existing_cm: Optional[dict],
) -> dict:
    """Assemble the final ConceptMap, merging into existing_cm if provided."""
    if existing_cm:
        cm = json.loads(json.dumps(existing_cm))  # deep copy
        groups = cm.setdefault("group", [])

        if target_system in ("SourceData", "") and groups:
            for g in groups:
                if g.get("target") and g["target"] != "SourceData":
                    target_system = g["target"]
                    break

        merge_group = next(
            (g for g in groups if g.get("source") == source_system), None
        )
        if merge_group is None:
            merge_group = {
                "source": source_system,
                "target": target_system,
                "element": [],
            }
            groups.append(merge_group)

        existing_codes = {e["code"] for e in merge_group.get("element", [])}
        new_elements = [e for e in elements if e["code"] not in existing_codes]
        merge_group["element"] = merge_group.get("element", []) + new_elements
        logger.info(
            "Merged %d REDCap entries into group (source=%s) for '%s'.",
            len(new_elements),
            source_system,
            field_name,
        )
        return cm

    return {
        "resourceType": "ConceptMap",
        "id": f"redcap-{field_name}",
        "url": cm_url,
        "name": f"REDCap_{field_name}",
        "title": f"REDCap code mapping for {field_name}",
        "status": "draft",
        "description": f"Auto-generated from REDCap data dictionary for field '{field_name}'.",
        "sourceUri": "SourceData",
        "group": [
            {
                "source": source_system,
                "target": target_system,
                "element": elements,
            }
        ],
    }
