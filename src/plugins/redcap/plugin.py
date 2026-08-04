"""
REDCap pipeline plugin.

Reads a REDCap data dictionary (via API, CSV, or JSON) and:
  1. Enriches the source StructureDefinition with field labels and data-type hints.
  2. Adds @FHIR-MAPPING-derived entries to the automapping table.
  3. Generates or supplements ConceptMaps for coded fields.

Configuration block example (project JSON):

  "plugins": [
    {
      "type": "redcap",

      // Source: one of "api", "csv", "json"
      "source": "api",

      // API source — URL and token can be literal strings or ${ENV_VAR} references
      "api_url": "${REDCAP_API_URL}",
      "api_token": "${REDCAP_API_TOKEN}",
      "api_verify_ssl": true,     // optional, default true

      // CSV or JSON source
      "codebook_path": "source_data/DataDictionary.csv",  // relative to CWD

      // Optional: explicit REDCap code → FHIR code mapping table.
      // This alignment is defined when the profile is created (e.g. from the
      // Kerndatensatz) and cannot be derived automatically from the codebook.
      // See codebook.py for supported formats (JSON or CSV).
      "fhir_code_mapping": "source_data/fhir_code_mapping.json",

      // Optional: base URL for generated ConceptMap URIs.
      "base_url": "http://example.org"
    }
  ]
"""

from typing import Optional
import logging
import re
logger = logging.getLogger(__name__)
from plugins.base import PipelinePlugin
from plugins.redcap.codebook import REDCapCodebook, load_fhir_code_mapping
from plugins.redcap.generators import build_concept_map


class REDCapPlugin(PipelinePlugin):
    """Pipeline plugin that translates REDCap codebook knowledge into FHIR constructs."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._codebook: Optional[REDCapCodebook] = None
        self._base_url: str = config.get("base_url", "http://example.org")
        # {field_name: {redcap_code: fhir_code}} — loaded lazily
        self._fhir_code_mapping: Optional[dict] = None

    @property
    def plugin_id(self) -> str:
        return "redcap"

    def _load_fhir_code_mapping(self) -> dict:
        if self._fhir_code_mapping is not None:
            return self._fhir_code_mapping
        path = self.config.get("fhir_code_mapping", "")
        if path:
            try:
                self._fhir_code_mapping = load_fhir_code_mapping(path)
            except Exception as e:
                logger.warning(
                    "REDCap plugin: could not load fhir_code_mapping from '%s': %s",
                    path,
                    e,
                )
                self._fhir_code_mapping = {}
        else:
            self._fhir_code_mapping = {}
        return self._fhir_code_mapping

    def _load_codebook(self) -> REDCapCodebook:
        if self._codebook is not None:
            return self._codebook

        source = self.config.get("source", "").lower()

        if source == "api":
            api_url = self.config.get("api_url", "")
            api_token = self.config.get("api_token", "")
            verify_ssl = self.config.get("api_verify_ssl", True)
            if not api_url or not api_token:
                raise ValueError(
                    "REDCap plugin (source=api) requires 'api_url' and 'api_token'."
                )
            self._codebook = REDCapCodebook.from_api(api_url, api_token, verify_ssl)

        elif source in ("csv", "json"):
            path = self.config.get("codebook_path", "")
            if not path:
                raise ValueError(
                    f"REDCap plugin (source={source}) requires 'codebook_path'."
                )
            if source == "csv":
                self._codebook = REDCapCodebook.from_csv(path)
            else:
                self._codebook = REDCapCodebook.from_json(path)

        else:
            raise ValueError(
                f"REDCap plugin: unknown source '{source}'. "
                "Use 'api', 'csv', or 'json'."
            )

        logger.info(
            "REDCap plugin: loaded codebook with %d field definitions.",
            len(self._codebook),
        )
        return self._codebook

    # REDCap text_validation_type to FHIR type code
    _VALIDATION_TO_FHIR_TYPE = {
        "integer": "integer",
        "number": "decimal",
        "float": "decimal",
        "date_dmy": "date",
        "date_mdy": "date",
        "date_ymd": "date",
        "datetime_dmy": "dateTime",
        "datetime_mdy": "dateTime",
        "datetime_ymd": "dateTime",
        "datetime_seconds_dmy": "dateTime",
        "datetime_seconds_mdy": "dateTime",
        "datetime_seconds_ymd": "dateTime",
        "time": "time",
        "time_mm_ss": "time",
        "email": "string",
        "phone": "string",
        "phone_australia": "string",
        "postalcode_german": "string",
        "zipcode": "string",
        "mrn_10d": "string",
        "mrn_generic": "string",
        "alpha_only": "string",
        "vmrn": "string",
    }

    def post_source_def(
        self,
        source_definition: dict,
        source_fields: list[dict],
    ) -> dict:
        """
        Enrich source field metadata from the REDCap codebook:
        - Set element descriptions from field_label.
        - Override element types from text_validation_type so the factory can
          narrow value[x] choice types (e.g. rauchen_tag → integer not string).
        """
        cb = self._load_codebook()
        field_names = {f.get("id", "").split(".")[-1] for f in source_fields}

        enriched_desc = 0
        enriched_type = 0
        enriched_units = 0
        snapshot = source_definition.get("snapshot", {})

        for elem in snapshot.get("element", []):
            field_name = elem.get("id", "").split(".")[-1]
            if field_name not in field_names:
                continue
            meta = cb.get_field(field_name)
            if not meta:
                continue

            # Enrich description from field_label
            label = meta.get("field_label", "").strip()
            if label and not elem.get("short"):
                elem["short"] = label
                enriched_desc += 1

            # Override type from text_validation_type when available
            validation = meta.get(
                "text_validation_type_or_show_slider_number", ""
            ).strip()
            fhir_type = self._VALIDATION_TO_FHIR_TYPE.get(validation)
            if fhir_type and elem.get("type"):
                # Replace first type entry with the validated type
                current_type = elem["type"]
                if isinstance(current_type, list) and current_type:
                    if isinstance(current_type[0], dict):
                        if current_type[0].get("code") != fhir_type:
                            current_type[0]["code"] = fhir_type
                            enriched_type += 1
                    elif (
                        isinstance(current_type[0], str)
                        and current_type[0] != fhir_type
                    ):
                        elem["type"] = [{"code": fhir_type}]
                        enriched_type += 1

            if self._enrich_allowed_units(elem, meta):
                enriched_units += 1

        logger.info(
            "REDCap plugin (post_source_def): %d descriptions, %d types, "
            "%d units enriched.",
            enriched_desc,
            enriched_type,
            enriched_units,
        )
        return source_definition

    # REDCap has no unit column: a numeric field states its unit in the free-text
    # Field Note ("kg", "cm"). Reading that convention is REDCap-specific and so
    # belongs here, but what it produces is not — it is the standard FHIR
    # `elementdefinition-allowedUnits` extension, which any consumer of the source
    # definition can act on without knowing REDCap exists.
    _ALLOWED_UNITS_URL = (
        "http://hl7.org/fhir/StructureDefinition/elementdefinition-allowedUnits"
    )
    _UCUM_SYSTEM = "http://unitsofmeasure.org"
    _NUMERIC_FHIR_TYPES = {"integer", "decimal", "positiveInt", "unsignedInt"}
    # Deliberately strict. A Field Note is free text and usually a sentence; only a
    # bare token can be a unit, so anything longer is left alone rather than pinned
    # as a unit that would then be emitted into every derived Quantity.
    _UNIT_TOKEN = re.compile(r"^[A-Za-z%\[\]/*.0-9^{}-]{1,12}$")

    def _enrich_allowed_units(self, elem: dict, meta: dict) -> bool:
        note = (meta.get("field_note") or "").strip()
        if not note or not self._UNIT_TOKEN.match(note):
            return False
        codes = {
            t.get("code")
            for t in (elem.get("type") or [])
            if isinstance(t, dict)
        }
        if not codes & self._NUMERIC_FHIR_TYPES:
            return False
        extensions = elem.setdefault("extension", [])
        if any(
            isinstance(ext, dict) and ext.get("url") == self._ALLOWED_UNITS_URL
            for ext in extensions
        ):
            return False
        extensions.append(
            {
                "url": self._ALLOWED_UNITS_URL,
                "valueCodeableConcept": {
                    "coding": [{"system": self._UCUM_SYSTEM, "code": note}]
                },
            }
        )
        return True

    def enrich_automapping(self, automapping: dict[str, str]) -> dict[str, str]:
        """
        Add mappings from @FHIR-MAPPING annotations that are not already
        present in the automapping table.

        Annotation format supported:
          Simple:  @FHIR-MAPPING='Patient/birthDate'
          Complex: @FHIR-MAPPING='{ "type": "Patient", "primaryElementPath": "..." }'
        """
        cb = self._load_codebook()
        added = 0

        # Build reverse index: field_name → source_id as used in the automapping
        existing_source_ids = set(automapping.keys())
        # Derive the source definition type prefix from the existing keys
        prefix = ""
        for k in existing_source_ids:
            if "." in k:
                prefix = k.rsplit(".", 1)[
                    0
                ]  # e.g. "Sourcedefinition_testing_kfdm_profile"
                break

        for field in cb.all_fields():
            ann = field.get("_fhir_annotation")
            if not ann:
                continue

            field_name = field.get("field_name", "")
            source_id = f"{prefix}.{field_name}" if prefix else field_name

            if source_id in existing_source_ids:
                continue  # Already mapped, respect existing automapping

            # Build the FHIR target path from the annotation
            fhir_type = ann.get("type", "")
            path = ann.get("primaryElementPath", "").replace("/", ".")
            if not fhir_type or not path:
                continue

            target_path = f"{fhir_type}.{path}"
            automapping[source_id] = target_path
            added += 1
            logger.debug(
                "REDCap plugin: added automapping %s → %s (from @FHIR-MAPPING)",
                source_id,
                target_path,
            )

        logger.info(
            "REDCap plugin (enrich_automapping): added %d entries from annotations.",
            added,
        )
        return automapping

    _FORM_STATUS_CHOICES = [
        {"code": "0", "label": "Incomplete"},
        {"code": "1", "label": "Unverified"},
        {"code": "2", "label": "Complete"},
    ]
    _FORM_STATUS_FHIR_DEFAULT = {
        "Complete": "completed",
        "Unverified": "in-progress",
        "Incomplete": "in-progress",
    }
    _QR_STATUS_SYSTEM = "http://hl7.org/fhir/questionnaire-answers-status"

    def get_field_metadata(self, field_name: str) -> Optional[dict]:
        """Return the codebook entry for a field, or None.

        ``<form>_complete`` status fields are synthesized — they carry fixed REDCap choices
        and are not present in the Data Dictionary metadata.
        """
        if field_name.endswith("_complete"):
            return {
                "field_name": field_name,
                "field_label": "Form completion status",
                "_choices": list(self._FORM_STATUS_CHOICES),
                "_fhir_annotation": {"primaryElementSystem": self._QR_STATUS_SYSTEM},
                "_form_status": True,
            }
        cb = self._load_codebook()
        return cb.get_field(field_name)

    def generate_concept_map(
        self,
        field_name: str,
        field_meta: dict,
        existing_cm: Optional[dict],
    ) -> Optional[dict]:
        """
        Generate or supplement a ConceptMap for a coded field.

        Called by the pipeline for every field that the static generator
        would create a ConceptMap for, and also for any coded field the
        plugin detects via the codebook.

        :param field_name: Source field name (last segment of the field id).
        :param field_meta: Codebook entry for this field (from get_field_metadata).
        :param existing_cm: Existing CM from static generator, if any.
        :returns: A FHIR ConceptMap dict, or None to leave existing CM unchanged.
        """
        choices = field_meta.get("_choices", [])
        if not choices:
            return None

        fhir_annotation = field_meta.get("_fhir_annotation")
        # Look up the explicit FHIR code mapping for this field (if provided)
        explicit_mapping = self._load_fhir_code_mapping().get(field_name)
        if field_meta.get("_form_status") and not explicit_mapping:
            explicit_mapping = dict(self._FORM_STATUS_FHIR_DEFAULT)
        cm = build_concept_map(
            field_name=field_name,
            choices=choices,
            base_url=self._base_url,
            fhir_annotation=fhir_annotation,
            existing_cm=existing_cm,
            explicit_mapping=explicit_mapping,
        )
        if cm:
            logger.info(
                "REDCap plugin: generated ConceptMap for '%s' (%d entries).",
                field_name,
                len(cm.get("group", [{}])[0].get("element", [])),
            )
        return cm
