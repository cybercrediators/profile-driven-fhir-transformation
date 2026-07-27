"""
REDCap codebook reader + FHIR code mapping table loader.

Codebook input sources (three options):
  - 'api'  : Live fetch via REDCap API (content=metadata)
  - 'csv'  : REDCap Data Dictionary export (CSV)
  - 'json' : REDCap API metadata export saved as JSON

FHIR code mapping table (optional, separate file):
  A user-provided mapping that bridges REDCap codes to FHIR codes.
  This alignment is defined when the profile is created (e.g. from the
  Kerndatensatz) and cannot be derived generically from the codebook alone.

  Supported formats:
  - JSON: {"field_name": {"redcap_code": "fhir_code", ...}, ...}
  - CSV:  field_name, redcap_code, fhir_code  (one row per code)

  Example JSON:
    {
      "aktbefinden": {"1": "SG", "2": "G", "3": "B", "4": "A", "5": "M"},
      "diabetes":    {"1": "Y",  "2": "N", "3": "U"}
    }

  Example CSV:
    field_name,redcap_code,fhir_code
    aktbefinden,1,SG
    aktbefinden,2,G
    diabetes,1,Y
    diabetes,2,N

Output (codebook): a dict keyed by field_name, each value a dict with the raw REDCap
field metadata (field_type, select_choices_or_calculations, field_annotation,
text_validation_type_or_show_slider_number, field_label, section_header, etc.)
"""

import csv
import json
from typing import Optional
import logging
logger = logging.getLogger(__name__)

def parse_choices(choices_str: str) -> list[dict]:
    """
    Parse a REDCap choices string into a list of {code, label} dicts.

    REDCap format: "code1, label1 | code2, label2 | ..."
    Calculated fields (contain '[') and sliders with only boundary labels
    are returned as an empty list.
    """
    if not choices_str or "[" in choices_str:
        return []

    result = []
    for part in choices_str.split("|"):
        part = part.strip()
        if not part:
            continue
        idx = part.find(",")
        if idx == -1:
            continue
        code = part[:idx].strip()
        label = part[idx + 1 :].strip()
        if code:
            result.append({"code": code, "label": label})
    return result


def parse_fhir_annotation(annotation: str) -> Optional[dict]:
    """
    Parse a REDCap @FHIR-MAPPING annotation.

    Two formats are supported:
      Simple:  @FHIR-MAPPING='ResourceType/elementPath'
      Complex: @FHIR-MAPPING='{ "type": "...", "primaryElementPath": "...", ... }'
    """
    if not annotation or "@FHIR-MAPPING=" not in annotation:
        return None
    try:
        start = annotation.index("@FHIR-MAPPING=") + len("@FHIR-MAPPING=")
        raw = annotation[start:].strip().strip("'\"")
        if raw.startswith("{"):
            return json.loads(raw)
        # Simple path form, e.g. "Patient/birthDate"
        parts = raw.split("/", 1)
        return {
            "type": parts[0],
            "primaryElementPath": parts[1] if len(parts) > 1 else "",
        }
    except Exception as e:
        logger.debug("Annotation is not a parseable FHIR mapping (%s: %s)", type(e).__name__, e)
        return None


def _from_api(api_url: str, api_token: str, verify_ssl: bool = True) -> list[dict]:
    """Fetch metadata directly from the REDCap API."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("'requests' package is required for REDCap API access.")

    logger.info("Fetching REDCap metadata from API: %s", api_url)
    resp = requests.post(
        api_url,
        data={
            "token": api_token,
            "content": "metadata",
            "format": "json",
            "returnFormat": "json",
        },
        timeout=30,
        verify=verify_ssl,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"REDCap API error: {data['error']}")
    logger.info("Fetched %d field definitions from REDCap API.", len(data))
    return data


def _from_json(path: str) -> list[dict]:
    """Load metadata from a previously saved REDCap API JSON export."""
    logger.info("Loading REDCap metadata from JSON: %s", path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d field definitions from JSON.", len(data))
    return data

def _from_csv(path: str) -> list[dict]:
    """Load metadata from a REDCap Data Dictionary CSV export."""
    logger.info("Loading REDCap metadata from CSV: %s", path)

    # REDCap CSV column names (standard export headers)
    FIELD_MAP = {
        "Variable / Field Name": "field_name",
        "Form Name": "form_name",
        "Section Header": "section_header",
        "Field Type": "field_type",
        "Field Label": "field_label",
        "Choices, Calculations, OR Slider Labels": "select_choices_or_calculations",
        "Field Note": "field_note",
        "Text Validation Type OR Show Slider Number": "text_validation_type_or_show_slider_number",
        "Text Validation Min": "text_validation_min",
        "Text Validation Max": "text_validation_max",
        "Identifier?": "identifier",
        "Branching Logic (Show field only if...)": "branching_logic",
        "Required Field?": "required_field",
        "Custom Alignment": "custom_alignment",
        "Question Number (surveys only)": "question_number",
        "Matrix Group Name": "matrix_group_name",
        "Matrix Ranking?": "matrix_ranking",
        "Field Annotation": "field_annotation",
    }

    records = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            for csv_col, field_key in FIELD_MAP.items():
                val = row.get(csv_col, "").strip()
                if val:
                    record[field_key] = val
            if record.get("field_name"):
                records.append(record)
    logger.info("Loaded %d field definitions from CSV.", len(records))
    return records

class REDCapCodebook:
    """
    Parsed REDCap data dictionary.

    Provides per-field access to choices (as parsed code/label pairs),
    FHIR annotations, and raw metadata.
    """

    def __init__(self, fields: list[dict]):
        self._fields: dict[str, dict] = {}
        for f in fields:
            name = f.get("field_name", "")
            if name:
                # Attach parsed choices and FHIR annotation for convenience
                f["_choices"] = parse_choices(
                    f.get("select_choices_or_calculations", "")
                )
                f["_fhir_annotation"] = parse_fhir_annotation(
                    f.get("field_annotation", "")
                )
                self._fields[name] = f

    @classmethod
    def from_api(
        cls, api_url: str, api_token: str, verify_ssl: bool = True
    ) -> "REDCapCodebook":
        return cls(_from_api(api_url, api_token, verify_ssl))

    @classmethod
    def from_json(cls, path: str) -> "REDCapCodebook":
        return cls(_from_json(path))

    @classmethod
    def from_csv(cls, path: str) -> "REDCapCodebook":
        return cls(_from_csv(path))

    def get_field(self, field_name: str) -> Optional[dict]:
        return self._fields.get(field_name)

    def all_fields(self) -> list[dict]:
        return list(self._fields.values())

    def fields_with_choices(self) -> list[dict]:
        return [f for f in self._fields.values() if f.get("_choices")]

    def __len__(self) -> int:
        return len(self._fields)

def load_fhir_code_mapping(path: str) -> dict[str, dict[str, str]]:
    """
    Load a FHIR code mapping table from a JSON or CSV file.

    Returns a dict of {field_name: {redcap_code: fhir_code}}.

    JSON format:
      {"aktbefinden": {"1": "SG", "2": "G", ...}, ...}

    CSV format (columns: field_name, redcap_code, fhir_code):
      aktbefinden,1,SG
      aktbefinden,2,G
      diabetes,1,Y
    """
    if not path:
        return {}

    ext = path.rsplit(".", 1)[-1].lower()
    result: dict[str, dict[str, str]] = {}

    if ext == "json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"FHIR code mapping JSON must be an object, got {type(data)}"
            )
        for field_name, codes in data.items():
            if isinstance(codes, dict):
                result[field_name] = {str(k): str(v) for k, v in codes.items()}
            else:
                logger.warning(
                    "Skipping field '%s': expected dict, got %s",
                    field_name,
                    type(codes),
                )

    elif ext == "csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            # Accept headers: field_name, redcap_code, fhir_code (case-insensitive)
            headers = {h.lower().strip(): h for h in (reader.fieldnames or [])}
            fn_col = headers.get("field_name") or headers.get("field name")
            rc_col = headers.get("redcap_code") or headers.get("redcap code")
            fc_col = headers.get("fhir_code") or headers.get("fhir code")
            if not (fn_col and rc_col and fc_col):
                raise ValueError(
                    f"FHIR code mapping CSV must have columns: field_name, redcap_code, fhir_code. "
                    f"Found: {list(reader.fieldnames or [])}"
                )
            for row in reader:
                fname = row.get(fn_col, "").strip()
                rcode = row.get(rc_col, "").strip()
                fcode = row.get(fc_col, "").strip()
                if fname and rcode and fcode:
                    result.setdefault(fname, {})[rcode] = fcode
    else:
        raise ValueError(
            f"Unsupported FHIR code mapping format: '{ext}'. Use 'json' or 'csv'."
        )

    total = sum(len(v) for v in result.values())
    logger.info(
        "Loaded FHIR code mapping: %d fields, %d code entries from %s",
        len(result),
        total,
        path,
    )
    return result
