"""Unit tests for the REDCap codebook reader and FHIR code mapping loader."""

import json

import pytest

from plugins.redcap.codebook import (
    REDCapCodebook,
    load_fhir_code_mapping,
    parse_choices,
    parse_fhir_annotation,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# parse_choices
# --------------------------------------------------------------------------- #


def test_parse_choices_basic():
    choices = parse_choices("1, männlich | 2, weiblich | 3, divers")
    assert choices == [
        {"code": "1", "label": "männlich"},
        {"code": "2", "label": "weiblich"},
        {"code": "3", "label": "divers"},
    ]


def test_parse_choices_label_with_embedded_comma():
    # Only the FIRST comma splits code from label - rest of label is preserved.
    choices = parse_choices("1, Yes, definitely")
    assert choices == [{"code": "1", "label": "Yes, definitely"}]


def test_parse_choices_empty_string_returns_empty_list():
    assert parse_choices("") == []
    assert parse_choices(None) == []


def test_parse_choices_calculated_field_returns_empty_list():
    # calc fields contain '[' in their formula and must not be parsed as choices
    assert parse_choices("rounddown(datediff([geburtsdatum], 'today', 'y'))") == []


def test_parse_choices_skips_malformed_parts():
    # A part with no comma is skipped, blank parts between pipes are skipped
    choices = parse_choices("1, one | garbage | | 2, two")
    assert choices == [{"code": "1", "label": "one"}, {"code": "2", "label": "two"}]


def test_parse_choices_strips_whitespace():
    choices = parse_choices("  1 ,  one  |  2 ,  two  ")
    assert choices == [{"code": "1", "label": "one"}, {"code": "2", "label": "two"}]


# --------------------------------------------------------------------------- #
# parse_fhir_annotation
# --------------------------------------------------------------------------- #


def test_parse_fhir_annotation_simple_form():
    ann = parse_fhir_annotation(" @FHIR-MAPPING='Patient/birthDate'")
    assert ann == {"type": "Patient", "primaryElementPath": "birthDate"}


def test_parse_fhir_annotation_simple_form_no_path():
    ann = parse_fhir_annotation("@FHIR-MAPPING='Patient'")
    assert ann == {"type": "Patient", "primaryElementPath": ""}


def test_parse_fhir_annotation_complex_json_form():
    raw = '@FHIR-MAPPING=\'{"type": "Observation", "primaryElementPath": "value"}\''
    ann = parse_fhir_annotation(raw)
    assert ann == {"type": "Observation", "primaryElementPath": "value"}


def test_parse_fhir_annotation_no_marker_returns_none():
    assert parse_fhir_annotation("just a regular note") is None
    assert parse_fhir_annotation("") is None
    assert parse_fhir_annotation(None) is None


def test_parse_fhir_annotation_malformed_json_returns_none():
    raw = "@FHIR-MAPPING='{not valid json'"
    assert parse_fhir_annotation(raw) is None


# --------------------------------------------------------------------------- #
# REDCapCodebook — construction & accessors
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_fields():
    return [
        {
            "field_name": "geburtsdatum",
            "field_label": "Geburtsdatum",
            "field_type": "text",
            "text_validation_type_or_show_slider_number": "date_dmy",
            "field_annotation": " @FHIR-MAPPING='Patient/birthDate'",
        },
        {
            "field_name": "geschlecht",
            "field_label": "Geschlecht:",
            "field_type": "radio",
            "select_choices_or_calculations": (
                "male, männlich | female, weiblich | other, divers | unknown, keine Angabe"
            ),
            "field_annotation": " @FHIR-MAPPING='Patient/gender'",
        },
        {
            "field_name": "alter",
            "field_label": "Alter",
            "field_type": "calc",
            "select_choices_or_calculations": "rounddown(datediff([geburtsdatum], 'today', 'y'))",
        },
        # No field_name -> must be skipped entirely
        {"field_label": "orphan row"},
    ]


def test_codebook_len_and_all_fields(sample_fields):
    cb = REDCapCodebook(sample_fields)
    assert len(cb) == 3
    assert {f["field_name"] for f in cb.all_fields()} == {
        "geburtsdatum",
        "geschlecht",
        "alter",
    }


def test_codebook_get_field_attaches_parsed_choices_and_annotation(sample_fields):
    cb = REDCapCodebook(sample_fields)
    geschlecht = cb.get_field("geschlecht")
    assert geschlecht["_choices"] == [
        {"code": "male", "label": "männlich"},
        {"code": "female", "label": "weiblich"},
        {"code": "other", "label": "divers"},
        {"code": "unknown", "label": "keine Angabe"},
    ]
    assert geschlecht["_fhir_annotation"] == {
        "type": "Patient",
        "primaryElementPath": "gender",
    }


def test_codebook_get_field_missing_returns_none(sample_fields):
    cb = REDCapCodebook(sample_fields)
    assert cb.get_field("does_not_exist") is None


def test_codebook_fields_with_choices_excludes_calc_and_free_text(sample_fields):
    cb = REDCapCodebook(sample_fields)
    names = {f["field_name"] for f in cb.fields_with_choices()}
    assert names == {"geschlecht"}


def test_codebook_field_without_name_is_skipped():
    cb = REDCapCodebook([{"field_label": "no name here"}])
    assert len(cb) == 0


# --------------------------------------------------------------------------- #
# REDCapCodebook.from_csv
# --------------------------------------------------------------------------- #

SAMPLE_CSV = (
    '"Variable / Field Name","Form Name","Section Header","Field Type","Field Label",'
    '"Choices, Calculations, OR Slider Labels","Field Note",'
    '"Text Validation Type OR Show Slider Number","Text Validation Min","Text Validation Max",'
    'Identifier?,"Branching Logic (Show field only if...)","Required Field?","Custom Alignment",'
    '"Question Number (surveys only)","Matrix Group Name","Matrix Ranking?","Field Annotation"\n'
    'record_id,waves,,text,"Record ID",,,,,,,,,,,,,\n'
    'geburtsdatum,waves,"Allgemeine Fragen",text,Geburtsdatum,,DD-MM-YYYY,date_dmy,,,,,,,,,,'
    ' @FHIR-MAPPING=\'Patient/birthDate\'\n'
    'geschlecht,waves,,radio,"Geschlecht:","male, männlich|female, weiblich",,,,,,,,RH,,,,'
    ' @FHIR-MAPPING=\'Patient/gender\'\n'
)


def test_from_csv_parses_standard_redcap_export(tmp_path):
    csv_path = tmp_path / "dictionary.csv"
    csv_path.write_text(SAMPLE_CSV, encoding="utf-8-sig")

    cb = REDCapCodebook.from_csv(str(csv_path))
    assert len(cb) == 3

    geschlecht = cb.get_field("geschlecht")
    assert geschlecht["field_label"] == "Geschlecht:"
    assert geschlecht["field_type"] == "radio"
    assert geschlecht["_choices"] == [
        {"code": "male", "label": "männlich"},
        {"code": "female", "label": "weiblich"},
    ]
    assert geschlecht["_fhir_annotation"] == {
        "type": "Patient",
        "primaryElementPath": "gender",
    }


def test_from_csv_skips_rows_without_field_name(tmp_path):
    csv_content = SAMPLE_CSV + ",waves,,text,orphan,,,,,,,,,,,,,\n"
    csv_path = tmp_path / "dictionary.csv"
    csv_path.write_text(csv_content, encoding="utf-8-sig")

    cb = REDCapCodebook.from_csv(str(csv_path))
    # The extra row has an empty field_name and must not appear
    assert len(cb) == 3


# --------------------------------------------------------------------------- #
# REDCapCodebook.from_json
# --------------------------------------------------------------------------- #


def test_from_json_loads_api_style_export(tmp_path, sample_fields):
    json_path = tmp_path / "metadata.json"
    json_path.write_text(json.dumps(sample_fields[:2]), encoding="utf-8")

    cb = REDCapCodebook.from_json(str(json_path))
    assert len(cb) == 2
    assert cb.get_field("geburtsdatum")["field_label"] == "Geburtsdatum"


# --------------------------------------------------------------------------- #
# REDCapCodebook.from_api
# --------------------------------------------------------------------------- #


def test_from_api_posts_request_and_builds_codebook(monkeypatch):
    """from_api must not touch the network — the 'requests' module is faked."""

    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"field_name": "record_id", "field_label": "Record ID"}]

    class FakeRequestsModule:
        @staticmethod
        def post(url, data=None, timeout=None, verify=None):
            calls["url"] = url
            calls["data"] = data
            calls["timeout"] = timeout
            calls["verify"] = verify
            return FakeResponse()

    import sys

    monkeypatch.setitem(sys.modules, "requests", FakeRequestsModule())

    cb = REDCapCodebook.from_api("https://redcap.example.org/api/", "TOKEN123", verify_ssl=False)

    assert len(cb) == 1
    assert calls["url"] == "https://redcap.example.org/api/"
    assert calls["data"]["token"] == "TOKEN123"
    assert calls["data"]["content"] == "metadata"
    assert calls["verify"] is False


def test_from_api_raises_on_error_payload(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": "invalid token"}

    class FakeRequestsModule:
        @staticmethod
        def post(url, data=None, timeout=None, verify=None):
            return FakeResponse()

    import sys

    monkeypatch.setitem(sys.modules, "requests", FakeRequestsModule())

    with pytest.raises(RuntimeError, match="invalid token"):
        REDCapCodebook.from_api("https://redcap.example.org/api/", "BAD")


# --------------------------------------------------------------------------- #
# load_fhir_code_mapping
# --------------------------------------------------------------------------- #


def test_load_fhir_code_mapping_empty_path_returns_empty_dict():
    assert load_fhir_code_mapping("") == {}


def test_load_fhir_code_mapping_json(tmp_path):
    data = {"aktbefinden": {"1": "SG", "2": "G"}, "diabetes": {"1": "Y", "2": "N"}}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = load_fhir_code_mapping(str(path))
    assert result == data


def test_load_fhir_code_mapping_json_non_dict_root_raises(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    with pytest.raises(ValueError, match="must be an object"):
        load_fhir_code_mapping(str(path))


def test_load_fhir_code_mapping_json_skips_non_dict_field_values(tmp_path):
    data = {"aktbefinden": {"1": "SG"}, "broken_field": "not-a-dict"}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = load_fhir_code_mapping(str(path))
    assert result == {"aktbefinden": {"1": "SG"}}


def test_load_fhir_code_mapping_csv(tmp_path):
    csv_content = (
        "field_name,redcap_code,fhir_code\n"
        "aktbefinden,1,SG\n"
        "aktbefinden,2,G\n"
        "diabetes,1,Y\n"
    )
    path = tmp_path / "mapping.csv"
    path.write_text(csv_content, encoding="utf-8-sig")

    result = load_fhir_code_mapping(str(path))
    assert result == {"aktbefinden": {"1": "SG", "2": "G"}, "diabetes": {"1": "Y"}}


def test_load_fhir_code_mapping_csv_case_insensitive_headers_with_spaces(tmp_path):
    csv_content = "Field Name,REDCap Code,FHIR Code\naktbefinden,1,SG\n"
    path = tmp_path / "mapping.csv"
    path.write_text(csv_content, encoding="utf-8-sig")

    result = load_fhir_code_mapping(str(path))
    assert result == {"aktbefinden": {"1": "SG"}}


def test_load_fhir_code_mapping_csv_missing_columns_raises(tmp_path):
    csv_content = "field_name,redcap_code\naktbefinden,1\n"
    path = tmp_path / "mapping.csv"
    path.write_text(csv_content, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="must have columns"):
        load_fhir_code_mapping(str(path))


def test_load_fhir_code_mapping_csv_skips_incomplete_rows(tmp_path):
    csv_content = (
        "field_name,redcap_code,fhir_code\n"
        "aktbefinden,1,SG\n"
        "aktbefinden,,G\n"  # missing redcap_code -> skipped
        ",2,G\n"  # missing field_name -> skipped
    )
    path = tmp_path / "mapping.csv"
    path.write_text(csv_content, encoding="utf-8-sig")

    result = load_fhir_code_mapping(str(path))
    assert result == {"aktbefinden": {"1": "SG"}}


def test_load_fhir_code_mapping_unsupported_extension_raises(tmp_path):
    path = tmp_path / "mapping.txt"
    path.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported FHIR code mapping format"):
        load_fhir_code_mapping(str(path))


# ── Field Note → elementdefinition-allowedUnits ───────────────────────────────
def _plugin_with(fields):
    from types import SimpleNamespace
    from plugins.redcap.plugin import REDCapPlugin

    plugin = object.__new__(REDCapPlugin)
    plugin._codebook = SimpleNamespace(get_field=lambda name: fields.get(name))
    return plugin


def _numeric_element(name="gewicht", code="decimal"):
    return {"id": f"Source.{name}", "path": f"Source.{name}", "type": [{"code": code}]}


def test_numeric_field_note_becomes_a_ucum_allowed_units_extension():
    plugin = _plugin_with({})
    elem = _numeric_element()
    assert plugin._enrich_allowed_units(elem, {"field_note": "kg"}) is True
    ext = elem["extension"][0]
    assert ext["url"].endswith("elementdefinition-allowedUnits")
    coding = ext["valueCodeableConcept"]["coding"][0]
    assert coding == {"system": "http://unitsofmeasure.org", "code": "kg"}


def test_prose_field_note_is_not_mistaken_for_a_unit():
    # Field Note is free text and usually a sentence; pinning one as a unit would
    # put it into every derived Quantity.
    plugin = _plugin_with({})
    elem = _numeric_element()
    assert (
        plugin._enrich_allowed_units(elem, {"field_note": "Bitte in kg angeben"})
        is False
    )
    assert "extension" not in elem or not elem["extension"]


def test_non_numeric_field_gets_no_unit():
    plugin = _plugin_with({})
    elem = _numeric_element("familienstand", "string")
    assert plugin._enrich_allowed_units(elem, {"field_note": "kg"}) is False


def test_empty_field_note_is_ignored():
    plugin = _plugin_with({})
    assert plugin._enrich_allowed_units(_numeric_element(), {"field_note": ""}) is False


def test_existing_allowed_units_is_not_overwritten():
    plugin = _plugin_with({})
    elem = _numeric_element()
    elem["extension"] = [{"url": plugin._ALLOWED_UNITS_URL, "valueCodeableConcept": {}}]
    assert plugin._enrich_allowed_units(elem, {"field_note": "kg"}) is False
    assert len(elem["extension"]) == 1
