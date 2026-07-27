"""Unit tests for the REDCap pipeline plugin (plugins/redcap/plugin.py)."""

import json

import pytest

from plugins.redcap.plugin import REDCapPlugin

pytestmark = pytest.mark.unit


CSV_CONTENT = (
    '"Variable / Field Name","Form Name","Section Header","Field Type","Field Label",'
    '"Choices, Calculations, OR Slider Labels","Field Note",'
    '"Text Validation Type OR Show Slider Number","Text Validation Min","Text Validation Max",'
    'Identifier?,"Branching Logic (Show field only if...)","Required Field?","Custom Alignment",'
    '"Question Number (surveys only)","Matrix Group Name","Matrix Ranking?","Field Annotation"\n'
    'geburtsdatum,waves,"Allgemeine Fragen",text,Geburtsdatum,,DD-MM-YYYY,date_dmy,,,,,,,,,,'
    ' @FHIR-MAPPING=\'Patient/birthDate\'\n'
    'geschlecht,waves,,radio,"Geschlecht:","male, männlich|female, weiblich",,,,,,,,RH,,,,'
    ' @FHIR-MAPPING=\'Patient/gender\'\n'
    'rauchen_tag,waves,,text,"Zigaretten pro Tag",,,integer,,,,,,,,,,\n'
)


@pytest.fixture
def csv_codebook_path(tmp_path):
    path = tmp_path / "dictionary.csv"
    path.write_text(CSV_CONTENT, encoding="utf-8-sig")
    return str(path)


@pytest.fixture
def csv_plugin(csv_codebook_path):
    return REDCapPlugin({"type": "redcap", "source": "csv", "codebook_path": csv_codebook_path})


# --------------------------------------------------------------------------- #
# plugin_id / construction
# --------------------------------------------------------------------------- #


def test_plugin_id():
    plugin = REDCapPlugin({})
    assert plugin.plugin_id == "redcap"


def test_default_base_url():
    plugin = REDCapPlugin({})
    assert plugin._base_url == "http://example.org"


def test_custom_base_url():
    plugin = REDCapPlugin({"base_url": "http://kfdm.example.org"})
    assert plugin._base_url == "http://kfdm.example.org"


# --------------------------------------------------------------------------- #
# _load_codebook — source dispatch & error handling
# --------------------------------------------------------------------------- #


def test_load_codebook_csv_source(csv_plugin):
    cb = csv_plugin._load_codebook()
    assert len(cb) == 3
    assert cb.get_field("geburtsdatum") is not None


def test_load_codebook_is_cached(csv_plugin, monkeypatch):
    from plugins.redcap import codebook as codebook_module

    calls = {"n": 0}
    original = codebook_module.REDCapCodebook.from_csv

    @classmethod
    def counting_from_csv(cls, path):
        calls["n"] += 1
        return original.__func__(cls, path)

    monkeypatch.setattr(codebook_module.REDCapCodebook, "from_csv", counting_from_csv)

    csv_plugin._load_codebook()
    csv_plugin._load_codebook()
    assert calls["n"] == 1


def test_load_codebook_json_source(tmp_path):
    fields = [{"field_name": "record_id", "field_label": "Record ID"}]
    json_path = tmp_path / "metadata.json"
    json_path.write_text(json.dumps(fields), encoding="utf-8")

    plugin = REDCapPlugin(
        {"type": "redcap", "source": "json", "codebook_path": str(json_path)}
    )
    cb = plugin._load_codebook()
    assert len(cb) == 1


def test_load_codebook_csv_missing_path_raises():
    plugin = REDCapPlugin({"type": "redcap", "source": "csv"})
    with pytest.raises(ValueError, match="codebook_path"):
        plugin._load_codebook()


def test_load_codebook_api_missing_credentials_raises():
    plugin = REDCapPlugin({"type": "redcap", "source": "api"})
    with pytest.raises(ValueError, match="api_url"):
        plugin._load_codebook()


def test_load_codebook_unknown_source_raises():
    plugin = REDCapPlugin({"type": "redcap", "source": "xml"})
    with pytest.raises(ValueError, match="unknown source"):
        plugin._load_codebook()


def test_load_codebook_api_source_uses_fake_requests(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"field_name": "record_id", "field_label": "Record ID"}]

    class FakeRequestsModule:
        @staticmethod
        def post(url, data=None, timeout=None, verify=None):
            return FakeResponse()

    import sys

    monkeypatch.setitem(sys.modules, "requests", FakeRequestsModule())

    plugin = REDCapPlugin(
        {
            "type": "redcap",
            "source": "api",
            "api_url": "https://redcap.example.org/api/",
            "api_token": "TOKEN",
        }
    )
    cb = plugin._load_codebook()
    assert len(cb) == 1


# --------------------------------------------------------------------------- #
# _load_fhir_code_mapping
# --------------------------------------------------------------------------- #


def test_load_fhir_code_mapping_no_path_returns_empty_dict():
    plugin = REDCapPlugin({})
    assert plugin._load_fhir_code_mapping() == {}


def test_load_fhir_code_mapping_loads_json(tmp_path):
    mapping = {"diabetes": {"1": "Y", "2": "N"}}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    plugin = REDCapPlugin({"fhir_code_mapping": str(path)})
    assert plugin._load_fhir_code_mapping() == mapping


def test_load_fhir_code_mapping_bad_path_is_swallowed_and_returns_empty(tmp_path):
    plugin = REDCapPlugin({"fhir_code_mapping": str(tmp_path / "does_not_exist.json")})
    # Errors are caught internally and logged; must not raise.
    assert plugin._load_fhir_code_mapping() == {}


def test_load_fhir_code_mapping_is_cached(tmp_path):
    mapping = {"diabetes": {"1": "Y"}}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    plugin = REDCapPlugin({"fhir_code_mapping": str(path)})
    first = plugin._load_fhir_code_mapping()
    path.write_text(json.dumps({"other": {"9": "Z"}}), encoding="utf-8")
    second = plugin._load_fhir_code_mapping()
    assert first is second
    assert second == mapping


# --------------------------------------------------------------------------- #
# post_source_def
# --------------------------------------------------------------------------- #


def _elem(id_, type_=None, short=None):
    e = {"id": id_}
    if type_ is not None:
        e["type"] = type_
    if short is not None:
        e["short"] = short
    return e


def test_post_source_def_enriches_description_and_type(csv_plugin):
    source_definition = {
        "snapshot": {
            "element": [
                _elem("Source.geburtsdatum", type_=[{"code": "string"}]),
                _elem("Source.geschlecht", type_=[{"code": "string"}]),
                _elem("Source.rauchen_tag", type_=[{"code": "string"}]),
            ]
        }
    }
    source_fields = [
        {"id": "Source.geburtsdatum"},
        {"id": "Source.geschlecht"},
        {"id": "Source.rauchen_tag"},
    ]

    result = csv_plugin.post_source_def(source_definition, source_fields)
    elements = {e["id"]: e for e in result["snapshot"]["element"]}

    # date_dmy -> date
    assert elements["Source.geburtsdatum"]["type"][0]["code"] == "date"
    assert elements["Source.geburtsdatum"]["short"] == "Geburtsdatum"

    # integer validation -> integer
    assert elements["Source.rauchen_tag"]["type"][0]["code"] == "integer"
    assert elements["Source.rauchen_tag"]["short"] == "Zigaretten pro Tag"

    # radio field has no text_validation_type -> type left unchanged, label still applied
    assert elements["Source.geschlecht"]["type"][0]["code"] == "string"
    assert elements["Source.geschlecht"]["short"] == "Geschlecht:"


def test_post_source_def_does_not_overwrite_existing_short(csv_plugin):
    source_definition = {
        "snapshot": {
            "element": [
                _elem("Source.geburtsdatum", type_=[{"code": "string"}], short="Pre-existing"),
            ]
        }
    }
    source_fields = [{"id": "Source.geburtsdatum"}]

    result = csv_plugin.post_source_def(source_definition, source_fields)
    assert result["snapshot"]["element"][0]["short"] == "Pre-existing"


def test_post_source_def_handles_string_type_entries(csv_plugin):
    """When elem['type'] is a list of bare strings (not dicts), a validated type
    replaces the whole list with a single {'code': ...} dict."""
    source_definition = {
        "snapshot": {"element": [_elem("Source.geburtsdatum", type_=["string"])]}
    }
    source_fields = [{"id": "Source.geburtsdatum"}]

    result = csv_plugin.post_source_def(source_definition, source_fields)
    assert result["snapshot"]["element"][0]["type"] == [{"code": "date"}]


def test_post_source_def_ignores_fields_not_in_source_fields(csv_plugin):
    source_definition = {
        "snapshot": {"element": [_elem("Source.geburtsdatum", type_=[{"code": "string"}])]}
    }
    # source_fields references a different field only
    result = csv_plugin.post_source_def(source_definition, [{"id": "Source.geschlecht"}])
    # untouched because "geburtsdatum" isn't in the field_names set
    assert result["snapshot"]["element"][0]["type"][0]["code"] == "string"
    assert "short" not in result["snapshot"]["element"][0]


def test_post_source_def_missing_snapshot_is_noop(csv_plugin):
    result = csv_plugin.post_source_def({}, [{"id": "Source.geburtsdatum"}])
    assert result == {}


def test_post_source_def_field_name_matches_but_absent_from_codebook(csv_plugin):
    """A source field can be present in source_fields (and thus in the
    field_names filter) yet absent from the codebook itself - e.g. a
    computed/system field with no Data Dictionary entry. Must be skipped
    without raising."""
    source_definition = {
        "snapshot": {"element": [_elem("Source.not_in_codebook", type_=[{"code": "string"}])]}
    }
    source_fields = [{"id": "Source.not_in_codebook"}]

    result = csv_plugin.post_source_def(source_definition, source_fields)
    elem = result["snapshot"]["element"][0]
    assert "short" not in elem
    assert elem["type"][0]["code"] == "string"


# --------------------------------------------------------------------------- #
# enrich_automapping
# --------------------------------------------------------------------------- #


def test_enrich_automapping_adds_annotation_derived_mappings(csv_plugin):
    automapping = {"Source.rauchen_tag": "Observation.valueInteger"}
    result = csv_plugin.enrich_automapping(automapping)

    assert result["Source.geburtsdatum"] == "Patient.birthDate"
    assert result["Source.geschlecht"] == "Patient.gender"
    # existing mapping is untouched
    assert result["Source.rauchen_tag"] == "Observation.valueInteger"


def test_enrich_automapping_respects_existing_mapping(csv_plugin):
    # geburtsdatum already mapped explicitly -> annotation must not override it
    automapping = {"Source.geburtsdatum": "Patient.deceasedDateTime"}
    result = csv_plugin.enrich_automapping(automapping)
    assert result["Source.geburtsdatum"] == "Patient.deceasedDateTime"


def test_enrich_automapping_no_prefix_when_automapping_empty(csv_plugin):
    result = csv_plugin.enrich_automapping({})
    # No existing keys -> no prefix could be derived -> bare field_name used as key
    assert result["geburtsdatum"] == "Patient.birthDate"


def test_enrich_automapping_skips_fields_without_annotation(csv_plugin):
    automapping = {}
    result = csv_plugin.enrich_automapping(automapping)
    # rauchen_tag has no @FHIR-MAPPING annotation and must not appear
    assert "rauchen_tag" not in result


def test_enrich_automapping_skips_annotation_missing_type_or_path(tmp_path):
    csv_content = (
        CSV_CONTENT
        + 'incomplete_ann,waves,,text,"Incomplete",,,,,,,,,,,,,'
        + ' @FHIR-MAPPING=\'OnlyType\'\n'
    )
    csv_path = tmp_path / "dict.csv"
    csv_path.write_text(csv_content, encoding="utf-8-sig")
    plugin = REDCapPlugin({"type": "redcap", "source": "csv", "codebook_path": str(csv_path)})

    result = plugin.enrich_automapping({})
    # Annotation resolves to type="OnlyType", primaryElementPath="" -> skipped
    assert "incomplete_ann" not in result


# --------------------------------------------------------------------------- #
# get_field_metadata
# --------------------------------------------------------------------------- #


def test_get_field_metadata_regular_field(csv_plugin):
    meta = csv_plugin.get_field_metadata("geschlecht")
    assert meta["field_label"] == "Geschlecht:"


def test_get_field_metadata_unknown_field_returns_none(csv_plugin):
    assert csv_plugin.get_field_metadata("does_not_exist") is None


def test_get_field_metadata_synthesizes_form_complete_status(csv_plugin):
    meta = csv_plugin.get_field_metadata("waves_complete")
    assert meta["_form_status"] is True
    assert meta["_choices"] == [
        {"code": "0", "label": "Incomplete"},
        {"code": "1", "label": "Unverified"},
        {"code": "2", "label": "Complete"},
    ]
    assert meta["_fhir_annotation"] == {
        "primaryElementSystem": "http://hl7.org/fhir/questionnaire-answers-status"
    }


# --------------------------------------------------------------------------- #
# generate_concept_map
# --------------------------------------------------------------------------- #


def test_generate_concept_map_returns_none_without_choices(csv_plugin):
    meta = csv_plugin.get_field_metadata("geburtsdatum")  # text field, no choices
    assert csv_plugin.generate_concept_map("geburtsdatum", meta, None) is None


def test_generate_concept_map_builds_cm_for_coded_field(csv_plugin):
    meta = csv_plugin.get_field_metadata("geschlecht")
    cm = csv_plugin.generate_concept_map("geschlecht", meta, None)
    assert cm["resourceType"] == "ConceptMap"
    codes = {e["code"] for e in cm["group"][0]["element"]}
    assert codes == {"male", "female"}


def test_generate_concept_map_uses_explicit_fhir_code_mapping(tmp_path, csv_codebook_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"geschlecht": {"male": "M", "female": "F"}}), encoding="utf-8"
    )
    plugin = REDCapPlugin(
        {
            "type": "redcap",
            "source": "csv",
            "codebook_path": csv_codebook_path,
            "fhir_code_mapping": str(mapping_path),
        }
    )
    meta = plugin.get_field_metadata("geschlecht")
    cm = plugin.generate_concept_map("geschlecht", meta, None)
    elements = {e["code"]: e["target"][0]["code"] for e in cm["group"][0]["element"]}
    assert elements == {"male": "M", "female": "F"}


def test_generate_concept_map_form_status_uses_default_mapping(csv_plugin):
    meta = csv_plugin.get_field_metadata("waves_complete")
    cm = csv_plugin.generate_concept_map("waves_complete", meta, None)
    elements = {e["code"]: e["target"][0]["code"] for e in cm["group"][0]["element"]}
    assert elements == {"0": "in-progress", "1": "in-progress", "2": "completed"}
