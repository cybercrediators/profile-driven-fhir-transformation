"""Unit tests for parse_element_definition / create_field_info / get_ref_profile.

Focus: the flattened ``field_info`` faithfully surfaces the profile facets (Tier-2 data
fidelity), and the element-skipping / fixed-value / reference / binding branches behave.
"""

from types import SimpleNamespace

import pytest

from fhir.resources.R4B.structuredefinition import (
    StructureDefinition,
    StructureDefinitionSnapshot,
)
from fhir.resources.R4B.elementdefinition import (
    ElementDefinition,
    ElementDefinitionType,
    ElementDefinitionBinding,
)

from parser.resource_parser import element_definition_parser as edp

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _elem(path, **kw):
    kw.setdefault("id", path)
    return ElementDefinition(path=path, **kw)


def _sd_obj(elements, url="http://example.org/sd/Test", res_type="Patient"):
    """A registry-object-like wrapper around a real StructureDefinition."""
    snap = StructureDefinitionSnapshot(element=elements)
    sd = StructureDefinition(
        url=url, name="Test", status="active", kind="resource",
        abstract=False, type=res_type, snapshot=snap,
    )
    return SimpleNamespace(data=sd, mappable_fields=[])


def _app_state(get_obj=None):
    used_by = []
    registry = SimpleNamespace(
        get_obj_by_name=get_obj or (lambda u: None),
        add_to_used_by=lambda src, tgt: used_by.append((src, tgt)),
    )
    state = SimpleNamespace(registry=registry, used_by=used_by)
    return state


# --------------------------------------------------------------------------- #
# create_field_info — Tier-2 data fidelity
# --------------------------------------------------------------------------- #


def test_create_field_info_basic_keys():
    fi = edp.create_field_info(_elem("Patient.name", min=1, max="1", short="A name"), [{"code": "HumanName"}])
    assert fi["path"] == "Patient.name"
    assert fi["cardinality"] == {"min": 1, "max": "1"}
    assert fi["description"] == "A name"
    assert fi["is_required"] is True
    assert fi["type"] == [{"code": "HumanName"}]


def test_create_field_info_surfaces_must_support_comment_maxlength_contentref():
    elem = _elem(
        "Patient.identifier", min=0, max="*",
        mustSupport=True, comment="populate when known", maxLength=64,
        contentReference="#Patient.contact",
    )
    fi = edp.create_field_info(elem, [{"code": "Identifier"}])
    assert fi["must_support"] is True
    assert fi["comment"] == "populate when known"
    assert fi["max_length"] == 64
    assert fi["content_reference"] == "#Patient.contact"


def test_create_field_info_captures_min_and_max_value_including_zero():
    elem = _elem("Observation.value", min=0, max="1", minValueInteger=0, maxValueInteger=100)
    fi = edp.create_field_info(elem, [{"code": "integer"}])
    # min of 0 is falsy but must still be captured (guard is `is not None`)
    assert fi["min_value"] == {"type": "Integer", "value": 0}
    assert fi["max_value"] == {"type": "Integer", "value": 100}


def test_create_field_info_omits_absent_facets():
    fi = edp.create_field_info(_elem("Patient.gender", min=0, max="1"), [{"code": "code"}])
    for absent in ("must_support", "comment", "max_length", "content_reference", "min_value", "max_value", "default_value", "sliceName"):
        assert absent not in fi


# --------------------------------------------------------------------------- #
# fixed / pattern / defaultValue separation (Correctness-C)
# --------------------------------------------------------------------------- #


def test_create_field_info_captures_default_value_separately():
    # defaultValue is a fallback, NOT a constraint — surfaced as default_value, never forced.
    elem = _elem("Patient.gender", min=0, max="1", defaultValueCode="unknown")
    fi = edp.create_field_info(elem, [{"code": "code"}])
    assert fi["default_value"] == {"type": "Code", "value": "unknown"}
    assert not fi["fixed_value"]  # default must not become a fixed/forced value


def test_default_value_only_element_is_not_treated_as_fixed():
    # An element with ONLY a defaultValue (no fixed/pattern) must not get a fixed_value, so
    # the generated map won't force it. It is still captured as default_value for fidelity.
    elem = _elem("Patient.gender", min=0, max="1", defaultValueCode="male")
    fi = edp.parse_element_definition({}, elem, _sd_obj([elem]), _app_state())
    assert not fi["fixed_value"]
    assert fi["default_value"] == {"type": "Code", "value": "male"}


def test_fixed_value_still_captured_and_wins_over_default():
    # fixed[x] is a real constraint and must still populate fixed_value (and not be shadowed
    # by a co-present defaultValue, which alphabetically used to sort first in dir()).
    elem = _elem(
        "Patient.gender", min=1, max="1",
        fixedCode="female", defaultValueCode="male",
    )
    fi = edp.parse_element_definition({}, elem, _sd_obj([elem]), _app_state())
    assert fi["fixed_value"] == "female"
    assert fi["default_value"] == {"type": "Code", "value": "male"}


def test_create_field_info_includes_slice_name_when_present():
    fi = edp.create_field_info(_elem("Patient.identifier", min=0, max="1", sliceName="mrn"), [{"code": "Identifier"}])
    assert fi["sliceName"] == "mrn"


# --------------------------------------------------------------------------- #
# get_ref_profile
# --------------------------------------------------------------------------- #


def test_get_ref_profile_single_target():
    elem = _elem("Obs.subject", type=[ElementDefinitionType(code="Reference", targetProfile=["http://p/Patient"])])
    assert edp.get_ref_profile(elem) == "http://p/Patient"


def test_get_ref_profile_non_reference_returns_none():
    elem = _elem("Obs.value", type=[ElementDefinitionType(code="Quantity")])
    assert edp.get_ref_profile(elem) is None


def test_get_ref_profile_multiple_targets_returns_none():
    elem = _elem("Obs.subject", type=[ElementDefinitionType(code="Reference", targetProfile=["http://p/A", "http://p/B"])])
    assert edp.get_ref_profile(elem) is None


# --------------------------------------------------------------------------- #
# parse_element_definition — branches
# --------------------------------------------------------------------------- #


def test_parse_skips_resource_root():
    elem = _elem("Patient", min=0, max="*")
    sd = _sd_obj([elem])
    assert edp.parse_element_definition({}, elem, sd, _app_state()) is None


def test_parse_skips_max_zero():
    elem = _elem("Patient.deceased", min=0, max="0")
    sd = _sd_obj([elem])
    assert edp.parse_element_definition({}, elem, sd, _app_state()) is None


def test_parse_fixed_value_writes_template_and_returns_field_info():
    elem = _elem("Patient.active", min=1, max="1", fixedBoolean=True)
    sd = _sd_obj([elem])
    res_dict = {}
    fi = edp.parse_element_definition(res_dict, elem, sd, _app_state())
    assert res_dict == {"active": True}
    assert fi["fixed_value"] is True


def test_parse_reference_records_reference_target():
    elem = _elem(
        "Patient.managingOrganization", min=0, max="1",
        type=[ElementDefinitionType(code="Reference", targetProfile=["http://p/Org"])],
    )
    sd = _sd_obj([elem])
    fi = edp.parse_element_definition({}, elem, sd, _app_state())
    assert fi["reference_target"] == "http://p/Org"


def test_parse_binding_expands_options_and_records_used_by(monkeypatch):
    monkeypatch.setattr(edp, "expand_valueset", lambda url, seen, st: [{"code": "M"}, {"code": "F"}])
    elem = _elem(
        "Patient.gender", min=0, max="1",
        type=[ElementDefinitionType(code="code")],
        binding=ElementDefinitionBinding(strength="required", valueSet="http://vs/gender"),
    )
    sd = _sd_obj([elem])
    state = _app_state()
    fi = edp.parse_element_definition({}, elem, sd, state)
    assert fi["valueSetUrl"] == "http://vs/gender"
    assert fi["binding_strength"] == "required"
    assert fi["options"] == [{"code": "M"}, {"code": "F"}]
    assert ("http://vs/gender", sd.data.url) in state.used_by


def test_parse_fixed_value_element_retains_binding_facets(monkeypatch):
    # a fixed/pattern element still carries its binding — the VS canonical and
    # strength must survive the early fixed-value branch (§L claim narrowing);
    # options stay unexpanded there (map-generation behavior unchanged)
    monkeypatch.setattr(edp, "expand_valueset", lambda url, seen, st: [{"code": "M"}])
    elem = _elem(
        "Patient.gender", min=0, max="1", fixedCode="female",
        type=[ElementDefinitionType(code="code")],
        binding=ElementDefinitionBinding(strength="required", valueSet="http://vs/gender"),
    )
    sd = _sd_obj([elem])
    state = _app_state()
    fi = edp.parse_element_definition({}, elem, sd, state)
    assert fi["fixed_value"] == "female"
    assert fi["valueSetUrl"] == "http://vs/gender"
    assert fi["binding_strength"] == "required"
    assert fi["options"] == []  # expansion deliberately skipped on this branch
    assert ("http://vs/gender", sd.data.url) in state.used_by


def test_parse_reference_element_retains_binding_facets():
    elem = _elem(
        "Patient.managingOrganization", min=0, max="1",
        type=[ElementDefinitionType(code="Reference", targetProfile=["http://p/Org"])],
        binding=ElementDefinitionBinding(strength="extensible", valueSet="http://vs/orgs"),
    )
    sd = _sd_obj([elem])
    fi = edp.parse_element_definition({}, elem, sd, _app_state())
    assert fi["reference_target"] == "http://p/Org"
    assert fi["valueSetUrl"] == "http://vs/orgs"
    assert fi["binding_strength"] == "extensible"


def test_parse_complex_type_is_expanded_via_fhir_resources():
    elem = _elem("Patient.name", min=0, max="*", type=[ElementDefinitionType(code="HumanName")])
    sd = _sd_obj([elem])
    fi = edp.parse_element_definition({}, elem, sd, _app_state())
    type_entry = fi["type"][0]
    assert "type_structure" in type_entry
    assert any(f["path"].endswith(".family") for f in type_entry["type_structure"])
