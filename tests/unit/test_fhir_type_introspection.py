"""Unit tests for the spec-derived FHIR type introspection.

These pin the behaviour that the primitive set, the marker->name map and the carrier
fallback are all derived from ``fhir.resources`` (not hand-maintained), and that complex
type expansion / annotation unwrapping behave as the parser relies on.
"""

import datetime
import decimal
import uuid

import pytest

from fhir.resources.R4B import fhirtypes as ft
from typing import Optional, List, Annotated, get_args

from parser.resource_parser import fhir_type_introspection as fti

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# derived primitive registry
# --------------------------------------------------------------------------- #


def test_primitive_set_has_the_21_r4b_primitives():
    # FHIR R4B defines exactly 21 primitive datatypes; the set is derived, not hard-coded.
    assert len(fti.PRIMITIVES) == 21
    # a representative spread, including the two whose markers are irregular
    for name in ("boolean", "string", "code", "base64Binary", "uuid", "integer64"):
        assert fti.is_primitive(name)


def test_complex_types_are_not_primitive():
    for name in ("CodeableConcept", "Period", "Reference", "BackboneElement"):
        assert not fti.is_primitive(name)


def test_marker_map_resolves_irregular_markers():
    # base64Binary / uuid carry markers (EncodedBytes / UuidVersion) whose class names do
    # NOT lowercase-map to the FHIR name — they must still resolve via the derived map.
    b64_marker = get_args(ft.Base64BinaryType)[1]
    uuid_marker = get_args(ft.UuidType)[1]
    code_marker = get_args(ft.CodeType)[1]
    assert fti._fhir_primitive_from_metadata([b64_marker]) == "base64Binary"
    assert fti._fhir_primitive_from_metadata([uuid_marker]) == "uuid"
    assert fti._fhir_primitive_from_metadata([code_marker]) == "code"


def test_marker_map_returns_none_for_non_primitive_metadata():
    assert fti._fhir_primitive_from_metadata([object()]) is None
    assert fti._fhir_primitive_from_metadata([]) is None


# --------------------------------------------------------------------------- #
# get_fhir_type_name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "py_type,expected",
    [
        (bool, "boolean"),
        (bytes, "base64Binary"),
        (decimal.Decimal, "decimal"),
        (uuid.UUID, "uuid"),
        (datetime.date, "date"),
        (datetime.time, "time"),
    ],
)
def test_unambiguous_carriers_are_library_derived(py_type, expected):
    assert fti.get_fhir_type_name(py_type) == expected


@pytest.mark.parametrize(
    "py_type,expected",
    [
        (str, "string"),
        (int, "integer"),
        (datetime.datetime, "dateTime"),
        (float, "decimal"),
    ],
)
def test_ambiguous_carriers_fall_back_to_most_general(py_type, expected):
    # str maps onto 8 primitives, int onto 4, datetime onto 2 — no unique inverse, so the
    # fallback picks the general one.
    assert fti.get_fhir_type_name(py_type) == expected


def test_complex_class_resolves_to_its_name():
    from fhir.resources.R4B.period import Period

    assert fti.get_fhir_type_name(Period) == "Period"


def test_type_suffix_is_stripped():
    class PeriodType:  # mimics a fhir.resources alias class name
        pass

    assert fti.get_fhir_type_name(PeriodType) == "Period"


def test_optional_union_is_unwrapped_to_concrete_member():
    assert fti.get_fhir_type_name(Optional[str]) == "string"


# --------------------------------------------------------------------------- #
# get_complex_type_fields
# --------------------------------------------------------------------------- #


def test_expands_named_complex_type_from_fhir_resources():
    fields = fti.get_complex_type_fields("HumanName", parent_path="Patient.name")
    by_name = {f["path"].split(".")[-1]: f for f in fields}
    assert {"family", "given", "use", "period"} <= set(by_name)
    assert by_name["family"]["type"] == "string"
    assert by_name["use"]["type"] == "code"
    # nested complex datatype expands recursively
    assert by_name["period"]["type"] == "Period"
    assert "type_structure" in by_name["period"]


def test_anonymous_backbone_subtype_emitted_as_backbone_element():
    # Timing.repeat is a TimingRepeat backbone sub-type (no module of its own); it must be
    # emitted as a generic BackboneElement while keeping its expanded children.
    fields = fti.get_complex_type_fields("Timing", parent_path="X.timing")
    repeat = next(f for f in fields if f["path"].endswith(".repeat"))
    assert repeat["type"] == "BackboneElement"
    assert "type_structure" in repeat


def test_unknown_type_returns_empty():
    assert fti.get_complex_type_fields("NotARealType") == []


def test_recursion_depth_is_capped():
    assert fti.get_complex_type_fields("HumanName", recursion_depth=6) == []


# --------------------------------------------------------------------------- #
# extract_inner_type
# --------------------------------------------------------------------------- #


def test_extract_inner_type_optional():
    inner, is_list, is_optional, _ = fti.extract_inner_type(Optional[str])
    assert inner is str and is_list is False and is_optional is True


def test_extract_inner_type_list():
    inner, is_list, is_optional, _ = fti.extract_inner_type(List[str])
    assert inner is str and is_list is True


def test_extract_inner_type_annotated_metadata():
    inner, _, _, metadata = fti.extract_inner_type(Annotated[str, "m1", "m2"])
    assert inner is str
    assert list(metadata) == ["m1", "m2"]


def test_extract_inner_type_plain():
    inner, is_list, is_optional, metadata = fti.extract_inner_type(int)
    assert inner is int and not is_list and not is_optional and metadata == []
