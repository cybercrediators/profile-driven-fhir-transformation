"""Unit tests for minimal-mode required-path collection in StructureMapGenerator.

Covers which fields survive minimal-mode pruning: mandatory (min>=1), fixed/pattern, and —
the behaviour added here — mustSupport fields (implementer obligations), while plain optional
unmapped fields are dropped.
"""

import pytest

from mapping.fml_map import StructureMapGenerator

pytestmark = pytest.mark.unit


def _collect(fields, create_references=False):
    # The method only reads self.create_references and recurses via real bound methods, so
    # bypass __init__ (which needs app_state, urls, etc.) and just set that one attribute.
    gen = object.__new__(StructureMapGenerator)
    gen.create_references = create_references
    collector = set()
    gen._collect_required_paths_from_fields(fields, collector)
    return collector


def _f(path, *, min=0, fixed=None, must_support=False, **kw):
    field = {
        "path": path,
        "is_required": min > 0,
        "cardinality": {"min": min, "max": "1"},
    }
    if fixed is not None:
        field["fixed_value"] = fixed
    if must_support:
        field["must_support"] = True
    field.update(kw)
    return field


def test_required_field_is_kept():
    assert "Patient.name" in _collect([_f("Patient.name", min=1)])


def test_fixed_value_field_is_kept():
    assert "Patient.active" in _collect([_f("Patient.active", min=0, fixed=True)])


def test_plain_optional_field_is_pruned():
    assert _collect([_f("Patient.gender", min=0)]) == set()


def test_must_support_optional_field_is_kept():
    # The behaviour under test: a mustSupport field (even at min 0) is an implementer
    # obligation and must survive minimal-mode pruning.
    assert "Patient.birthDate" in _collect(
        [_f("Patient.birthDate", min=0, must_support=True)]
    )


def test_must_support_child_pulls_in_ancestor_container():
    # A nested mustSupport field retains its (optional, non-mustSupport) parent container,
    # because keeping a path adds all its ancestors.
    field = _f(
        "Patient.contact",
        min=0,
        type="BackboneElement",
        children=[_f("Patient.contact.name", min=0, must_support=True)],
    )
    paths = _collect([field])
    assert {"Patient.contact", "Patient.contact.name"} <= paths


def test_optional_non_must_support_backbone_with_optional_children_is_pruned():
    # Regression guard: an optional backbone whose children are only conditionally required
    # (no mustSupport, no min>=1) is still dropped.
    field = _f(
        "Patient.link",
        min=0,
        type="BackboneElement",
        children=[_f("Patient.link.other", min=0)],
    )
    assert _collect([field]) == set()
