"""Resolving a mapping target that names a `value[x]` choice inside a slice.

A profile may narrow a slice's `value[x]` to one datatype without profiling that
datatype's children. The children are then synthesized from `fhir.resources` and keep
the *unsliced* path, so no single field record carries both the slice and the concrete
choice — and an authored target like `component:back.valueQuantity.value` matches
nothing even though every part of it is real.

These drive the resolver directly. The existing slice tests hand `create_field_rules` an
already-canonical `automapped_mappings` dict, i.e. they start *after* resolution, which
is why none of them could see this.
"""

from types import SimpleNamespace

import pytest

from mapping.fml_map import StructureMapGenerator

pytestmark = pytest.mark.unit


def _generator():
    generator = object.__new__(StructureMapGenerator)
    generator.mapping_diagnostics = []
    generator.current_profile_name = "test"
    # `_choice_suffix` is the only factory behaviour the resolver needs.
    generator.factory = SimpleNamespace(
        _choice_suffix=lambda code: code[:1].upper() + code[1:] if code else ""
    )
    return generator


def _quantity_children(owner_path):
    """Quantity's expanded structure — synthesized, so paths keep the unsliced parent."""
    return [
        {"path": f"{owner_path}.value", "id": "Quantity.value", "type": "decimal"},
        {"path": f"{owner_path}.unit", "id": "Quantity.unit", "type": "string"},
        {"path": f"{owner_path}.system", "id": "Quantity.system", "type": "uri"},
    ]


def _sliced_choice_field(slice_id, types=("Quantity",), owner="Observation.component"):
    """A slice whose `value[x]` is narrowed but whose datatype children are expanded."""
    return {
        "path": slice_id,
        "id": slice_id,
        "children": [
            {
                "path": f"{owner}.value[x]",
                "id": f"{slice_id}.value[x]",
                "type": [
                    {"code": t, "type_structure": _quantity_children(f"{owner}.value[x]")}
                    for t in types
                ],
            }
        ],
    }


def _index(*identities):
    return {identity: identity for identity in identities}


# ── the marfoglia shape: optional component slice narrowed to Quantity ────────
def test_concrete_choice_inside_a_slice_resolves_to_the_canonical_key():
    gen = _generator()
    slice_id = "Observation.component:back"
    fields = {slice_id: _sliced_choice_field(slice_id)}
    resolved = gen._resolve_sliced_choice_target(
        "Observation.component:back.valueQuantity.value", _index(slice_id), fields
    )
    assert resolved == "Observation.component:back.value[x]:valueQuantity.value"


def test_a_datatype_child_that_does_not_exist_is_refused():
    gen = _generator()
    slice_id = "Observation.component:back"
    fields = {slice_id: _sliced_choice_field(slice_id)}
    assert (
        gen._resolve_sliced_choice_target(
            "Observation.component:back.valueQuantity.nosuch", _index(slice_id), fields
        )
        is None
    )


def test_a_choice_type_the_slice_does_not_allow_is_refused():
    gen = _generator()
    slice_id = "Observation.component:back"
    fields = {slice_id: _sliced_choice_field(slice_id)}
    assert (
        gen._resolve_sliced_choice_target(
            "Observation.component:back.valueString", _index(slice_id), fields
        )
        is None
    )


# ── the AMP/LCI shape: a primitive answer inside a QuestionnaireResponse item ──
def test_primitive_answer_choice_inside_an_item_slice_resolves():
    gen = _generator()
    slice_id = "QuestionnaireResponse.item:proNopro.answer"
    fields = {
        slice_id: {
            "path": slice_id,
            "id": slice_id,
            "children": [
                {
                    "path": "QuestionnaireResponse.item.answer.value[x]",
                    "id": f"{slice_id}.value[x]",
                    "type": [{"code": "string"}],
                }
            ],
        }
    }
    resolved = gen._resolve_sliced_choice_target(
        "QuestionnaireResponse.item:proNopro.answer.valueString",
        _index(slice_id),
        fields,
    )
    assert resolved == (
        "QuestionnaireResponse.item:proNopro.answer.value[x]:valueString"
    )


# ── sibling slices must not cross-assign ──────────────────────────────────────
def test_sibling_slices_resolve_to_their_own_identity():
    gen = _generator()
    back, stump = "Observation.component:back", "Observation.component:stump"
    fields = {back: _sliced_choice_field(back), stump: _sliced_choice_field(stump)}
    index = _index(back, stump)
    assert gen._resolve_sliced_choice_target(
        f"{back}.valueQuantity.value", index, fields
    ) == f"{back}.value[x]:valueQuantity.value"
    assert gen._resolve_sliced_choice_target(
        f"{stump}.valueQuantity.value", index, fields
    ) == f"{stump}.value[x]:valueQuantity.value"


# ── the guards: exact-first, and no guessing ──────────────────────────────────
def test_a_target_already_carrying_the_x_marker_is_left_alone():
    # Anything with `[x]` is either an exact key already or genuinely absent;
    # re-interpreting it could redirect a lookup that works.
    gen = _generator()
    slice_id = "Observation.component:back"
    assert (
        gen._resolve_sliced_choice_target(
            f"{slice_id}.value[x]:valueQuantity.value",
            _index(slice_id),
            {slice_id: _sliced_choice_field(slice_id)},
        )
        is None
    )


def test_an_unsliced_target_is_left_alone():
    gen = _generator()
    assert (
        gen._resolve_sliced_choice_target(
            "Observation.component.valueQuantity.value", {}, {}
        )
        is None
    )


def test_duplicate_element_records_collapse_to_one_key():
    # A slice can carry the same choice element twice (the parser keeps both the
    # snapshot child and a re-sliced copy). Both spell the same canonical key, so this
    # is not ambiguity and must resolve rather than be refused.
    #
    # The ambiguity branch in the resolver is defensive: two elements of one slice
    # would have to spell the *same* segment as *different* canonical keys, which the
    # `<base>` + `<concreteType>` decomposition makes unreachable for the shapes seen
    # so far. It is kept so a future shape reports instead of silently picking one.
    gen = _generator()
    slice_id = "Observation.component:back"
    owner = "Observation.component"
    child = {
        "path": f"{owner}.value[x]",
        "id": f"{slice_id}.value[x]",
        "type": [
            {
                "code": "Quantity",
                "type_structure": _quantity_children(f"{owner}.value[x]"),
            }
        ],
    }
    field = {"path": slice_id, "id": slice_id, "children": [child, dict(child)]}
    resolved = gen._resolve_sliced_choice_target(
        f"{slice_id}.valueQuantity.value", _index(slice_id), {slice_id: field}
    )
    assert resolved == f"{slice_id}.value[x]:valueQuantity.value"
    assert not gen.mapping_diagnostics
