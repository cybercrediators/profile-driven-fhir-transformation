"""Characterization of the legacy deterministic automapping path (WP3).

These tests describe what the current triple-flatten path *does*, not what it
should do. They exist so the WP3 candidate API can be layered on top without
changing a single deterministic selection, and so that a later `automapper-v2`
migration has an explicit record of the behaviour it replaces.

The path being frozen is:

    fml_map.flatten_profile_fields
        -> FMLMap._flatten_fields_recursively
            -> FMLAutomapper._flatten_target_fields   (the third flatten)

Several assertions below record behaviour that is arguably wrong (duplicate
candidates, synthetic `_virtual_path` values that can be emitted as real mapping
targets). They are deliberately pinned rather than fixed: WP3 explicitly defers
that change to a separately evaluated migration.
"""

from types import SimpleNamespace

import pytest

from mapping.fml_creator.fml_automapper import FMLAutomapper
from mapping.fml_map import StructureMapGenerator

pytestmark = pytest.mark.unit


def target_field(path, *, eid=None, description="", type_code="string", **extra):
    """A target field shaped like `conv_mappable` output."""

    field = {
        "path": path,
        "id": eid or path,
        "type": type_code,
        "cardinality": {"min": 0, "max": "1"},
        "description": description,
        "is_required": False,
        "fixed_value": [],
        "options": [],
    }
    field.update(extra)
    return field


def source_field(path, *, eid=None, type_code="string", **extra):
    field = {
        "path": path,
        "id": eid or path.split(".")[-1],
        "type": type_code,
    }
    field.update(extra)
    return field


@pytest.fixture
def mapper():
    # use_word2vec=False keeps scoring on the deterministic TF-IDF/Jaccard path;
    # the GloVe branch needs an optional dependency and a background load.
    return FMLAutomapper(use_word2vec=False)


# --- the third flatten -----------------------------------------------------


def test_third_flatten_keeps_only_text_bearing_keys(mapper):
    """Type, cardinality, and slice context are dropped before scoring.

    This is why the call sites can only emit a path: the matched dict no longer
    carries enough to validate the target. The WP3 candidate API has to carry
    provenance back to the original field to recover it.
    """
    fields = [
        target_field(
            "Patient.birthDate",
            type_code="date",
            sliceName="s",
            cardinality={"min": 1, "max": "1"},
        )
    ]

    flat = mapper._flatten_target_fields(fields)

    assert set(flat[0]) == {"id", "path", "description", "short", "definition", "_virtual_path"}
    assert "type" not in flat[0]
    assert "cardinality" not in flat[0]
    assert "sliceName" not in flat[0]


def test_top_level_virtual_path_equals_the_real_path(mapper):
    flat = mapper._flatten_target_fields([target_field("Patient.birthDate")])

    assert flat[0]["_virtual_path"] == "Patient.birthDate"
    assert flat[0]["path"] == "Patient.birthDate"


def test_nested_virtual_path_is_synthesised_from_the_parent_chain(mapper):
    """`_virtual_path` is built from parent + last segment, not from the real path."""

    fields = [
        target_field(
            "Patient.contact",
            type_code="BackboneElement",
            type_structure=[target_field("ContactPoint.value")],
        )
    ]

    flat = mapper._flatten_target_fields(fields)
    nested = flat[1]

    # The child's own path belongs to the datatype, not to the profile...
    assert nested["path"] == "ContactPoint.value"
    # ...while the synthetic path is rebased under the parent.
    assert nested["_virtual_path"] == "Patient.contact.value"


def test_third_flatten_recurses_into_type_structure_but_not_children(mapper):
    fields = [
        target_field(
            "Patient.name",
            type_structure=[target_field("HumanName.family")],
            children=[target_field("Patient.name.given")],
        )
    ]

    flat = mapper._flatten_target_fields(fields)

    assert [f["path"] for f in flat] == ["Patient.name", "HumanName.family"]


def test_pre_flattened_children_become_duplicate_candidates(mapper):
    """The characteristic duplication of the triple flatten.

    `_flatten_fields_recursively` has already hoisted `type_structure` entries to
    the top level, and `_flatten_target_fields` then recurses into them again. The
    same element therefore appears twice with two different synthetic paths.
    """
    child = target_field("HumanName.family")
    parent = target_field("Patient.name", type_structure=[child])
    # what `_flatten_fields_recursively` hands over: parent *and* hoisted child
    pre_flattened = [parent, child]

    flat = mapper._flatten_target_fields(pre_flattened)

    virtual_paths = [f["_virtual_path"] for f in flat]
    assert virtual_paths == [
        "Patient.name",
        "Patient.name.family",  # nested occurrence, rebased
        "HumanName.family",     # hoisted occurrence, unrebased
    ]
    # Both occurrences point at one real element.
    assert sum(1 for f in flat if f["path"] == "HumanName.family") == 2


# --- selection semantics ---------------------------------------------------


def test_exact_name_match_is_selected(mapper):
    targets = [target_field("Patient.gender"), target_field("Patient.birthDate")]

    match, score = mapper.find_mapping(source_field("src.birthDate"), targets)

    assert match["path"] == "Patient.birthDate"
    assert score >= mapper.threshold


def test_score_below_threshold_returns_no_match_and_zero(mapper):
    targets = [target_field("Patient.managingOrganization")]

    match, score = mapper.find_mapping(source_field("src.xyzzy"), targets)

    assert match is None
    # The score is reported as 0.0 on rejection, not as the score actually reached.
    assert score == 0.0


def test_threshold_is_inclusive(mapper):
    targets = [target_field("Patient.gender")]
    exact = source_field("src.gender")

    _, score = mapper.find_mapping(exact, targets)
    mapper.threshold = score

    match, _ = mapper.find_mapping(exact, targets)
    assert match is not None


def test_empty_target_list_is_not_an_error(mapper):
    assert mapper.find_mapping(source_field("src.gender"), []) == (None, 0.0)


def test_first_of_equally_scoring_targets_wins(mapper):
    """Ties resolve to the earliest candidate: every comparison is a strict `>`."""

    targets = [
        target_field("Patient.gender", eid="Patient.gender#a"),
        target_field("Patient.gender", eid="Patient.gender#b"),
    ]

    match, _ = mapper.find_mapping(source_field("src.gender"), targets)

    assert match["id"] == "Patient.gender#a"


def test_scoring_uses_the_virtual_path_not_the_real_path(mapper):
    """A nested candidate is scored on its synthetic path.

    This is what makes the duplicate occurrences behave differently: the rebased
    copy matches a source field named after the profile path, the hoisted copy
    matches one named after the datatype path.
    """
    child = target_field("HumanName.family")
    targets = [target_field("Patient.name", type_structure=[child])]

    match, score = mapper.find_mapping(source_field("src.nameFamily"), targets)

    assert match is not None
    assert match["_virtual_path"] == "Patient.name.family"
    assert score > 0


def test_selection_is_deterministic_across_repeated_calls(mapper):
    targets = [
        target_field("Patient.birthDate", description="date of birth"),
        target_field("Patient.deceasedDateTime", description="date of death"),
        target_field("Patient.gender"),
    ]
    src = source_field("src.birthDate")

    results = {mapper.find_mapping(src, targets)[0]["path"] for _ in range(5)}

    assert len(results) == 1


# --- what the call site emits ----------------------------------------------


def _generator(mapper):
    smg = object.__new__(StructureMapGenerator)
    smg.factory = SimpleNamespace(explicit_mapping_targets=set())
    smg.custom_mapping_table = None
    smg.automapper = mapper
    return smg


def test_emitted_target_is_the_real_path_when_it_differs_from_the_virtual_one(mapper):
    """`_get_automapped_mappings` emits **both** paths when they diverge.

    A nested candidate carries a real datatype path and a synthetic profile path.
    The call site cannot tell which one downstream needs, so it records both — the
    synthetic one becomes an addressable mapping target in its own right.
    """
    smg = _generator(mapper)
    child = target_field("HumanName.family")
    targets = [target_field("Patient.name", type_structure=[child])]

    paths, mappings = smg._get_automapped_mappings(
        None, mapper, [source_field("src.nameFamily")], targets, "Patient", "Patient"
    )

    assert paths == {"HumanName.family", "Patient.name.family"}
    assert mappings == {
        "HumanName.family": "nameFamily",
        "Patient.name.family": "nameFamily",
    }
    assert smg.factory.explicit_mapping_targets == paths


def test_only_one_target_is_emitted_when_the_paths_agree(mapper):
    smg = _generator(mapper)
    targets = [target_field("Patient.birthDate")]

    paths, mappings = smg._get_automapped_mappings(
        None, mapper, [source_field("src.birthDate")], targets, "Patient", "Patient"
    )

    assert paths == {"Patient.birthDate"}
    assert mappings == {"Patient.birthDate": "birthDate"}


def test_unmatched_source_fields_emit_nothing(mapper):
    smg = _generator(mapper)

    paths, mappings = smg._get_automapped_mappings(
        None,
        mapper,
        [source_field("src.xyzzy")],
        [target_field("Patient.managingOrganization")],
        "Patient",
        "Patient",
    )

    assert paths == set()
    assert mappings == {}
