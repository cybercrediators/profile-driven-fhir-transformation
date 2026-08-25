"""The WP3 candidate API over the frozen deterministic automapper.

Two properties matter here and are tested separately:

1. **Equivalence.** `find_candidates` must not perturb deterministic
   automapping. Its rank-1 candidate is the exact target `find_mapping` returns,
   and its flattened population is byte-identical to the legacy flatten.
2. **Exactness of the LLM-visible set.** Only candidates that resolve to one
   specific target-tree element may be offered to a model. Everything else stays
   in the list for deterministic compatibility but is withheld.
"""

import itertools
import random

import pytest

from mapping.fml_creator.fml_automapper import (
    EXCLUDED_CARDINALITY,
    EXCLUDED_DATATYPE,
    EXCLUDED_PROHIBITED,
    EXCLUDED_UNRESOLVED,
    RESOLUTION_NOT_ATTEMPTED,
    AutomapCandidate,
    FMLAutomapper,
)
from mapping.target_tree import (
    RESOLUTION_AMBIGUOUS_PATH,
    RESOLUTION_EXACT_ID,
    RESOLUTION_NOT_FOUND,
    RESOLUTION_UNIQUE_PATH,
    TargetTree,
)

pytestmark = pytest.mark.unit


# --- fixtures --------------------------------------------------------------


def target_field(path, *, eid=None, description="", type_code="string", **extra):
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
    field = {"path": path, "id": eid or path.split(".")[-1], "type": type_code}
    field.update(extra)
    return field


def snapshot(*elements):
    return {
        "resourceType": "StructureDefinition",
        "type": "Patient",
        "snapshot": {"element": list(elements)},
    }


def element(eid, path, *, minimum=0, maximum="1", types=("string",), slice_name=None):
    out = {
        "id": eid,
        "path": path,
        "min": minimum,
        "max": maximum,
        "type": [{"code": code} for code in types],
    }
    if slice_name:
        out["sliceName"] = slice_name
    return out


@pytest.fixture
def mapper():
    return FMLAutomapper(use_word2vec=False)


@pytest.fixture
def patient_tree():
    return TargetTree.from_snapshot(
        snapshot(
            element("Patient", "Patient", types=()),
            element("Patient.gender", "Patient.gender", types=("code",)),
            element("Patient.birthDate", "Patient.birthDate", types=("date",)),
            element("Patient.name", "Patient.name", maximum="*", types=("HumanName",)),
            element("Patient.name.family", "Patient.name.family", types=("string",)),
            element("Patient.deceasedBoolean", "Patient.deceasedBoolean", types=("boolean",)),
            element("Patient.photo", "Patient.photo", maximum="0", types=("Attachment",)),
        )
    )


# --- equivalence with the frozen legacy path --------------------------------


def test_provenance_flatten_produces_the_legacy_flatten_exactly(mapper):
    """The twin flatten must not drift from `_flatten_target_fields`."""

    fields = [
        target_field(
            "Patient.name",
            type_structure=[
                target_field("HumanName.family"),
                target_field(
                    "HumanName.period", type_structure=[target_field("Period.start")]
                ),
            ],
        ),
        target_field("Patient.gender"),
    ]

    legacy = mapper._flatten_target_fields(fields)
    twin = [flat for flat, _ in mapper._flatten_target_candidates(fields)]

    assert twin == legacy


def test_provenance_flatten_pairs_each_entry_with_its_source_field(mapper):
    child = target_field("HumanName.family", type_code="string")
    parent = target_field("Patient.name", type_code="HumanName", type_structure=[child])

    pairs = mapper._flatten_target_candidates([parent])

    assert pairs[0][1] is parent
    assert pairs[1][1] is child


def test_per_target_tfidf_scores_agree_with_the_legacy_winner(mapper):
    targets = [
        target_field("Patient.gender", description="administrative gender"),
        target_field("Patient.birthDate", description="the date of birth"),
        target_field("Patient.name"),
    ]
    flat = mapper._flatten_target_fields(targets)

    for name in ("gender", "birthDate", "name", "unrelated"):
        src = source_field(f"src.{name}")
        scores = mapper._tfidf_scores(src, flat)
        legacy_match, legacy_score = mapper._find_match_vectorized(src, flat)

        best_index, best_score = -1, 0.0
        for index, score in enumerate(scores):
            if score > best_score:
                best_index, best_score = index, score

        if legacy_match is None:
            assert best_index == -1
        else:
            assert flat[best_index] == legacy_match
            assert best_score == pytest.approx(legacy_score)


def test_weighted_components_reconstruct_the_legacy_weighted_score(mapper):
    targets = [target_field("Patient.birthDate", description="date of birth")]
    flat = mapper._flatten_target_fields(targets)
    src = source_field("src.birthDate")

    row = mapper._score_table(src, flat)[0]
    legacy = mapper._calculate_weighted_score(
        mapper._extract_features(src), mapper._extract_features(flat[0])
    )

    assert row.weighted == pytest.approx(legacy)
    assert row.weighted == pytest.approx(
        0.5 * row.name_sim + 0.3 * row.path_sim + 0.2 * row.desc_sim
    )


CORPUS = [
    target_field("Patient.gender", description="administrative gender"),
    target_field("Patient.birthDate", description="the date of birth"),
    target_field("Patient.deceasedBoolean", description="indicates if deceased"),
    target_field(
        "Patient.name",
        description="a name associated with the patient",
        type_structure=[
            target_field("HumanName.family", description="family name"),
            target_field("HumanName.given", description="given names"),
        ],
    ),
    target_field("Patient.managingOrganization", description="organization"),
    # duplicate occurrences, as `_flatten_fields_recursively` hoists them
    target_field("HumanName.family", description="family name"),
    target_field("HumanName.given", description="given names"),
]

SOURCE_NAMES = [
    "gender",
    "birthDate",
    "familyName",
    "given",
    "deceased",
    "organization",
    "name",
    "xyzzy",
    "family",
    "patient_gender",
]


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_rank_one_candidate_is_exactly_the_legacy_selection(mapper, name):
    src = source_field(f"src.{name}")

    legacy_match, legacy_score = mapper.find_mapping(src, CORPUS)
    candidates = mapper.find_candidates(src, CORPUS, top_k=None)
    winner = next((c for c in candidates if c.is_legacy_selection), None)

    if legacy_match is None:
        # Below threshold: no candidate may claim to be the legacy selection.
        assert winner is None or not winner.meets_legacy_threshold
        return

    assert winner is not None
    assert winner is candidates[0]
    assert winner.legacy_emitted_path == (
        legacy_match.get("path") or legacy_match.get("_virtual_path")
    )
    assert winner.legacy_virtual_path == legacy_match.get("_virtual_path")
    assert winner.scores.legacy == pytest.approx(legacy_score)
    assert winner.meets_legacy_threshold


@pytest.mark.parametrize("seed", range(12))
def test_equivalence_holds_on_shuffled_corpora(mapper, seed):
    """Order changes the legacy tie-break, so equivalence must survive shuffling."""

    rng = random.Random(seed)
    targets = CORPUS[:]
    rng.shuffle(targets)
    src = source_field(f"src.{rng.choice(SOURCE_NAMES)}")

    legacy_match, _ = mapper.find_mapping(src, targets)
    candidates = mapper.find_candidates(src, targets, top_k=None)

    if legacy_match is None:
        assert all(not c.meets_legacy_threshold for c in candidates)
        return

    assert candidates[0].is_legacy_selection
    assert candidates[0].scoring_path == legacy_match.get(
        "_virtual_path", legacy_match.get("path")
    )


def test_tie_resolves_to_the_same_candidate_as_find_mapping(mapper):
    targets = [
        target_field("Patient.gender", eid="Patient.gender#a"),
        target_field("Patient.gender", eid="Patient.gender#b"),
    ]
    src = source_field("src.gender")

    legacy_match, _ = mapper.find_mapping(src, targets)
    candidates = mapper.find_candidates(src, targets, top_k=None)

    assert candidates[0].element_id == legacy_match["id"] == "Patient.gender#a"


def test_candidates_are_produced_below_the_threshold(mapper):
    """LLM reranking needs the population, not only the accepted selection."""

    src = source_field("src.xyzzy")

    assert mapper.find_mapping(src, CORPUS)[0] is None
    candidates = mapper.find_candidates(src, CORPUS, top_k=None)
    assert candidates
    assert all(not c.meets_legacy_threshold for c in candidates)


def test_empty_target_list_yields_no_candidates(mapper):
    assert mapper.find_candidates(source_field("src.gender"), []) == []


# --- candidate shape and identity ------------------------------------------


def test_candidate_ids_are_unique_and_stable(mapper):
    src = source_field("src.family")

    first = mapper.find_candidates(src, CORPUS, top_k=None)
    second = mapper.find_candidates(src, CORPUS, top_k=None)

    ids = [c.candidate_id for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c.candidate_id for c in second]


def test_candidate_id_survives_unrelated_corpus_changes(mapper):
    """Identity is content-derived, so adding a target must not renumber others."""

    src = source_field("src.gender")
    before = {
        c.scoring_path: c.candidate_id
        for c in mapper.find_candidates(src, CORPUS, top_k=None)
    }

    extended = [target_field("Patient.active", description="whether active")] + CORPUS
    after = {
        c.scoring_path: c.candidate_id
        for c in mapper.find_candidates(src, extended, top_k=None)
    }

    for path, candidate_id in before.items():
        assert after[path] == candidate_id


def test_byte_identical_duplicates_are_disambiguated_by_ordinal(mapper):
    duplicate = target_field("Patient.gender")
    candidates = mapper.find_candidates(
        source_field("src.gender"), [duplicate, dict(duplicate)], top_k=None
    )

    ids = sorted(c.candidate_id for c in candidates)
    assert len(set(ids)) == 2
    assert ids[1] == f"{ids[0]}-2"


def test_the_three_paths_are_reported_separately(mapper):
    child = target_field("HumanName.family")
    targets = [target_field("Patient.name", type_structure=[child])]

    nested = next(
        c
        for c in mapper.find_candidates(source_field("src.family"), targets, top_k=None)
        if c.element_path == "HumanName.family"
    )

    assert nested.scoring_path == "Patient.name.family"
    assert nested.legacy_virtual_path == "Patient.name.family"
    assert nested.legacy_emitted_path == "HumanName.family"


def test_validation_context_lost_by_the_flatten_is_recovered(mapper):
    targets = [
        target_field(
            "Patient.name",
            eid="Patient.name:official",
            type_code="HumanName",
            cardinality={"min": 1, "max": "*"},
            sliceName="official",
        )
    ]

    candidate = mapper.find_candidates(source_field("src.name"), targets, top_k=None)[0]

    assert candidate.element_id == "Patient.name:official"
    assert candidate.element_path == "Patient.name"
    assert candidate.types == ("HumanName",)
    assert candidate.min == 1
    assert candidate.max == "*"
    assert candidate.slice_name == "official"


def test_choice_types_are_carried_as_multiple_types(mapper):
    targets = [
        target_field(
            "Patient.deceased[x]",
            type_code="choice",
            choice_types=["boolean", "dateTime"],
        )
    ]

    candidate = mapper.find_candidates(source_field("src.deceased"), targets, top_k=None)[0]

    assert candidate.types == ("boolean", "dateTime")


def test_top_k_truncates_after_ordering(mapper):
    src = source_field("src.family")

    full = mapper.find_candidates(src, CORPUS, top_k=None)
    limited = mapper.find_candidates(src, CORPUS, top_k=3)

    assert limited == full[:3]
    assert len(limited) == 3


def test_tail_is_ordered_by_descending_score_then_index(mapper):
    candidates = mapper.find_candidates(source_field("src.family"), CORPUS, top_k=None)
    tail = candidates[1:]

    keys = [(-c.scores.legacy, c.flat_index) for c in tail]
    assert keys == sorted(keys)


def test_no_tree_means_nothing_is_offered_to_a_model(mapper):
    candidates = mapper.find_candidates(source_field("src.gender"), CORPUS, top_k=None)

    assert all(c.resolution_status == RESOLUTION_NOT_ATTEMPTED for c in candidates)
    assert all(not c.llm_visible for c in candidates)
    assert all(c.canonical_target_id is None for c in candidates)


# --- canonical resolution ---------------------------------------------------


def test_element_id_resolves_exactly(mapper, patient_tree):
    targets = [target_field("Patient.gender", eid="Patient.gender", type_code="code")]

    candidate = mapper.find_candidates(
        source_field("src.gender", type_code="code"), targets, target_tree=patient_tree
    )[0]

    assert candidate.resolution_status == RESOLUTION_EXACT_ID
    assert candidate.resolution_key == "element-id"
    assert candidate.canonical_target_id == "Patient.gender"
    assert candidate.canonical_target_path == "Patient.gender"
    assert candidate.llm_visible


def test_synthetic_virtual_path_resolves_when_the_tree_contains_it(mapper, patient_tree):
    """The rebased path is a legitimate key when it names one real element."""

    child = target_field("HumanName.family")
    targets = [target_field("Patient.name", type_code="HumanName", type_structure=[child])]

    nested = next(
        c
        for c in mapper.find_candidates(
            source_field("src.family"), targets, top_k=None, target_tree=patient_tree
        )
        if c.element_path == "HumanName.family"
    )

    # The key that resolved is the synthetic path; the status reports *how* it
    # matched, and here that path is itself an ElementDefinition.id in the tree.
    assert nested.resolution_key == "virtual-path"
    assert nested.resolution_status == RESOLUTION_EXACT_ID
    assert nested.canonical_target_id == "Patient.name.family"
    assert nested.llm_visible


def test_unique_path_match_resolves_when_the_id_differs_from_the_path(mapper):
    tree = TargetTree.from_snapshot(
        snapshot(
            element("Patient", "Patient", types=()),
            element(
                "Patient.identifier:mrn",
                "Patient.identifier",
                types=("Identifier",),
                slice_name="mrn",
            ),
        )
    )
    targets = [target_field("Patient.identifier", eid="unknown-id")]

    candidate = mapper.find_candidates(
        source_field("src.identifier"), targets, target_tree=tree
    )[0]

    assert candidate.resolution_status == RESOLUTION_UNIQUE_PATH
    assert candidate.canonical_target_id == "Patient.identifier:mrn"
    assert candidate.llm_visible


def test_unresolvable_candidate_is_kept_but_withheld(mapper, patient_tree):
    """Deterministic mode still sees it; the model does not."""

    targets = [target_field("Contact.telecom", eid="Contact.telecom")]

    candidate = mapper.find_candidates(
        source_field("src.telecom"), targets, target_tree=patient_tree
    )[0]

    assert candidate.resolution_status == RESOLUTION_NOT_FOUND
    assert candidate.exclusion_reason == EXCLUDED_UNRESOLVED
    assert not candidate.llm_visible
    assert candidate.canonical_target_id is None
    # still present for deterministic compatibility
    assert candidate.legacy_emitted_path == "Contact.telecom"


def test_ambiguous_path_is_withheld_rather_than_resolved_to_the_first(mapper):
    """Two slices share a path and the unsliced element is not in the snapshot."""

    tree = TargetTree.from_snapshot(
        snapshot(
            element("Patient", "Patient", types=()),
            element(
                "Patient.identifier:mrn",
                "Patient.identifier",
                types=("Identifier",),
                slice_name="mrn",
            ),
            element(
                "Patient.identifier:ssn",
                "Patient.identifier",
                types=("Identifier",),
                slice_name="ssn",
            ),
        )
    )
    targets = [target_field("Patient.identifier", eid="unknown-id")]

    candidate = mapper.find_candidates(
        source_field("src.identifier"), targets, target_tree=tree
    )[0]

    assert candidate.resolution_status == RESOLUTION_AMBIGUOUS_PATH
    assert candidate.exclusion_reason == EXCLUDED_UNRESOLVED
    assert not candidate.llm_visible


def test_canonical_annotation_overrides_stale_flatten_context(mapper, patient_tree):
    """The tree is authoritative for the constraints used to validate a target."""

    targets = [
        target_field(
            "Patient.name",
            eid="Patient.name",
            type_code="string",  # deliberately wrong on the field
            cardinality={"min": 0, "max": "1"},
        )
    ]

    candidate = mapper.find_candidates(
        source_field("src.name"), targets, target_tree=patient_tree
    )[0]

    assert candidate.types == ("HumanName",)
    assert candidate.max == "*"


# --- LLM-visibility pruning -------------------------------------------------


def test_prohibited_target_is_pruned(mapper, patient_tree):
    targets = [target_field("Patient.photo", eid="Patient.photo", type_code="Attachment")]

    candidate = mapper.find_candidates(
        source_field("src.photo"), targets, target_tree=patient_tree
    )[0]

    assert candidate.exclusion_reason == EXCLUDED_PROHIBITED
    assert not candidate.llm_visible


def test_repeating_source_into_single_target_is_pruned(mapper, patient_tree):
    targets = [target_field("Patient.gender", eid="Patient.gender", type_code="code")]
    src = source_field("src.gender", type_code="code", cardinality={"min": 0, "max": "*"})

    candidate = mapper.find_candidates(src, targets, target_tree=patient_tree)[0]

    assert candidate.exclusion_reason == EXCLUDED_CARDINALITY
    assert not candidate.llm_visible


def test_repeating_source_into_repeating_target_is_kept(mapper, patient_tree):
    targets = [target_field("Patient.name", eid="Patient.name", type_code="HumanName")]
    src = source_field("src.name", type_code="HumanName", cardinality={"min": 0, "max": "*"})

    candidate = mapper.find_candidates(src, targets, target_tree=patient_tree)[0]

    assert candidate.llm_visible


def test_mutually_exclusive_primitive_families_are_pruned(mapper, patient_tree):
    targets = [
        target_field("Patient.birthDate", eid="Patient.birthDate", type_code="date")
    ]
    src = source_field("src.birthDate", type_code="boolean")

    candidate = mapper.find_candidates(src, targets, target_tree=patient_tree)[0]

    assert candidate.exclusion_reason == EXCLUDED_DATATYPE
    assert not candidate.llm_visible


@pytest.mark.parametrize(
    "source_type,target_eid,target_type",
    [
        # text converts to everything: a source system carries dates as strings
        ("string", "Patient.birthDate", "date"),
        ("code", "Patient.deceasedBoolean", "boolean"),
        # same family
        ("dateTime", "Patient.birthDate", "date"),
        # unknown/complex types never prune
        ("HumanName", "Patient.name", "HumanName"),
    ],
)
def test_datatype_prune_stays_narrow(
    mapper, patient_tree, source_type, target_eid, target_type
):
    targets = [target_field(target_eid, eid=target_eid, type_code=target_type)]
    src = source_field("src.value", type_code=source_type)

    candidate = mapper.find_candidates(src, targets, target_tree=patient_tree)[0]

    assert candidate.exclusion_reason != EXCLUDED_DATATYPE


def test_a_choice_target_is_kept_when_any_type_is_compatible(mapper):
    tree = TargetTree.from_snapshot(
        snapshot(
            element("Patient", "Patient", types=()),
            element(
                "Patient.deceased[x]",
                "Patient.deceased[x]",
                types=("boolean", "dateTime"),
            ),
        )
    )
    targets = [target_field("Patient.deceased[x]", eid="Patient.deceased[x]")]
    src = source_field("src.deceasedDate", type_code="dateTime")

    candidate = mapper.find_candidates(src, targets, target_tree=tree)[0]

    assert candidate.llm_visible


def test_llm_visible_subset_is_a_strict_subset_of_the_full_population(
    mapper, patient_tree
):
    src = source_field("src.family")

    candidates = mapper.find_candidates(
        src, CORPUS, top_k=None, target_tree=patient_tree
    )
    visible = [c for c in candidates if c.llm_visible]

    assert visible
    assert len(visible) < len(candidates)
    # nothing was dropped from the deterministic view
    assert len(candidates) == len(mapper.find_candidates(src, CORPUS, top_k=None))
    assert all(c.canonical_target_id for c in visible)
    assert all(c.exclusion_reason is None for c in visible)


def test_annotation_does_not_change_the_legacy_selection(mapper, patient_tree):
    """Pruning is an LLM-side filter, never a change to deterministic ranking."""

    for name in SOURCE_NAMES:
        src = source_field(f"src.{name}")
        plain = mapper.find_candidates(src, CORPUS, top_k=None)
        annotated = mapper.find_candidates(
            src, CORPUS, top_k=None, target_tree=patient_tree
        )

        assert [c.candidate_id for c in plain] == [c.candidate_id for c in annotated]
        assert [c.is_legacy_selection for c in plain] == [
            c.is_legacy_selection for c in annotated
        ]


def test_candidate_order_is_stable_for_identical_input(mapper, patient_tree):
    src = source_field("src.family")

    runs = [
        tuple(
            (c.candidate_id, c.llm_visible)
            for c in mapper.find_candidates(
                src, CORPUS, top_k=None, target_tree=patient_tree
            )
        )
        for _ in range(5)
    ]

    assert len(set(runs)) == 1


def test_candidate_is_hashable_and_comparable(mapper):
    """Frozen dataclass: a run report can put candidates in a set or dict."""

    candidates = mapper.find_candidates(source_field("src.gender"), CORPUS, top_k=2)

    assert isinstance(candidates[0], AutomapCandidate)
    assert len(set(candidates)) == 2
    assert candidates[0] != candidates[1]


def test_no_pair_of_llm_visible_candidates_shares_a_canonical_target(
    mapper, patient_tree
):
    """The duplicate occurrences must not become two ways to say the same thing."""

    visible = mapper.find_llm_candidates(
        source_field("src.family"), CORPUS, patient_tree, top_k=None
    )

    canonical_ids = [c.canonical_target_id for c in visible]
    duplicates = [
        target_id
        for target_id, group in itertools.groupby(sorted(canonical_ids))
        if len(list(group)) > 1
    ]
    assert not duplicates, f"duplicate canonical targets offered to the model: {duplicates}"


# --- compiler-addressable target identity ----------------------------------


def test_slice_qualifier_survives_in_the_mapping_target_path(mapper):
    """A snapshot path drops `:sliceName`; the mapping table needs it.

    Authored tables in this repository address slices as
    `Procedure.code.coding:sct.code`, so emitting the FHIR path would silently
    map to the unsliced element instead.
    """
    tree = TargetTree.from_snapshot(
        snapshot(
            element("Patient", "Patient", types=()),
            element(
                "Patient.identifier:mrn",
                "Patient.identifier",
                types=("Identifier",),
                slice_name="mrn",
            ),
        )
    )
    targets = [target_field("Patient.identifier", eid="Patient.identifier:mrn")]

    candidate = mapper.find_candidates(
        source_field("src.identifier"), targets, target_tree=tree
    )[0]

    assert candidate.mapping_target_path == "Patient.identifier:mrn"
    assert candidate.canonical_target_id == "Patient.identifier:mrn"
    # retained as metadata only, and demonstrably lossy
    assert candidate.canonical_target_path == "Patient.identifier"
    assert candidate.llm_visible


def test_choice_qualifier_survives_in_the_mapping_target_path(mapper):
    tree = TargetTree.from_snapshot(
        snapshot(
            element("Procedure", "Procedure", types=()),
            element(
                "Procedure.performed[x]:performedPeriod",
                "Procedure.performed[x]",
                types=("Period",),
                slice_name="performedPeriod",
            ),
        )
    )
    targets = [
        target_field(
            "Procedure.performed[x]", eid="Procedure.performed[x]:performedPeriod"
        )
    ]

    candidate = mapper.find_candidates(
        source_field("src.performed"), targets, target_tree=tree
    )[0]

    assert candidate.mapping_target_path == "Procedure.performed[x]:performedPeriod"
    assert candidate.canonical_target_path == "Procedure.performed[x]"


def test_two_slices_of_one_element_are_distinguishable_targets(mapper):
    """The regression this guards: both slices collapsing onto one path."""

    tree = TargetTree.from_snapshot(
        snapshot(
            element("Procedure", "Procedure", types=()),
            element("Procedure.code", "Procedure.code", types=("CodeableConcept",)),
            element(
                "Procedure.code.coding:sct",
                "Procedure.code.coding",
                types=("Coding",),
                slice_name="sct",
            ),
            element(
                "Procedure.code.coding:ops",
                "Procedure.code.coding",
                types=("Coding",),
                slice_name="ops",
            ),
        )
    )
    targets = [
        target_field("Procedure.code.coding", eid="Procedure.code.coding:sct"),
        target_field("Procedure.code.coding", eid="Procedure.code.coding:ops"),
    ]

    visible = mapper.find_llm_candidates(
        source_field("src.code"), targets, tree, top_k=None
    )

    assert {c.mapping_target_path for c in visible} == {
        "Procedure.code.coding:sct",
        "Procedure.code.coding:ops",
    }
    # ...whereas the FHIR path cannot tell them apart at all
    assert {c.canonical_target_path for c in visible} == {"Procedure.code.coding"}


def test_mapping_target_path_is_absent_without_resolution(mapper):
    candidate = mapper.find_candidates(source_field("src.gender"), CORPUS)[0]

    assert candidate.mapping_target_path is None


# --- the LLM-visible selection ---------------------------------------------


def test_llm_candidates_do_not_waste_slots_on_withheld_targets(mapper, patient_tree):
    """A pruned rank-one candidate must not consume the only slot."""

    targets = [
        # rank 1 for "photo", but prohibited by the profile
        target_field("Patient.photo", eid="Patient.photo", type_code="Attachment"),
        target_field("Patient.gender", eid="Patient.gender", type_code="code"),
    ]
    src = source_field("src.photo")

    raw = mapper.find_candidates(src, targets, top_k=1, target_tree=patient_tree)
    assert raw[0].exclusion_reason == EXCLUDED_PROHIBITED
    assert not raw[0].llm_visible

    offered = mapper.find_llm_candidates(src, targets, patient_tree, top_k=1)
    assert len(offered) == 1
    assert offered[0].llm_visible
    assert offered[0].mapping_target_path == "Patient.gender"


def test_llm_candidates_deduplicate_by_addressable_target(mapper, patient_tree):
    """One element reached twice by the triple flatten is offered once."""

    child = target_field("HumanName.family")
    targets = [
        target_field("Patient.name", type_code="HumanName", type_structure=[child]),
        # the hoisted occurrence, addressed by its profile id this time
        target_field("Patient.name.family", eid="Patient.name.family"),
    ]

    offered = mapper.find_llm_candidates(
        source_field("src.family"), targets, patient_tree, top_k=None
    )
    family = [c for c in offered if c.mapping_target_path == "Patient.name.family"]

    assert len(family) == 1


def test_deduplication_keeps_the_highest_ranked_occurrence(mapper, patient_tree):
    child = target_field("HumanName.family")
    targets = [
        target_field("Patient.name", type_code="HumanName", type_structure=[child]),
        target_field("Patient.name.family", eid="Patient.name.family"),
    ]
    src = source_field("src.family")

    ordered = mapper.find_candidates(src, targets, top_k=None, target_tree=patient_tree)
    expected = next(
        c for c in ordered if c.llm_visible and c.mapping_target_path == "Patient.name.family"
    )
    offered = mapper.find_llm_candidates(src, targets, patient_tree, top_k=None)

    assert expected.candidate_id in {c.candidate_id for c in offered}


def test_llm_candidates_preserve_relative_ranking(mapper, patient_tree):
    src = source_field("src.family")

    ordered = mapper.find_candidates(src, CORPUS, top_k=None, target_tree=patient_tree)
    offered = mapper.find_llm_candidates(src, CORPUS, patient_tree, top_k=None)

    positions = [
        next(i for i, c in enumerate(ordered) if c.candidate_id == picked.candidate_id)
        for picked in offered
    ]
    assert positions == sorted(positions)


def test_llm_candidates_return_nothing_when_every_target_is_withheld(mapper, patient_tree):
    targets = [target_field("Unknown.thing", eid="Unknown.thing")]

    assert mapper.find_llm_candidates(source_field("src.thing"), targets, patient_tree) == []


def test_llm_candidates_respect_top_k_after_filtering(mapper, patient_tree):
    src = source_field("src.family")

    all_visible = mapper.find_llm_candidates(src, CORPUS, patient_tree, top_k=None)
    limited = mapper.find_llm_candidates(src, CORPUS, patient_tree, top_k=2)

    assert len(all_visible) >= 2
    assert limited == all_visible[:2]


def test_find_candidates_returns_the_complete_population_by_default(mapper):
    """The audit view must not silently truncate."""

    flat = mapper._flatten_target_fields(CORPUS)

    assert len(mapper.find_candidates(source_field("src.family"), CORPUS)) == len(flat)


# --- the embedding model must settle before scoring -------------------------


class _SlowLoader:
    """Stands in for the background GloVe load, flipping `model_ready` on join."""

    def __init__(self, mapper):
        self.mapper = mapper
        self.joined = False

    def is_alive(self):
        return not self.joined

    def join(self):
        self.joined = True
        # what `_load_model` does on success: the similarity function switches
        # from Jaccard to embeddings from this point on
        self.mapper.model_ready = True
        self.mapper.model = {}


@pytest.fixture
def pending_mapper():
    """A mapper configured exactly as the pipeline configures it: GloVe pending."""

    mapper = FMLAutomapper(use_word2vec=False)
    mapper.use_word2vec = True
    mapper.model_ready = False
    mapper.load_thread = _SlowLoader(mapper)
    return mapper


def test_find_mapping_waits_for_the_embedding_model(pending_mapper):
    pending_mapper.find_mapping(source_field("src.gender"), CORPUS)

    assert pending_mapper.load_thread.joined


def test_find_candidates_waits_for_the_embedding_model(pending_mapper):
    """Without this, one candidate table could mix Jaccard and embedding scores.

    `_compute_similarity` re-reads `model_ready` on every call, so a load that
    completes midway through `_score_table` would score the first candidates one
    way and the rest another.
    """
    pending_mapper.find_candidates(source_field("src.gender"), CORPUS)

    assert pending_mapper.load_thread.joined


def test_model_state_cannot_change_during_a_candidate_table(pending_mapper):
    observed = []
    original = FMLAutomapper._compute_similarity

    def spy(self, tokens1, tokens2):
        observed.append(self.model_ready)
        return original(self, tokens1, tokens2)

    pending_mapper._compute_similarity = spy.__get__(pending_mapper)
    pending_mapper.find_candidates(source_field("src.gender"), CORPUS)

    assert observed, "no similarity comparison was made"
    assert len(set(observed)) == 1, "scoring mode changed mid-table"
