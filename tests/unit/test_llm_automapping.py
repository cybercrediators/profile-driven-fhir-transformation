"""Optional LLM automapping mode (WP4).

The property under test throughout: the model reranks, it does not author. Every
target it can name was produced and canonically resolved by the deterministic
matcher, so a hallucinated path has no way into the mapping table.

No test contacts a provider; a scripted fake backend stands in for one.
"""

import json

import pytest

from llm.errors import LLMConfigurationError
from llm.models import LLMResponse, LLMUsage, ModelSettings, StructuredOutputMode
from mapping.fml_creator.fml_automapper import FMLAutomapper
from mapping.fml_creator.fml_helper import rebase_to_resource_identity
from mapping.fml_creator.llm_automapping import (
    UNMAPPED,
    CandidateSelection,
    LLMAutomappingStrategy,
    ResourceTargets,
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


PATIENT_SNAPSHOT = {
    "resourceType": "StructureDefinition",
    "type": "Patient",
    "snapshot": {
        "element": [
            element("Patient", "Patient", types=()),
            element("Patient.gender", "Patient.gender", types=("code",)),
            element("Patient.birthDate", "Patient.birthDate", types=("date",)),
            element(
                "Patient.identifier:mrn",
                "Patient.identifier",
                types=("Identifier",),
                slice_name="mrn",
            ),
            element(
                "Patient.photo", "Patient.photo", maximum="0", types=("Attachment",)
            ),
        ]
    },
}

PATIENT_TARGETS = [
    target_field(
        "Patient.gender",
        eid="Patient.gender",
        type_code="code",
        description="administrative gender",
    ),
    target_field(
        "Patient.birthDate",
        eid="Patient.birthDate",
        type_code="date",
        description="the date of birth",
    ),
    target_field(
        "Patient.identifier",
        eid="Patient.identifier:mrn",
        type_code="Identifier",
        description="medical record number",
    ),
    target_field(
        "Patient.photo",
        eid="Patient.photo",
        type_code="Attachment",
        description="image of the patient",
    ),
]


class ScriptedBackend:
    """Fake LLM client returning canned selections, keyed by source field id."""

    provider = "openai-compatible"

    def __init__(self, answers=None, default=UNMAPPED, cache_hits=()):
        self.answers = answers or {}
        self.default = default
        self.cache_hits = set(cache_hits)
        self.calls = []

    def complete(
        self, *, messages, response_model, settings, feature="llm", context=None
    ):
        source_id = (context or {}).get("source_id")
        self.calls.append(
            {
                "source_id": source_id,
                "feature": feature,
                "prompt": messages[-1].content,
                "context": context,
                "settings": settings,
            }
        )
        answer = self.answers.get(source_id, self.default)
        if callable(answer):
            answer = answer(len([c for c in self.calls if c["source_id"] == source_id]))
        value = response_model(candidate_id=answer, confidence=0.9, reason="scripted")
        return LLMResponse(
            value=value,
            raw_text=value.model_dump_json(),
            provider=self.provider,
            model=settings.model,
            usage=LLMUsage(total_tokens=10),
            cache_hit=source_id in self.cache_hits,
        )


class ExplodingBackend:
    provider = "openai-compatible"

    def __init__(self, error):
        self.error = error

    def complete(self, **_):
        raise self.error


def make_strategy(backend, **kwargs):
    return LLMAutomappingStrategy(
        FMLAutomapper(use_word2vec=False),
        backend,
        ModelSettings(
            model="test-model", structured_output=StructuredOutputMode.JSON_SCHEMA
        ),
        **kwargs,
    )


def patient_resource(res_id="example-patient", targets=None, snapshot=None):
    return ResourceTargets(
        res_type="Patient",
        res_id=res_id,
        target_fields=targets if targets is not None else PATIENT_TARGETS,
        profile=snapshot if snapshot is not None else PATIENT_SNAPSHOT,
    )


# --- selection --------------------------------------------------------------


def test_selected_candidate_becomes_a_mapping_table_entry():
    backend = ScriptedBackend(answers={"gender": "c1"})
    strategy = make_strategy(backend)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )

    assert table == {"gender": "example-patient.gender"}


def test_unmapped_answer_produces_no_entry():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )

    assert table == {}


def test_slice_qualifier_reaches_the_mapping_table():
    """The WP3 fix has to survive all the way into the emitted target."""

    backend = ScriptedBackend(answers={"mrn": lambda _: "c1"})
    strategy = make_strategy(backend, top_k=1, second_pass_top_k=None)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.mrn", type_code="Identifier")]
    )

    assert table == {"mrn": "example-patient.identifier:mrn"}


def test_target_is_rebased_onto_the_profile_identity():
    backend = ScriptedBackend(answers={"gender": "c1"})
    strategy = make_strategy(backend)

    table = strategy.build_mapping_table(
        [patient_resource(res_id="my-profile")], [source_field("src.gender")]
    )

    assert table["gender"].startswith("my-profile.")


def test_resource_type_prefix_is_kept_when_there_is_no_distinct_identity():
    backend = ScriptedBackend(answers={"gender": "c1"})
    strategy = make_strategy(backend)

    table = strategy.build_mapping_table(
        [patient_resource(res_id="Patient")], [source_field("src.gender")]
    )

    assert table == {"gender": "Patient.gender"}


# --- the model cannot author a target ---------------------------------------


def test_an_unknown_candidate_id_is_rejected_not_repaired():
    backend = ScriptedBackend(answers={"gender": "c99"})
    strategy = make_strategy(backend, second_pass_top_k=None)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )
    report = strategy.report(table)

    assert table == {}
    assert report["summary"]["rejected_selections"] == 1
    assert report["fields"][0]["attempts"][0]["rejected_reason"].startswith(
        "unknown-candidate-id:"
    )


def test_a_free_text_path_is_rejected():
    """The answer space is offer ids; a path is simply not one of them."""

    backend = ScriptedBackend(answers={"gender": "Patient.gender"})
    strategy = make_strategy(backend, second_pass_top_k=None)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )

    assert table == {}


def test_a_hallucinated_path_never_reaches_the_table():
    backend = ScriptedBackend(answers={"gender": "Patient.thisDoesNotExist"})
    strategy = make_strategy(backend, second_pass_top_k=None)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )

    assert "Patient.thisDoesNotExist" not in table.values()
    assert table == {}


def test_a_leaf_below_an_unexpanded_complex_element_is_still_offered():
    """The bug this pins cost the model every name field in a real project.

    A profile snapshot stops at the element it constrains: a minimal Patient
    lists ``Patient.name`` and nothing beneath it. The candidate list, however,
    comes from the type-introspected flatten, so ``Patient.name.given`` is a
    candidate — and if the target tree is built from the snapshot alone it
    resolves against nothing, is marked invisible, and is never shown. The model
    then answers ``unmapped`` for a field the deterministic matcher maps
    correctly, and looks worse than it is.
    """

    snapshot = {
        "resourceType": "StructureDefinition",
        "type": "Patient",
        "snapshot": {
            "element": [
                element("Patient", "Patient", types=()),
                # Constrained, but not expanded — exactly what a real profile does.
                element("Patient.name", "Patient.name", maximum="*", types=("HumanName",)),
            ]
        },
    }
    targets = [
        target_field("Patient.name", eid="Patient.name", type_code="HumanName"),
        target_field("Patient.name.given", eid="Patient.name.given"),
        target_field("Patient.name.family", eid="Patient.name.family"),
    ]
    backend = ScriptedBackend()
    strategy = make_strategy(backend)

    strategy.build_mapping_table(
        [patient_resource(targets=targets, snapshot=snapshot)],
        [source_field("src.givenName")],
    )

    assert backend.calls, "the model was not asked at all"
    offered = backend.calls[0]["prompt"]
    # Rebased onto the profile identity, as every offer is.
    assert "example-patient.name.given" in offered
    assert "example-patient.name.family" in offered
    # And it is the *top* offer, which is the point: the deterministic scorer
    # always ranked it highly; the model was simply never shown it.
    assert offered.index("name.given") < offered.index("name.family")


def test_each_profiles_deterministic_winner_survives_the_top_k_cut():
    """Scores are only comparable within one profile, and the pool is not.

    Candidates from every profile compete for the same handful of slots, so a
    field whose correct target lives in a small profile can have that target
    buried under unrelated candidates from a larger one. On a real
    three-profile project ``encounterId`` was shown fifteen candidates and the
    deterministic winner — the correct answer — was not among them, from any
    profile. The model answered ``unmapped`` and said exactly why.

    The premise of this mode is that the model reranks *what the deterministic
    matcher found*; that only holds if what the matcher would have selected is
    on the list.
    """

    # Ten candidates that all mention the source term, so they own every slot
    # a pure score ordering has to give away.
    wide_targets = [
        target_field(
            f"Patient.filler{index}",
            eid=f"Patient.filler{index}",
            description="gender gender gender",
        )
        for index in range(10)
    ]
    wide_snapshot = {
        "resourceType": "StructureDefinition",
        "type": "Patient",
        "snapshot": {
            "element": [element("Patient", "Patient", types=())]
            + [
                element(f"Patient.filler{index}", f"Patient.filler{index}")
                for index in range(10)
            ]
        },
    }
    # One weakly-matching candidate: enough to be its own profile's rank-1,
    # nowhere near enough to win a slot against the ten above.
    narrow_targets = [
        target_field(
            "Patient.birthDate",
            eid="Patient.birthDate",
            description="administrative gender code",
        )
    ]
    narrow_snapshot = {
        "resourceType": "StructureDefinition",
        "type": "Patient",
        "snapshot": {
            "element": [
                element("Patient", "Patient", types=()),
                element("Patient.birthDate", "Patient.birthDate"),
            ]
        },
    }
    backend = ScriptedBackend()
    strategy = make_strategy(backend, top_k=3, second_pass_top_k=None)

    strategy.build_mapping_table(
        [
            patient_resource(res_id="wide", targets=wide_targets, snapshot=wide_snapshot),
            patient_resource(
                res_id="narrow", targets=narrow_targets, snapshot=narrow_snapshot
            ),
        ],
        [source_field("src.gender")],
    )

    assert backend.calls
    prompt = backend.calls[0]["prompt"]
    # It would lose every slot on score; it is offered because its own profile
    # would have selected it.
    assert "narrow.birthDate" in prompt


def test_a_prohibited_target_is_never_offered():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, top_k=99, second_pass_top_k=None)

    strategy.build_mapping_table([patient_resource()], [source_field("src.photo")])

    offered = {
        offer["target"]
        for attempt in strategy.report()["fields"][0]["attempts"]
        for offer in attempt["offers"]
    }
    assert not any("photo" in target for target in offered)


def test_rejected_field_does_not_disturb_other_fields():
    """Invalid output for one field must not partially corrupt the table."""

    backend = ScriptedBackend(answers={"gender": "c99", "birthDate": "c1"})
    strategy = make_strategy(backend, second_pass_top_k=None)

    table = strategy.build_mapping_table(
        [patient_resource()],
        [source_field("src.gender"), source_field("src.birthDate")],
    )

    assert "gender" not in table
    assert table["birthDate"] == "example-patient.birthDate"


def test_a_target_can_be_allocated_to_only_one_source_field():
    """Independent model calls must not create a last-writer-wins mapping."""

    backend = ScriptedBackend(default="c1")
    strategy = make_strategy(backend, top_k=1, second_pass_top_k=None)
    only_gender = [PATIENT_TARGETS[0]]

    table = strategy.build_mapping_table(
        [patient_resource(targets=only_gender)],
        [
            source_field("src.first", eid="first"),
            source_field("src.second", eid="second"),
        ],
    )
    report = strategy.report(table)

    assert table == {"first": "example-patient.gender"}
    assert len(set(table.values())) == len(table)
    assert len(backend.calls) == 1
    assert report["fields"][1]["rejected_reason"] == ("all-candidates-already-assigned")
    assert report["summary"]["allocation_rejections"] == 1


# --- prompting and offers ---------------------------------------------------


def test_offers_are_enumerated_and_the_prompt_names_them():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, top_k=3, second_pass_top_k=None)

    strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])

    prompt = backend.calls[0]["prompt"]
    assert "c1:" in prompt
    assert "example-patient.gender" in prompt
    assert "unmapped" in prompt


def test_prompt_carries_the_source_field_identity():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, second_pass_top_k=None)

    strategy.build_mapping_table(
        [patient_resource()],
        [source_field("src.gender", type_code="code", description="patient sex")],
    )

    prompt = backend.calls[0]["prompt"]
    assert "src.gender" in prompt
    assert "code" in prompt
    assert "patient sex" in prompt


def test_top_k_bounds_the_offered_set():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, top_k=2, second_pass_top_k=None)

    strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])

    offers = strategy.report()["fields"][0]["attempts"][0]["offers"]
    assert len(offers) == 2


def test_offer_order_is_deterministic_across_runs():
    def run():
        backend = ScriptedBackend(default=UNMAPPED)
        strategy = make_strategy(backend, top_k=3, second_pass_top_k=None)
        strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])
        return backend.calls[0]["prompt"]

    assert run() == run()


def test_candidates_are_pooled_across_profiles():
    """A source field competes against every profile's targets at once."""

    observation_snapshot = {
        "resourceType": "StructureDefinition",
        "type": "Observation",
        "snapshot": {
            "element": [
                element("Observation", "Observation", types=()),
                element(
                    "Observation.effectiveDateTime",
                    "Observation.effectiveDateTime",
                    types=("dateTime",),
                ),
            ]
        },
    }
    resources = [
        patient_resource(),
        ResourceTargets(
            res_type="Observation",
            res_id="example-observation",
            target_fields=[
                target_field(
                    "Observation.effectiveDateTime",
                    eid="Observation.effectiveDateTime",
                    type_code="dateTime",
                )
            ],
            profile=observation_snapshot,
        ),
    ]
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, top_k=99, second_pass_top_k=None)

    strategy.build_mapping_table(resources, [source_field("src.effectiveDateTime")])

    assert len(backend.calls) == 1, "one call per source field, not per profile"
    resources_offered = {
        offer["resource"]
        for offer in strategy.report()["fields"][0]["attempts"][0]["offers"]
    }
    assert resources_offered == {"example-patient", "example-observation"}


def test_a_field_with_no_offerable_target_costs_no_provider_call():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend)
    empty = ResourceTargets("Patient", "example-patient", [], PATIENT_SNAPSHOT)

    table = strategy.build_mapping_table([empty], [source_field("src.gender")])

    assert table == {}
    assert backend.calls == []


def test_a_profile_without_a_target_tree_is_skipped_and_recorded():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend)
    broken = ResourceTargets("Patient", "broken", PATIENT_TARGETS, profile=None)

    strategy.build_mapping_table([broken], [source_field("src.gender")])

    assert strategy.report()["skipped_profiles"] == [
        {"resource": "broken", "reason": "no-target-tree"}
    ]


# --- the bounded second pass ------------------------------------------------


def test_second_pass_retries_an_unmapped_field_with_a_wider_set():
    answers = {"gender": lambda call: UNMAPPED if call == 1 else "c1"}
    backend = ScriptedBackend(answers=answers)
    strategy = make_strategy(backend, top_k=1, second_pass_top_k=4)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )

    assert len(backend.calls) == 2
    assert table == {"gender": "example-patient.gender"}
    attempts = strategy.report(table)["fields"][0]["attempts"]
    assert [a["pass"] for a in attempts] == [1, 2]
    assert len(attempts[0]["offers"]) < len(attempts[1]["offers"])


def test_second_pass_is_skipped_when_it_would_offer_nothing_new():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, top_k=99, second_pass_top_k=200)

    strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])

    assert len(backend.calls) == 1


def test_second_pass_can_be_disabled():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, top_k=1, second_pass_top_k=None)

    strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])

    assert len(backend.calls) == 1


def test_an_accepted_first_pass_does_not_trigger_a_second():
    backend = ScriptedBackend(answers={"gender": "c1"})
    strategy = make_strategy(backend, top_k=1, second_pass_top_k=4)

    strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])

    assert len(backend.calls) == 1


# --- provider failure -------------------------------------------------------


def test_provider_failure_aborts_rather_than_silently_changing_mode():
    from llm.errors import LLMProviderError

    strategy = make_strategy(
        ExplodingBackend(LLMProviderError("upstream is down", 503))
    )

    with pytest.raises(LLMProviderError):
        strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])


def test_timeout_is_not_swallowed():
    from llm.errors import LLMTimeoutError

    strategy = make_strategy(ExplodingBackend(LLMTimeoutError("too slow")))

    with pytest.raises(LLMTimeoutError):
        strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])


def test_an_unusable_answer_is_a_rejected_attempt_not_a_dead_run():
    """The provider answered and the model was wrong — a different thing from
    the provider being unusable. Aborting there throws away every field already
    resolved and makes one malformed reply fatal to a whole project."""

    from llm.errors import LLMResponseValidationError

    strategy = make_strategy(
        ExplodingBackend(LLMResponseValidationError("bad envelope", "{}"))
    )

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )

    assert table == {}
    report = strategy.report(table)
    assert report["summary"]["unusable_responses"] >= 1
    reasons = [
        attempt["rejected_reason"]
        for field in report["fields"]
        for attempt in field["attempts"]
    ]
    assert any("unusable-response" in (reason or "") for reason in reasons)


def test_one_unusable_answer_does_not_cost_the_fields_around_it():
    class OneBadAnswer(ScriptedBackend):
        def complete(self, **kwargs):
            source = kwargs["context"]["source_id"]
            if source.endswith("birthDate"):
                from llm.errors import LLMResponseValidationError

                raise LLMResponseValidationError("bad envelope", "{}")
            return super().complete(**kwargs)

    backend = OneBadAnswer(answers={"gender": "c1"})
    strategy = make_strategy(backend)

    table = strategy.build_mapping_table(
        [patient_resource()],
        [source_field("src.birthDate"), source_field("src.gender")],
    )

    assert table  # the good field still mapped


# --- the report -------------------------------------------------------------


def test_report_records_offers_selection_and_provenance():
    backend = ScriptedBackend(answers={"gender": "c1"})
    strategy = make_strategy(backend, top_k=2, second_pass_top_k=None)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )
    report = strategy.report(table)

    assert report["report_version"] == 1
    assert report["mode"] == "llm"
    assert report["model"] == "test-model"
    assert report["mapping_table"] == table
    assert report["proposed_mapping_table"] == table
    assert report["compiled_mapping_table"] == table

    field = report["fields"][0]
    assert field["source_id"] == "gender"
    assert field["final_target"] == "example-patient.gender"
    assert field["selected_offer_id"] == "c1"

    attempt = field["attempts"][0]
    assert attempt["selection"]["confidence"] == 0.9
    assert attempt["selection"]["reason"] == "scripted"
    offer = attempt["offers"][0]
    assert offer["element_id"] == "Patient.gender"
    assert offer["target"] == "example-patient.gender"
    assert "score" in offer


def test_report_distinguishes_model_proposals_from_compiler_output():
    backend = ScriptedBackend(answers={"gender": "c1", "birthDate": "c1"})
    strategy = make_strategy(backend, second_pass_top_k=None)
    proposed = strategy.build_mapping_table(
        [patient_resource()],
        [source_field("src.gender"), source_field("src.birthDate")],
    )
    compiled = {"gender": proposed["gender"]}

    report = strategy.report(
        proposed,
        compiled_mapping_table=compiled,
        compiler_diagnostics=[
            {
                "code": "mapping-target-path-not-found",
                "source": "birthDate",
                "target": proposed["birthDate"],
            }
        ],
    )

    assert report["proposed_mapping_table"] == proposed
    assert report["compiled_mapping_table"] == compiled
    assert report["mapping_table"] == compiled
    assert report["summary"]["selected"] == 2
    assert report["summary"]["mapped"] == 1
    assert report["summary"]["compiler_rejections"] == 1
    assert report["compiler_rejections"][0]["source_id"] == "birthDate"


def test_report_counts_cache_hits_separately_from_provider_calls():
    backend = ScriptedBackend(answers={"gender": "c1"}, cache_hits={"gender"})
    strategy = make_strategy(backend)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )
    summary = strategy.report(table)["summary"]

    assert summary["cache_hits"] == 1
    assert summary["provider_calls"] == 0


def test_report_summary_counts_mapped_and_unmapped():
    backend = ScriptedBackend(answers={"gender": "c1"}, default=UNMAPPED)
    strategy = make_strategy(backend, second_pass_top_k=None)

    table = strategy.build_mapping_table(
        [patient_resource()],
        [source_field("src.gender"), source_field("src.birthDate")],
    )
    summary = strategy.report(table)["summary"]

    assert summary["source_fields"] == 2
    assert summary["mapped"] == 1
    assert summary["unmapped"] == 1


def test_report_is_json_serialisable():
    backend = ScriptedBackend(answers={"gender": "c1"})
    strategy = make_strategy(backend)

    table = strategy.build_mapping_table(
        [patient_resource()], [source_field("src.gender")]
    )

    json.dumps(strategy.report(table))


# --- cache identity ---------------------------------------------------------


def test_context_identifies_the_offered_set():
    """A changed candidate list must not reuse an answer for the old one."""

    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, top_k=2, second_pass_top_k=None)

    strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])

    context = backend.calls[0]["context"]
    assert context["source_id"] == "gender"
    assert [entry[1] for entry in context["offers"]] == [
        offer["element_id"]
        for offer in strategy.report()["fields"][0]["attempts"][0]["offers"]
    ]


def test_calls_are_partitioned_under_the_automapping_feature():
    backend = ScriptedBackend(default=UNMAPPED)
    strategy = make_strategy(backend, second_pass_top_k=None)

    strategy.build_mapping_table([patient_resource()], [source_field("src.gender")])

    assert backend.calls[0]["feature"] == "automapping"


# --- the answer schema ------------------------------------------------------


def test_selection_schema_is_strict_mode_compatible():
    """WP2 rejects an optional field that cannot be null; `reason` must qualify."""

    from llm.structured_output import json_schema_for

    schema = json_schema_for(CandidateSelection, strict=True)

    assert set(schema["required"]) == {"candidate_id", "confidence", "reason"}
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("given", "expected"),
    [(0.85, 0.85), (85, 0.85), (100, 1.0), (-0.5, 0.0), ("0.4", 0.4), ("70", 0.7)],
)
def test_selection_confidence_accepts_a_percentage_and_never_fails_a_run(given, expected):
    """Nothing reads ``confidence``; a bound on it can only reject an otherwise
    usable answer. A model replying ``100`` used to abort a whole project's
    automapping on the field where it happened."""

    selection = CandidateSelection(candidate_id="c1", confidence=given, reason=None)

    assert selection.confidence == pytest.approx(expected)


@pytest.mark.parametrize("top_k", [0, -1, "many", True])
def test_top_k_must_be_a_positive_integer(top_k):
    with pytest.raises(LLMConfigurationError, match="automapping_top_k"):
        make_strategy(ScriptedBackend(), top_k=top_k)


def test_second_pass_width_accepts_numeric_strings_and_is_normalised():
    strategy = make_strategy(ScriptedBackend(), top_k="2", second_pass_top_k="4")

    assert strategy.top_k == 2
    assert strategy.second_pass_top_k == 4


def test_second_pass_width_must_be_wider_than_the_first_pass():
    with pytest.raises(LLMConfigurationError, match="second_pass_top_k"):
        make_strategy(ScriptedBackend(), top_k=5, second_pass_top_k=5)


# --- the shared rebase helper -----------------------------------------------


# --- generator wiring -------------------------------------------------------


def _generator(llm_automapping=None, automapper=None):
    """A `StructureMapGenerator` with only the automapping collaborators wired."""

    from mapping.fml_map import StructureMapGenerator

    smg = object.__new__(StructureMapGenerator)
    smg.automapper = automapper or FMLAutomapper(use_word2vec=False)
    smg.llm_automapping = llm_automapping
    smg.map_name = "structure_map_proj"
    smg.mapping_diagnostics = []
    smg._llm_proposed_mapping_table = None
    smg._llm_compiled_mapping_table = {}
    smg._llm_diagnostics_start = 0
    return smg


def test_generator_routes_to_the_llm_strategy_when_one_is_present(monkeypatch):
    backend = ScriptedBackend(answers={"gender": "c1"})
    strategy = make_strategy(backend)
    smg = _generator(llm_automapping=strategy)
    monkeypatch.setattr(smg, "_iter_mappable_resources", lambda resources: iter(()))
    monkeypatch.setattr(smg, "_save_llm_automapping_report", lambda report: None)
    called = []
    monkeypatch.setattr(
        strategy,
        "build_mapping_table",
        lambda targets, fields: (
            called.append(("llm", fields)) or {"gender": "P.gender"}
        ),
    )

    table = smg._build_llm_automapping([], [source_field("src.gender")])

    assert table == {"gender": "P.gender"}
    assert called and called[0][0] == "llm"


def test_generator_writes_the_report_beside_the_mapping_table(monkeypatch):
    backend = ScriptedBackend(answers={"gender": "c1"})
    strategy = make_strategy(backend)
    smg = _generator(llm_automapping=strategy)
    monkeypatch.setattr(smg, "_iter_mappable_resources", lambda resources: iter(()))
    written = {}

    class FakeIO:
        ProjectFolders = type(
            "F", (), {"SOURCE_DATA": type("V", (), {"value": "source_data"})}
        )

        def store_project_file(self, folder, filename, content, mode, overwrite):
            written["filename"] = filename
            written["content"] = content

    smg.app_state = type("S", (), {"dataIO": FakeIO()})()

    smg._build_llm_automapping([], [source_field("src.gender")])
    smg._finalize_llm_automapping_report()

    assert written["filename"] == "structure_map_proj_llm_automapping_report.json"
    payload = json.loads(written["content"])
    assert payload["mode"] == "llm"
    assert payload["report_version"] == 1


def test_report_write_failure_aborts_the_llm_generation(monkeypatch):
    """A successful audited mode may not silently omit its required report."""

    strategy = make_strategy(ScriptedBackend(default=UNMAPPED))
    smg = _generator(llm_automapping=strategy)
    monkeypatch.setattr(smg, "_iter_mappable_resources", lambda resources: iter(()))

    class FailingIO:
        ProjectFolders = type(
            "F", (), {"SOURCE_DATA": type("V", (), {"value": "source_data"})}
        )

        def store_project_file(self, *_, **__):
            raise OSError("read-only filesystem")

    smg.app_state = type("S", (), {"dataIO": FailingIO()})()

    assert smg._build_llm_automapping([], [source_field("src.gender")]) == {}
    with pytest.raises(OSError, match="read-only filesystem"):
        smg._finalize_llm_automapping_report()


def test_deterministic_generation_never_constructs_an_llm_strategy():
    """`--auto-mapping` alone must leave the LLM path entirely uninvolved."""

    from mapping.fml_map import StructureMapGenerator
    import inspect

    signature = inspect.signature(StructureMapGenerator.__init__)
    assert signature.parameters["llm_automapping"].default is None


def test_pipeline_controller_default_leaves_llm_automapping_unset():
    from controller.pipeline_controller import pipeline_controller as module
    import inspect

    signature = inspect.signature(module.PipelineController.__init__)
    assert signature.parameters["auto_mapping_mode"].default == "deterministic"


@pytest.mark.parametrize(
    "path,res_type,res_id,expected",
    [
        ("Patient.gender", "Patient", "my-profile", "my-profile.gender"),
        ("Patient.gender", "Patient", "Patient", "Patient.gender"),
        ("Patient.gender", "Patient", None, "Patient.gender"),
        ("Patient.identifier:mrn", "Patient", "p", "p.identifier:mrn"),
        # a path that does not belong to this resource type is left alone
        ("Observation.value", "Patient", "p", "Observation.value"),
        (None, "Patient", "p", None),
    ],
)
def test_rebase_to_resource_identity(path, res_type, res_id, expected):
    assert rebase_to_resource_identity(path, res_type, res_id) == expected
