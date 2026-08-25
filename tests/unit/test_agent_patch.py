"""The agent patch protocol: envelope, policy, and guarded application (WP5).

The safety property under test is narrow and absolute: a proposal either applies
exactly as written to the revision it was written against, or it is rejected with
structured evidence and the base document is byte-for-byte untouched. There is no
third outcome, and in particular there is no partial application.
"""

import copy
import json

import pytest

from agent.models import (
    PATCH_SCHEMA_VERSION,
    AgentPatch,
    PatchOp,
    RejectionCode,
    operation,
)
from agent.patch import (
    apply_patch,
    canonical_json,
    canonical_sha256,
    describe_pointers,
    raw_sha256,
)
from agent.policy import PatchPolicy, pointer_tokens, unescape_token, validate

pytestmark = pytest.mark.unit


# --- fixtures ---------------------------------------------------------------


def base_map():
    """A small but structurally realistic StructureMap: nested, repeated arrays."""

    return {
        "resourceType": "StructureMap",
        "id": "example-map",
        "url": "http://example.org/StructureMap/example-map",
        "name": "ExampleMap",
        "status": "draft",
        "structure": [
            {"url": "http://example.org/StructureDefinition/src", "mode": "source"},
            {"url": "http://hl7.org/fhir/StructureDefinition/Patient", "mode": "target"},
        ],
        "group": [
            {
                "name": "TransformPatient",
                "typeMode": "none",
                "input": [
                    {"name": "source", "mode": "source", "type": "Src"},
                    {"name": "target", "mode": "target", "type": "Patient"},
                ],
                "rule": [
                    {
                        "name": "map-gender",
                        "source": [{"context": "source", "element": "sex"}],
                        "target": [
                            {"context": "target", "element": "gender", "transform": "copy"}
                        ],
                    },
                    {
                        "name": "map-name",
                        "source": [{"context": "source", "variable": "s"}],
                        "target": [
                            {
                                "context": "target",
                                "element": "name",
                                "variable": "vName",
                                "transform": "create",
                            }
                        ],
                        "rule": [
                            {
                                "name": "set-family",
                                "source": [{"context": "s", "element": "last"}],
                                "target": [
                                    {
                                        "context": "vName",
                                        "element": "family",
                                        "transform": "copy",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            }
        ],
    }


GENDER_TRANSFORM = "/group/0/rule/0/target/0/transform"


def envelope(operations, *, base=None, **overrides):
    document = base if base is not None else base_map()
    fields = {
        "schema_version": PATCH_SCHEMA_VERSION,
        "map_url": document["url"],
        "map_id": document["id"],
        "base_sha256": canonical_sha256(document),
        "diagnostic_ids": ["diag-abc123"],
        "patch": operations,
    }
    fields.update(overrides)
    return AgentPatch(**fields)


def op(kind, path, value=...):
    if value is ...:
        return operation(kind, path)
    return operation(kind, path, value)


def guarded_replace(path, old, new):
    """The shape every mutation must take: assert, then change."""

    return [op(PatchOp.TEST, path, old), op(PatchOp.REPLACE, path, new)]


def codes(result):
    return {rejection.code for rejection in result.rejections}


# --- canonical hashing ------------------------------------------------------


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_digest_survives_a_reserialisation_round_trip():
    """The reason the gate is canonical rather than raw bytes."""

    document = base_map()
    reformatted = json.loads(json.dumps(document, indent=4, sort_keys=False))

    assert canonical_sha256(document) == canonical_sha256(reformatted)


def test_raw_digest_does_not_survive_reformatting():
    document = base_map()

    compact = raw_sha256(json.dumps(document, separators=(",", ":")))
    pretty = raw_sha256(json.dumps(document, indent=2))

    assert compact != pretty


def test_digest_changes_when_content_changes():
    document = base_map()
    before = canonical_sha256(document)
    document["group"][0]["rule"][0]["name"] = "map-gender-renamed"

    assert canonical_sha256(document) != before


# --- happy path -------------------------------------------------------------


def test_guarded_replace_applies():
    document = base_map()
    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"))
    )

    assert result.applied
    assert result.candidate["group"][0]["rule"][0]["target"][0]["transform"] == "cast"


def test_application_never_mutates_the_input():
    document = base_map()
    untouched = copy.deepcopy(document)

    apply_patch(document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast")))

    assert document == untouched


def test_candidate_is_a_distinct_object():
    document = base_map()
    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"))
    )

    result.candidate["group"][0]["rule"][0]["name"] = "mutated"
    assert document["group"][0]["rule"][0]["name"] == "map-gender"


def test_application_is_reproducible():
    document = base_map()
    proposal = envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"))

    first = apply_patch(document, proposal)
    second = apply_patch(document, proposal)

    assert first.candidate_sha256 == second.candidate_sha256
    assert first.candidate == second.candidate


def test_candidate_hash_matches_the_candidate():
    document = base_map()
    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"))
    )

    assert result.candidate_sha256 == canonical_sha256(result.candidate)


def test_the_base_digest_cannot_be_supplied_by_the_caller():
    """Regression: a caller-supplied digest made the staleness check vacuous.

    `apply_patch(changed, proposal, base_sha256=proposal.base_sha256)` used to
    assert that a value equals itself, disabling the one guarantee the envelope
    exists to provide. The parameter is gone; the digest always comes from the
    document actually being patched.
    """
    import inspect

    assert "base_sha256" not in inspect.signature(apply_patch).parameters


def test_a_stale_patch_is_rejected_however_it_is_invoked():
    original = base_map()
    proposal = envelope(
        [op(PatchOp.ADD, "/group/0/rule/-", {"name": "new", "source": [], "target": []})],
        base=original,
    )

    changed = base_map()
    changed["group"][0]["rule"][0]["name"] = "renamed"

    result = apply_patch(changed, proposal)

    assert result.rejected
    assert RejectionCode.STALE_BASE.value in codes(result)


# --- array insertion and removal --------------------------------------------


def test_appending_a_rule_to_an_array():
    document = base_map()
    new_rule = {
        "name": "map-birthdate",
        "source": [{"context": "source", "element": "dob"}],
        "target": [{"context": "target", "element": "birthDate", "transform": "copy"}],
    }

    result = apply_patch(
        document, envelope([op(PatchOp.ADD, "/group/0/rule/-", new_rule)])
    )

    assert result.applied
    rules = result.candidate["group"][0]["rule"]
    assert len(rules) == 3
    assert rules[2]["name"] == "map-birthdate"


def test_inserting_a_rule_at_an_index_shifts_the_rest():
    document = base_map()
    new_rule = {"name": "map-first", "source": [], "target": []}

    result = apply_patch(
        document, envelope([op(PatchOp.ADD, "/group/0/rule/0", new_rule)])
    )

    names = [rule["name"] for rule in result.candidate["group"][0]["rule"]]
    assert names == ["map-first", "map-gender", "map-name"]


def test_removing_a_rule_requires_and_honours_its_guard():
    document = base_map()
    existing = document["group"][0]["rule"][0]

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.TEST, "/group/0/rule/0", existing),
                op(PatchOp.REMOVE, "/group/0/rule/0"),
            ]
        ),
    )

    assert result.applied
    names = [rule["name"] for rule in result.candidate["group"][0]["rule"]]
    assert names == ["map-name"]


def test_editing_a_nested_rule():
    document = base_map()
    path = "/group/0/rule/1/rule/0/target/0/transform"

    result = apply_patch(document, envelope(guarded_replace(path, "copy", "cast")))

    assert result.applied
    nested = result.candidate["group"][0]["rule"][1]["rule"][0]
    assert nested["target"][0]["transform"] == "cast"


# --- stale array indices ----------------------------------------------------


def test_a_stale_index_fails_its_guard_instead_of_editing_the_wrong_rule():
    """The property that makes positional addressing acceptable at all.

    A patch written when `map-gender` was rule 0 must not silently edit whatever
    now occupies index 0.
    """
    original = base_map()
    proposal = envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"), base=original)

    # the map is edited underneath the proposal: a rule is prepended
    shifted = base_map()
    shifted["group"][0]["rule"].insert(0, {"name": "inserted", "source": [], "target": []})

    result = apply_patch(shifted, proposal)

    assert result.rejected
    # caught by the revision binding, before application is even attempted
    assert RejectionCode.STALE_BASE.value in codes(result)
    assert shifted["group"][0]["rule"][0]["name"] == "inserted"


def test_a_guard_that_does_not_hold_rejects_even_with_a_matching_base_hash():
    """Defence in depth: the guard is checked against the document, not the hash."""

    document = base_map()
    proposal = envelope(
        # asserts a transform value the map does not have at that pointer
        guarded_replace(GENDER_TRANSFORM, "translate", "cast"),
        base=document,
    )

    result = apply_patch(document, proposal)

    assert result.rejected
    assert RejectionCode.TEST_FAILED.value in codes(result)
    assert document["group"][0]["rule"][0]["target"][0]["transform"] == "copy"


def test_a_pointer_past_the_end_of_an_array_is_rejected():
    document = base_map()

    result = apply_patch(
        document,
        envelope(guarded_replace("/group/0/rule/99/target/0/transform", "copy", "cast")),
    )

    assert result.rejected
    assert codes(result) & {
        RejectionCode.TEST_FAILED.value,
        RejectionCode.INVALID_POINTER.value,
    }


# --- policy: guards ---------------------------------------------------------



def test_an_unguarded_replace_is_guarded_by_the_tool():
    """The guard is the tool's job now, not the model's.

    Providers are asked for a schema with no `test` operation at all (see
    `ProposedOperation`), because transcribing the current value was what they
    got wrong: `test-failed` and `unguarded-mutation` together were 39 of 58
    rejections in one corpus run, none of them about the repair itself. The
    invariant that every mutation is guarded is unchanged — only its author is.
    """
    document = base_map()

    result = apply_patch(
        document, envelope([op(PatchOp.REPLACE, GENDER_TRANSFORM, "cast")])
    )

    assert result.applied
    assert result.normalized_patch[0] == {
        "op": "test",
        "path": GENDER_TRANSFORM,
        "value": "copy",
    }
    # Asserted against the base, so it can never be the value being written.
    assert result.normalized_patch[1]["value"] == "cast"



def test_an_unguarded_remove_is_guarded_by_the_tool():
    document = base_map()

    result = apply_patch(document, envelope([op(PatchOp.REMOVE, "/group/0/rule/0")]))

    assert result.applied
    guard = result.normalized_patch[0]
    assert guard["op"] == "test" and guard["path"] == "/group/0/rule/0"
    # The whole rule, verbatim: removing by index is only safe if the thing at
    # that index is still the thing the proposal was written against.
    assert guard["value"] == base_map()["group"][0]["rule"][0]



def test_a_guard_on_a_different_path_still_does_not_count():
    """A stray `test` never stands in for the mutation's own guard.

    Replayed envelopes and older builds can still carry model-authored guards,
    so injection has to decide whether one is *the* guard. It matches on the
    immediately preceding operation and the identical pointer; anything else
    gets its own guard injected regardless.
    """
    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.TEST, "/group/0/rule/1/target/0/transform", "create"),
                op(PatchOp.REPLACE, GENDER_TRANSFORM, "cast"),
            ]
        ),
    )

    assert result.applied
    ops = result.normalized_patch
    assert [entry["op"] for entry in ops] == ["test", "test", "replace"]
    assert ops[1] == {"op": "test", "path": GENDER_TRANSFORM, "value": "copy"}



def test_a_guard_must_still_be_immediately_before_its_mutation():
    """Adjacency is the point: an intervening op can change what is asserted."""
    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.TEST, GENDER_TRANSFORM, "copy"),
                op(PatchOp.TEST, "/group/0/name", "TransformPatient"),
                op(PatchOp.REPLACE, GENDER_TRANSFORM, "cast"),
            ]
        ),
    )

    assert result.applied
    ops = result.normalized_patch
    # The model's own guard, two places back, is not adjacent; a fresh one is
    # injected so the replace is preceded by an assertion on its own pointer.
    assert ops[-2] == {"op": "test", "path": GENDER_TRANSFORM, "value": "copy"}
    assert ops[-1]["op"] == "replace"


def test_add_does_not_require_a_guard():
    """The base hash already binds an insertion to a known revision."""

    document = base_map()

    result = apply_patch(
        document,
        envelope([op(PatchOp.ADD, "/group/0/rule/-", {"name": "x", "source": [], "target": []})]),
    )

    assert result.applied


# --- policy: identity, scope, shape -----------------------------------------


@pytest.mark.parametrize("field", ["resourceType", "id", "url", "name", "version"])
def test_identity_fields_are_protected(field):
    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.TEST, f"/{field}", document.get(field)),
                op(PatchOp.REPLACE, f"/{field}", "hijacked"),
            ]
        ),
    )

    assert result.rejected
    assert RejectionCode.PROTECTED_FIELD.value in codes(result)


def test_reading_an_identity_field_as_a_guard_is_allowed():
    """Protection is about changing identity, not about referring to it."""

    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.TEST, "/url", document["url"]),
                *guarded_replace(GENDER_TRANSFORM, "copy", "cast"),
            ]
        ),
    )

    assert result.applied


def test_root_replacement_is_rejected():
    document = base_map()

    result = apply_patch(document, envelope([op(PatchOp.REPLACE, "", {"gone": True})]))

    assert RejectionCode.ROOT_REPLACEMENT.value in codes(result)


def test_a_pointer_to_an_unknown_top_level_field_is_rejected():
    document = base_map()

    result = apply_patch(
        document, envelope(guarded_replace("/notAField/0", "a", "b"))
    )

    assert RejectionCode.OUT_OF_SCOPE_POINTER.value in codes(result)


def test_status_remains_editable():
    """Identity is protected; ordinary content is not frozen."""

    document = base_map()

    result = apply_patch(document, envelope(guarded_replace("/status", "draft", "active")))

    assert result.applied


def test_a_malformed_pointer_is_rejected():
    document = base_map()

    result = apply_patch(document, envelope(guarded_replace("group/0", "a", "b")))

    assert RejectionCode.INVALID_POINTER.value in codes(result)


def test_a_replace_without_a_value_is_not_constructible():
    """The discriminated union makes the shape a schema fact, not a policy check."""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        op(PatchOp.REPLACE, GENDER_TRANSFORM)


def test_remove_rejects_a_surplus_value():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        op(PatchOp.REMOVE, "/group/0/rule/0", {"anything": True})


def test_an_explicit_null_is_a_legal_value():
    document = base_map()

    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", None))
    )

    assert result.applied
    assert result.candidate["group"][0]["rule"][0]["target"][0]["transform"] is None


def test_move_and_copy_are_not_expressible():
    """Deferred ops are excluded by the schema, not merely by policy."""

    from pydantic import ValidationError

    for unsupported in ("move", "copy"):
        with pytest.raises(ValidationError):
            op(unsupported, "/group/0", None)


def test_an_unsupported_op_is_rejected_when_policy_narrows_the_set():
    document = base_map()
    policy = PatchPolicy(allowed_ops=frozenset({PatchOp.TEST, PatchOp.REPLACE}))

    result = apply_patch(
        document,
        envelope([op(PatchOp.ADD, "/group/0/rule/-", {"name": "x"})]),
        policy=policy,
    )

    assert RejectionCode.UNSUPPORTED_OP.value in codes(result)


# --- policy: bounds ---------------------------------------------------------


def test_an_empty_patch_is_rejected():
    document = base_map()

    result = apply_patch(document, envelope([]))

    assert RejectionCode.EMPTY_PATCH.value in codes(result)


def test_a_patch_that_only_asserts_is_rejected():
    document = base_map()

    result = apply_patch(document, envelope([op(PatchOp.TEST, GENDER_TRANSFORM, "copy")]))

    assert RejectionCode.NO_EFFECT.value in codes(result)


def test_a_patch_that_applies_but_changes_nothing_is_rejected():
    """Otherwise a loop could 'make progress' indefinitely without progressing."""

    document = base_map()

    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "copy"))
    )

    assert result.rejected
    assert RejectionCode.NO_EFFECT.value in codes(result)


def test_too_many_operations_is_rejected():
    document = base_map()
    policy = PatchPolicy(max_operations=2)

    result = apply_patch(
        document,
        envelope(
            guarded_replace(GENDER_TRANSFORM, "copy", "cast")
            + [op(PatchOp.ADD, "/group/0/rule/-", {"name": "x"})]
        ),
        policy=policy,
    )

    assert RejectionCode.TOO_MANY_OPERATIONS.value in codes(result)


def test_an_oversized_payload_is_rejected():
    document = base_map()
    policy = PatchPolicy(max_payload_bytes=200)

    result = apply_patch(
        document,
        envelope([op(PatchOp.ADD, "/group/0/rule/-", {"name": "x" * 500})]),
        policy=policy,
    )

    assert RejectionCode.PAYLOAD_TOO_LARGE.value in codes(result)


def test_the_schema_itself_pins_the_patch_version():
    """The model cannot answer with a version WP5 would only reject later.

    As a free ``int`` this field admitted 0, 2, 3 and date-like values, and 19
    attempts in one batch were spent producing an unusable patch.
    """

    import pydantic  # noqa: PLC0415

    with pytest.raises(pydantic.ValidationError):
        envelope(
            guarded_replace(GENDER_TRANSFORM, "copy", "cast"),
            schema_version=PATCH_SCHEMA_VERSION + 1,
        )


def test_an_unsupported_schema_version_is_still_rejected_by_policy():
    """Defence in depth: a patch that reached WP5 without going through the
    schema — a cached envelope from an older build, say — must not apply."""

    document = base_map()
    stale = AgentPatch.model_construct(
        schema_version=PATCH_SCHEMA_VERSION + 1,
        map_url=document["url"],
        map_id=document["id"],
        base_sha256=canonical_sha256(document),
        diagnostic_ids=["diag-abc123"],
        patch=guarded_replace(GENDER_TRANSFORM, "copy", "cast"),
    )

    assert RejectionCode.MALFORMED_OP.value in codes(apply_patch(document, stale))


# --- rejection reporting ----------------------------------------------------


def test_every_rejection_is_collected_not_just_the_first():
    """A retry that fixes one fault per attempt burns the attempt budget."""

    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [op(PatchOp.REPLACE, "/notAField", "x")],
            base_sha256="0" * 64,
        ),
    )

    assert {
        RejectionCode.STALE_BASE.value,
        RejectionCode.UNGUARDED_MUTATION.value,
        RejectionCode.OUT_OF_SCOPE_POINTER.value,
    } <= codes(result)


def test_a_rejection_locates_the_offending_operation():
    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                *guarded_replace(GENDER_TRANSFORM, "copy", "cast"),
                op(PatchOp.REPLACE, "/notAField", "x"),
            ]
        ),
    )

    offending = [
        rejection
        for rejection in result.rejections
        if rejection.code == RejectionCode.OUT_OF_SCOPE_POINTER.value
    ]
    assert offending[0].operation_index == 2
    assert offending[0].path == "/notAField"



def test_a_rejected_patch_yields_no_candidate():
    document = base_map()

    # A pointer whose current value cannot be resolved: nothing to assert, so no
    # guard can be injected and the mutation stays refused.
    result = apply_patch(
        document, envelope([op(PatchOp.REPLACE, "/group/0/rule/0/target/0/nope", "x")])
    )

    assert RejectionCode.UNGUARDED_MUTATION.value in codes(result)
    assert result.candidate is None
    assert result.candidate_sha256 is None



def test_the_normalised_patch_is_recorded_even_on_rejection():
    document = base_map()

    result = apply_patch(
        document, envelope([op(PatchOp.REPLACE, "/group/0/rule/0/target/0/nope", "x")])
    )

    assert result.rejected
    assert result.normalized_patch == [
        {"op": "replace", "path": "/group/0/rule/0/target/0/nope", "value": "x"}
    ]


def test_remove_normalises_without_a_value_key():
    """RFC 6902 defines no `value` for remove; emitting one is non-conformant."""

    operations = [op(PatchOp.TEST, "/group/0/rule/0", {}), op(PatchOp.REMOVE, "/group/0/rule/0")]

    normalized = envelope(operations).as_rfc6902()

    assert "value" not in normalized[1]
    assert "value" in normalized[0]


def test_report_dict_is_json_serialisable():
    document = base_map()
    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"))
    )

    json.dumps(result.as_report_dict())


# --- pointer helpers --------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [("plain", "plain"), ("a~1b", "a/b"), ("a~0b", "a~b"), ("~01", "~1")],
)
def test_rfc6901_token_unescaping(token, expected):
    assert unescape_token(token) == expected


def test_pointer_tokens_splits_and_unescapes():
    assert pointer_tokens("/group/0/rule") == ["group", "0", "rule"]
    assert pointer_tokens("") == []


def test_a_pointer_must_start_with_a_slash():
    with pytest.raises(ValueError):
        pointer_tokens("group/0")


def test_describe_pointers_labels_every_rule_including_nested():
    entries = describe_pointers(base_map())

    by_pointer = {entry["pointer"]: entry["label"] for entry in entries}
    assert by_pointer["/group/0/rule/0"] == "TransformPatient › map-gender"
    assert by_pointer["/group/0/rule/1"] == "TransformPatient › map-name"
    assert (
        by_pointer["/group/0/rule/1/rule/0"]
        == "TransformPatient › map-name › set-family"
    )


def test_described_pointers_actually_resolve():
    document = base_map()

    for entry in describe_pointers(document):
        node = document
        for token in pointer_tokens(entry["pointer"]):
            node = node[int(token)] if isinstance(node, list) else node[token]
        assert node["name"] == entry["name"]


# --- policy is usable standalone --------------------------------------------


def test_validate_reports_admissible_as_an_empty_list():
    document = base_map()
    proposal = envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"))

    assert validate(proposal, document, canonical_sha256(document)) == []


def test_policy_is_reportable():
    assert PatchPolicy().as_report_dict()["allowed_ops"] == [
        "add",
        "remove",
        "replace",
        "test",
    ]


# --- sequential application -------------------------------------------------


def test_an_earlier_insert_invalidates_a_later_index():
    """Regression: per-operation validation missed RFC 6902's sequencing.

    Inserting at `/group/0/rule/0` and then editing `/group/0/rule/1` looks fine
    operation by operation, but by the time the second op runs, index 1 is the
    rule that *was* index 0. The `test` guard does not catch it whenever the two
    rules happen to share the asserted value — which is exactly when a mistake is
    hardest to notice.
    """
    document = base_map()
    # make the two rules indistinguishable to a guard on this pointer
    document["group"][0]["rule"][1]["target"][0]["transform"] = "copy"

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.ADD, "/group/0/rule/0", {"name": "NEW", "source": [], "target": []}),
                op(PatchOp.TEST, "/group/0/rule/1/target/0/transform", "copy"),
                op(PatchOp.REPLACE, "/group/0/rule/1/target/0/transform", "cast"),
            ],
            base=document,
        ),
    )

    assert result.rejected
    assert RejectionCode.SHIFTED_POINTER.value in codes(result)
    assert document["group"][0]["rule"][0]["name"] == "map-gender"


def test_an_earlier_remove_invalidates_a_later_index():
    document = base_map()
    existing = document["group"][0]["rule"][0]

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.TEST, "/group/0/rule/0", existing),
                op(PatchOp.REMOVE, "/group/0/rule/0"),
                op(PatchOp.TEST, "/group/0/rule/1/name", "map-name"),
                op(PatchOp.REPLACE, "/group/0/rule/1/name", "renamed"),
            ],
            base=document,
        ),
    )

    assert RejectionCode.SHIFTED_POINTER.value in codes(result)


def test_appending_does_not_invalidate_later_pointers():
    """An append cannot shift an existing index, so it must stay permitted."""

    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.ADD, "/group/0/rule/-", {"name": "NEW", "source": [], "target": []}),
                *guarded_replace(GENDER_TRANSFORM, "copy", "cast"),
            ]
        ),
    )

    assert result.applied
    rules = result.candidate["group"][0]["rule"]
    assert rules[0]["target"][0]["transform"] == "cast"
    assert rules[-1]["name"] == "NEW"


def test_a_structural_change_to_one_array_does_not_block_another():
    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.ADD, "/structure/0", {"url": "http://x", "mode": "source"}),
                *guarded_replace(GENDER_TRANSFORM, "copy", "cast"),
            ]
        ),
    )

    assert RejectionCode.SHIFTED_POINTER.value not in codes(result)


def test_a_structural_change_placed_last_is_allowed():
    """The ordering WP7 should prefer: edits first, then structural changes."""

    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                *guarded_replace(GENDER_TRANSFORM, "copy", "cast"),
                op(PatchOp.ADD, "/group/0/rule/0", {"name": "NEW", "source": [], "target": []}),
            ]
        ),
    )

    assert result.applied


# --- add as a disguised replacement -----------------------------------------



def test_an_add_onto_an_existing_member_is_guarded_but_a_fresh_one_is_not():
    """RFC 6902 §4.1: `add` onto an existing object member replaces it.

    So it is guarded like a replacement. Onto a member that does not exist yet
    it is a pure insert with nothing to assert, and injecting a guard there
    would be injecting a falsehood.
    """
    overwrite = apply_patch(
        base_map(), envelope([op(PatchOp.ADD, GENDER_TRANSFORM, "cast")])
    )
    assert [entry["op"] for entry in overwrite.normalized_patch] == ["test", "add"]
    assert overwrite.normalized_patch[0]["value"] == "copy"

    insert = apply_patch(
        base_map(),
        envelope([op(PatchOp.ADD, "/group/0/rule/0/target/0/variable", "v")]),
    )
    assert [entry["op"] for entry in insert.normalized_patch] == ["add"]


def test_a_guarded_add_onto_an_existing_member_is_allowed():
    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [op(PatchOp.TEST, GENDER_TRANSFORM, "copy"), op(PatchOp.ADD, GENDER_TRANSFORM, "cast")]
        ),
    )

    assert result.applied


def test_an_add_of_a_genuinely_new_member_needs_no_guard():
    document = base_map()

    result = apply_patch(
        document,
        envelope([op(PatchOp.ADD, "/group/0/rule/0/documentation", "why this rule exists")]),
    )

    assert result.applied


def test_an_array_insertion_needs_no_guard():
    document = base_map()

    result = apply_patch(
        document,
        envelope([op(PatchOp.ADD, "/group/0/rule/0", {"name": "NEW", "source": [], "target": []})]),
    )

    assert result.applied


# --- envelope identity ------------------------------------------------------


def test_a_patch_for_a_different_map_url_is_rejected():
    document = base_map()

    result = apply_patch(
        document,
        envelope(
            guarded_replace(GENDER_TRANSFORM, "copy", "cast"),
            map_url="http://example.org/StructureMap/some-other-map",
        ),
    )

    assert result.rejected
    assert RejectionCode.WRONG_MAP.value in codes(result)


def test_a_patch_for_a_different_map_id_is_rejected():
    """A canonical URL alone is not identity — two files can share one mid-rename."""

    document = base_map()

    result = apply_patch(
        document,
        envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"), map_id="other-map"),
    )

    assert RejectionCode.WRONG_MAP.value in codes(result)


def test_a_base_that_is_not_a_structure_map_is_rejected():
    document = base_map()
    document["resourceType"] = "Questionnaire"

    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"), base=document)
    )

    assert RejectionCode.NOT_A_STRUCTURE_MAP.value in codes(result)


# --- deep structural validity -----------------------------------------------


def test_a_bogus_field_deeper_than_the_first_segment_is_rejected():
    """Regression: scope checking only looked at the first pointer segment.

    `/group/0/notAGroupField` passes every per-operation rule and still produces
    a structurally invalid map. Parsing the candidate with the R4B model catches
    it at any depth without a hand-maintained table of legal paths.
    """
    document = base_map()

    result = apply_patch(
        document, envelope([op(PatchOp.ADD, "/group/0/notAGroupField", {"junk": 1})])
    )

    assert result.rejected
    assert RejectionCode.INVALID_CANDIDATE.value in codes(result)


def test_a_value_of_the_wrong_shape_is_rejected():
    document = base_map()

    result = apply_patch(
        document,
        envelope(
            [
                op(PatchOp.TEST, "/group/0/rule", document["group"][0]["rule"]),
                op(PatchOp.REPLACE, "/group/0/rule", "not-a-list-of-rules"),
            ]
        ),
    )

    assert result.rejected
    assert RejectionCode.INVALID_CANDIDATE.value in codes(result)


def test_a_structurally_valid_edit_still_passes():
    document = base_map()

    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast"))
    )

    assert result.applied


# --- pointer allowlist (WP7 prompt scoping) ---------------------------------


def test_pointers_outside_the_offered_fragment_are_rejected():
    document = base_map()
    policy = PatchPolicy(allowed_pointer_prefixes=("/group/0/rule/1",))

    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast")), policy=policy
    )

    assert RejectionCode.OUT_OF_SCOPE_POINTER.value in codes(result)


def test_pointers_inside_the_offered_fragment_are_allowed():
    document = base_map()
    policy = PatchPolicy(allowed_pointer_prefixes=("/group/0/rule/0",))

    result = apply_patch(
        document, envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast")), policy=policy
    )

    assert result.applied


def test_an_empty_allowlist_means_no_restriction():
    document = base_map()

    result = apply_patch(
        document,
        envelope(guarded_replace(GENDER_TRANSFORM, "copy", "cast")),
        policy=PatchPolicy(allowed_pointer_prefixes=()),
    )

    assert result.applied


# --- strict structured output -----------------------------------------------


def test_the_envelope_can_be_requested_in_strict_json_schema_mode():
    """Regression: WP5 violated the schema rule WP2 enforces.

    The agent loop asks a provider for an `AgentPatch` under the default
    json-schema mode, so this contract has to satisfy the same strictness rule
    every other output model does.
    """
    from llm.structured_output import json_schema_for

    schema = json_schema_for(AgentPatch, strict=True)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version",
        "map_url",
        "map_id",
        "base_sha256",
        "diagnostic_ids",
        "patch",
        "rationale",
        "provenance",
    }


def test_remove_carries_no_nullable_value_placeholder():
    from llm.structured_output import json_schema_for

    schema = json_schema_for(AgentPatch, strict=True)
    remove = schema["$defs"]["RemoveOperation"]

    assert "value" not in remove["properties"]
    assert set(remove["required"]) == {"op", "path"}


def test_the_operation_union_is_discriminated_on_op():
    from llm.structured_output import json_schema_for

    schema = json_schema_for(AgentPatch, strict=True)
    items = schema["properties"]["patch"]["items"]

    # a tagged union: the schema names `op` as the tag and maps each value
    assert items["discriminator"]["propertyName"] == "op"
    assert set(items["discriminator"]["mapping"]) == {"test", "add", "replace", "remove"}
    assert {branch["$ref"].rsplit("/", 1)[-1] for branch in items["oneOf"]} == {
        "TestOperation",
        "AddOperation",
        "ReplaceOperation",
        "RemoveOperation",
    }


# --- optional-dependency isolation ------------------------------------------


def test_importing_the_agent_package_loads_no_patch_library():
    """`jsonpatch` is an agent-mode dependency, imported when a patch is applied."""

    import subprocess
    import sys as _sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(repo / 'src')!r})\n"
        "import agent, agent.models, agent.policy, agent.patch\n"
        "print('LOADED' if 'jsonpatch' in sys.modules else 'ABSENT')\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", script], capture_output=True, text=True, cwd=repo, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ABSENT"


def test_missing_jsonpatch_reports_agent_mode_and_the_extra(monkeypatch):
    import builtins
    import sys as _sys

    from llm.errors import LLMDependencyError

    from agent import patch as patch_module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jsonpatch":
            raise ImportError("No module named 'jsonpatch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(_sys.modules, "jsonpatch", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(LLMDependencyError) as excinfo:
        patch_module._load_jsonpatch()

    assert "agent mode" in str(excinfo.value)
    assert "pip install -e '.[llm]'" in str(excinfo.value)


def test_the_value_node_is_typed_in_the_schema_the_provider_receives():
    """An untyped node becomes an object-only grammar under strict mode.

    Pydantic's `JsonValue` serializes to `{}`, and strict hardening leaves it
    alone because there is no object node to harden. Asked in isolation for the
    bare string `"modifierExtension"` with objects forbidden in the prompt, the
    endpoint answered `{"type": "string", "value": "modifierExtension"}` — it
    described the string rather than emitting it. Every repair that replaces a
    scalar leaf was unrepresentable.
    """
    from agent.models import ProposedPatch
    from llm.structured_output import json_schema_for

    schema = json_schema_for(ProposedPatch, strict=True)

    for name in ("AddOperation", "ReplaceOperation"):
        value = schema["$defs"][name]["properties"]["value"]
        types = {branch.get("type") for branch in value["anyOf"]}
        assert "string" in types, f"{name} cannot carry a bare string"
        assert {"boolean", "integer", "number", "object", "array", "null"} <= types


def test_a_patch_value_may_be_a_bare_scalar():
    """The model-side half: the union must still validate what it advertises."""
    from agent.models import ProposedPatch

    proposal = ProposedPatch.model_validate(
        {
            "schema_version": PATCH_SCHEMA_VERSION,
            "map_url": "http://e/x",
            "map_id": "x",
            "base_sha256": "0" * 64,
            "diagnostic_ids": [],
            "patch": [
                {"op": "replace", "path": "/group/0/rule/2/target/0/element",
                 "value": "modifierExtension"},
                {"op": "add", "path": "/group/0/rule/0/rule", "value": [{"name": "r"}]},
                {"op": "replace", "path": "/n", "value": 3},
                {"op": "replace", "path": "/b", "value": True},
            ],
            "rationale": "shapes",
        }
    )
    values = [op.value for op in proposal.patch]

    assert values[0] == "modifierExtension"
    assert values[1] == [{"name": "r"}]
    # `bool` precedes `int` in the union, so `True` must not arrive as `1`.
    assert values[2] == 3 and isinstance(values[2], int)
    assert values[3] is True and isinstance(values[3], bool)
