"""The shared prompt-template package.

Prompts live as `.j2` files so their wording is reviewable as prose. These tests
cover the rendering contract and the few pieces of guidance that exist because a
real run got them wrong — those are regressions, not decoration.
"""

import builtins
import sys

import pytest

from agent.context import (
    AgentContext,
    ContextListStatus,
    InsertionPoint,
    TargetElementBrief,
)
from llm.errors import LLMDependencyError
from llm.models import StructuredOutputMode
from llm.prompts import (
    AGENT_FIX_SYSTEM,
    AGENT_FIX_USER,
    AUTOMAPPING_SYSTEM,
    AUTOMAPPING_USER,
    PROMPT_DIR,
    available_templates,
    render,
)
from mapping.fml_creator.fml_automapper import AutomapCandidate, CandidateScores
from mapping.fml_creator.llm_automapping import UNMAPPED, Offer, _source_context

pytestmark = pytest.mark.unit


def _candidate(**overrides):
    base = dict(
        candidate_id="cand-1",
        scoring_path="Patient.name.family",
        legacy_emitted_path="HumanName.family",
        legacy_virtual_path="Patient.name.family",
        scores=CandidateScores(tfidf=0.91, weighted=0.4),
        flat_index=0,
        types=("string",),
        min=0,
        max="1",
        canonical_target_id="Patient.name.family",
    )
    base.update(overrides)
    return AutomapCandidate(**base)


def _offer(**overrides):
    base = dict(
        offer_id="c1",
        candidate=_candidate(),
        emitted_target="example-patient.name.family",
        res_type="Patient",
        res_id="example-patient",
        description="The family name",
    )
    base.update(overrides)
    return Offer(**base)


def _user_prompt(offers=None, source=None):
    return render(
        AUTOMAPPING_USER,
        source=_source_context(
            source
            or {
                "id": "familyName",
                "path": "src.familyName",
                "type": "string",
                "description": "Surname",
            }
        ),
        offers=offers if offers is not None else [_offer()],
        unmapped_token=UNMAPPED,
        pass_index=1,
        is_retry=False,
    )


# --- the package ------------------------------------------------------------


def test_templates_ship_with_the_package():
    # One directory for every prompt in the project: automapping (WP4) and agent
    # mode (WP7) share conventions rather than each growing their own home.
    assert available_templates() == [
        "agent_fix_system.j2",
        "agent_fix_user.j2",
        "automapping_system.j2",
        "automapping_user.j2",
    ]
    assert (PROMPT_DIR / AUTOMAPPING_SYSTEM).is_file()


def test_unknown_template_is_an_error():
    from jinja2 import TemplateNotFound

    with pytest.raises(TemplateNotFound):
        render("no_such_template.j2")


def test_a_missing_variable_fails_loudly():
    """A silently blank section would still be cached as if it were correct."""

    from jinja2 import UndefinedError

    with pytest.raises(UndefinedError):
        render(AUTOMAPPING_SYSTEM)  # unmapped_token not supplied


def test_missing_jinja2_reports_the_optional_extra(monkeypatch):
    from llm import prompts

    prompts._environment.cache_clear()
    monkeypatch.delitem(sys.modules, "jinja2", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jinja2":
            raise ImportError("No module named 'jinja2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(LLMDependencyError) as excinfo:
            render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)
        assert "jinja2" in str(excinfo.value)
        assert "pip install -e '.[llm]'" in str(excinfo.value)
    finally:
        prompts._environment.cache_clear()


# --- the user prompt --------------------------------------------------------


def test_user_prompt_enumerates_the_offers():
    prompt = _user_prompt(
        offers=[
            _offer(),
            _offer(offer_id="c2", emitted_target="example-patient.gender"),
        ]
    )

    assert "c1: example-patient.name.family" in prompt
    assert "c2: example-patient.gender" in prompt


def test_user_prompt_carries_the_candidate_facets():
    prompt = _user_prompt()

    assert "type=string" in prompt
    assert "card=0..1" in prompt
    assert "description=The family name" in prompt
    assert "similarity=0.910" in prompt


def test_user_prompt_shows_a_slice_qualifier():
    offer = _offer(
        candidate=_candidate(
            slice_name="mrn", canonical_target_id="Patient.identifier:mrn"
        ),
        emitted_target="example-patient.identifier:mrn",
    )

    prompt = _user_prompt(offers=[offer])

    assert "example-patient.identifier:mrn" in prompt
    assert "slice=mrn" in prompt


def test_user_prompt_omits_absent_source_facets():
    prompt = _user_prompt(source={"id": "bare"})

    # only the source block, not the trailing instructions (which mention "a path:")
    header = prompt.split("Candidate targets", 1)[0]
    assert "id: bare" in header
    assert "path:" not in header
    assert "description:" not in header
    assert "cardinality:" not in header


def test_user_prompt_states_the_answer_space():
    prompt = _user_prompt()

    assert UNMAPPED in prompt
    assert "Do not invent an id or a path" in prompt


# --- the system prompt ------------------------------------------------------


def test_system_prompt_describes_a_bounded_attempt_not_an_exhaustive_universe():
    prompt = render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)

    assert "only permitted" in prompt
    assert "current attempt" in prompt
    assert "exhaustive" not in prompt
    assert "does not exist" not in prompt


def test_system_prompt_distinguishes_scalar_leaves_from_complex_targets():
    """Adapted from the prototype, and confirmed by a real run.

    A WP4 smoke run mapped three distinct scalar source fields onto the complex
    `minimal-patient-profile.name`. The prototype's automap.j2 already carried
    a container warning. Keep that protection without rejecting valid mappings
    from complex source datatypes to whole complex FHIR elements.
    """
    prompt = render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)

    assert "scalar source" in prompt
    assert "Patient.name.family" in prompt
    assert "HumanName" in prompt
    assert "Patient.name" in prompt
    assert "complex element is valid" in prompt


def test_system_prompt_treats_similarity_as_retrieval_evidence():
    prompt = render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)

    assert "retrieval evidence" in prompt
    assert "lower-ranked candidate" in prompt


def test_system_prompt_carries_the_fhir_semantics_hints():
    prompt = render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)

    for hint in ("identifier.value", "name.family", "birthDate", "coding.code"):
        assert hint in prompt, f"missing FHIR hint: {hint}"


def test_system_prompt_prefers_unmapped_over_a_forced_match():
    prompt = render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)

    assert "unmapped over forcing" in prompt


def test_system_prompt_interpolates_the_unmapped_token():
    prompt = render(AUTOMAPPING_SYSTEM, unmapped_token="none-of-these")

    assert "none-of-these" in prompt
    assert "{{" not in prompt


def test_no_template_leaks_unrendered_syntax():
    prompt = _user_prompt() + render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)

    assert "{{" not in prompt
    assert "{%" not in prompt


def test_prompts_are_provider_neutral():
    """Nothing here may assume a particular vendor or model family."""

    for name in available_templates():
        text = (PROMPT_DIR / name).read_text(encoding="utf-8").lower()
        for vendor in ("openai", "gpt-", "claude", "anthropic", "gemini", "llama"):
            assert vendor not in text, f"{name} mentions {vendor}"


def test_structured_output_mode_is_not_described_in_the_prompt():
    """Schema instructions belong to the WP2 layer, not to these templates.

    Duplicating them here would let the two drift, and in the server-constrained
    modes the schema is not sent in the prompt at all.
    """
    system = render(AUTOMAPPING_SYSTEM, unmapped_token=UNMAPPED)

    assert StructuredOutputMode.JSON_SCHEMA.value not in system
    assert "json schema" not in system.lower()


# --- agent repair prompts ---------------------------------------------------


def _agent_context():
    statuses = {
        name: ContextListStatus(total=total, shown=shown, truncated=shown < total)
        for name, total, shown in (
            ("pointers", 5, 1),
            ("source_fields", 91, 1),
            ("mapping_rows", 2, 1),
            ("required_elements", 2, 1),
            ("prohibited_paths", 2, 1),
        )
    }
    return AgentContext(
        map_url="http://example.org/StructureMap/map",
        map_id="map",
        map_sha256="abc123",
        resource_type="Patient",
        source_type="Source",
        findings=[],
        insertion_points=[
            InsertionPoint(
                pointer="/group/0/rule/-",
                label="Append a top-level rule",
                target_paths=["Patient.active"],
            )
        ],
        source_fields=[{"name": "enabled", "type": "boolean"}],
        mapping_rows=[
            {"source": "enabled", "target": "Patient.active", "fixed_value": False}
        ],
        required_elements=[
            TargetElementBrief(
                path="Patient.active",
                cardinality="1..1",
                required=True,
                fixed_value=False,
            )
        ],
        prohibited_paths=["Patient.deceased[x]"],
        repair_hints=[
            {
                "path": "Observation.effective",
                "source_field": "when",
                "options": [
                    {
                        "element": "effective",
                        "type": "dateTime",
                        "transform": "cast",
                    }
                ],
            }
        ],
        list_status=statuses,
    )


def test_agent_system_prompt_matches_the_enforced_safety_contract():
    prompt = render(AGENT_FIX_SYSTEM)

    for field in ("resourceType", "id", "url", "name", "version"):
        assert f"`{field}`" in prompt
    assert "untrusted data" in prompt
    assert "Authorized insertion points" in prompt
    assert "leaves the baseline unchanged" in prompt
    assert "byte-for-byte" not in prompt
    assert "remove` plus `add" not in prompt
    assert "pointer of every rule" not in prompt
    assert 'element="effective", cast(src, "dateTime")' in prompt
    assert "effectiveDateTimeDateTime" in prompt


def test_agent_user_prompt_renders_falsy_constants_and_bounded_context():
    prompt = render(AGENT_FIX_USER, context=_agent_context(), map_json="{}")

    assert prompt.count("`false`") == 2
    assert "Showing 1 of 91 project source fields" in prompt
    assert "omitted field is not authorized" in prompt
    assert "/group/0/rule/-" in prompt
    # The base element name, never the concrete choice variant — Matchbox
    # derives the variant from the transform and doubling it yields
    # `effectiveDateTimeDateTime`.
    assert '"element": "effective"' in prompt
    assert "effectiveDateTime" not in prompt
    # A target entry without a context writes nothing: the entry is accepted,
    # the transform runs, and the element is still missing afterwards.
    assert '"contextType": "variable"' in prompt
    assert "Never address an elided marker" not in prompt  # map is not pruned


def test_each_member_pointer_gets_its_own_line():
    """`trim_blocks` eats the newline after a block tag.

    A line ending in `{% endif %}` therefore runs into the next one, which
    collapsed the whole member list onto a single unreadable line while every
    value in it was still correct — invisible to any test that only checked the
    pointers were present.
    """
    from agent.context import AgentContext, RulePointer, _editable_members
    from llm.prompts import AGENT_FIX_USER, render

    rule = {
        "name": "map-gender",
        "source": [{"context": "src", "element": "gender"}],
        "target": [{"context": "tgt", "element": "gender", "transform": "copy"}],
    }
    pointer = RulePointer(
        pointer="/group/0/rule/3",
        label="G › map-gender",
        name="map-gender",
        focused=True,
        target_variables=["tgt"],
        members=_editable_members(rule, "/group/0/rule/3"),
    )
    context = AgentContext(map_url="http://e/x", map_id="x", pointers=[pointer])

    rendered = render(AGENT_FIX_USER, context=context, map_json="{}")

    assert '    - `/group/0/rule/3/source/0/element` (string) = "gender"\n' in rendered
    assert "    - `/group/0/rule/3/target/-` — append a new entry here\n" in rendered
    # The line above the list must not have been absorbed into it either.
    assert "  - variables in scope: source none; target `tgt`\n" in rendered


def test_the_system_prompt_no_longer_teaches_a_protocol_the_schema_forbids():
    from llm.prompts import AGENT_FIX_SYSTEM, render

    rendered = render(AGENT_FIX_SYSTEM)

    assert '"op": "test"' not in rendered
    assert "write a `test`.**" in rendered
