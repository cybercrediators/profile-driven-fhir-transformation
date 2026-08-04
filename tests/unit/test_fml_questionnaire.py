"""Unit tests for QuestionnaireMapCreator (fml_questionnaire.py).

Builds a small, realistic Questionnaire (a group with a nested display item, a coded
single-select, a multi-select "checkbox" item, and an unmapped leaf) plus a
QuestionnaireResponse-style mapping table, and exercises `generate_group` end to end —
covering group/display recursion, mapped/unmapped/multi-source leaves, coded (ConceptMap +
$translate) answers, and the status rule. All ConceptMap generation is offline: answer
options are supplied directly (answerOption), so `expand_valueset`/the terminology
connector is never invoked.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from fhir.resources.R4B.questionnaire import (
    Questionnaire,
    QuestionnaireItem,
    QuestionnaireItemAnswerOption,
    QuestionnaireItemEnableWhen,
    QuestionnaireItemInitial,
)
from fhir.resources.R4B.coding import Coding
from fhir.resources.R4B.expression import Expression
from fhir.resources.R4B.extension import Extension
from fhir.resources.R4B.quantity import Quantity
from fhir.resources.R4B.reference import Reference

from data_handling.data_io import DataIO
from mapping.fml_creator.fml_factory import FMLRuleFactory
from mapping.fml_creator.fml_questionnaire import QuestionnaireMapCreator

pytestmark = pytest.mark.unit


class _FakeDataIO:
    ProjectFolders = DataIO.ProjectFolders

    def __init__(self):
        self.stored = []

    def project_file_exists(self, folder, filename):
        return False

    def store_project_file(self, folder, filename, content, mode="STR", overwrite=False):
        self.stored.append((folder, filename, content))


@pytest.fixture
def factory():
    f = object.__new__(FMLRuleFactory)
    f.app_state = SimpleNamespace(dataIO=_FakeDataIO())
    f.map_url = "http://example.org/fml"
    f.overwrite = False
    f.plugins = []
    return f


def _coding_option(code, system, display=None):
    return QuestionnaireItemAnswerOption.model_construct(
        valueCoding=Coding.model_construct(code=code, system=system, display=display)
    )


@pytest.fixture
def questionnaire():
    grp = QuestionnaireItem.model_construct(
        linkId="grp",
        type="group",
        text="Group",
        item=[
            QuestionnaireItem.model_construct(linkId="q1", type="boolean", text="Q1"),
            QuestionnaireItem.model_construct(
                linkId="display1",
                type="display",
                text="A display item",
                item=[QuestionnaireItem.model_construct(linkId="q1b", type="string", text="Q1b")],
            ),
        ],
    )
    q2 = QuestionnaireItem.model_construct(
        linkId="q2",
        type="choice",
        text="Q2",
        answerOption=[
            _coding_option("yes", "http://example.org/cs", "Yes"),
            _coding_option("no", "http://example.org/cs", "No"),
        ],
    )
    q3 = QuestionnaireItem.model_construct(
        linkId="q3",
        type="choice",
        text="Q3",
        answerOption=[
            _coding_option("a", "http://example.org/cs2"),
            _coding_option("b", "http://example.org/cs2"),
        ],
    )
    q4 = QuestionnaireItem.model_construct(linkId="q4", type="string", text="Q4")
    return Questionnaire.model_construct(name="TestQuestionnaire", item=[grp, q2, q3, q4])


@pytest.fixture
def mapping_table():
    return {
        "Src.flag": "QuestionnaireResponse.item[q1]",
        "Src.q1bVal": "QuestionnaireResponse.item[q1b]",
        "Src.q2Val": "QuestionnaireResponse.item[q2]",
        "Src.q3ValA": "QuestionnaireResponse.item[q3]",
        "Src.q3ValB": "QuestionnaireResponse.item[q3]",
        "Src.status": "QuestionnaireResponse.status",
    }


@pytest.fixture
def creator(questionnaire, factory):
    return QuestionnaireMapCreator(SimpleNamespace(data=questionnaire), factory=factory)


def _enable_expression(rule):
    """The compiled enableWhen carried on a rule.

    Enablement is compiled exactly as before, but it is no longer emitted as a
    rule source: matchbox cannot evaluate a source whose context is the target
    being built and aborts the whole map. The expression is retained as
    documentation so the map still records the requirement.
    """
    doc = rule.documentation or ""
    marker = "INACTIVE"
    assert marker in doc, f"expected an inactive-enableWhen note, got {doc!r}"
    return doc.split("): ", 1)[1]


def _by_name(rules, name):
    return next(r for r in rules if r.name == name)


# ── index-building helpers ───────────────────────────────────────────────────────
def test_build_linkid_source_index_groups_multi_source_fields(creator, mapping_table):
    idx = creator._build_linkid_source_index(mapping_table)
    assert idx["q1"] == ["Src.flag"]
    assert idx["q3"] == ["Src.q3ValA", "Src.q3ValB"]


def test_get_status_source_field(creator, mapping_table):
    assert creator._get_status_source_field(mapping_table) == "Src.status"
    assert creator._get_status_source_field({}) is None


# ── generate_group: overall shape ────────────────────────────────────────────────
def test_generate_group_overall_shape(creator, mapping_table):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    assert group.name == "TestQuestionnaire"
    # status rule first, then the subject reference placeholder (wired by the
    # bundle assembler), then one outer rule per top-level item (grp, q2, q3, q4)
    assert group.rule[0].name == "set-status"
    assert group.rule[1].name == "TODO-resolve-reference-QuestionnaireResponse-subject"
    top_names = [r.name for r in group.rule[2:]]
    assert top_names == ["rule-grp", "rule-q2", "rule-q3", "rule-q4"]


def test_questionnaire_link_rule_emitted_when_canonical_known(questionnaire, factory):
    q = Questionnaire.model_construct(
        name=questionnaire.name,
        item=questionnaire.item,
        url="http://example.org/Questionnaire/TestQ",
    )
    creator = QuestionnaireMapCreator(SimpleNamespace(data=q), factory=factory)
    group = creator.generate_group("Source", "TestQ")
    rule = next(r for r in group.rule if r.name == "assign-questionnaire")
    assert rule.target[0].element == "questionnaire"
    assert rule.target[0].parameter[0].valueString == "http://example.org/Questionnaire/TestQ"


def test_no_questionnaire_link_rule_without_canonical(creator):
    group = creator.generate_group("Source", "TestQ")
    assert not any(r.name == "assign-questionnaire" for r in group.rule)


def test_unmapped_required_status_keeps_todo_scaffold(creator):
    group = creator.generate_group("Source", "TestQ")
    status_rule = _by_name(group.rule, "set-status")
    assert status_rule.source[0].element == "TODO_MAP_STATUS"
    assert status_rule.target[0].element == "status"
    assert status_rule.target[0].parameter[0].valueId == "srcStatus"


def test_subject_placeholder_rule_shape(creator):
    group = creator.generate_group("Source", "TestQ")
    rule = next(r for r in group.rule
                if r.name == "TODO-resolve-reference-QuestionnaireResponse-subject")
    # documentation must match BundleService._collect_todo_refs' regex contract
    assert rule.documentation.startswith("Reference<QuestionnaireResponse.subject>")
    assert "Patient" in rule.documentation  # default when no subjectType declared
    assert rule.target is None
    assert rule.source[0].context == "Source"


def test_status_rule_uses_mapped_source(creator, mapping_table):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    status_rule = group.rule[0]
    assert status_rule.source[0].element == "status"  # local_element_name("Src.status")
    assert status_rule.target[0].element == "status"
    # No ConceptMap is needed for an already-compatible response status code.
    assert status_rule.target[0].transform == "copy"
    assert status_rule.target[0].parameter[0].valueId == "srcStatus"


# ── group + display recursion: display produces no rule of its own ──────────────
def test_group_item_flattens_nested_display_child(creator, mapping_table):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    grp_rule = _by_name(group.rule, "rule-grp")
    # assign-linkId, assign-text, then q1's rule and q1b's rule (display1 itself never
    # appears — its child is spliced directly into the parent's rule list).
    names = [r.name for r in grp_rule.rule]
    assert names == ["assign-linkId", "assign-text", "rule-q1", "rule-q1b"]


def test_mapped_boolean_leaf_item_uses_copy_answer_rule(creator, mapping_table):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    grp_rule = _by_name(group.rule, "rule-grp")
    q1_rule = _by_name(grp_rule.rule, "rule-q1")
    assert q1_rule.source[0].element == "flag"
    answer_rule = _by_name(q1_rule.rule, "answer-rule-q1")
    set_val = answer_rule.rule[0]
    assert set_val.target[0].element == "valueBoolean"
    assert set_val.target[0].transform == "copy"


def test_mapped_string_leaf_under_display_uses_string_value_type(creator, mapping_table):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    grp_rule = _by_name(group.rule, "rule-grp")
    q1b_rule = _by_name(grp_rule.rule, "rule-q1b")
    answer_rule = _by_name(q1b_rule.rule, "answer-rule-q1b")
    assert answer_rule.rule[0].target[0].element == "valueString"


# ── coded single-select: ConceptMap + $translate ─────────────────────────────────
def test_coded_choice_item_generates_concept_map_and_translate(creator, mapping_table, factory):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    q2_rule = _by_name(group.rule, "rule-q2")
    answer_rule = _by_name(q2_rule.rule, "answer-rule-q2")
    coding_rule = answer_rule.rule[0]
    assert coding_rule.name == "set-coding"
    assert coding_rule.target[0].transform == "create"
    assert coding_rule.target[0].parameter[0].valueString == "Coding"

    code_rule = _by_name(coding_rule.rule, "set-code")
    assert code_rule.target[0].transform == "translate"
    cm_url = code_rule.target[0].parameter[1].valueString
    assert "ConceptMap/cm-item-q2-" in cm_url
    assert any("cm-item-q2-" in name for _, name, _ in factory.app_state.dataIO.stored)

    system_rule = _by_name(coding_rule.rule, "set-system")
    assert system_rule.target[0].parameter[0].valueString == "http://example.org/cs"


# ── multi-source checkbox item ───────────────────────────────────────────────────
def test_multi_source_item_emits_one_checkbox_rule_per_source(creator, mapping_table):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    q3_rule = _by_name(group.rule, "rule-q3")
    names = [r.name for r in q3_rule.rule]
    assert names == ["assign-linkId", "assign-text", "cb-q3ValA", "cb-q3ValB"]

    cb_a = _by_name(q3_rule.rule, "cb-q3ValA")
    assert cb_a.source[0].element == "q3ValA"
    assert cb_a.source[0].condition == "$this = '1'"
    coding1 = cb_a.rule[0]
    assert coding1.name == "set-coding-1"
    code1 = coding1.rule[0]
    assert code1.name == "set-code-1"
    assert code1.target[0].parameter[0].valueString == "a"
    system1 = coding1.rule[1]
    assert system1.name == "set-system-1"
    assert system1.target[0].parameter[0].valueString == "http://example.org/cs2"

    cb_b = _by_name(q3_rule.rule, "cb-q3ValB")
    code2 = cb_b.rule[0].rule[0]
    assert code2.name == "set-code-2"
    assert code2.target[0].parameter[0].valueString == "b"


# ── unmapped leaf: placeholder answer rule ───────────────────────────────────────
def test_unmapped_leaf_item_gets_placeholder_answer_rule(creator, mapping_table):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    q4_rule = _by_name(group.rule, "rule-q4")
    # No source element for the outer item (nothing mapped for q4).
    assert q4_rule.source[0].element is None
    placeholder = _by_name(q4_rule.rule, "answer-rule-q4")
    assert placeholder.source[0].element == "TODO-map-q4"
    set_val = placeholder.rule[0]
    assert set_val.target[0].element == "valueString"


def test_unmapped_item_uses_typed_initial_value_instead_of_placeholder(factory):
    item = QuestionnaireItem.model_construct(
        linkId="initial-int",
        type="integer",
        initial=[QuestionnaireItemInitial.model_construct(valueInteger=7)],
    )
    questionnaire = Questionnaire.model_construct(name="Initials", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group("Source", "Target")
    item_rule = _by_name(group.rule, "rule-initial-int")
    assert not any(
        (source.element or "").startswith("TODO-")
        for rule in item_rule.rule
        for source in (rule.source or [])
    )
    initial = _by_name(item_rule.rule, "initial-rule-initial-int-1")
    set_initial = initial.rule[0]
    assert set_initial.target[0].element == "valueInteger"
    assert set_initial.target[0].parameter[0].valueInteger == 7


def test_initial_selected_answer_option_is_used_as_default(factory):
    item = QuestionnaireItem.model_construct(
        linkId="selected",
        type="choice",
        answerOption=[
            QuestionnaireItemAnswerOption.model_construct(
                valueInteger=2, initialSelected=True
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(name="Selected", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group("Source", "Target")
    item_rule = _by_name(group.rule, "rule-selected")
    initial = _by_name(item_rule.rule, "initial-rule-selected-1")
    assert initial.rule[0].target[0].parameter[0].valueInteger == 2


def test_complex_coding_initial_is_emitted_recursively(factory):
    item = QuestionnaireItem.model_construct(
        linkId="initial-code",
        type="choice",
        initial=[
            QuestionnaireItemInitial.model_construct(
                valueCoding=Coding.model_construct(
                    system="http://example.org/system",
                    code="answer",
                    display="Answer",
                )
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(name="CodingInitial", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group("Source", "Target")
    initial = _by_name(
        _by_name(group.rule, "rule-initial-code").rule,
        "initial-rule-initial-code-1",
    )
    fixed = initial.rule[0]
    assert fixed.target[0].element == "value"
    assert fixed.target[0].transform == "create"
    assert fixed.target[0].parameter[0].valueString == "Coding"
    assert {
        rule.target[0].element for rule in fixed.rule
    } == {"system", "code", "display"}


def test_mapped_repeating_group_scopes_child_sources_to_group_item(factory):
    child = QuestionnaireItem.model_construct(
        linkId="group-value", type="string", text="Value"
    )
    repeated_group = QuestionnaireItem.model_construct(
        linkId="repeated-group",
        type="group",
        repeats=True,
        item=[child],
    )
    questionnaire = Questionnaire.model_construct(
        name="RepeatedGroups", item=[repeated_group]
    )
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={
            "Src.groups": "QuestionnaireResponse.item[repeated-group]",
            "Src.groups.value": "QuestionnaireResponse.item[group-value]",
        },
    )
    group_rule = _by_name(group.rule, "rule-repeated-group")
    child_rule = _by_name(group_rule.rule, "rule-group-value")
    assert group_rule.source[0].element == "groups"
    assert child_rule.source[0].context == "srcVal"
    assert child_rule.source[0].element == "value"


def test_mapped_string_item_enforces_questionnaire_max_length(factory):
    item = QuestionnaireItem.model_construct(
        linkId="short", type="string", maxLength=12
    )
    questionnaire = Questionnaire.model_construct(name="Bounds", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.short": "QuestionnaireResponse.item[short]"},
    )
    item_rule = _by_name(group.rule, "rule-short")
    assert item_rule.source[0].check == "($this.toString().length() <= 12)"


def test_primitive_enable_when_is_compiled_against_response_answers(factory):
    trigger = QuestionnaireItem.model_construct(
        linkId="trigger", type="boolean"
    )
    dependent = QuestionnaireItem.model_construct(
        linkId="dependent",
        type="string",
        enableWhen=[
            QuestionnaireItemEnableWhen.model_construct(
                question="trigger",
                operator="=",
                answerBoolean=True,
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(
        name="Enablement", item=[trigger, dependent]
    )
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={
            "Src.flag": "QuestionnaireResponse.item[trigger]",
            "Src.value": "QuestionnaireResponse.item[dependent]",
        },
    )
    item_rule = _by_name(group.rule, "rule-dependent")
    assert all(s.context != "Target" for s in item_rule.source)
    assert _enable_expression(item_rule) == (
        "($this.repeat(item).where(linkId = 'trigger').answer.value"
        ".where($this = true).exists())"
    )


def test_enable_when_any_combines_primitive_conditions(factory):
    first = QuestionnaireItem.model_construct(linkId="first", type="integer")
    second = QuestionnaireItem.model_construct(linkId="second", type="string")
    dependent = QuestionnaireItem.model_construct(
        linkId="dependent",
        type="string",
        enableBehavior="any",
        enableWhen=[
            QuestionnaireItemEnableWhen.model_construct(
                question="first", operator=">", answerInteger=2
            ),
            QuestionnaireItemEnableWhen.model_construct(
                question="second", operator="=", answerString="yes"
            ),
        ],
    )
    questionnaire = Questionnaire.model_construct(
        name="EnableAny", item=[first, second, dependent]
    )
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={
            "Src.first": "QuestionnaireResponse.item[first]",
            "Src.second": "QuestionnaireResponse.item[second]",
            "Src.value": "QuestionnaireResponse.item[dependent]",
        },
    )
    condition = _enable_expression(_by_name(group.rule, "rule-dependent"))
    assert condition == (
        "($this.repeat(item).where(linkId = 'first').answer.value"
        ".where($this > 2).exists()) or "
        "($this.repeat(item).where(linkId = 'second').answer.value"
        ".where($this = 'yes').exists())"
    )


def test_unresolvable_enable_when_is_diagnosed_and_left_ungated(factory):
    dependent = QuestionnaireItem.model_construct(
        linkId="dependent",
        type="string",
        enableWhen=[
            QuestionnaireItemEnableWhen.model_construct(
                question="coded",
                operator=">",
                answerCoding=Coding.model_construct(
                    system="http://example.org/system", code="yes"
                ),
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(
        name="EnableDiagnostic", item=[dependent]
    )
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.value": "QuestionnaireResponse.item[dependent]"},
    )
    assert len(_by_name(group.rule, "rule-dependent").source) == 1
    assert {
        diagnostic["code"] for diagnostic in factory.diagnostics
    } == {"questionnaire-enablewhen-authoring-required"}


@pytest.mark.parametrize(
    "answer_property,answer,expected_fragments",
    [
        (
            "answerCoding",
            Coding.model_construct(
                system="http://example.org/system", code="yes"
            ),
            ("system = 'http://example.org/system'", "code = 'yes'"),
        ),
        (
            "answerQuantity",
            Quantity.model_construct(
                value=Decimal("4.5"),
                system="http://unitsofmeasure.org",
                code="mg",
            ),
            (
                "value >= 4.5",
                "system = 'http://unitsofmeasure.org'",
                "code = 'mg'",
            ),
        ),
        (
            "answerReference",
            Reference.model_construct(reference="Patient/example"),
            ("reference = 'Patient/example'",),
        ),
    ],
)
def test_complex_enable_when_compiles_target_answer_predicate(
    factory, answer_property, answer, expected_fragments
):
    enable_when = QuestionnaireItemEnableWhen.model_construct(
        question="trigger", operator=">=" if answer_property == "answerQuantity" else "="
    )
    setattr(enable_when, answer_property, answer)
    dependent = QuestionnaireItem.model_construct(
        linkId="dependent",
        type="string",
        enableWhen=[enable_when],
    )
    questionnaire = Questionnaire.model_construct(
        name="ComplexEnablement", item=[dependent]
    )
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.value": "QuestionnaireResponse.item[dependent]"},
    )
    condition = _enable_expression(_by_name(group.rule, "rule-dependent"))
    assert all(fragment in condition for fragment in expected_fragments)


def test_enable_when_expression_is_evaluated_against_response(factory):
    item = QuestionnaireItem.model_construct(
        linkId="dependent",
        type="string",
        extension=[
            Extension.model_construct(
                url=(
                    "http://hl7.org/fhir/uv/sdc/StructureDefinition/"
                    "sdc-questionnaire-enableWhenExpression"
                ),
                valueExpression=Expression.model_construct(
                    language="text/fhirpath",
                    expression=(
                        "%resource.item.where(linkId='trigger').answer.exists()"
                    ),
                ),
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(
        name="ExpressionEnablement", item=[item]
    )
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.value": "QuestionnaireResponse.item[dependent]"},
    )
    condition = _enable_expression(_by_name(group.rule, "rule-dependent"))
    assert condition == "$this.item.where(linkId='trigger').answer.exists()"


@pytest.mark.parametrize(
    "extension_name,rule_name",
    [
        ("sdc-questionnaire-initialExpression", "initial-expression-rule-derived"),
        (
            "sdc-questionnaire-calculatedExpression",
            "calculated-expression-rule-derived",
        ),
    ],
)
def test_authored_fhirpath_expression_emits_evaluate_answer(
    factory, extension_name, rule_name
):
    item = QuestionnaireItem.model_construct(
        linkId="derived",
        type="integer",
        extension=[
            Extension.model_construct(
                url=(
                    "http://hl7.org/fhir/uv/sdc/StructureDefinition/"
                    f"{extension_name}"
                ),
                valueExpression=Expression.model_construct(
                    language="text/fhirpath",
                    expression="%resource.item.answer.value.sum()",
                ),
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(name="Expressions", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group("Source", "Target")
    item_rule = _by_name(group.rule, "rule-derived")
    expression_rule = _by_name(item_rule.rule, rule_name)
    evaluate = expression_rule.rule[0].target[0]
    assert evaluate.element == "valueInteger"
    assert evaluate.transform == "evaluate"
    assert evaluate.parameter[0].valueId == "qrExpressionContext"
    assert evaluate.parameter[1].valueString == "$this.item.answer.value.sum()"


def test_unsupported_expression_context_is_diagnosed_and_keeps_placeholder(factory):
    item = QuestionnaireItem.model_construct(
        linkId="external",
        type="string",
        extension=[
            Extension.model_construct(
                url=(
                    "http://hl7.org/fhir/uv/sdc/StructureDefinition/"
                    "sdc-questionnaire-calculatedExpression"
                ),
                valueExpression=Expression.model_construct(
                    language="text/fhirpath",
                    expression="%patient.name.first().family",
                ),
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(name="Expressions", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group("Source", "Target")
    item_rule = _by_name(group.rule, "rule-external")
    assert _by_name(item_rule.rule, "answer-rule-external")
    assert {
        diagnostic["code"] for diagnostic in factory.diagnostics
    } == {"questionnaire-expression-context-unavailable"}


def test_answer_expression_is_not_mistaken_for_a_selected_answer(factory):
    item = QuestionnaireItem.model_construct(
        linkId="dynamic-options",
        type="choice",
        extension=[
            Extension.model_construct(
                url=(
                    "http://hl7.org/fhir/uv/sdc/StructureDefinition/"
                    "sdc-questionnaire-answerExpression"
                ),
                valueExpression=Expression.model_construct(
                    language="text/fhirpath",
                    expression="%resource.item.answer.value",
                ),
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(
        name="AnswerExpression", item=[item]
    )
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group("Source", "Target")
    assert _by_name(
        _by_name(group.rule, "rule-dynamic-options").rule,
        "answer-rule-dynamic-options",
    )
    assert {
        diagnostic["code"] for diagnostic in factory.diagnostics
    } == {"questionnaire-answer-expression-candidate-only"}


def test_repeating_question_creates_one_item_and_iterates_answers(factory):
    item = QuestionnaireItem.model_construct(
        linkId="tags", type="string", repeats=True
    )
    questionnaire = Questionnaire.model_construct(name="Repeating", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.tags": "QuestionnaireResponse.item[tags]"},
    )
    item_rule = _by_name(group.rule, "rule-tags")
    assert item_rule.source[0].element is None
    assert item_rule.source[0].variable == "srcItemRoot"
    answer_rule = _by_name(item_rule.rule, "answer-rule-tags")
    assert answer_rule.source[0].context == "srcItemRoot"
    assert answer_rule.source[0].element == "tags"
    assert answer_rule.source[0].variable == "srcVal"


def test_checkbox_sources_create_one_item_with_multiple_answers(
    creator, mapping_table
):
    group = creator.generate_group("Source", "Target", mapping_table=mapping_table)
    item_rule = _by_name(group.rule, "rule-q3")
    assert item_rule.source[0].element is None
    assert item_rule.source[0].variable == "srcItemRoot"


def test_nonrepeating_question_rejects_repeating_source_collection(factory):
    item = QuestionnaireItem.model_construct(
        linkId="single", type="string", repeats=False
    )
    questionnaire = Questionnaire.model_construct(name="Cardinality", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )
    factory.source_field_max = {"values": "*"}

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.values": "QuestionnaireResponse.item[single]"},
    )
    item_rule = _by_name(group.rule, "rule-single")
    assert item_rule.source[0].element is None
    assert item_rule.source[0].check == "((values.count()) <= 1)"
    assert _by_name(item_rule.rule, "answer-rule-single").source[0].element == "values"


def test_required_question_guards_missing_source_before_item_creation(factory):
    item = QuestionnaireItem.model_construct(
        linkId="required", type="string", required=True
    )
    questionnaire = Questionnaire.model_construct(name="Required", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.value": "QuestionnaireResponse.item[required]"},
    )
    item_rule = _by_name(group.rule, "rule-required")
    assert item_rule.source[0].element is None
    assert item_rule.source[0].check == "((value.count()) >= 1)"
    assert _by_name(item_rule.rule, "answer-rule-required").source[0].element == "value"


def test_answer_constraints_are_attached_to_mapped_value(factory):
    item = QuestionnaireItem.model_construct(
        linkId="score",
        type="choice",
        answerOption=[
            QuestionnaireItemAnswerOption.model_construct(valueInteger=1),
            QuestionnaireItemAnswerOption.model_construct(valueInteger=3),
        ],
        extension=[
            Extension.model_construct(
                url="http://hl7.org/fhir/StructureDefinition/minValue",
                valueInteger=1,
            ),
            Extension.model_construct(
                url="http://hl7.org/fhir/StructureDefinition/maxValue",
                valueInteger=3,
            ),
        ],
    )
    questionnaire = Questionnaire.model_construct(name="Constraints", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.score": "QuestionnaireResponse.item[score]"},
    )
    check = _by_name(group.rule, "rule-score").source[0].check
    assert "($this >= 1)" in check
    assert "($this <= 3)" in check
    assert "($this = 1 or $this = 3)" in check
    assert (
        _by_name(_by_name(group.rule, "rule-score").rule, "answer-rule-score")
        .rule[0]
        .target[0]
        .element
        == "valueInteger"
    )


def test_questionnaire_occurrence_extensions_guard_answer_count(factory):
    item = QuestionnaireItem.model_construct(
        linkId="tags",
        type="string",
        repeats=True,
        extension=[
            Extension.model_construct(
                url=(
                    "http://hl7.org/fhir/StructureDefinition/"
                    "questionnaire-minOccurs"
                ),
                valueInteger=2,
            ),
            Extension.model_construct(
                url=(
                    "http://hl7.org/fhir/StructureDefinition/"
                    "questionnaire-maxOccurs"
                ),
                valueInteger=4,
            ),
        ],
    )
    questionnaire = Questionnaire.model_construct(name="Occurrences", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.tags": "QuestionnaireResponse.item[tags]"},
    )
    check = _by_name(group.rule, "rule-tags").source[0].check
    assert check == "((tags.count()) >= 2) and ((tags.count()) <= 4)"


def test_child_question_is_nested_below_parent_answer(factory):
    child = QuestionnaireItem.model_construct(
        linkId="child", type="string"
    )
    parent = QuestionnaireItem.model_construct(
        linkId="parent", type="string", item=[child]
    )
    questionnaire = Questionnaire.model_construct(name="Nested", item=[parent])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={
            "Src.parent": "QuestionnaireResponse.item[parent]",
            "Src.child": "QuestionnaireResponse.item[child]",
        },
    )
    parent_rule = _by_name(group.rule, "rule-parent")
    answer_rule = _by_name(parent_rule.rule, "answer-rule-parent")
    child_rule = _by_name(answer_rule.rule, "rule-child")
    assert child_rule.target[0].context == "tAns"
    assert child_rule.target[0].element == "item"


def test_repeating_question_children_require_authored_correlation(factory):
    child = QuestionnaireItem.model_construct(
        linkId="child", type="string"
    )
    parent = QuestionnaireItem.model_construct(
        linkId="parent", type="string", repeats=True, item=[child]
    )
    questionnaire = Questionnaire.model_construct(name="Nested", item=[parent])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    creator.generate_group(
        "Source",
        "Target",
        mapping_table={
            "Src.parent": "QuestionnaireResponse.item[parent]",
            "Src.child": "QuestionnaireResponse.item[child]",
        },
    )
    assert {
        diagnostic["code"] for diagnostic in factory.diagnostics
    } == {"questionnaire-nested-answer-correlation-required"}


def test_date_enable_when_uses_fhirpath_date_literal(factory):
    item = QuestionnaireItem.model_construct(
        linkId="dependent",
        type="string",
        enableWhen=[
            QuestionnaireItemEnableWhen.model_construct(
                question="date",
                operator=">",
                answerDate="2025-01-01",
            )
        ],
    )
    questionnaire = Questionnaire.model_construct(name="Dates", item=[item])
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=questionnaire), factory=factory
    )

    group = creator.generate_group(
        "Source",
        "Target",
        mapping_table={"Src.value": "QuestionnaireResponse.item[dependent]"},
    )
    condition = _enable_expression(_by_name(group.rule, "rule-dependent"))
    assert ".where($this > @2025-01-01).exists()" in condition


def test_checkbox_integer_option_keeps_integer_literal_type(creator):
    rules = creator._fixed_option_value_rules(
        {"element": "valueInteger", "value": 3},
        "integer",
        1,
    )
    parameter = rules[0].target[0].parameter[0]
    assert parameter.valueInteger == 3
    assert parameter.valueString is None


# ── _get_value_type: pure mapping table ──────────────────────────────────────────
@pytest.mark.parametrize(
    "item_type,expected",
    [
        ("boolean", "valueBoolean"),
        ("decimal", "valueDecimal"),
        ("choice", "valueCoding"),
        ("quantity", "valueQuantity"),
        ("unknown-type", "valueString"),
    ],
)
def test_get_value_type(creator, item_type, expected):
    assert creator._get_value_type(item_type) == expected


# ── Label-valued options fed by a coded source (REDCap shape) ─────────────────
#
# kfdm's Questionnaire declares `answerOption.valueString: "Hausarzt"` while the
# REDCap export supplies the codebook's code `"1"`. Copying the raw value stores a
# bare "1", and enumerating the labels in a source `check` fails for every record —
# fatally, because a failed FML check aborts the whole map for that record. The
# plugin already holds the code list; it just was never consulted on this path.

class _CodebookPlugin:
    """Minimal stand-in for the REDCap plugin's codebook surface."""

    plugin_id = "fake-redcap"

    def __init__(self, field, choices):
        self._field = field
        self._choices = choices

    def get_field_metadata(self, field_name):
        if field_name != self._field:
            return None
        return {"_choices": self._choices}

    def generate_concept_map(self, field_name, field_meta, existing_cm):
        return {
            "resourceType": "ConceptMap",
            "status": "draft",
            "group": [{
                "element": [
                    {"code": c["code"],
                     "target": [{"code": c["label"], "equivalence": "equivalent"}]}
                    for c in field_meta["_choices"]
                ]
            }],
        }


_CHOICES = [{"code": "1", "label": "Hausarzt"}, {"code": "2", "label": "Radiologe"}]


def _label_choice_questionnaire():
    return Questionnaire.model_construct(
        name="LabelChoice",
        status="active",
        item=[
            QuestionnaireItem.model_construct(
                linkId="1.1",
                type="choice",
                text="Wer hat Sie aufgeklaert?",
                answerOption=[
                    QuestionnaireItemAnswerOption.model_construct(valueString="Hausarzt"),
                    QuestionnaireItemAnswerOption.model_construct(valueString="Radiologe"),
                ],
            )
        ],
    )


def _build(factory, mapping_table):
    creator = QuestionnaireMapCreator(
        SimpleNamespace(data=_label_choice_questionnaire()), factory=factory
    )
    return creator.generate_group("Source", "TargetQR", mapping_table)


def _all_rules(group):
    out = []

    def walk(rs):
        for r in rs or []:
            out.append(r)
            walk(r.rule)

    walk(group.rule)
    return out


def test_coded_source_gets_a_translate_instead_of_a_label_check(factory):
    factory.plugins = [_CodebookPlugin("erstdiagnose_durch", _CHOICES)]
    group = _build(factory, {"Source.erstdiagnose_durch": "QuestionnaireResponse.item[1.1]"})
    rules = _all_rules(group)

    checks = [s.check for r in rules for s in (r.source or []) if s.check]
    assert not any("Hausarzt" in (c or "") for c in checks), (
        "label enumeration must not be checked against the untranslated source"
    )
    translates = [
        t for r in rules for t in (r.target or []) if t.transform == "translate"
    ]
    assert translates, "no translate emitted for a coded source"
    assert translates[0].element == "valueString"


def test_label_check_is_kept_when_no_plugin_knows_the_field(factory):
    """Without a code list there is nothing to translate — behaviour must not change."""
    factory.plugins = []
    group = _build(factory, {"Source.erstdiagnose_durch": "QuestionnaireResponse.item[1.1]"})
    rules = _all_rules(group)

    checks = [s.check for r in rules for s in (r.source or []) if s.check]
    assert any("Hausarzt" in (c or "") for c in checks)
    assert not [t for r in rules for t in (r.target or []) if t.transform == "translate"]


def test_no_translate_when_the_options_already_are_the_codes(factory):
    """A Questionnaire listing the codes needs no translation — translating would
    replace a correct code with a display string."""
    factory.plugins = [
        _CodebookPlugin("erstdiagnose_durch",
                        [{"code": "Hausarzt", "label": "Hausarzt (GP)"},
                         {"code": "Radiologe", "label": "Radiologe (Rad)"}])
    ]
    group = _build(factory, {"Source.erstdiagnose_durch": "QuestionnaireResponse.item[1.1]"})
    rules = _all_rules(group)

    assert not [t for r in rules for t in (r.target or []) if t.transform == "translate"]


def test_unexecutable_enablewhen_is_documented_not_silently_dropped(factory):
    """matchbox cannot evaluate a target-context rule source (verified: it fails
    even with a trivial condition), and emitting one aborts the entire map. The
    condition is kept as rule documentation so the generated map still records
    what the Questionnaire requires and why it is inactive."""
    q = Questionnaire.model_construct(
        name="EnableWhenQ",
        status="active",
        item=[
            QuestionnaireItem.model_construct(
                linkId="1", type="string", text="Trigger"
            ),
            QuestionnaireItem.model_construct(
                linkId="2",
                type="string",
                text="Dependent",
                enableWhen=[
                    QuestionnaireItemEnableWhen.model_construct(
                        question="1", operator="=", answerString="yes"
                    )
                ],
            ),
        ],
    )
    creator = QuestionnaireMapCreator(SimpleNamespace(data=q), factory=factory)
    group = creator.generate_group(
        "Source", "TargetQR",
        {"Source.trigger": "QuestionnaireResponse.item[1]",
         "Source.dependent": "QuestionnaireResponse.item[2]"},
    )

    def walk(rs, out):
        for r in rs or []:
            out.append(r)
            walk(r.rule, out)
        return out

    rules = walk(group.rule, [])
    dependent = next(r for r in rules if r.name == "rule-2")
    # No source may point at the target root — that is the construct matchbox rejects.
    assert all(s.context != "TargetQR" for s in dependent.source)
    assert dependent.documentation and "INACTIVE" in dependent.documentation
    assert "repeat(item)" in dependent.documentation
    assert any(
        d.get("code") == "questionnaire-enablewhen-not-emitted"
        for d in factory.diagnostics
    )
