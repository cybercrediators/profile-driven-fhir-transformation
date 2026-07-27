"""Unit tests for QuestionnaireMapCreator (fml_questionnaire.py).

Builds a small, realistic Questionnaire (a group with a nested display item, a coded
single-select, a multi-select "checkbox" item, and an unmapped leaf) plus a
QuestionnaireResponse-style mapping table, and exercises `generate_group` end to end —
covering group/display recursion, mapped/unmapped/multi-source leaves, coded (ConceptMap +
$translate) answers, and the status rule. All ConceptMap generation is offline: answer
options are supplied directly (answerOption), so `expand_valueset`/the terminology
connector is never invoked.
"""

from types import SimpleNamespace

import pytest

from fhir.resources.R4B.questionnaire import (
    Questionnaire,
    QuestionnaireItem,
    QuestionnaireItemAnswerOption,
)
from fhir.resources.R4B.coding import Coding

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


def test_subject_placeholder_rule_shape(creator):
    group = creator.generate_group("Source", "TestQ")
    rule = next(r for r in group.rule
                if r.name == "TODO-resolve-reference-QuestionnaireResponse-subject")
    # documentation must match BundleService._collect_todo_refs' regex contract
    assert rule.documentation.startswith("Reference<QuestionnaireResponse.subject>")
    assert "Patient" in rule.documentation  # default when no subjectType declared
    assert rule.target[0].element == "subject"
    assert rule.source[0].context == "Source"


def test_status_rule_uses_mapped_source(creator, mapping_table):
    group = creator.generate_group("Source", "TestQ", mapping_table=mapping_table)
    status_rule = group.rule[0]
    assert status_rule.source[0].element == "status"  # local_element_name("Src.status")
    assert status_rule.target[0].element == "status"
    # No ConceptMap possible for a bare status mapping (no options) -> TODO copy fallback.
    assert status_rule.target[0].transform == "copy"


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
