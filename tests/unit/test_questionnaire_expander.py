"""Unit tests for the slimmed questionnaire_expander.

It no longer builds a (discarded) flattened view; its job is the answerValueSet pre-pass:
expand each coded item's value set and record the questionnaire -> value set dependency,
recursing into nested items.
"""

from types import SimpleNamespace

import pytest

from fhir.resources.R4B.questionnaire import Questionnaire, QuestionnaireItem

from data_handling.registry.registry_object import RegistryObject
from parser.resource_parser import questionnaire_expander as qe

pytestmark = pytest.mark.unit


@pytest.fixture
def captured(monkeypatch):
    """Record expand_valueset calls and used_by edges instead of doing real expansion."""
    expanded = []
    used_by = []
    monkeypatch.setattr(qe, "expand_valueset", lambda url, seen, st: expanded.append(url) or [])
    state = SimpleNamespace(
        registry=SimpleNamespace(add_to_used_by=lambda src, tgt: used_by.append((src, tgt)))
    )
    return SimpleNamespace(expanded=expanded, used_by=used_by, state=state)


def _questionnaire_ro(items, url="http://q/Test"):
    q = Questionnaire(status="active", url=url, item=items)
    return RegistryObject(q, "Questionnaire")


def test_none_questionnaire_is_noop(captured):
    out, _ = qe.parse_questionnaire(None, captured.state)
    assert out is None
    assert captured.expanded == []


def test_already_processed_is_noop(captured):
    ro = _questionnaire_ro([QuestionnaireItem(linkId="a", type="choice", answerValueSet="http://vs/a")])
    ro.set_processed()
    qe.parse_questionnaire(ro, captured.state)
    assert captured.expanded == []


def test_no_items_is_noop(captured):
    # A questionnaire with no items is an error case and returns before being marked
    # processed (matching the original behaviour) — nothing is pre-warmed.
    q = Questionnaire(status="active", url="http://q/Empty")
    ro = RegistryObject(q, "Questionnaire")
    qe.parse_questionnaire(ro, captured.state)
    assert captured.expanded == []
    assert not ro.is_processed()


def test_prewarms_value_sets_and_records_used_by(captured):
    ro = _questionnaire_ro([
        QuestionnaireItem(linkId="top", type="choice", answerValueSet="http://vs/top"),
        QuestionnaireItem(linkId="plain", type="string"),  # no value set
    ])
    qe.parse_questionnaire(ro, captured.state)
    assert captured.expanded == ["http://vs/top"]
    assert ("http://vs/top", "http://q/Test") in captured.used_by
    assert ro.is_processed()


def test_recurses_into_nested_items(captured):
    nested = QuestionnaireItem(
        linkId="group", type="group",
        item=[QuestionnaireItem(linkId="child", type="choice", answerValueSet="http://vs/child")],
    )
    ro = _questionnaire_ro([nested])
    qe.parse_questionnaire(ro, captured.state)
    assert "http://vs/child" in captured.expanded
