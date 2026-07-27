"""Unit tests for expand_valueset.

The terminology server is stubbed to always miss (returns None) so the manual compose-based
expansion path is exercised; resolve_url is stubbed to hand back controlled ValueSet /
CodeSystem models keyed by url.
"""

from types import SimpleNamespace

import pytest

from fhir.resources.R4B.valueset import (
    ValueSet,
    ValueSetCompose,
    ValueSetComposeInclude,
    ValueSetComposeIncludeConcept,
    ValueSetComposeIncludeFilter,
)
from fhir.resources.R4B.codesystem import CodeSystem, CodeSystemConcept

from parser.resource_parser import value_expander as ve

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _stub_terminology(monkeypatch):
    """Force the manual expansion path (no external terminology server)."""
    monkeypatch.setattr(
        ve, "TerminologyConnector",
        lambda app_state: SimpleNamespace(
            query_terminology_server=lambda url: None,
            extract_values=lambda r: [],
        ),
    )


def _state(objs=None):
    store = objs or {}
    registry = SimpleNamespace(
        get_obj_by_name=lambda u: store.get(u),
        add_to_used_by=lambda src, tgt: None,
    )
    return SimpleNamespace(registry=registry, conf={})


def _vs(includes, url="http://vs/x"):
    return ValueSet(status="active", url=url, compose=ValueSetCompose(include=includes))


# --------------------------------------------------------------------------- #


def test_returns_cached_expansion_without_resolving(monkeypatch):
    cached = [{"code": "C", "display": "c", "system": "s"}]
    obj = SimpleNamespace(processed=0, expanded_values=cached)
    monkeypatch.setattr(ve, "resolve_url", lambda *a, **k: pytest.fail("must not resolve"))
    out = ve.expand_valueset("http://vs/x", [], _state({"http://vs/x": obj}))
    assert out == cached


def test_non_valueset_returns_empty(monkeypatch):
    monkeypatch.setattr(ve, "resolve_url", lambda url, st: {"resourceType": "Basic"})
    assert ve.expand_valueset("http://vs/x", [], _state()) == []


def test_include_concept_options(monkeypatch):
    vs = _vs([
        ValueSetComposeInclude(
            system="http://cs/g",
            concept=[
                ValueSetComposeIncludeConcept(code="M", display="Male"),
                ValueSetComposeIncludeConcept(code="F", display="Female"),
            ],
        )
    ])
    monkeypatch.setattr(ve, "resolve_url", lambda url, st: vs)
    out = ve.expand_valueset("http://vs/x", [], _state())
    assert {o["code"] for o in out} == {"M", "F"}
    assert all(o["system"] == "http://cs/g" for o in out)


def test_include_system_with_complete_codesystem(monkeypatch):
    vs = _vs([ValueSetComposeInclude(system="http://cs/complete")])
    cs = CodeSystem(
        status="active", content="complete", url="http://cs/complete",
        concept=[CodeSystemConcept(code="a", display="A"), CodeSystemConcept(code="b", display="B")],
    )
    resolved = {"http://vs/x": vs, "http://cs/complete": cs}
    monkeypatch.setattr(ve, "resolve_url", lambda url, st: resolved.get(url))
    out = ve.expand_valueset("http://vs/x", [], _state())
    assert {o["code"] for o in out} == {"a", "b"}


def test_include_system_without_codesystem_emits_placeholder(monkeypatch):
    vs = _vs([ValueSetComposeInclude(system="http://cs/unknown")])
    resolved = {"http://vs/x": vs}
    monkeypatch.setattr(ve, "resolve_url", lambda url, st: resolved.get(url))
    out = ve.expand_valueset("http://vs/x", [], _state())
    assert out == [{"code": "FROM_CS", "display": "from system: http://cs/unknown", "system": "http://cs/unknown"}]


def test_nested_valueset_is_expanded(monkeypatch):
    outer = _vs([ValueSetComposeInclude(valueSet=["http://vs/inner"])], url="http://vs/outer")
    inner = _vs([
        ValueSetComposeInclude(
            system="http://cs/i",
            concept=[ValueSetComposeIncludeConcept(code="x", display="X")],
        )
    ], url="http://vs/inner")
    resolved = {"http://vs/outer": outer, "http://vs/inner": inner}
    monkeypatch.setattr(ve, "resolve_url", lambda url, st: resolved.get(url))
    out = ve.expand_valueset("http://vs/outer", [], _state())
    assert [o["code"] for o in out] == ["x"]


def test_include_filter_emits_filter_marker(monkeypatch):
    vs = _vs([
        ValueSetComposeInclude(
            system="http://cs/f",
            filter=[ValueSetComposeIncludeFilter(property="concept", op="is-a", value="123")],
        )
    ])
    monkeypatch.setattr(ve, "resolve_url", lambda url, st: vs)
    out = ve.expand_valueset("http://vs/x", [], _state())
    assert len(out) == 1
    assert out[0]["code"].startswith("FILTER:")
