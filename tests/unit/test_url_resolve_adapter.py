"""Unit tests for the URL resolve adapters (data_handling/url_resolver/url_resolve_adapter.py).

``requests`` is monkeypatched at the module level -- no network calls are made.
"""

import json

import pytest
import requests

import data_handling.url_resolver.url_resolve_adapter as adapter_module
from data_handling.url_resolver.url_resolve_adapter import (
    Resolvers,
    SimplifierURLResolveAdapter,
    StandardURLResolveAdapter,
    URLResolveAdapterBase,
)

pytestmark = pytest.mark.unit


class FakeResponse:
    def __init__(self, json_data=None, status_ok=True, json_error=False):
        self._json_data = json_data
        self.status_ok = status_ok
        self.json_error = json_error
        self.text = "not-json" if json_error else json.dumps(json_data or {})

    def raise_for_status(self):
        if not self.status_ok:
            raise requests.exceptions.HTTPError("bad status")

    def json(self):
        if self.json_error:
            raise json.JSONDecodeError("bad json", "doc", 0)
        return self._json_data


# --------------------------------------------------------------------------- #
# http_json_call (shared by both adapters, exercised via base class directly)
# --------------------------------------------------------------------------- #


def test_http_json_call_returns_parsed_json(monkeypatch):
    observed = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        observed.update(url=url, headers=headers, params=params, timeout=timeout)
        return FakeResponse({"resourceType": "StructureDefinition"})

    monkeypatch.setattr(adapter_module.requests, "get", fake_get)

    adapter = StandardURLResolveAdapter({})
    result = adapter.http_json_call("http://x/sd", headers={"Accept": "a"}, params={"p": 1})

    assert result == {"resourceType": "StructureDefinition"}
    assert observed == {
        "url": "http://x/sd",
        "headers": {"Accept": "a"},
        "params": {"p": 1},
        "timeout": 10,
    }


def test_http_json_call_returns_none_on_request_exception(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(adapter_module.requests, "get", fake_get)

    adapter = StandardURLResolveAdapter({})
    assert adapter.http_json_call("http://x") is None


def test_http_json_call_returns_none_on_http_error(monkeypatch):
    def fake_get(*args, **kwargs):
        return FakeResponse(status_ok=False)

    monkeypatch.setattr(adapter_module.requests, "get", fake_get)

    adapter = StandardURLResolveAdapter({})
    assert adapter.http_json_call("http://x") is None


def test_http_json_call_returns_none_on_json_decode_error(monkeypatch):
    def fake_get(*args, **kwargs):
        return FakeResponse(json_error=True)

    monkeypatch.setattr(adapter_module.requests, "get", fake_get)

    adapter = StandardURLResolveAdapter({})
    assert adapter.http_json_call("http://x") is None


def test_base_query_is_abstract():
    with pytest.raises(TypeError):
        URLResolveAdapterBase({})


# --------------------------------------------------------------------------- #
# StandardURLResolveAdapter.query
# --------------------------------------------------------------------------- #


def test_standard_adapter_query_success(monkeypatch):
    resource = {"resourceType": "StructureDefinition", "url": "http://x/sd"}
    monkeypatch.setattr(
        StandardURLResolveAdapter, "http_json_call", lambda self, url, headers=None, params=None: resource
    )
    adapter = StandardURLResolveAdapter({})
    assert adapter.query("http://x/sd") == resource


def test_standard_adapter_query_none_result(monkeypatch):
    monkeypatch.setattr(
        StandardURLResolveAdapter, "http_json_call", lambda self, url, headers=None, params=None: None
    )
    adapter = StandardURLResolveAdapter({})
    assert adapter.query("http://x/sd") is None


def test_standard_adapter_query_empty_dict_result(monkeypatch):
    # an empty dict is falsy -> also treated as "could not resolve"
    monkeypatch.setattr(
        StandardURLResolveAdapter, "http_json_call", lambda self, url, headers=None, params=None: {}
    )
    adapter = StandardURLResolveAdapter({})
    assert adapter.query("http://x/sd") is None


# --------------------------------------------------------------------------- #
# SimplifierURLResolveAdapter.query
# --------------------------------------------------------------------------- #


def test_simplifier_adapter_query_raises_without_api_url():
    adapter = SimplifierURLResolveAdapter({"simplifier_api_url": None})
    with pytest.raises(ValueError):
        adapter.query("http://x/sd")


def test_simplifier_adapter_query_uses_kwarg_api_url_when_conf_missing(monkeypatch):
    # the api_url kwarg must be used for the actual HTTP call, not just to
    # satisfy the guard (fixed 2026-07-06 — previously queried conf's None)
    called = {}

    def fake_http_json_call(self, url, headers=None, params=None):
        called["url"] = url
        called["params"] = params
        return {"resourceType": "Bundle", "url": "http://x/sd", "entry": [{"resource": {"resourceType": "StructureDefinition"}}]}

    monkeypatch.setattr(SimplifierURLResolveAdapter, "http_json_call", fake_http_json_call)

    adapter = SimplifierURLResolveAdapter({"simplifier_api_url": None})
    result = adapter.query("http://x/sd", api_url="http://simplifier")

    assert result == {"resourceType": "StructureDefinition"}
    assert called["url"] == "http://simplifier"


def test_simplifier_adapter_query_no_version_returns_first_entry(monkeypatch):
    bundle = {
        "url": "http://x/sd",
        "entry": [{"resource": {"resourceType": "StructureDefinition", "version": "1.0"}}],
    }
    monkeypatch.setattr(
        SimplifierURLResolveAdapter,
        "http_json_call",
        lambda self, url, headers=None, params=None: bundle,
    )
    adapter = SimplifierURLResolveAdapter({"simplifier_api_url": "http://simplifier"})
    result = adapter.query("http://x/sd")
    assert result == {"resourceType": "StructureDefinition", "version": "1.0"}


def test_simplifier_adapter_query_selects_matching_version(monkeypatch):
    """A version match must return the unwrapped resource — same shape as the
    no-match fallback (fixed 2026-07-06; previously returned the raw bundle
    entry ``{"resource": {...}}`` on a match)."""
    bundle = {
        "url": "http://x/sd",
        "entry": [
            {"resource": {"resourceType": "StructureDefinition", "version": "1.0"}},
            {"resource": {"resourceType": "StructureDefinition", "version": "2.0"}},
        ],
    }
    monkeypatch.setattr(
        SimplifierURLResolveAdapter,
        "http_json_call",
        lambda self, url, headers=None, params=None: bundle,
    )
    adapter = SimplifierURLResolveAdapter({"simplifier_api_url": "http://simplifier"})
    result = adapter.query("http://x/sd|2.0")
    assert result == {"resourceType": "StructureDefinition", "version": "2.0"}


def test_simplifier_adapter_query_version_not_found_falls_back_to_first(monkeypatch):
    bundle = {
        "url": "http://x/sd",
        "entry": [{"resource": {"resourceType": "StructureDefinition", "version": "1.0"}}],
    }
    monkeypatch.setattr(
        SimplifierURLResolveAdapter,
        "http_json_call",
        lambda self, url, headers=None, params=None: bundle,
    )
    adapter = SimplifierURLResolveAdapter({"simplifier_api_url": "http://simplifier"})
    result = adapter.query("http://x/sd|9.9")
    assert result == {"resourceType": "StructureDefinition", "version": "1.0"}


def test_simplifier_adapter_query_none_result_returns_none(monkeypatch):
    monkeypatch.setattr(
        SimplifierURLResolveAdapter,
        "http_json_call",
        lambda self, url, headers=None, params=None: None,
    )
    adapter = SimplifierURLResolveAdapter({"simplifier_api_url": "http://simplifier"})
    assert adapter.query("http://x/sd") is None


def test_simplifier_adapter_query_no_entry_key_returns_none(monkeypatch):
    # res.get("entry") is None -> falls straight through, `res` stays None
    monkeypatch.setattr(
        SimplifierURLResolveAdapter,
        "http_json_call",
        lambda self, url, headers=None, params=None: {"url": "http://x/sd"},
    )
    adapter = SimplifierURLResolveAdapter({"simplifier_api_url": "http://simplifier"})
    assert adapter.query("http://x/sd") is None


# --------------------------------------------------------------------------- #
# Resolvers enum
# --------------------------------------------------------------------------- #


def test_resolvers_enum_maps_to_adapter_classes():
    assert Resolvers.STANDARD.value is StandardURLResolveAdapter
    assert Resolvers.SIMPLIFIER.value is SimplifierURLResolveAdapter
