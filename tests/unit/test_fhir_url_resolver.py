"""Unit tests for resolve_url / resolve_add_to_registry.

Focus: the single return contract (always a fhir object, or a dict fallback) across
the registry / cache / local-package / network paths, and the None-registry guard.
"""

import types

import pytest

from data_handling.url_resolver import fhir_url_resolver as mod

pytestmark = pytest.mark.unit


class FakeCache:
    def __init__(self, store=None):
        self.store = store or {}
        self.added = []

    def get_resource_from_cache(self, url):
        return self.store.get(url)

    def add_resource_to_cache(self, obj):
        self.added.append(obj)


def _state(registry=None, cache=None, conf=None):
    return types.SimpleNamespace(
        registry=registry, cache=cache or FakeCache(), conf=conf or {}
    )


# --------------------------------------------------------------------------- #
# resolve_url
# --------------------------------------------------------------------------- #


def test_resolve_url_returns_registry_object_data():
    data = object()
    registry = types.SimpleNamespace(
        registry_objects={"http://x": types.SimpleNamespace(data=data)}
    )
    assert mod.resolve_url("http://x", _state(registry=registry)) is data


def test_resolve_url_none_registry_does_not_crash(monkeypatch):
    # The None-registry branch must not dereference registry_objects.
    monkeypatch.setattr(mod, "get_resource_from_local_package", lambda u, c: None)
    resolver = types.SimpleNamespace(query=lambda url: None)
    assert mod.resolve_url("http://x", _state(registry=None), resolver) is None


def test_resolve_url_cache_hit_returns_converted_object(monkeypatch):
    marker = object()
    monkeypatch.setattr(mod.utils, "json_to_obj", lambda content, rt: marker)
    cache = FakeCache({"http://x": {"resourceType": "Patient", "url": "http://x"}})
    state = _state(registry=types.SimpleNamespace(registry_objects={}), cache=cache)
    assert mod.resolve_url("http://x", state) is marker


def test_resolve_url_network_path_converts_object_and_caches_raw_dict(monkeypatch):
    # The core fix: the network-resolve path returns a converted object (like the
    # other paths) while the *raw dict* is what gets cached.
    marker = object()
    raw = {"resourceType": "StructureDefinition", "url": "http://x"}
    monkeypatch.setattr(mod, "get_resource_from_local_package", lambda u, c: None)
    monkeypatch.setattr(mod.utils, "json_to_obj", lambda content, rt: marker)
    cache = FakeCache()
    state = _state(registry=types.SimpleNamespace(registry_objects={}), cache=cache)
    resolver = types.SimpleNamespace(query=lambda url: raw)

    result = mod.resolve_url("http://x", state, resolver)

    assert result is marker
    assert cache.added == [raw]


def test_resolve_url_returns_none_when_unresolved(monkeypatch):
    monkeypatch.setattr(mod, "get_resource_from_local_package", lambda u, c: None)
    state = _state(registry=types.SimpleNamespace(registry_objects={}))
    resolver = types.SimpleNamespace(query=lambda url: None)
    assert mod.resolve_url("http://x", state, resolver) is None


# --------------------------------------------------------------------------- #
# resolve_add_to_registry
# --------------------------------------------------------------------------- #


def test_resolve_add_to_registry_derives_type_from_object(monkeypatch):
    class Patient:
        url = "http://x"

    obj = Patient()
    monkeypatch.setattr(mod, "resolve_url", lambda u, s, r=None: obj)
    added = {}
    registry = types.SimpleNamespace(
        get_obj_by_name=lambda u: None,
        add_fhir_object=lambda o, rt: added.update(obj=o, res_type=rt) or "REG",
    )

    result = mod.resolve_add_to_registry("http://x", _state(registry=registry))

    assert result == "REG"
    assert added["obj"] is obj
    assert added["res_type"] == "Patient"


def test_resolve_add_to_registry_handles_dict_fallback(monkeypatch):
    raw = {"resourceType": "ValueSet", "url": "http://x"}
    monkeypatch.setattr(mod, "resolve_url", lambda u, s, r=None: raw)
    added = {}
    registry = types.SimpleNamespace(
        get_obj_by_name=lambda u: None,
        add_fhir_object=lambda o, rt: added.update(res_type=rt) or "REG",
    )

    mod.resolve_add_to_registry("http://x", _state(registry=registry))

    assert added["res_type"] == "ValueSet"


def test_resolve_add_to_registry_returns_none_when_unresolved(monkeypatch):
    monkeypatch.setattr(mod, "resolve_url", lambda u, s, r=None: None)
    registry = types.SimpleNamespace(get_obj_by_name=lambda u: None)
    assert mod.resolve_add_to_registry("http://x", _state(registry=registry)) is None


def test_resolve_add_to_registry_returns_existing_without_resolving(monkeypatch):
    existing = object()
    registry = types.SimpleNamespace(get_obj_by_name=lambda u: existing)

    def _boom(*a, **k):
        raise AssertionError("resolve_url must not be called when already registered")

    monkeypatch.setattr(mod, "resolve_url", _boom)
    assert mod.resolve_add_to_registry("http://x", _state(registry=registry)) is existing
