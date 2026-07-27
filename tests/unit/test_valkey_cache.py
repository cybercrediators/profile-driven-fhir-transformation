"""Unit tests for ValkeyConnector (data_handling/caching/valkey_cache.py).

The real ``valkey`` package is installed (used for its exception types), but no live
Valkey server is required or contacted -- ``valkey.Valkey`` is monkeypatched with a
small fake client so ``connect()`` never opens a socket.
"""

import json
from types import SimpleNamespace

import pytest
import valkey

import data_handling.caching.valkey_cache as valkey_cache_module
from data_handling.caching.valkey_cache import ValkeyConnector

pytestmark = pytest.mark.unit


class FakeValkeyClient:
    """Records constructor args and lets tests control ping()/command behaviour."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.ping_result = True
        self.store = {}
        FakeValkeyClient.instances.append(self)

    def ping(self):
        if self.ping_result is not True:
            raise self.ping_result
        return True

    @classmethod
    def from_url(cls, url):
        # classmethod like the real client (valkey_cache.py calls
        # ``valkey.Valkey.from_url(url)`` since the 2026-07-06 fix)
        inst = cls()
        inst.from_url_arg = url
        return inst


def make_connector(connection=None):
    """Build a ValkeyConnector bypassing __init__/connect() with a fake connection."""
    connector = ValkeyConnector.__new__(ValkeyConnector)
    connector.connection = connection if connection is not None else SimpleNamespace()
    return connector


@pytest.fixture(autouse=True)
def reset_fake_instances():
    FakeValkeyClient.instances = []
    yield
    FakeValkeyClient.instances = []


# --------------------------------------------------------------------------- #
# connect() / __init__
# --------------------------------------------------------------------------- #


def test_connect_uses_host_port_db_when_no_url(monkeypatch):
    monkeypatch.setattr(valkey_cache_module.valkey, "Valkey", FakeValkeyClient)

    connector = ValkeyConnector("localhost", 6379, 0)

    assert isinstance(connector.connection, FakeValkeyClient)
    assert connector.connection.init_args == ("localhost", 6379, 0)


def test_connect_uses_from_url_when_url_provided(monkeypatch):
    monkeypatch.setattr(valkey_cache_module.valkey, "Valkey", FakeValkeyClient)

    connector = ValkeyConnector(None, None, None, valkey_url="valkey://host:1234/0")

    assert connector.connection.from_url_arg == "valkey://host:1234/0"


def test_connect_returns_none_on_connection_error(monkeypatch, capsys):
    monkeypatch.setattr(valkey_cache_module.valkey, "Valkey", FakeValkeyClient)

    class RaisingClient(FakeValkeyClient):
        def ping(self):
            raise valkey.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr(valkey_cache_module.valkey, "Valkey", RaisingClient)

    connector = ValkeyConnector("localhost", 6379, 0)

    assert connector.connection is None
    assert "Could not connect to Valkey" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# get_resource_from_cache
# --------------------------------------------------------------------------- #


def test_get_resource_from_cache_found():
    resource = {"resourceType": "Patient", "url": "http://x/p"}
    fake_conn = SimpleNamespace(get=lambda url: json.dumps(resource).encode("utf-8"))
    connector = make_connector(fake_conn)

    assert connector.get_resource_from_cache("http://x/p") == resource


def test_get_resource_from_cache_missing_returns_none():
    fake_conn = SimpleNamespace(get=lambda url: None)
    connector = make_connector(fake_conn)

    assert connector.get_resource_from_cache("http://missing") is None


# --------------------------------------------------------------------------- #
# add_resource_to_cache
# --------------------------------------------------------------------------- #


def test_add_resource_to_cache_without_url_returns_none():
    connector = make_connector()
    assert connector.add_resource_to_cache({"resourceType": "Patient"}) is None


def test_add_resource_to_cache_success_calls_set():
    stored = {}
    fake_conn = SimpleNamespace(set=lambda url, value: stored.update({url: value}))
    connector = make_connector(fake_conn)

    resource = {"resourceType": "Patient", "url": "http://x/p"}
    result = connector.add_resource_to_cache(resource)

    assert result == resource
    assert json.loads(stored["http://x/p"]) == resource


# --------------------------------------------------------------------------- #
# remove_resource_from_cache
# --------------------------------------------------------------------------- #


def test_remove_resource_from_cache_found(capsys):
    fake_conn = SimpleNamespace(delete=lambda url: 1)
    connector = make_connector(fake_conn)
    connector.remove_resource_from_cache("http://x/p")
    # no exception is the main assertion; log output isn't captured by capsys (uses logging)


def test_remove_resource_from_cache_not_found():
    fake_conn = SimpleNamespace(delete=lambda url: 0)
    connector = make_connector(fake_conn)
    connector.remove_resource_from_cache("http://missing")  # should not raise


# --------------------------------------------------------------------------- #
# update_resource_in_cache
# --------------------------------------------------------------------------- #


def test_update_resource_in_cache_without_url_returns_none():
    connector = make_connector()
    assert connector.update_resource_in_cache({"resourceType": "Patient"}) is None


def test_update_resource_in_cache_success():
    stored = {}
    fake_conn = SimpleNamespace(set=lambda url, value: stored.update({url: value}))
    connector = make_connector(fake_conn)

    resource = {"resourceType": "Patient", "url": "http://x/p"}
    result = connector.update_resource_in_cache(resource)

    assert result == resource
    assert json.loads(stored["http://x/p"]) == resource


# --------------------------------------------------------------------------- #
# clear_cache
# --------------------------------------------------------------------------- #


def test_clear_cache_calls_flushdb():
    calls = []
    fake_conn = SimpleNamespace(flushdb=lambda: calls.append("flushed"))
    connector = make_connector(fake_conn)
    connector.clear_cache(force=True)
    assert calls == ["flushed"]


# --------------------------------------------------------------------------- #
# list_cached_resources
# --------------------------------------------------------------------------- #


def test_list_cached_resources_decodes_bytes_keys():
    fake_conn = SimpleNamespace(
        scan_iter=lambda match=None: iter([b"http://x/a", b"http://x/b"])
    )
    connector = make_connector(fake_conn)
    assert connector.list_cached_resources() == ["http://x/a", "http://x/b"]


def test_list_cached_resources_forwards_filter_pattern():
    observed = {}

    def scan_iter(match=None):
        observed["match"] = match
        return iter([])

    fake_conn = SimpleNamespace(scan_iter=scan_iter)
    connector = make_connector(fake_conn)
    connector.list_cached_resources(filter_pattern="patient*")
    assert observed["match"] == "patient*"


# --------------------------------------------------------------------------- #
# show_stats
# --------------------------------------------------------------------------- #


def test_show_stats_with_dbsize_prints_info(capsys):
    fake_conn = SimpleNamespace(
        dbsize=lambda: 42,
        info=lambda: {
            "used_memory_human": "1M",
            "used_memory_peak_human": "2M",
            "connected_clients": 3,
            "uptime_in_seconds": 100,
        },
    )
    connector = make_connector(fake_conn)
    connector.show_stats()
    out = capsys.readouterr().out
    assert "Total keys: 42" in out
    assert "Memory used: 1M" in out
    assert "Connected clients: 3" in out


def test_show_stats_without_dbsize_prints_fallback(capsys):
    # SimpleNamespace with no dbsize attribute -> hasattr() is False
    connector = make_connector(SimpleNamespace())
    connector.show_stats()
    assert "Stats operation not supported" in capsys.readouterr().out
