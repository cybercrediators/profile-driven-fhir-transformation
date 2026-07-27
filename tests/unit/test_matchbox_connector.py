"""Unit tests for MatchboxConnector (controller/connector/matchbox_connector.py).

``requests`` is monkeypatched at the module level -- no network calls are made.
"""

import json

import pytest
import requests

import controller.connector.matchbox_connector as matchbox_connector_module
from controller.connector.matchbox_connector import MatchboxConnector

pytestmark = pytest.mark.unit


class FakeResponse:
    def __init__(self, json_data=None, status_ok=True, content=b"{}", json_error=False, text="body"):
        self._json_data = json_data
        self.status_ok = status_ok
        self.content = content
        self.json_error = json_error
        self.text = text

    def raise_for_status(self):
        if not self.status_ok:
            raise requests.exceptions.HTTPError("bad status")

    def json(self):
        if self.json_error:
            raise json.JSONDecodeError("bad json", "doc", 0)
        return self._json_data


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_raises_without_connection_config():
    with pytest.raises(ValueError):
        MatchboxConnector(None)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://mb", "http://mb/fhir/"),
        ("http://mb/", "http://mb/fhir/"),
    ],
)
def test_init_normalizes_trailing_slash(url, expected):
    connector = MatchboxConnector({"url": url})
    assert connector.matchbox_uri == expected


def test_init_default_headers():
    connector = MatchboxConnector({"url": "http://mb"})
    assert connector.headers == {"Content-Type": "application/fhir+json"}


def test_init_custom_headers():
    connector = MatchboxConnector({"url": "http://mb"}, headers={"X-Test": "1"})
    assert connector.headers == {"X-Test": "1"}


def test_init_missing_url_key_still_builds_fhir_suffix():
    # a non-empty dict without "url" is still truthy -> passes the guard, and
    # matchbox_uri falls back to just "fhir/".
    connector = MatchboxConnector({"other_key": "value"})
    assert connector.matchbox_uri == "fhir/"


# --------------------------------------------------------------------------- #
# send_request
# --------------------------------------------------------------------------- #


def test_send_request_success_returns_json(monkeypatch):
    observed = {}

    def fake_request(method, url, headers=None, params=None, **kwargs):
        observed.update(method=method, url=url, headers=headers, params=params, kwargs=kwargs)
        return FakeResponse(json_data={"resourceType": "Bundle"})

    monkeypatch.setattr(matchbox_connector_module.requests, "request", fake_request)

    connector = MatchboxConnector({"url": "http://mb"})
    result = connector.send_request("StructureDefinition/1", "GET", params={"a": 1})

    assert result == {"resourceType": "Bundle"}
    assert observed["method"] == "GET"
    assert observed["url"] == "http://mb/fhir/StructureDefinition/1"
    assert observed["headers"] == {"Content-Type": "application/fhir+json"}
    assert observed["params"] == {"a": 1}


def test_send_request_merges_extra_headers(monkeypatch):
    observed = {}

    def fake_request(method, url, headers=None, params=None, **kwargs):
        observed["headers"] = headers
        return FakeResponse(json_data={})

    monkeypatch.setattr(matchbox_connector_module.requests, "request", fake_request)

    connector = MatchboxConnector({"url": "http://mb"})
    connector.send_request("x", "GET", headers={"Authorization": "Bearer t"})

    assert observed["headers"] == {
        "Content-Type": "application/fhir+json",
        "Authorization": "Bearer t",
    }
    # original connector headers must not be mutated by the per-call merge
    assert connector.headers == {"Content-Type": "application/fhir+json"}


def test_send_request_empty_body_returns_none(monkeypatch):
    def fake_request(method, url, headers=None, params=None, **kwargs):
        return FakeResponse(content=b"")

    monkeypatch.setattr(matchbox_connector_module.requests, "request", fake_request)

    connector = MatchboxConnector({"url": "http://mb"})
    assert connector.send_request("x", "DELETE") is None


def test_send_request_http_error_returns_none(monkeypatch):
    def fake_request(method, url, headers=None, params=None, **kwargs):
        return FakeResponse(status_ok=False, text="server error")

    monkeypatch.setattr(matchbox_connector_module.requests, "request", fake_request)

    connector = MatchboxConnector({"url": "http://mb"})
    assert connector.send_request("x", "GET") is None


def test_send_request_connection_error_returns_none(monkeypatch):
    def fake_request(method, url, headers=None, params=None, **kwargs):
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(matchbox_connector_module.requests, "request", fake_request)

    connector = MatchboxConnector({"url": "http://mb"})
    assert connector.send_request("x", "GET") is None


def test_send_request_json_decode_error_returns_response_text(monkeypatch):
    def fake_request(method, url, headers=None, params=None, **kwargs):
        return FakeResponse(json_error=True, text="<html>not json</html>")

    monkeypatch.setattr(matchbox_connector_module.requests, "request", fake_request)

    connector = MatchboxConnector({"url": "http://mb"})
    result = connector.send_request("x", "GET")
    assert result == "<html>not json</html>"


def test_send_request_forwards_extra_kwargs(monkeypatch):
    observed = {}

    def fake_request(method, url, headers=None, params=None, **kwargs):
        observed["kwargs"] = kwargs
        return FakeResponse(json_data={})

    monkeypatch.setattr(matchbox_connector_module.requests, "request", fake_request)

    connector = MatchboxConnector({"url": "http://mb"})
    connector.send_request("x", "POST", json={"a": 1})
    assert observed["kwargs"] == {"json": {"a": 1}}
