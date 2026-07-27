"""Unit tests for TerminologyConnector (controller/connector/terminology_connector.py).

``requests`` is monkeypatched at the module level -- no network calls are made.
``app_state`` only needs a ``.conf`` attribute here, so a SimpleNamespace stands in
for the real AppState dataclass (duck typing; no Registry/DataIO construction needed).
"""

from types import SimpleNamespace

import pytest
import requests

import controller.connector.terminology_connector as terminology_connector_module
from controller.connector.terminology_connector import TerminologyConnector

pytestmark = pytest.mark.unit


class FakeResponse:
    def __init__(self, json_data=None, status_ok=True, raise_exc=None):
        self._json_data = json_data
        self.status_ok = status_ok
        self.raise_exc = raise_exc

    def raise_for_status(self):
        if self.raise_exc:
            raise self.raise_exc
        if not self.status_ok:
            raise requests.exceptions.HTTPError("bad status")

    def json(self):
        return self._json_data


def make_connector(uri="http://terminology/"):
    app_state = SimpleNamespace(conf={"terminology_server_uri": uri})
    return TerminologyConnector(app_state)


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_builds_expand_endpoint():
    connector = make_connector("http://terminology/")
    assert connector.endpoint == "http://terminology/ValueSet/$expand"


def test_init_default_headers():
    connector = make_connector()
    assert connector.headers == {
        "Accept": "application/fhir+json",
        "Content-Type": "application/fhir+json",
    }


# --------------------------------------------------------------------------- #
# query_terminology_server
# --------------------------------------------------------------------------- #


def test_query_terminology_server_success_returns_expansion(monkeypatch):
    observed = {}

    def fake_get(endpoint, headers=None, params=None):
        observed.update(endpoint=endpoint, headers=headers, params=params)
        return FakeResponse(json_data={"expansion": {"contains": [{"code": "a"}]}})

    monkeypatch.setattr(terminology_connector_module.requests, "get", fake_get)

    connector = make_connector()
    result = connector.query_terminology_server("http://vs/1")

    assert result == {"contains": [{"code": "a"}]}
    assert observed["params"] == {"url": "http://vs/1"}


def test_query_terminology_server_forwards_optional_params(monkeypatch):
    observed = {}

    def fake_get(endpoint, headers=None, params=None):
        observed["params"] = params
        return FakeResponse(json_data={"expansion": {}})

    monkeypatch.setattr(terminology_connector_module.requests, "get", fake_get)

    connector = make_connector()
    connector.query_terminology_server("http://vs/1", filter_str="abc", count=10, offset=5)

    assert observed["params"] == {
        "url": "http://vs/1",
        "filter": "abc",
        "count": 10,
        "offset": 5,
    }


def test_query_terminology_server_no_expansion_key_returns_none(monkeypatch):
    def fake_get(endpoint, headers=None, params=None):
        return FakeResponse(json_data={"resourceType": "OperationOutcome"})

    monkeypatch.setattr(terminology_connector_module.requests, "get", fake_get)

    connector = make_connector()
    assert connector.query_terminology_server("http://vs/1") is None


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.HTTPError("http error"),
        requests.exceptions.ConnectionError("connection error"),
        requests.exceptions.Timeout("timeout"),
        requests.exceptions.RequestException("generic error"),
    ],
)
def test_query_terminology_server_handles_request_exceptions(monkeypatch, exc):
    def fake_get(endpoint, headers=None, params=None):
        return FakeResponse(raise_exc=exc)

    monkeypatch.setattr(terminology_connector_module.requests, "get", fake_get)

    connector = make_connector()
    assert connector.query_terminology_server("http://vs/1") is None


# --------------------------------------------------------------------------- #
# extract_codes / extract_values
# --------------------------------------------------------------------------- #


def test_extract_codes_maps_expected_fields():
    connector = make_connector()
    expansion = {
        "contains": [
            {"code": "a", "display": "Alpha", "system": "http://sys"},
            {"code": "b", "display": "Beta", "system": "http://sys"},
        ]
    }
    result = connector.extract_codes(expansion)
    assert result == [
        {"code": "a", "display": "Alpha", "system": "http://sys"},
        {"code": "b", "display": "Beta", "system": "http://sys"},
    ]


def test_extract_codes_handles_none_input():
    connector = make_connector()
    assert connector.extract_codes(None) == []


def test_extract_codes_handles_missing_contains_key():
    connector = make_connector()
    assert connector.extract_codes({}) == []


def test_extract_values_is_alias_for_extract_codes():
    connector = make_connector()
    assert connector.extract_values == connector.extract_codes
    assert connector.extract_values.__func__ is connector.extract_codes.__func__
