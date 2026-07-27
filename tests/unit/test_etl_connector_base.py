import pytest

from controller.etl_controller.etl_controller import ETLConnector


pytestmark = pytest.mark.unit


class _DummyConnector(ETLConnector):
    """Concrete ETLConnector implementing all abstract methods for base-class tests."""

    def __init__(self, socket_conf, fake_socket_client):
        super().__init__(socket_conf)
        # Override the real socket client with a fake to avoid I/O.
        self.socket_client = fake_socket_client
        self.logged = []
        self.errors = []

    def validate_setup(self):
        return "ok"

    def get_status(self):
        return {"status": "running"}

    def ingest_source_data(self, raw_payload):
        return raw_payload

    def transform(self, payload, bundle=True, structure_map_url=None):
        return payload

    def transform_batch(self, payloads, bundle=True):
        return list(payloads)

    def validate_resource(self, resource, profile_url=None):
        return resource

    def return_data(self, data, metadata=None):
        return data

    def emit_log(self, level: str, message: str, **details) -> None:
        self.logged.append((level, message, details))

    def emit_error(self, message: str, **details):
        self.errors.append((message, details))
        return {"error": message, **details}


def test_socket_request_success():
    class FakeSocketClient:
        def __init__(self):
            self.calls = []

        def send_request(self, method, params=None, data=None):
            self.calls.append((method, params, data))
            return {"status": "success", "data": {"ok": True}}

    fake_sc = FakeSocketClient()
    connector = _DummyConnector({"type": "UNIX"}, fake_sc)

    result = connector._socket_request("status", params={"a": 1}, data={"b": 2})

    assert result == {"ok": True}
    assert fake_sc.calls == [("status", {"a": 1}, {"b": 2})]
    assert connector.errors == []


def test_socket_request_failure_calls_emit_error():
    class FakeSocketClient:
        def send_request(self, method, params=None, data=None):
            return {"status": "error", "message": "boom"}

    connector = _DummyConnector({"type": "UNIX"}, FakeSocketClient())
    result = connector._socket_request("status")

    # the underlying message is wrapped: "Socket call 'status' failed: boom"
    assert "boom" in result["error"]
    assert "boom" in connector.errors[0][0]


def test_socket_request_exception_is_caught_and_reported():
    class ExplodingSocketClient:
        def send_request(self, method, params=None, data=None):
            raise ConnectionError("down")

    connector = _DummyConnector({"type": "UNIX"}, ExplodingSocketClient())
    result = connector._socket_request("status")

    assert result["error"] == "down"
    assert connector.errors[0][0] == "down"
