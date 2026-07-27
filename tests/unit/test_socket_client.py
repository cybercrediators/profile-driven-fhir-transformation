import json
import socket as socket_module
from types import SimpleNamespace

import pytest

import view.socket_client as socket_client_module
from view.socket_client import SocketClient


class _FakeSocket:
    def __init__(self, response_bytes: bytes):
        self._response = response_bytes
        self.sent = b""

    def sendall(self, data: bytes):
        self.sent += data

    def recv(self, _: int) -> bytes:
        if self._response:
            data = self._response
            self._response = b""
            return data
        return b""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_client(monkeypatch):
    # Provide a fake socket connection that returns a success response once.
    # send_request reads until it sees a newline, so include it.
    fake_sock = _FakeSocket(b'{"status": "success", "data": {"ok": true}}\n')

    def _connect(self):
        return fake_sock

    client = SocketClient({"type": "UNIX", "path": "/tmp/test.sock"})
    monkeypatch.setattr(SocketClient, "_connect", _connect, raising=True)
    return SimpleNamespace(client=client, sock=fake_sock)


def test_send_request_builds_payload_and_returns_response(fake_client):
    payload = fake_client.client.send_request(
        method="status",
        params={"foo": "bar"},
        data={"hello": "world"},
    )

    # Verify outbound JSON contains method/params/data
    sent_text = fake_client.sock.sent.decode("utf-8").strip()
    sent_obj = json.loads(sent_text)
    assert sent_obj["method"] == "status"
    assert sent_obj["params"] == {"foo": "bar"}
    assert sent_obj["data"] == {"hello": "world"}

    # Verify we parsed the response from the fake socket (full envelope is returned)
    assert payload == {"status": "success", "data": {"ok": True}}


def test_send_request_defaults_params(fake_client):
    fake_client.sock.sent = b""  # reset capture
    _ = fake_client.client.send_request(method="status")
    sent_obj = json.loads(fake_client.sock.sent.decode("utf-8"))
    assert sent_obj["params"] == {}


# --------------------------------------------------------------------------- #
# __init__ (str/default constructor path)
# --------------------------------------------------------------------------- #


def test_init_with_default_args_uses_socket_path_arg():
    client = SocketClient()
    assert client.socket_path == "/tmp/fsh_nifi_bridge.sock"


def test_init_with_str_type_uses_socket_path_arg():
    client = SocketClient("UNIX", socket_path="/tmp/custom.sock")
    assert client.socket_path == "/tmp/custom.sock"


def test_init_with_dict_conf_reads_path_key():
    client = SocketClient({"path": "/tmp/from_conf.sock"})
    assert client.socket_path == "/tmp/from_conf.sock"


def test_init_with_dict_conf_defaults_when_no_path_key():
    client = SocketClient({})
    assert client.socket_path == "/tmp/fsh_nifi_bridge.sock"


# --------------------------------------------------------------------------- #
# _connect()
# --------------------------------------------------------------------------- #


class FakeRawSocket:
    def __init__(self, connect_exc=None):
        self.connect_exc = connect_exc
        self.connected_to = None

    def connect(self, path):
        if self.connect_exc:
            raise self.connect_exc
        self.connected_to = path


def test_connect_success_returns_connected_socket(monkeypatch):
    fake_sock = FakeRawSocket()
    monkeypatch.setattr(
        socket_client_module.socket, "socket", lambda family, kind: fake_sock
    )
    client = SocketClient(socket_path="/tmp/test.sock")
    result = client._connect()
    assert result is fake_sock
    assert fake_sock.connected_to == "/tmp/test.sock"


def test_connect_file_not_found_raises_connection_error(monkeypatch):
    fake_sock = FakeRawSocket(connect_exc=FileNotFoundError())
    monkeypatch.setattr(
        socket_client_module.socket, "socket", lambda family, kind: fake_sock
    )
    client = SocketClient(socket_path="/tmp/missing.sock")
    with pytest.raises(ConnectionError, match="UNIX socket not found"):
        client._connect()


def test_connect_connection_refused_raises_connection_error(monkeypatch):
    fake_sock = FakeRawSocket(connect_exc=ConnectionRefusedError())
    monkeypatch.setattr(
        socket_client_module.socket, "socket", lambda family, kind: fake_sock
    )
    client = SocketClient(socket_path="/tmp/refused.sock")
    with pytest.raises(ConnectionError, match="Connection refused"):
        client._connect()


# --------------------------------------------------------------------------- #
# send_request(): error/edge paths through _connect()
# --------------------------------------------------------------------------- #


def test_send_request_returns_none_when_connect_raises_connection_error(monkeypatch):
    client = SocketClient(socket_path="/tmp/test.sock")

    def fail_connect(self):
        raise ConnectionError("bridge server not running")

    monkeypatch.setattr(SocketClient, "_connect", fail_connect, raising=True)
    assert client.send_request("status") is None


def test_send_request_returns_none_on_unexpected_exception(monkeypatch):
    client = SocketClient(socket_path="/tmp/test.sock")

    def fail_connect(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(SocketClient, "_connect", fail_connect, raising=True)
    assert client.send_request("status") is None


def test_send_request_returns_none_on_empty_response(monkeypatch):
    class EmptySocket:
        def sendall(self, data):
            pass

        def recv(self, n):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    client = SocketClient(socket_path="/tmp/test.sock")
    monkeypatch.setattr(SocketClient, "_connect", lambda self: EmptySocket(), raising=True)
    assert client.send_request("status") is None
