"""Unit tests for SocketServer (controller/socket_controller/socket_controller.py) and
serve_forever_process (controller/socket_controller/socket_controller_process.py).

SocketServer.__init__ constructs a real PipelineController (which in turn wants a
real project on disk, matchbox connection, etc.) -- far too heavy for a unit test.
We monkeypatch the module-level `PipelineController` name that socket_controller.py
imports with a lightweight FakePipelineController, then construct SocketServer for
real so its actual __init__ logic (dispatcher wiring, worker/thread config, shared
validation state) gets exercised.

Framing/dispatch is tested against a real UNIX socketpair so `_handle_connection`'s
recv/decode/newline-framing loop runs unmodified against real socket objects (no
mocking of the socket itself, per the "no network" constraint -- socketpair never
leaves the machine).
"""

import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

import controller.socket_controller.socket_controller as socket_controller_module
import controller.socket_controller.socket_controller_process as scp_module
from controller.socket_controller.socket_controller import SocketServer
from controller.socket_controller.socket_controller_process import serve_forever_process

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakePipelineController:
    """Stand-in for PipelineController: records calls, returns configurable results."""

    def __init__(self, conf):
        self.conf = conf
        self.app_state = SimpleNamespace(
            conf=conf,
            registry=SimpleNamespace(registry_objects={}),
            cache=SimpleNamespace(get_resource_from_cache=lambda url: None),
        )
        self.structure_map_urls = []
        self.plugins = []
        self.ensure_processed_called = False
        self.validate_setup_result = True
        self.validate_setup_raises = None
        self.sync_changed_result = {"changed": []}
        self.prepare_matchbox_result = {"uploaded": []}
        self.transform_data_result = None
        self.transform_data_queue = None  # per-call results (batch flow)
        self.validate_data_result = {"ok": True}

    def _ensure_processed(self):
        self.ensure_processed_called = True

    def validate_setup(self, read_only=True):
        if self.validate_setup_raises:
            raise self.validate_setup_raises
        return self.validate_setup_result

    def sync_changed_resources(self):
        return self.sync_changed_result

    def prepare_matchbox_setup(self, force_upload=False):
        return self.prepare_matchbox_result

    def transform_data(self, input_data, structure_map_url=None, map_outputs=None):
        if self.transform_data_queue is not None:
            return self.transform_data_queue.pop(0) if self.transform_data_queue else None
        return self.transform_data_result

    def validate_data(self, resource_obj, profile_url):
        return self.validate_data_result


def make_server(monkeypatch, conf=None, **fake_overrides):
    """Build a real SocketServer with a FakePipelineController injected."""
    created = {}

    def _factory(conf):
        fake = FakePipelineController(conf)
        for k, v in fake_overrides.items():
            setattr(fake, k, v)
        created["pc"] = fake
        return fake

    monkeypatch.setattr(socket_controller_module, "PipelineController", _factory)
    server = SocketServer(conf=conf or {}, run_validation_scheduler=False)
    return server, created["pc"]


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_builds_dispatcher_and_default_config(monkeypatch):
    server, fake_pc = make_server(monkeypatch, conf={})
    assert fake_pc.ensure_processed_called is True
    assert server.worker_processes == 1
    assert server.thread_workers == 8
    assert server.running is True
    assert set(server.dispatcher.keys()) == {
        "status", "stop", "transform_data", "transform_batch",
        "validate_data", "validate_setup", "prepare_matchbox", "sync_resources",
    }
    assert server._mp_stop_event is None


def test_init_reads_socket_conf_overrides(monkeypatch):
    conf = {
        "socket_connection": {
            "worker_processes": 1,
            "max_workers": 3,
            "validation_interval": 42,
            "path": "/tmp/custom.sock",
        }
    }
    server, _ = make_server(monkeypatch, conf=conf)
    assert server.thread_workers == 3
    assert server.validation_interval == 42
    assert server.socket_conf["path"] == "/tmp/custom.sock"


def test_init_with_shared_validation_state(monkeypatch):
    shared_status = {"ok": True, "timestamp": 1.0, "error": None}
    lock = threading.Lock()
    server, _ = make_server(
        monkeypatch,
        conf={},
    )
    # constructed directly to hit the "provided" branch
    def _factory(conf):
        return FakePipelineController(conf)

    monkeypatch.setattr(socket_controller_module, "PipelineController", _factory)
    server2 = SocketServer(
        conf={}, shared_validation_status=shared_status, shared_validation_lock=lock,
        run_validation_scheduler=False,
    )
    assert server2._validation_status is shared_status
    assert server2._validation_status_lock is lock


# --------------------------------------------------------------------------- #
# get_status / stop_server
# --------------------------------------------------------------------------- #


def test_get_status_reports_root_resources(monkeypatch):
    server, fake_pc = make_server(monkeypatch)
    obj1 = SimpleNamespace(is_root=True, data=SimpleNamespace(url="http://x/root"))
    obj2 = SimpleNamespace(is_root=False, data=SimpleNamespace(url="http://x/nonroot"))
    fake_pc.app_state.registry.registry_objects = {"a": obj1, "b": obj2}
    status = server.get_status()
    assert status["service"] == "FSH-NiFi Bridge"
    assert status["status"] == "running"
    assert status["processed_resources"] == 2
    assert status["root_resources"] == ["http://x/root"]
    assert status["validate_setup"] == {"ok": None, "timestamp": None, "error": None}


def test_stop_server_sets_flags_and_returns_message(monkeypatch):
    server, _ = make_server(monkeypatch)
    result = server.stop_server()
    assert server.running is False
    assert server._stop_event.is_set()
    assert result == {"message": "Server is shutting down."}


def test_stop_server_joins_running_validation_thread(monkeypatch):
    server, _ = make_server(monkeypatch)
    done = threading.Event()

    def _loop():
        done.wait(timeout=2)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    server._validation_thread = thread
    done.set()
    result = server.stop_server()
    assert result == {"message": "Server is shutting down."}
    assert not thread.is_alive()


def test_stop_server_sets_mp_stop_event_when_present(monkeypatch):
    server, _ = make_server(monkeypatch)
    server._mp_stop_event = threading.Event()
    server.stop_server()
    assert server._mp_stop_event.is_set()


# --------------------------------------------------------------------------- #
# prepare_matchbox_setup / sync_resources
# --------------------------------------------------------------------------- #


def test_prepare_matchbox_setup_delegates(monkeypatch):
    server, fake_pc = make_server(monkeypatch, prepare_matchbox_result={"x": 1})
    assert server.prepare_matchbox_setup() == {"x": 1}


def test_sync_resources_force_true_uses_prepare_matchbox(monkeypatch):
    server, fake_pc = make_server(monkeypatch, prepare_matchbox_result="all-changed")
    result = server.sync_resources(force=True)
    assert result == {"changed": "all-changed"}


def test_sync_resources_force_false_uses_sync_changed(monkeypatch):
    server, fake_pc = make_server(monkeypatch, sync_changed_result="some-changed")
    result = server.sync_resources(force=False)
    assert result == {"changed": "some-changed"}


# --------------------------------------------------------------------------- #
# validate_setup
# --------------------------------------------------------------------------- #


def test_validate_setup_success_updates_status(monkeypatch):
    server, fake_pc = make_server(monkeypatch, validate_setup_result=True)
    result = server.validate_setup()
    assert result is True
    assert server._validation_status["ok"] is True
    assert server._validation_status["error"] is None


def test_validate_setup_failure_updates_status_with_error(monkeypatch):
    server, fake_pc = make_server(monkeypatch, validate_setup_result=False)
    result = server.validate_setup()
    assert result is False
    assert server._validation_status["ok"] is False
    assert server._validation_status["error"] == "setup check failed"


# --------------------------------------------------------------------------- #
# validate_data
# --------------------------------------------------------------------------- #


def test_validate_data_no_matchbox_connection_raises(monkeypatch):
    server, _ = make_server(monkeypatch, conf={})
    with pytest.raises(ValueError, match="No matchbox connection"):
        server.validate_data(data=json.dumps({"resourceType": "Patient"}))


def test_validate_data_success(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, fake_pc = make_server(monkeypatch, conf=conf, validate_data_result={"ok": True})
    result = server.validate_data(data=json.dumps({"resourceType": "Patient"}))
    assert result == {"ok": True}


def test_validate_data_unparseable_input_raises_runtime_error(monkeypatch):
    # Unparseable JSON input must produce a handled error: the same
    # raise-RuntimeError shape transform_data/transform_batch use, which
    # _handle_connection converts to a {"status": "error", ...} response.
    # (Previously this path crashed with an UnboundLocalError because
    # `response` was only assigned inside a skipped walrus-`if` body.)
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, _ = make_server(monkeypatch, conf=conf)
    with pytest.raises(RuntimeError, match="Failed to parse input data"):
        server.validate_data(data="not valid json")


def test_validate_data_unparseable_input_yields_error_response_over_socket(monkeypatch):
    # End-to-end over the socket protocol: the RuntimeError surfaces as a
    # proper error response, not an unhandled crash or dropped connection.
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, _ = make_server(monkeypatch, conf=conf)
    response = _send_and_handle(
        server,
        {"method": "validate_data", "params": {"data": "not valid json"}},
    )
    assert response == {"status": "error", "message": "Failed to parse input data!"}


# --------------------------------------------------------------------------- #
# transform_data / transform_batch
# --------------------------------------------------------------------------- #


def test_transform_data_no_matchbox_connection_raises(monkeypatch):
    server, _ = make_server(monkeypatch, conf={})
    with pytest.raises(ValueError, match="No matchbox connection"):
        server.transform_data(data=json.dumps({"resourceType": "Patient"}))


def test_transform_data_unparseable_input_raises_runtime_error(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, _ = make_server(monkeypatch, conf=conf)
    with pytest.raises(RuntimeError, match="Failed to parse input data"):
        server.transform_data(data="not valid json")


def test_transform_data_no_resources_raises_runtime_error(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, fake_pc = make_server(monkeypatch, conf=conf, transform_data_result=None)
    with pytest.raises(RuntimeError, match="Failed execution of the data transformation"):
        server.transform_data(data=json.dumps({"resourceType": "Patient"}))


def test_transform_data_returns_resources_list(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    resources = [{"resourceType": "Patient"}]
    server, fake_pc = make_server(monkeypatch, conf=conf, transform_data_result=resources)
    result = server.transform_data(data=json.dumps({"resourceType": "Foo"}))
    assert result == resources


def test_transform_data_bundle_true_wraps_result(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    resources = [{"resourceType": "Patient"}]
    server, fake_pc = make_server(monkeypatch, conf=conf, transform_data_result=resources)
    result = server.transform_data(data=json.dumps({"resourceType": "Foo"}), bundle=True)
    assert result["resourceType"] == "Bundle"
    assert result["entry"][0]["resource"] == {"resourceType": "Patient"}


def test_transform_batch_no_matchbox_connection_raises(monkeypatch):
    server, _ = make_server(monkeypatch, conf={})
    with pytest.raises(ValueError, match="No matchbox connection"):
        server.transform_batch(data=json.dumps([{"resourceType": "Patient"}]))


def test_transform_batch_requires_json_array(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, _ = make_server(monkeypatch, conf=conf)
    with pytest.raises(ValueError, match="requires a JSON array"):
        server.transform_batch(data=json.dumps({"resourceType": "Patient"}))


def test_transform_batch_empty_result_raises_runtime_error(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, fake_pc = make_server(monkeypatch, conf=conf, transform_data_result=None)
    with pytest.raises(RuntimeError, match="Failed execution of the batch data transformation"):
        server.transform_batch(data=json.dumps([{"resourceType": "Patient"}]))


def test_transform_batch_returns_batch(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    batch = [[{"resourceType": "Patient"}], [{"resourceType": "Observation"}]]
    server, fake_pc = make_server(
        monkeypatch, conf=conf, transform_data_queue=list(batch)
    )
    result = server.transform_batch(data=json.dumps([{"a": 1}, {"a": 2}]))
    assert result == batch


def test_transform_batch_skips_failed_records(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, fake_pc = make_server(
        monkeypatch,
        conf=conf,
        transform_data_queue=[[{"resourceType": "Patient"}], None],
    )
    result = server.transform_batch(data=json.dumps([{"a": 1}, {"a": 2}]))
    assert result == [[{"resourceType": "Patient"}]]


def test_transform_batch_bundle_true_wraps_each(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    batch = [[{"resourceType": "Patient"}]]
    server, fake_pc = make_server(
        monkeypatch, conf=conf, transform_data_queue=list(batch)
    )
    result = server.transform_batch(data=json.dumps([{"a": 1}]), bundle=True)
    assert len(result) == 1
    assert result[0]["resourceType"] == "Bundle"


# --------------------------------------------------------------------------- #
# _build_bundle_with_context
# --------------------------------------------------------------------------- #


def test_build_bundle_with_context_loads_structure_maps_from_cache(monkeypatch):
    server, fake_pc = make_server(monkeypatch)
    sm = {"resourceType": "StructureMap", "url": "http://x/sm", "group": []}
    fake_pc.structure_map_urls = ["http://x/sm", "http://x/missing"]
    fake_pc.app_state.cache.get_resource_from_cache = lambda url: sm if url == "http://x/sm" else None
    resources = [{"resourceType": "Patient"}]
    result = server._build_bundle_with_context(resources)
    assert result["resourceType"] == "Bundle"
    assert result["entry"][0]["resource"] == {"resourceType": "Patient"}


def test_build_bundle_with_context_uses_configured_external_references(monkeypatch):
    defaults = [
        {
            "path": "Observation.subject",
            "reference": "Patient/eval-patient-1",
        }
    ]
    server, _ = make_server(
        monkeypatch,
        conf={
            "matchbox_connection": {"url": "http://mb"},
            "external_reference_defaults": defaults,
        },
    )
    observation = {"resourceType": "Observation"}

    result = server._build_bundle_with_context([observation])

    assert result["entry"][0]["resource"]["subject"] == {
        "reference": "Patient/eval-patient-1",
        "type": "Patient",
    }


# --------------------------------------------------------------------------- #
# _small_json_conv
# --------------------------------------------------------------------------- #


def test_small_json_conv_valid(monkeypatch):
    server, _ = make_server(monkeypatch)
    assert server._small_json_conv('{"a": 1}') == {"a": 1}


def test_small_json_conv_invalid_returns_empty_dict(monkeypatch, caplog):
    server, _ = make_server(monkeypatch)
    with caplog.at_level("ERROR"):
        result = server._small_json_conv("not json")
    assert result == {}
    assert any("Failed to parse JSON data" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# _get_validation_error / _update_validation_status
# --------------------------------------------------------------------------- #


def test_get_validation_error_exempt_methods_always_none(monkeypatch):
    server, _ = make_server(monkeypatch)
    server._update_validation_status(ok=False, error="broken")
    for method in ("status", "validate_setup", "sync_resources", "stop"):
        assert server._get_validation_error(method) is None


def test_get_validation_error_blocks_when_invalid(monkeypatch):
    server, _ = make_server(monkeypatch)
    server._update_validation_status(ok=False, error="broken setup")
    assert server._get_validation_error("transform_data") == "broken setup"


def test_get_validation_error_default_message_when_no_error_text(monkeypatch):
    server, _ = make_server(monkeypatch)
    server._update_validation_status(ok=False, error=None)
    assert server._get_validation_error("transform_data") == "Validation failed; request blocked."


def test_get_validation_error_none_when_ok(monkeypatch):
    server, _ = make_server(monkeypatch)
    server._update_validation_status(ok=True)
    assert server._get_validation_error("transform_data") is None


def test_get_validation_error_none_when_unknown(monkeypatch):
    server, _ = make_server(monkeypatch)
    # ok is still None (never validated) -> not blocked
    assert server._get_validation_error("transform_data") is None


def test_update_validation_status_success_clears_error(monkeypatch):
    server, _ = make_server(monkeypatch)
    server._update_validation_status(ok=True, error="stale")
    assert server._validation_status["ok"] is True
    assert server._validation_status["error"] is None
    assert server._validation_status["timestamp"] is not None


# --------------------------------------------------------------------------- #
# _start_validation_scheduler
# --------------------------------------------------------------------------- #


def test_start_validation_scheduler_runs_once_and_spawns_thread(monkeypatch):
    server, fake_pc = make_server(monkeypatch, validate_setup_result=True)
    server.validation_interval = 0.01
    server._start_validation_scheduler()
    try:
        assert server._validation_thread is not None
        assert server._validation_thread.is_alive()
        assert server._validation_status["ok"] is True
    finally:
        server._stop_event.set()
        server._validation_thread.join(timeout=2)


def test_start_validation_scheduler_logs_initial_failure(monkeypatch, caplog):
    server, fake_pc = make_server(monkeypatch)
    fake_pc.sync_changed_result = None

    def _raise():
        raise RuntimeError("sync boom")

    server.pipeline_controller.sync_changed_resources = _raise
    server.validation_interval = 0.01
    with caplog.at_level("ERROR"):
        server._start_validation_scheduler()
    try:
        assert any("Initial validation failed" in r.message for r in caplog.records)
    finally:
        server._stop_event.set()
        server._validation_thread.join(timeout=2)


def test_validation_scheduler_loop_survives_periodic_failures(monkeypatch, caplog):
    server, fake_pc = make_server(monkeypatch)
    call_count = {"n": 0}

    def _flaky_validate_setup(read_only=True):
        call_count["n"] += 1
        if call_count["n"] > 1:
            raise RuntimeError("periodic boom")
        return True

    server.pipeline_controller.validate_setup = _flaky_validate_setup
    server.validation_interval = 0.01
    with caplog.at_level("ERROR"):
        server._start_validation_scheduler()
        # allow at least one periodic loop iteration
        time.sleep(0.1)
    server._stop_event.set()
    server._validation_thread.join(timeout=2)
    assert call_count["n"] >= 1


# --------------------------------------------------------------------------- #
# _handle_connection / handle_connection_safe -- real UNIX socketpair framing
# --------------------------------------------------------------------------- #


def _send_and_handle(server, request: dict):
    """Send `request` over a real socketpair to _handle_connection, return the parsed reply."""
    client, srv_conn = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
        client.shutdown(socket.SHUT_WR)
        server._handle_connection(srv_conn)
        client.settimeout(2)
        # The server writes one newline-terminated JSON response and then leaves
        # the connection open (closing it is the caller's/context-manager's job,
        # not _handle_connection's) -- so read until the "\n" framing terminator
        # rather than until EOF, which would never arrive here.
        reply = b""
        while b"\n" not in reply:
            chunk = client.recv(4096)
            if not chunk:
                break
            reply += chunk
        return json.loads(reply.decode("utf-8"))
    finally:
        client.close()
        srv_conn.close()


def test_handle_connection_dispatches_known_method(monkeypatch):
    server, fake_pc = make_server(monkeypatch)
    response = _send_and_handle(server, {"method": "status", "params": {}})
    assert response["status"] == "success"
    assert response["data"]["service"] == "FSH-NiFi Bridge"


def test_handle_connection_rejects_unknown_method(monkeypatch):
    server, _ = make_server(monkeypatch)
    response = _send_and_handle(server, {"method": "not-a-method", "params": {}})
    assert response == {"status": "error", "message": "Unknown method: not-a-method"}


def test_handle_connection_validates_params_dict(monkeypatch):
    server, _ = make_server(monkeypatch)
    response = _send_and_handle(server, {"method": "status", "params": [1, 2, 3]})
    assert response == {"status": "error", "message": "params must be a JSON object"}


def test_handle_connection_missing_method(monkeypatch):
    server, _ = make_server(monkeypatch)
    response = _send_and_handle(server, {"params": {}})
    assert response == {"status": "error", "message": "Missing method in request"}


def test_handle_connection_invalid_json(monkeypatch):
    server, _ = make_server(monkeypatch)
    client, srv_conn = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.sendall(b"{not json\n")
        client.shutdown(socket.SHUT_WR)
        server._handle_connection(srv_conn)
        client.settimeout(2)
        reply = client.recv(4096)
        assert json.loads(reply.decode("utf-8")) == {
            "status": "error", "message": "Invalid JSON request",
        }
    finally:
        client.close()
        srv_conn.close()


def test_handle_connection_empty_buffer_sends_nothing(monkeypatch):
    server, _ = make_server(monkeypatch)
    client, srv_conn = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.shutdown(socket.SHUT_WR)  # close write side without sending -> recv() returns b""
        server._handle_connection(srv_conn)
        srv_conn.settimeout(0.2)
        client.settimeout(0.2)
        with pytest.raises((socket.timeout, OSError)):
            client.recv(4096)
    finally:
        client.close()
        srv_conn.close()


def test_handle_connection_params_as_json_string(monkeypatch):
    server, fake_pc = make_server(monkeypatch)
    response = _send_and_handle(
        server, {"method": "status", "params": json.dumps({})}
    )
    assert response["status"] == "success"


def test_handle_connection_merges_data_field_into_params(monkeypatch):
    conf = {"matchbox_connection": {"url": "http://mb"}}
    server, fake_pc = make_server(
        monkeypatch, conf=conf, transform_data_result=[{"resourceType": "Patient"}]
    )
    response = _send_and_handle(
        server,
        {
            "method": "transform_data",
            "params": {},
            "data": json.dumps({"resourceType": "Foo"}),
        },
    )
    assert response["status"] == "success"
    assert response["data"] == [{"resourceType": "Patient"}]


def test_handle_connection_blocked_by_failed_validation(monkeypatch):
    server, _ = make_server(monkeypatch)
    server._update_validation_status(ok=False, error="setup broken")
    response = _send_and_handle(
        server, {"method": "transform_batch", "params": {"data": "[]"}}
    )
    assert response == {"status": "error", "message": "setup broken"}


def test_handle_connection_dispatcher_exception_is_caught(monkeypatch):
    server, fake_pc = make_server(monkeypatch)

    def _boom():
        raise RuntimeError("dispatcher exploded")

    server.dispatcher["status"] = _boom
    response = _send_and_handle(server, {"method": "status", "params": {}})
    assert response == {"status": "error", "message": "dispatcher exploded"}


def test_handle_connection_safe_swallows_exceptions(monkeypatch, caplog):
    server, _ = make_server(monkeypatch)

    class ExplodingConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def recv(self, n):
            raise RuntimeError("socket exploded")

    with caplog.at_level("ERROR"):
        server.handle_connection_safe(ExplodingConn())
    assert any("Unhandled connection error" in r.message for r in caplog.records)


def test_handle_connection_safe_happy_path(monkeypatch):
    server, fake_pc = make_server(monkeypatch)
    client, srv_conn = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.sendall((json.dumps({"method": "status", "params": {}}) + "\n").encode())
        client.shutdown(socket.SHUT_WR)
        server.handle_connection_safe(srv_conn)  # closes srv_conn itself (context manager)
        client.settimeout(2)
        reply = client.recv(4096)
        assert json.loads(reply.decode("utf-8"))["status"] == "success"
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# start_server -- real accept loop over a real UNIX socket, single-process mode
# --------------------------------------------------------------------------- #


def test_start_server_single_process_accepts_and_serves(monkeypatch, tmp_path):
    sock_path = tmp_path / "bridge.sock"
    conf = {"socket_connection": {"path": str(sock_path), "worker_processes": 1, "max_workers": 2}}
    server, fake_pc = make_server(monkeypatch, conf=conf)

    # start_server ends with sys.exit(0); make that a plain return from the thread's
    # perspective instead of letting SystemExit propagate out of the thread target.
    monkeypatch.setattr(socket_controller_module.sys, "exit", lambda code=0: None)

    thread = threading.Thread(target=server.start_server, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 5
        while not sock_path.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert sock_path.exists(), "server did not bind its UNIX socket in time"

        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(sock_path))
        client.sendall((json.dumps({"method": "status", "params": {}}) + "\n").encode())
        client.settimeout(3)
        reply = client.recv(4096)
        client.close()
        response = json.loads(reply.decode("utf-8"))
        assert response["status"] == "success"
        assert response["data"]["service"] == "FSH-NiFi Bridge"
    finally:
        server.stop_server()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_start_server_removes_stale_socket_file(monkeypatch, tmp_path):
    import stat

    sock_path = tmp_path / "stale.sock"
    sock_path.write_text("stale")  # pre-existing (non-socket) file at the socket path
    conf = {"socket_connection": {"path": str(sock_path), "worker_processes": 1, "max_workers": 2}}
    server, _ = make_server(monkeypatch, conf=conf)
    monkeypatch.setattr(socket_controller_module.sys, "exit", lambda code=0: None)

    def _is_bound_socket():
        try:
            return stat.S_ISSOCK(sock_path.stat().st_mode)
        except FileNotFoundError:
            return False

    thread = threading.Thread(target=server.start_server, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 5
        while not _is_bound_socket() and time.time() < deadline:
            time.sleep(0.02)
        assert _is_bound_socket(), "stale file was not replaced with a real bound socket"
    finally:
        server.stop_server()
        thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# serve_forever_process (socket_controller_process.py)
# --------------------------------------------------------------------------- #


class FakeWorkerServer:
    """Stand-in for the SocketServer constructed inside serve_forever_process."""

    def __init__(self, conf, shared_validation_status, shared_validation_lock, run_validation_scheduler):
        self.init_args = (conf, shared_validation_status, shared_validation_lock, run_validation_scheduler)
        self._run_validation_scheduler = run_validation_scheduler
        self.socket_conf = None
        self.worker_processes = None
        self.running = None
        self._stop_event = None
        self._mp_stop_event = None
        self.validation_interval = None
        self.scheduler_started = False
        self.handled = []

    def _start_validation_scheduler(self):
        self.scheduler_started = True

    def handle_connection_safe(self, conn):
        self.handled.append(conn)


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeServerSocket:
    """Stand-in for socket.fromfd(...): scripted accept() results, records settimeout/close."""

    def __init__(self, accept_results):
        self._results = list(accept_results)
        self.closed = False
        self.timeout = None

    def settimeout(self, t):
        self.timeout = t

    def accept(self):
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


def _patch_worker(monkeypatch, accept_results):
    fake_socket = FakeServerSocket(accept_results)
    monkeypatch.setattr(scp_module.socket, "fromfd", lambda fd, fam, typ: fake_socket)
    monkeypatch.setattr(socket_controller_module, "SocketServer", FakeWorkerServer)
    return fake_socket


def test_serve_forever_process_handles_connection_then_stops(monkeypatch):
    stop_event = threading.Event()
    conn = FakeConn()

    def _accept_then_stop():
        stop_event.set()
        return (conn, None)

    fake_socket = FakeServerSocket([])
    fake_socket._results = [(conn, None)]
    monkeypatch.setattr(scp_module.socket, "fromfd", lambda fd, fam, typ: fake_socket)
    monkeypatch.setattr(socket_controller_module, "SocketServer", FakeWorkerServer)

    # After the single accept() is consumed, make the *next* accept() call flip
    # stop_event and raise socket.timeout so the loop exits cleanly.
    original_accept = fake_socket.accept

    def accept_then_stop():
        if fake_socket._results:
            return original_accept()
        stop_event.set()
        raise socket.timeout()

    fake_socket.accept = accept_then_stop

    serve_forever_process(
        server_fd=99,
        conf={"socket_connection": {}},
        stop_event=stop_event,
        shared_validation_status={},
        shared_validation_lock=threading.Lock(),
        run_validation_scheduler=False,
        validation_interval=0,
    )

    assert fake_socket.closed is True
    assert stop_event.is_set()


def test_serve_forever_process_starts_validation_scheduler(monkeypatch):
    stop_event = threading.Event()
    stop_event.set()  # exit the accept loop immediately
    fake_socket = FakeServerSocket([socket.timeout()])
    monkeypatch.setattr(scp_module.socket, "fromfd", lambda fd, fam, typ: fake_socket)

    created = {}

    class RecordingWorkerServer(FakeWorkerServer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created["srv"] = self

    monkeypatch.setattr(socket_controller_module, "SocketServer", RecordingWorkerServer)

    serve_forever_process(
        server_fd=1,
        conf={"socket_connection": {}},
        stop_event=stop_event,
        shared_validation_status={},
        shared_validation_lock=threading.Lock(),
        run_validation_scheduler=True,
        validation_interval=5,
    )
    assert created["srv"].scheduler_started is True
    assert created["srv"].validation_interval == 5
    assert created["srv"].worker_processes == 1


def test_serve_forever_process_timeout_continues_loop(monkeypatch):
    stop_event = threading.Event()
    conn = FakeConn()
    fake_socket = FakeServerSocket([socket.timeout(), (conn, None)])
    monkeypatch.setattr(scp_module.socket, "fromfd", lambda fd, fam, typ: fake_socket)

    created = {}

    class RecordingWorkerServer(FakeWorkerServer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created["srv"] = self

        def handle_connection_safe(self, c):
            super().handle_connection_safe(c)
            stop_event.set()

    monkeypatch.setattr(socket_controller_module, "SocketServer", RecordingWorkerServer)

    serve_forever_process(
        server_fd=1,
        conf={"socket_connection": {}},
        stop_event=stop_event,
        shared_validation_status={},
        shared_validation_lock=threading.Lock(),
        run_validation_scheduler=False,
        validation_interval=0,
    )
    assert created["srv"].handled == [conn]


def test_serve_forever_process_oserror_before_stop_breaks_and_logs(monkeypatch, caplog):
    stop_event = threading.Event()
    fake_socket = FakeServerSocket([OSError("bad fd")])
    monkeypatch.setattr(scp_module.socket, "fromfd", lambda fd, fam, typ: fake_socket)
    monkeypatch.setattr(socket_controller_module, "SocketServer", FakeWorkerServer)

    with caplog.at_level("ERROR"):
        serve_forever_process(
            server_fd=1,
            conf={"socket_connection": {}},
            stop_event=stop_event,
            shared_validation_status={},
            shared_validation_lock=threading.Lock(),
            run_validation_scheduler=False,
            validation_interval=0,
        )
    assert fake_socket.closed is True
    assert any("Worker socket error" in r.message for r in caplog.records)


def test_serve_forever_process_oserror_after_stop_breaks_silently(monkeypatch, caplog):
    stop_event = threading.Event()
    stop_event.set()
    fake_socket = FakeServerSocket([OSError("bad fd")])
    monkeypatch.setattr(scp_module.socket, "fromfd", lambda fd, fam, typ: fake_socket)
    monkeypatch.setattr(socket_controller_module, "SocketServer", FakeWorkerServer)

    with caplog.at_level("ERROR"):
        serve_forever_process(
            server_fd=1,
            conf={"socket_connection": {}},
            stop_event=stop_event,
            shared_validation_status={},
            shared_validation_lock=threading.Lock(),
            run_validation_scheduler=False,
            validation_interval=0,
        )
    assert not any("Worker socket error" in r.message for r in caplog.records)


def test_serve_forever_process_generic_exception_logged_and_loop_ends(monkeypatch, caplog):
    stop_event = threading.Event()
    fake_socket = FakeServerSocket([RuntimeError("boom")])
    monkeypatch.setattr(scp_module.socket, "fromfd", lambda fd, fam, typ: fake_socket)

    created = {}

    class RecordingWorkerServer(FakeWorkerServer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created["srv"] = self

    monkeypatch.setattr(socket_controller_module, "SocketServer", RecordingWorkerServer)

    call_count = {"n": 0}
    original_accept = fake_socket.accept

    def accept_once_then_stop():
        call_count["n"] += 1
        if call_count["n"] > 1:
            stop_event.set()
            raise socket.timeout()
        return original_accept()

    fake_socket.accept = accept_once_then_stop

    with caplog.at_level("ERROR"):
        serve_forever_process(
            server_fd=1,
            conf={"socket_connection": {}},
            stop_event=stop_event,
            shared_validation_status={},
            shared_validation_lock=threading.Lock(),
            run_validation_scheduler=False,
            validation_interval=0,
        )
    assert any("Worker error" in r.message for r in caplog.records)
