"""Unit tests for the CacheCLI view layer (view/cache_cli.py).

CacheController is monkeypatched at the module level so no disk/valkey cache
backend is actually constructed.
"""

import pytest

import view.cache_cli as cache_cli_module
from view.cache_cli import CacheCLI

pytestmark = pytest.mark.unit


class FakeCacheController:
    def __init__(self, conf):
        self.conf = conf
        self.calls = []

    def clear_cache(self, force=False):
        self.calls.append(("clear_cache", force))

    def get_resource(self, res_url, output_path):
        self.calls.append(("get_resource", res_url, output_path))

    def delete_resource(self, res_url):
        self.calls.append(("delete_resource", res_url))

    def list_resources(self, filter_pattern):
        self.calls.append(("list_resources", filter_pattern))

    def show_stats(self):
        self.calls.append(("show_stats",))

    def add_resource(self, file_path):
        self.calls.append(("add_resource", file_path))


@pytest.fixture
def fake_controller(monkeypatch):
    holder = {}

    def _make(conf):
        controller = FakeCacheController(conf)
        holder["instance"] = controller
        return controller

    monkeypatch.setattr(cache_cli_module, "CacheController", _make)
    return holder


def make_cli(conf=None):
    return CacheCLI(conf if conf is not None else {})


@pytest.mark.parametrize(
    "action,kwargs,expected_call",
    [
        ("clear", {"force": True}, ("clear_cache", True)),
        ("get", {"res_url": "http://x/p", "output_path": "out.json"}, ("get_resource", "http://x/p", "out.json")),
        ("delete", {"res_url": "http://x/p"}, ("delete_resource", "http://x/p")),
        ("list", {"filter_pattern": "patient"}, ("list_resources", "patient")),
        ("stats", {}, ("show_stats",)),
        ("add", {"file_path": "in.json"}, ("add_resource", "in.json")),
    ],
)
def test_handle_cache_command_dispatches(fake_controller, action, kwargs, expected_call):
    cli = make_cli()
    cli.handle_cache_command(action, **kwargs)
    assert fake_controller["instance"].calls == [expected_call]


def test_handle_cache_command_constructs_controller_with_conf(fake_controller):
    conf = {"external_cache_service": "DISK"}
    cli = make_cli(conf)
    cli.handle_cache_command("stats")
    assert fake_controller["instance"].conf is conf


def test_handle_cache_command_unknown_action_exits(fake_controller, capsys):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_cache_command("bogus-action")
    assert exc.value.code == 1
    assert "No cache action specified" in capsys.readouterr().out
    assert fake_controller["instance"].calls == []


def test_handle_cache_command_none_action_exits(fake_controller):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_cache_command(None)
    assert exc.value.code == 1
