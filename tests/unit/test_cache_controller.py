"""Unit tests for CacheController (controller/external_services/cache_controller.py).

Uses the real DISK cache backend (data_handling.caching.disk_cache.DiskResourceCache)
rooted at tmp_path -- it is pure local-filesystem code, so exercising it is not a
"real external service" in the sense the task guards against (no network/valkey).
For the error/exception paths and the bytes-key decode branch (Valkey-specific)
we substitute a tiny local fake cache object instead of touching the real backend.
"""

import json

import pytest

from controller.external_services.cache_controller import CacheController

pytestmark = pytest.mark.unit


def make_controller(tmp_path, **conf_overrides):
    conf = {"external_cache_service": "DISK", "cache_args": {"cache_dir": str(tmp_path)}}
    conf.update(conf_overrides)
    return CacheController(conf)


class RaisingCache:
    """Fake cache whose methods raise, to exercise CacheController's except branches."""

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise RuntimeError(f"boom in {name}")

        return _raise


class BytesKeyCache:
    """Fake cache returning a MIXED str/bytes key list from list_cached_resources.

    list_resources must decode bytes->str (Valkey keys) BEFORE sorting; a mixed
    list previously made sorted() raise ``TypeError: '<' not supported between
    instances of 'str' and 'bytes'``, which the broad except turned into a
    sys.exit(1). The mixed list here asserts the fixed decode-then-sort order.
    """

    def list_cached_resources(self, filter_pattern=None):
        return [b"http://example.org/c", "http://example.org/a", b"http://example.org/b"]


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_unsupported_service_exits(tmp_path):
    conf = {"external_cache_service": "NOPE", "cache_args": {}}
    with pytest.raises(SystemExit) as exc:
        CacheController(conf)
    assert exc.value.code == 1


def test_init_disk_backend_success(tmp_path):
    controller = make_controller(tmp_path)
    assert controller.cache.cache_dir == tmp_path


# --------------------------------------------------------------------------- #
# add_resource
# --------------------------------------------------------------------------- #


def test_add_resource_success(tmp_path, capsys):
    controller = make_controller(tmp_path)
    src = tmp_path / "res.json"
    src.write_text(json.dumps({"resourceType": "Patient", "url": "http://x/patient"}))
    controller.add_resource(str(src))
    out = capsys.readouterr().out
    assert "Resource added to cache: http://x/patient" in out
    assert controller.cache.get_resource_from_cache("http://x/patient") is not None


def test_add_resource_missing_file_exits(tmp_path, capsys):
    controller = make_controller(tmp_path)
    with pytest.raises(SystemExit) as exc:
        controller.add_resource(str(tmp_path / "missing.json"))
    assert exc.value.code == 1
    assert "File not found" in capsys.readouterr().out


def test_add_resource_no_url_field_exits(tmp_path):
    controller = make_controller(tmp_path)
    src = tmp_path / "res.json"
    src.write_text(json.dumps({"resourceType": "Patient"}))
    with pytest.raises(SystemExit) as exc:
        controller.add_resource(str(src))
    assert exc.value.code == 1


def test_add_resource_invalid_json_exits(tmp_path):
    controller = make_controller(tmp_path)
    src = tmp_path / "res.json"
    src.write_text("{not valid json")
    with pytest.raises(SystemExit) as exc:
        controller.add_resource(str(src))
    assert exc.value.code == 1


def test_add_resource_cache_failure_prints_message(tmp_path, capsys):
    controller = make_controller(tmp_path)
    src = tmp_path / "res.json"
    src.write_text(json.dumps({"resourceType": "Patient", "url": "http://x/p"}))
    controller.cache.add_resource_to_cache = lambda resource: None
    controller.add_resource(str(src))
    assert "Failed to add resource to cache" in capsys.readouterr().out


def test_add_resource_unexpected_exception_exits(tmp_path):
    controller = make_controller(tmp_path)
    controller.cache = RaisingCache()
    src = tmp_path / "res.json"
    src.write_text(json.dumps({"resourceType": "Patient", "url": "http://x/p"}))
    with pytest.raises(SystemExit) as exc:
        controller.add_resource(str(src))
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# get_resource
# --------------------------------------------------------------------------- #


def test_get_resource_not_found_prints_message(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.get_resource("http://missing")
    assert "Resource not found: http://missing" in capsys.readouterr().out


def test_get_resource_prints_to_console(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.cache.add_resource_to_cache({"resourceType": "Patient", "url": "http://x/p"})
    controller.get_resource("http://x/p")
    out = capsys.readouterr().out
    assert '"resourceType": "Patient"' in out


def test_get_resource_writes_output_file(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.cache.add_resource_to_cache({"resourceType": "Patient", "url": "http://x/p"})
    out_file = tmp_path / "out" / "resource.json"
    controller.get_resource("http://x/p", output_file=str(out_file))
    assert out_file.exists()
    assert json.loads(out_file.read_text())["url"] == "http://x/p"
    assert "Resource saved to" in capsys.readouterr().out


def test_get_resource_exception_exits(tmp_path):
    controller = make_controller(tmp_path)
    controller.cache = RaisingCache()
    with pytest.raises(SystemExit) as exc:
        controller.get_resource("http://x")
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# delete_resource
# --------------------------------------------------------------------------- #


def test_delete_resource_not_found(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.delete_resource("http://missing")
    assert "Resource not found: http://missing" in capsys.readouterr().out


def test_delete_resource_success(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.cache.add_resource_to_cache({"resourceType": "Patient", "url": "http://x/p"})
    controller.delete_resource("http://x/p")
    assert "Resource deleted: http://x/p" in capsys.readouterr().out
    assert controller.cache.get_resource_from_cache("http://x/p") is None


def test_delete_resource_exception_exits(tmp_path):
    controller = make_controller(tmp_path)
    controller.cache = RaisingCache()
    with pytest.raises(SystemExit) as exc:
        controller.delete_resource("http://x")
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# list_resources
# --------------------------------------------------------------------------- #


def test_list_resources_empty_no_filter(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.list_resources()
    assert "No cached resources found" in capsys.readouterr().out


def test_list_resources_empty_with_filter(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.list_resources(filter_pattern="patient")
    assert "No cached resources matching 'patient'" in capsys.readouterr().out


def test_list_resources_prints_sorted_urls(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.cache.add_resource_to_cache({"resourceType": "Patient", "url": "http://x/b"})
    controller.cache.add_resource_to_cache({"resourceType": "Patient", "url": "http://x/a"})
    controller.list_resources()
    out = capsys.readouterr().out
    assert "Found 2 cached resource(s)" in out
    assert out.index("http://x/a") < out.index("http://x/b")


def test_list_resources_decodes_mixed_bytes_and_str_keys_before_sorting(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.cache = BytesKeyCache()  # mixed str/bytes list
    controller.list_resources()  # must not TypeError/sys.exit(1)
    out = capsys.readouterr().out
    assert "Found 3 cached resource(s)" in out
    a, b, c = (out.index(f"http://example.org/{k}") for k in ("a", "b", "c"))
    assert a < b < c  # decoded first, then sorted


def test_list_resources_exception_exits(tmp_path):
    controller = make_controller(tmp_path)
    controller.cache = RaisingCache()
    with pytest.raises(SystemExit) as exc:
        controller.list_resources()
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# show_stats
# --------------------------------------------------------------------------- #


def test_show_stats_success(tmp_path, capsys):
    controller = make_controller(tmp_path)
    controller.show_stats()
    assert "Disk Cache Stats" in capsys.readouterr().out


def test_show_stats_exception_exits(tmp_path):
    controller = make_controller(tmp_path)
    controller.cache = RaisingCache()
    with pytest.raises(SystemExit) as exc:
        controller.show_stats()
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# clear_cache
# --------------------------------------------------------------------------- #


def test_clear_cache_prompt_declined(tmp_path, monkeypatch, capsys):
    controller = make_controller(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    controller.clear_cache(force=False)
    assert "Cache clear cancelled" in capsys.readouterr().out


def test_clear_cache_prompt_accepted(tmp_path, monkeypatch, capsys):
    controller = make_controller(tmp_path)
    controller.cache.add_resource_to_cache({"resourceType": "Patient", "url": "http://x/p"})
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    controller.clear_cache(force=False)
    assert "All cached resources have been removed" in capsys.readouterr().out
    assert controller.cache.list_cached_resources() == []


def test_clear_cache_force_skips_prompt(tmp_path, monkeypatch, capsys):
    controller = make_controller(tmp_path)

    def fail_input(prompt=""):
        raise AssertionError("input() should not be called when force=True")

    monkeypatch.setattr("builtins.input", fail_input)
    controller.clear_cache(force=True)
    assert "All cached resources have been removed" in capsys.readouterr().out


def test_clear_cache_exception_exits(tmp_path):
    controller = make_controller(tmp_path)
    controller.cache = RaisingCache()
    with pytest.raises(SystemExit) as exc:
        controller.clear_cache(force=True)
    assert exc.value.code == 1
