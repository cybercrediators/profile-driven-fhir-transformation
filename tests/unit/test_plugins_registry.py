"""Unit tests for the plugin registry / loader (plugins/registry.py)."""

import pytest

from plugins.redcap.plugin import REDCapPlugin
from plugins.registry import _resolve_config_values, _resolve_env, load_plugins

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# _resolve_env
# --------------------------------------------------------------------------- #


def test_resolve_env_expands_set_variable(monkeypatch):
    monkeypatch.setenv("REDCAP_TEST_TOKEN", "secret123")
    assert _resolve_env("${REDCAP_TEST_TOKEN}") == "secret123"


def test_resolve_env_unset_variable_returned_unchanged(monkeypatch):
    monkeypatch.delenv("REDCAP_UNSET_VAR", raising=False)
    assert _resolve_env("${REDCAP_UNSET_VAR}") == "${REDCAP_UNSET_VAR}"


def test_resolve_env_bare_string_returned_unchanged():
    assert _resolve_env("plain-value") == "plain-value"
    assert _resolve_env("csv") == "csv"


def test_resolve_env_partial_braces_not_expanded():
    # Must look like a full ${VAR} reference, not just contain '$' or braces.
    assert _resolve_env("${INCOMPLETE") == "${INCOMPLETE"
    assert _resolve_env("prefix_${VAR}_suffix") == "prefix_${VAR}_suffix"


# --------------------------------------------------------------------------- #
# _resolve_config_values
# --------------------------------------------------------------------------- #


def test_resolve_config_values_expands_top_level_strings(monkeypatch):
    monkeypatch.setenv("REDCAP_API_URL", "https://redcap.example.org/api/")
    config = {"type": "redcap", "source": "api", "api_url": "${REDCAP_API_URL}"}
    resolved = _resolve_config_values(config)
    assert resolved["api_url"] == "https://redcap.example.org/api/"
    assert resolved["source"] == "api"


def test_resolve_config_values_recurses_into_nested_dicts(monkeypatch):
    monkeypatch.setenv("NESTED_VAR", "nested-value")
    config = {"outer": {"inner": "${NESTED_VAR}", "plain": 42}}
    resolved = _resolve_config_values(config)
    assert resolved["outer"]["inner"] == "nested-value"
    assert resolved["outer"]["plain"] == 42


def test_resolve_config_values_leaves_non_string_values_untouched():
    config = {"api_verify_ssl": True, "count": 3, "items": ["a", "b"]}
    resolved = _resolve_config_values(config)
    assert resolved == config


# --------------------------------------------------------------------------- #
# load_plugins
# --------------------------------------------------------------------------- #


def test_load_plugins_empty_or_none_returns_empty_list():
    assert load_plugins([]) == []
    assert load_plugins(None) == []


def test_load_plugins_instantiates_redcap_plugin(tmp_path):
    csv_path = tmp_path / "dict.csv"
    csv_path.write_text("Variable / Field Name,Field Type\nrecord_id,text\n", encoding="utf-8")

    configs = [{"type": "redcap", "source": "csv", "codebook_path": str(csv_path)}]
    plugins = load_plugins(configs)

    assert len(plugins) == 1
    assert isinstance(plugins[0], REDCapPlugin)
    assert plugins[0].plugin_id == "redcap"
    # codebook_path must have passed through config resolution unchanged
    assert plugins[0].config["codebook_path"] == str(csv_path)


def test_load_plugins_resolves_env_vars_in_config(monkeypatch, tmp_path):
    monkeypatch.setenv("REDCAP_TOKEN_ENV", "tok-abc")
    configs = [
        {
            "type": "redcap",
            "source": "api",
            "api_url": "https://redcap.example.org/api/",
            "api_token": "${REDCAP_TOKEN_ENV}",
        }
    ]
    plugins = load_plugins(configs)
    assert plugins[0].config["api_token"] == "tok-abc"


def test_load_plugins_skips_unknown_plugin_type(caplog):
    configs = [{"type": "not_a_real_plugin"}]
    with caplog.at_level("WARNING"):
        plugins = load_plugins(configs)
    assert plugins == []
    assert any("Unknown plugin type" in r.message for r in caplog.records)


def test_load_plugins_type_lookup_is_case_insensitive(tmp_path):
    csv_path = tmp_path / "dict.csv"
    csv_path.write_text("Variable / Field Name,Field Type\nrecord_id,text\n", encoding="utf-8")
    configs = [{"type": "REDCAP", "source": "csv", "codebook_path": str(csv_path)}]
    plugins = load_plugins(configs)
    assert len(plugins) == 1
    assert isinstance(plugins[0], REDCapPlugin)


def test_load_plugins_missing_type_key_is_skipped(caplog):
    with caplog.at_level("WARNING"):
        plugins = load_plugins([{"source": "csv"}])
    assert plugins == []


def test_load_plugins_continues_after_constructor_failure(monkeypatch, caplog):
    """If a plugin class raises during construction, load_plugins must log the
    error and continue rather than propagating the exception."""
    import plugins.redcap.plugin as redcap_plugin_module

    class _ExplodingPlugin(redcap_plugin_module.REDCapPlugin):
        def __init__(self, config):
            raise RuntimeError("boom")

    monkeypatch.setattr(redcap_plugin_module, "REDCapPlugin", _ExplodingPlugin)

    with caplog.at_level("ERROR"):
        plugins = load_plugins([{"type": "redcap", "source": "csv", "codebook_path": "x.csv"}])

    assert plugins == []
    assert any("Failed to load plugin" in r.message for r in caplog.records)
