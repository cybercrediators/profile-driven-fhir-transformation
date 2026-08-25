"""Resolution of optional LLM settings — from the environment, and only there.

The property under test throughout: **no LLM configuration is read from
`conf/*.json`**. Endpoint, model, and everything adjacent to a credential live
in `.env` / the environment, so a committed per-project file cannot carry them
and there is exactly one place to look when a run talks to the wrong model.
"""

from pathlib import Path

import pytest

from llm.errors import LLMConfigurationError, LLMCredentialsError
from llm.models import (
    DEFAULT_API_KEY_ENV,
    OPENAI_COMPATIBLE,
    LLMSettings,
    StructuredOutputMode,
    load_dotenv,
)

pytestmark = pytest.mark.unit


def test_the_environment_supplies_every_value():
    settings = LLMSettings.from_env(
        env={
            "LLM_MODEL_NAME": "gpt-4o-mini",
            "LLM_API_BASE_URL": "http://localhost:3000/v1",
            "LLM_MAX_OUTPUT_TOKENS": "2048",
            "LLM_TEMPERATURE": "0.2",
            "LLM_TIMEOUT_S": "45",
            "LLM_SEED": "7",
        },
        project_path="/projects/demo/",
    )

    assert settings.model == "gpt-4o-mini"
    assert settings.base_url == "http://localhost:3000/v1"
    assert settings.max_output_tokens == 2048
    assert settings.temperature == 0.2
    assert settings.timeout_s == 45.0
    assert settings.seed == 7
    assert settings.provider == OPENAI_COMPATIBLE
    assert settings.structured_output is StructuredOutputMode.JSON_SCHEMA


def test_settings_take_nothing_from_a_project_configuration():
    """`from_env` has no configuration parameter at all, so a stray `llm`
    section cannot influence a run even by accident."""

    import inspect

    parameters = inspect.signature(LLMSettings.from_env).parameters
    assert set(parameters) == {"env", "project_path", "dotenv"}


def test_llm_backend_openai_is_accepted_as_a_migration_alias():
    settings = LLMSettings.from_env(env={"LLM_MODEL_NAME": "m", "LLM_BACKEND": "openai"})

    assert settings.provider == OPENAI_COMPATIBLE


def test_unsupported_provider_is_rejected_rather_than_defaulted():
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        LLMSettings.from_env(env={"LLM_MODEL_NAME": "m", "LLM_BACKEND": "ollama-native"})


def test_missing_model_names_the_variable_and_says_where_it_is_not_read_from():
    with pytest.raises(LLMConfigurationError, match="LLM_MODEL_NAME") as excinfo:
        LLMSettings.from_env(env={})

    assert "conf/*.json" in str(excinfo.value)


def test_unsupported_structured_output_mode_is_rejected():
    with pytest.raises(LLMConfigurationError, match="LLM_STRUCTURED_OUTPUT"):
        LLMSettings.from_env(
            env={"LLM_MODEL_NAME": "m", "LLM_STRUCTURED_OUTPUT": "freeform"}
        )


def test_non_numeric_token_limit_is_reported_clearly():
    with pytest.raises(LLMConfigurationError, match="LLM_MAX_OUTPUT_TOKENS"):
        LLMSettings.from_env(env={"LLM_MODEL_NAME": "m", "LLM_MAX_OUTPUT_TOKENS": "many"})


def test_a_non_boolean_cache_flag_is_reported_clearly():
    with pytest.raises(LLMConfigurationError, match="LLM_CACHE_ENABLED"):
        LLMSettings.from_env(env={"LLM_MODEL_NAME": "m", "LLM_CACHE_ENABLED": "maybe"})


@pytest.mark.parametrize(
    ("raw", "expected"), [("false", False), ("0", False), ("true", True), ("", True)]
)
def test_cache_flag_accepts_the_usual_spellings(raw, expected):
    settings = LLMSettings.from_env(
        env={"LLM_MODEL_NAME": "m", "LLM_CACHE_ENABLED": raw}
    )
    assert settings.cache_enabled is expected


def test_cache_directory_defaults_beside_the_project_artifacts():
    settings = LLMSettings.from_env(
        env={"LLM_MODEL_NAME": "m"}, project_path="/projects/demo/"
    )

    assert settings.cache_dir == Path("/projects/demo/llm_cache")


def test_explicit_cache_directory_overrides_the_project_default():
    settings = LLMSettings.from_env(
        env={"LLM_MODEL_NAME": "m", "LLM_CACHE_DIR": "/tmp/c"},
        project_path="/projects/demo/",
    )

    assert settings.cache_dir == Path("/tmp/c")


def test_api_key_is_never_stored_on_the_settings_object():
    settings = LLMSettings.from_env(env={"LLM_MODEL_NAME": "m"})

    assert settings.api_key_env == DEFAULT_API_KEY_ENV
    assert settings.resolve_api_key({DEFAULT_API_KEY_ENV: "secret-value"}) == "secret-value"
    assert "secret-value" not in repr(settings)


def test_custom_api_key_variable_is_honoured():
    settings = LLMSettings.from_env(
        env={"LLM_MODEL_NAME": "m", "LLM_API_KEY_ENV": "MY_PROJECT_KEY"}
    )

    assert settings.resolve_api_key({"MY_PROJECT_KEY": "k"}) == "k"


def test_absent_credential_raises_instead_of_sending_an_empty_key():
    settings = LLMSettings.from_env(env={"LLM_MODEL_NAME": "m"})

    with pytest.raises(LLMCredentialsError, match=DEFAULT_API_KEY_ENV):
        settings.resolve_api_key({})


def test_provider_fingerprint_excludes_credentials():
    settings = LLMSettings.from_env(
        env={
            "LLM_MODEL_NAME": "m",
            "LLM_API_BASE_URL": "http://h/v1",
            "LLM_API_KEY": "sk-x",
        }
    )

    fingerprint = settings.provider_fingerprint()

    assert fingerprint == {
        "provider": OPENAI_COMPATIBLE,
        "base_url": "http://h/v1",
        "extra_body": {},
    }
    assert "sk-x" not in str(fingerprint)


def test_extra_body_is_json_and_part_of_the_cache_identity():
    """`extra_body` reaches the endpoint, so it changes what the model returns."""

    thinking_on = LLMSettings.from_env(
        env={"LLM_MODEL_NAME": "m", "LLM_EXTRA_BODY": '{"enable_thinking": true}'}
    )
    thinking_off = LLMSettings.from_env(
        env={"LLM_MODEL_NAME": "m", "LLM_EXTRA_BODY": '{"enable_thinking": false}'}
    )

    assert thinking_on.extra_body == {"enable_thinking": True}
    assert thinking_on.provider_fingerprint() != thinking_off.provider_fingerprint()


def test_malformed_extra_body_is_rejected():
    with pytest.raises(LLMConfigurationError, match="LLM_EXTRA_BODY"):
        LLMSettings.from_env(env={"LLM_MODEL_NAME": "m", "LLM_EXTRA_BODY": "not json"})


# --- the .env loader ----------------------------------------------------------


def test_dotenv_is_loaded_without_a_dependency(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        "LLM_MODEL_NAME=qwen3\n"
        'LLM_API_BASE_URL="http://localhost:11434/v1"\n'
        "MALFORMED\n"
    )
    env: dict = {}

    load_dotenv(path, env=env)

    assert env["LLM_MODEL_NAME"] == "qwen3"
    # Quotes are stripped; a line without `=` is skipped rather than fatal.
    assert env["LLM_API_BASE_URL"] == "http://localhost:11434/v1"
    assert "MALFORMED" not in env


def test_an_exported_variable_wins_over_the_file(tmp_path):
    """An export, a CI secret, and docker-compose's env_file must all beat the
    checked-out file, or a developer's local .env silently overrides CI."""

    path = tmp_path / ".env"
    path.write_text("LLM_MODEL_NAME=from-file\n")
    env = {"LLM_MODEL_NAME": "from-environment"}

    load_dotenv(path, env=env)

    assert env["LLM_MODEL_NAME"] == "from-environment"


def test_a_missing_dotenv_is_not_an_error(tmp_path):
    env: dict = {}
    load_dotenv(tmp_path / "nope.env", env=env)
    assert env == {}
