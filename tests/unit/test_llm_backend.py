"""OpenAI-compatible adapter, client factory, and optional-dependency isolation.

The adapter is exercised against a fake endpoint that mimics the SDK's response
shape. No test contacts a provider.
"""

import ast
import builtins
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Optional

import pytest
from pydantic import BaseModel

from llm.backend import StructuredOutputClient, build_client
from llm.cache import CachingLLMClient, LLMCache
from llm.errors import (
    LLMConfigurationError,
    LLMCredentialsError,
    LLMDependencyError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from llm.models import (
    OPENAI_COMPATIBLE,
    LLMMessage,
    LLMSettings,
    ModelSettings,
    StructuredOutputMode,
)
from llm.providers import openai_compat
from llm.providers.openai_compat import OpenAICompatibleClient

pytestmark = pytest.mark.unit

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


class Answer(BaseModel):
    candidate_id: str
    reason: Optional[str] = None


class FakeEndpoint:
    """Minimal stand-in for ``openai.OpenAI``'s chat-completions surface."""

    def __init__(self, content=None, raise_error=None, usage=True, choices=None):
        self._content = content if content is not None else '{"candidate_id": "c-1", "reason": null}'
        self._raise_error = raise_error
        self._usage = usage
        self._choices = choices
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        if self._raise_error is not None:
            raise self._raise_error
        choices = self._choices
        if choices is None:
            choices = [
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content), finish_reason="stop"
                )
            ]
        return SimpleNamespace(
            model=request.get("model"),
            choices=choices,
            usage=(
                SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16)
                if self._usage
                else None
            ),
        )


def _settings(**overrides) -> LLMSettings:
    """Settings straight from the dataclass — no configuration file involved."""

    base = {"model": "gpt-4o-mini", "base_url": "http://localhost:3000/v1"}
    base.update(overrides)
    mode = base.get("structured_output")
    if isinstance(mode, str):
        base["structured_output"] = StructuredOutputMode(mode)
    return LLMSettings(**base)


def _client(endpoint, settings=None) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(settings or _settings(), client=endpoint)


def _messages():
    return [LLMMessage.system("Be exact."), LLMMessage.user("Pick one.")]


def _complete(client, **overrides):
    kwargs = {
        "messages": _messages(),
        "response_model": Answer,
        "settings": ModelSettings(model="gpt-4o-mini", max_output_tokens=256),
        "feature": "automapping",
    }
    kwargs.update(overrides)
    return client.complete(**kwargs)


# --- request construction -------------------------------------------------


def test_request_carries_plain_message_records():
    endpoint = FakeEndpoint()

    _complete(_client(endpoint))

    request = endpoint.requests[0]
    assert request["messages"] == [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "Pick one."},
    ]
    assert request["model"] == "gpt-4o-mini"
    assert request["max_tokens"] == 256
    assert request["timeout"] == 120.0


def test_strict_json_schema_response_format_is_sent():
    endpoint = FakeEndpoint()

    _complete(_client(endpoint))

    response_format = endpoint.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_requested_output_mode_is_sent_verbatim_and_never_downgraded():
    for mode, expected in (
        (StructuredOutputMode.JSON_SCHEMA, "json_schema"),
        (StructuredOutputMode.JSON_OBJECT, "json_object"),
    ):
        endpoint = FakeEndpoint()
        _complete(
            _client(endpoint),
            settings=ModelSettings(model="m", structured_output=mode),
        )
        assert endpoint.requests[0]["response_format"]["type"] == expected

    endpoint = FakeEndpoint()
    _complete(
        _client(endpoint),
        settings=ModelSettings(model="m", structured_output=StructuredOutputMode.PROMPT),
    )
    assert "response_format" not in endpoint.requests[0]


def test_extra_body_is_forwarded_for_compatible_servers():
    endpoint = FakeEndpoint()
    settings = _settings(extra_body={"enable_thinking": True})

    _complete(_client(endpoint, settings))

    assert endpoint.requests[0]["extra_body"] == {"enable_thinking": True}


def test_seed_is_only_sent_when_configured():
    endpoint = FakeEndpoint()
    _complete(_client(endpoint), settings=ModelSettings(model="m"))
    assert "seed" not in endpoint.requests[0]

    endpoint = FakeEndpoint()
    _complete(_client(endpoint), settings=ModelSettings(model="m", seed=3))
    assert endpoint.requests[0]["seed"] == 3


# --- response handling ----------------------------------------------------


def test_valid_response_is_returned_with_usage_and_provenance():
    response = _complete(_client(FakeEndpoint()))

    assert response.value.candidate_id == "c-1"
    assert response.provider == OPENAI_COMPATIBLE
    assert response.model == "gpt-4o-mini"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 16
    assert response.cache_hit is False
    assert response.finish_reason == "stop"


def test_missing_usage_is_tolerated():
    response = _complete(_client(FakeEndpoint(usage=False)))

    assert response.usage.total_tokens is None


def test_schema_violating_response_is_rejected():
    endpoint = FakeEndpoint(content='{"wrong_field": 1}')

    with pytest.raises(LLMResponseValidationError, match="did not satisfy"):
        _complete(_client(endpoint))


def test_empty_choices_are_rejected():
    with pytest.raises(LLMResponseValidationError, match="no choices"):
        _complete(_client(FakeEndpoint(choices=[])))


def test_absent_message_content_is_rejected_with_the_finish_reason():
    choices = [SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="length")]

    with pytest.raises(LLMResponseValidationError, match="finish_reason=length"):
        _complete(_client(FakeEndpoint(choices=choices)))


# --- error mapping --------------------------------------------------------


def test_sdk_timeout_is_translated_to_a_timeout_error(monkeypatch):
    class FakeTimeout(Exception):
        pass

    endpoint = FakeEndpoint(raise_error=FakeTimeout("too slow"))
    client = _client(endpoint)
    monkeypatch.setattr(
        client, "_openai", SimpleNamespace(APITimeoutError=FakeTimeout, APIError=None)
    )

    with pytest.raises(LLMTimeoutError, match="did not respond"):
        _complete(client)


def test_sdk_api_error_is_translated_to_a_provider_error(monkeypatch):
    class FakeAPIError(Exception):
        status_code = 429

    endpoint = FakeEndpoint(raise_error=FakeAPIError("rate limited"))
    client = _client(endpoint)
    monkeypatch.setattr(
        client,
        "_openai",
        SimpleNamespace(APITimeoutError=None, APIError=FakeAPIError, APIConnectionError=None),
    )

    with pytest.raises(LLMProviderError) as excinfo:
        _complete(client)
    assert excinfo.value.status_code == 429


def test_unrecognised_exceptions_are_not_swallowed():
    endpoint = FakeEndpoint(raise_error=RuntimeError("bug in the adapter"))

    with pytest.raises(RuntimeError, match="bug in the adapter"):
        _complete(_client(endpoint))


def test_missing_sdk_raises_an_actionable_dependency_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "openai", raising=False)

    with pytest.raises(LLMDependencyError) as excinfo:
        openai_compat._load_openai_module()

    assert "pip install -e '.[llm]'" in str(excinfo.value)


def test_missing_credential_fails_at_construction_not_mid_run(monkeypatch):
    monkeypatch.setattr(
        openai_compat,
        "_load_openai_module",
        lambda: SimpleNamespace(OpenAI=lambda **_: FakeEndpoint()),
    )

    with pytest.raises(LLMCredentialsError, match="LLM_API_KEY"):
        OpenAICompatibleClient(_settings(), env={})


def test_credentials_never_appear_in_the_error_message(monkeypatch):
    monkeypatch.setattr(
        openai_compat,
        "_load_openai_module",
        lambda: SimpleNamespace(OpenAI=lambda **_: FakeEndpoint()),
    )
    client = OpenAICompatibleClient(_settings(), env={"LLM_API_KEY": "sk-secret"})
    endpoint = client._client

    _complete(client)

    assert "sk-secret" not in json.dumps(endpoint.requests, default=str)


# --- factory --------------------------------------------------------------


def test_build_client_wraps_the_adapter_in_its_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        openai_compat,
        "_load_openai_module",
        lambda: SimpleNamespace(OpenAI=lambda **_: FakeEndpoint()),
    )
    settings = _settings(cache_dir=str(tmp_path / "llm_cache"))

    client = build_client(settings, env={"LLM_API_KEY": "k"})

    # The output contract is applied outside the cache, so the cache keys on the
    # messages that are really sent.
    assert isinstance(client, StructuredOutputClient)
    assert isinstance(client.inner, CachingLLMClient)
    assert isinstance(client.inner.inner, OpenAICompatibleClient)
    assert client.inner.provider_fingerprint == settings.provider_fingerprint()


def test_build_client_honours_a_disabled_cache(monkeypatch):
    monkeypatch.setattr(
        openai_compat,
        "_load_openai_module",
        lambda: SimpleNamespace(OpenAI=lambda **_: FakeEndpoint()),
    )
    settings = _settings(cache_enabled=False)

    client = build_client(settings, env={"LLM_API_KEY": "k"})

    assert isinstance(client, StructuredOutputClient)
    assert isinstance(client.inner, OpenAICompatibleClient)


def test_build_client_accepts_an_injected_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        openai_compat,
        "_load_openai_module",
        lambda: SimpleNamespace(OpenAI=lambda **_: FakeEndpoint()),
    )
    cache = LLMCache(tmp_path / "custom")

    client = build_client(_settings(), env={"LLM_API_KEY": "k"}, cache=cache)

    assert client.inner.cache is cache


def test_prompt_mode_actually_puts_the_schema_in_the_prompt(monkeypatch, tmp_path):
    """Prompt-constrained mode must constrain the prompt, not just the parser."""

    endpoint = FakeEndpoint()
    monkeypatch.setattr(
        openai_compat, "_load_openai_module", lambda: SimpleNamespace(OpenAI=lambda **_: endpoint)
    )
    settings = _settings(
        structured_output="prompt", cache_dir=str(tmp_path / "llm_cache")
    )
    client = build_client(settings, env={"LLM_API_KEY": "k"})

    client.complete(
        messages=_messages(),
        response_model=Answer,
        settings=settings.model_settings(),
        feature="automapping",
    )

    sent = endpoint.requests[0]["messages"]
    assert "response_format" not in endpoint.requests[0]
    assert sent[-1]["role"] == "user"
    assert "candidate_id" in sent[-1]["content"]
    assert "single JSON document" in sent[-1]["content"]
    # the original instruction survives
    assert sent[-1]["content"].startswith("Pick one.")


def test_prompt_mode_schema_is_part_of_the_cached_request(monkeypatch, tmp_path):
    """The stored entry must describe the prompt that was actually sent."""

    endpoint = FakeEndpoint()
    monkeypatch.setattr(
        openai_compat, "_load_openai_module", lambda: SimpleNamespace(OpenAI=lambda **_: endpoint)
    )
    settings = _settings(
        structured_output="prompt", cache_dir=str(tmp_path / "llm_cache")
    )
    client = build_client(settings, env={"LLM_API_KEY": "k"})

    client.complete(
        messages=_messages(),
        response_model=Answer,
        settings=settings.model_settings(),
        feature="automapping",
    )

    entry = next((tmp_path / "llm_cache" / "automapping").glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    assert "single JSON document" in payload["request"]["messages"][-1]["content"]


def test_server_constrained_modes_leave_the_messages_untouched(monkeypatch):
    endpoint = FakeEndpoint()
    monkeypatch.setattr(
        openai_compat, "_load_openai_module", lambda: SimpleNamespace(OpenAI=lambda **_: endpoint)
    )
    settings = _settings(cache_enabled=False)
    client = build_client(settings, env={"LLM_API_KEY": "k"})

    client.complete(
        messages=_messages(),
        response_model=Answer,
        settings=settings.model_settings(),
    )

    assert endpoint.requests[0]["messages"] == [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "Pick one."},
    ]


def test_prompt_mode_appends_a_user_turn_when_there_is_none():
    recorded = {}

    class Recorder:
        provider = OPENAI_COMPATIBLE

        def complete(self, *, messages, response_model, settings, feature="llm", context=None):
            recorded["messages"] = messages
            return SimpleNamespace(value=None)

    StructuredOutputClient(Recorder()).complete(
        messages=[LLMMessage.system("Only a system turn.")],
        response_model=Answer,
        settings=ModelSettings(model="m", structured_output=StructuredOutputMode.PROMPT),
    )

    assert recorded["messages"][-1].role.value == "user"
    assert "single JSON document" in recorded["messages"][-1].content


def test_changing_extra_body_misses_the_cache(monkeypatch, tmp_path):
    """`enable_thinking` and friends change the answer, so they change the key."""

    calls = []

    def endpoint_factory(**_):
        endpoint = FakeEndpoint()
        calls.append(endpoint)
        return endpoint

    monkeypatch.setattr(
        openai_compat, "_load_openai_module", lambda: SimpleNamespace(OpenAI=endpoint_factory)
    )
    cache_dir = str(tmp_path / "llm_cache")

    def run(extra_body):
        settings = _settings(cache_dir=cache_dir, extra_body=extra_body)
        client = build_client(settings, env={"LLM_API_KEY": "k"})
        return client.complete(
            messages=_messages(),
            response_model=Answer,
            settings=settings.model_settings(),
            feature="automapping",
        )

    assert run({"enable_thinking": True}).cache_hit is False
    assert run({"enable_thinking": True}).cache_hit is True
    assert run({"enable_thinking": False}).cache_hit is False


def test_timeout_is_deliberately_outside_the_cache_identity():
    """A deadline bounds the wait, not the content.

    A cache entry exists only because a call completed; a timeout raises rather
    than storing a partial answer. Keying on it would discard every cached
    response whenever the deadline is tuned, with none of them having become
    wrong.
    """

    quick = ModelSettings(model="m", timeout_s=5.0)
    patient = ModelSettings(model="m", timeout_s=600.0)

    assert quick.cache_fingerprint() == patient.cache_fingerprint()


def test_build_client_rejects_an_unregistered_provider():
    settings = _settings()
    settings = LLMSettings(**{**settings.__dict__, "provider": "anthropic-native"})

    with pytest.raises(LLMConfigurationError, match="No adapter is registered"):
        build_client(settings)


# --- optional-dependency isolation ----------------------------------------


def _import_graph_probe(statements: str) -> str:
    """Run *statements* in a clean interpreter and report whether openai loaded."""

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})\n"
        f"{statements}\n"
        "print('LOADED' if 'openai' in sys.modules else 'ABSENT')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_importing_the_cli_does_not_load_a_provider_sdk():
    assert _import_graph_probe("import view.options; view.options.get_args") == "ABSENT"


def test_importing_the_pipeline_controller_does_not_load_a_provider_sdk():
    assert (
        _import_graph_probe(
            "from controller.pipeline_controller.pipeline_controller import PipelineController"
        )
        == "ABSENT"
    )


def test_importing_the_llm_package_itself_does_not_load_a_provider_sdk():
    """The SDK must load only inside build_client, after an LLM mode is chosen."""

    assert _import_graph_probe("import llm, llm.backend, llm.cache") == "ABSENT"


def _module_probe(statements: str, module: str) -> str:
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})\n"
        f"{statements}\n"
        f"print('LOADED' if {module!r} in sys.modules else 'ABSENT')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_jinja2_is_optional_too():
    """Prompt templating is an LLM-mode dependency, not a core one."""

    assert _module_probe("import llm, llm.backend, llm.cache", "jinja2") == "ABSENT"
    assert (
        _module_probe("from view import options; options.get_args", "jinja2") == "ABSENT"
    )
    # ...and it does load once a prompt is actually rendered
    assert (
        _module_probe(
            "from llm.prompts import render, AUTOMAPPING_SYSTEM\n"
            "render(AUTOMAPPING_SYSTEM, unmapped_token='unmapped')",
            "jinja2",
        )
        == "LOADED"
    )


def test_no_langchain_or_langgraph_import_in_the_llm_package():
    """Prose may name LangChain to explain its absence; code may not import it."""

    def _is_banned(module: str) -> bool:
        root = (module or "").split(".", 1)[0]
        return root.startswith("langchain") or root == "langgraph"

    for path in (REPO_ROOT / "src" / "llm").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(_is_banned(a.name) for a in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert not _is_banned(node.module or ""), path


def test_llm_extra_declares_no_langchain_dependency():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extra = pyproject.split("llm = [", 1)[1].split("]", 1)[0]

    assert "openai" in extra
    assert "langchain" not in extra
    assert "langgraph" not in extra
