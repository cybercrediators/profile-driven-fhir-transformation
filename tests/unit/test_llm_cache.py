"""Content-addressed LLM response cache and its invalidation behaviour."""

import json
from typing import Optional

import pytest
from pydantic import BaseModel

from llm.cache import CachingLLMClient, LLMCache
from llm.errors import LLMResponseValidationError
from llm.models import (
    OPENAI_COMPATIBLE,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    ModelSettings,
    StructuredOutputMode,
)

pytestmark = pytest.mark.unit

API_KEY = "sk-do-not-persist-this"
FINGERPRINT = {"provider": OPENAI_COMPATIBLE, "base_url": "http://localhost:3000/v1"}


class Answer(BaseModel):
    candidate_id: str
    reason: Optional[str] = None


class RecordingClient:
    """Stand-in provider that counts calls and returns a scripted payload."""

    provider = OPENAI_COMPATIBLE

    def __init__(self, raw_text: str = '{"candidate_id": "c-1", "reason": null}'):
        self.raw_text = raw_text
        self.calls = 0

    def complete(self, *, messages, response_model, settings, feature="llm", context=None):
        self.calls += 1
        return LLMResponse(
            value=response_model.model_validate_json(self.raw_text),
            raw_text=self.raw_text,
            provider=self.provider,
            model=settings.model,
            usage=LLMUsage(input_tokens=11, output_tokens=3, total_tokens=14),
            cache_hit=False,
            finish_reason="stop",
        )


def _settings(**overrides) -> ModelSettings:
    base = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_output_tokens": 256,
        "structured_output": StructuredOutputMode.JSON_SCHEMA,
    }
    base.update(overrides)
    return ModelSettings(**base)


def _messages(user: str = "Pick a candidate."):
    return [LLMMessage.system("You are a mapper."), LLMMessage.user(user)]


def _wrapped(tmp_path, inner=None):
    inner = inner or RecordingClient()
    cache = LLMCache(tmp_path / "llm_cache")
    return inner, cache, CachingLLMClient(inner, cache, FINGERPRINT)


def _call(client, **overrides):
    kwargs = {
        "messages": _messages(),
        "response_model": Answer,
        "settings": _settings(),
        "feature": "automapping",
    }
    kwargs.update(overrides)
    return client.complete(**kwargs)


def test_first_call_reaches_the_provider_and_reports_no_cache_hit(tmp_path):
    inner, _, client = _wrapped(tmp_path)

    response = _call(client)

    assert inner.calls == 1
    assert response.cache_hit is False
    assert response.value.candidate_id == "c-1"


def test_identical_request_is_served_from_cache(tmp_path):
    inner, _, client = _wrapped(tmp_path)

    _call(client)
    response = _call(client)

    assert inner.calls == 1
    assert response.cache_hit is True
    assert response.value.candidate_id == "c-1"
    assert response.usage.total_tokens == 14


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"messages": _messages("A different question.")}, id="prompt"),
        pytest.param({"feature": "agent-fix"}, id="feature"),
        pytest.param({"context": {"map_sha256": "abc"}}, id="context"),
        pytest.param({"settings": _settings(model="other-model")}, id="model"),
        pytest.param({"settings": _settings(temperature=0.7)}, id="temperature"),
        pytest.param({"settings": _settings(max_output_tokens=512)}, id="max_tokens"),
        pytest.param({"settings": _settings(seed=42)}, id="seed"),
        pytest.param(
            {"settings": _settings(structured_output=StructuredOutputMode.JSON_OBJECT)},
            id="output_mode",
        ),
    ],
)
def test_any_change_to_request_identity_misses_the_cache(tmp_path, overrides):
    inner, _, client = _wrapped(tmp_path)

    _call(client)
    _call(client, **overrides)

    assert inner.calls == 2


def test_changing_the_output_schema_misses_the_cache(tmp_path):
    class WiderAnswer(BaseModel):
        candidate_id: str
        reason: Optional[str] = None
        confidence: Optional[float] = None

    inner = RecordingClient('{"candidate_id": "c-1", "reason": null, "confidence": null}')
    cache = LLMCache(tmp_path / "llm_cache")
    client = CachingLLMClient(inner, cache, FINGERPRINT)

    _call(client, response_model=Answer)
    _call(client, response_model=WiderAnswer)

    assert inner.calls == 2


def test_changing_the_provider_endpoint_misses_the_cache(tmp_path):
    inner = RecordingClient()
    cache = LLMCache(tmp_path / "llm_cache")

    _call(CachingLLMClient(inner, cache, FINGERPRINT))
    _call(
        CachingLLMClient(
            inner, cache, {"provider": OPENAI_COMPATIBLE, "base_url": "http://elsewhere/v1"}
        )
    )

    assert inner.calls == 2


def test_context_change_alone_invalidates_an_otherwise_identical_prompt(tmp_path):
    inner, _, client = _wrapped(tmp_path)

    _call(client, context={"map_sha256": "aaa"})
    _call(client, context={"map_sha256": "aaa"})
    assert inner.calls == 1

    _call(client, context={"map_sha256": "bbb"})
    assert inner.calls == 2


def test_cached_entry_records_the_prompt_and_raw_response(tmp_path):
    _, cache, client = _wrapped(tmp_path)

    _call(client)

    entries = list((tmp_path / "llm_cache" / "automapping").glob("*.json"))
    assert len(entries) == 1
    payload = json.loads(entries[0].read_text(encoding="utf-8"))

    assert payload["request"]["feature"] == "automapping"
    assert payload["request"]["messages"][1]["content"] == "Pick a candidate."
    assert payload["response"]["raw_text"] == '{"candidate_id": "c-1", "reason": null}'
    assert payload["response"]["usage"]["total_tokens"] == 14


def test_no_credential_ever_reaches_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", API_KEY)
    _, _, client = _wrapped(tmp_path)

    _call(client, messages=[LLMMessage.user("Pick a candidate.")])

    for path in (tmp_path / "llm_cache").rglob("*.json"):
        assert API_KEY not in path.read_text(encoding="utf-8")


def test_cached_text_is_revalidated_rather_than_trusted(tmp_path):
    """A cache read must not be a way around schema validation."""

    _, cache, client = _wrapped(tmp_path)
    _call(client)

    entry_path = next((tmp_path / "llm_cache" / "automapping").glob("*.json"))
    payload = json.loads(entry_path.read_text(encoding="utf-8"))
    payload["response"]["raw_text"] = '{"unexpected": true}'
    entry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LLMResponseValidationError):
        _call(client)


def test_corrupt_entry_is_treated_as_a_miss_not_a_failure(tmp_path):
    inner, _, client = _wrapped(tmp_path)
    _call(client)

    entry_path = next((tmp_path / "llm_cache" / "automapping").glob("*.json"))
    entry_path.write_text("{not json", encoding="utf-8")

    response = _call(client)

    assert inner.calls == 2
    assert response.cache_hit is False


def test_disabled_cache_never_reads_or_writes(tmp_path):
    inner = RecordingClient()
    cache = LLMCache(tmp_path / "llm_cache", enabled=False)
    client = CachingLLMClient(inner, cache, FINGERPRINT)

    _call(client)
    _call(client)

    assert inner.calls == 2
    assert not (tmp_path / "llm_cache").exists()


def test_clear_removes_only_the_named_feature(tmp_path):
    _, cache, client = _wrapped(tmp_path)
    _call(client, feature="automapping")
    _call(client, feature="agent-fix")

    assert cache.clear("automapping") == 1
    assert not list((tmp_path / "llm_cache" / "automapping").glob("*.json"))
    assert list((tmp_path / "llm_cache" / "agent-fix").glob("*.json"))


def test_key_material_is_inspectable_and_credential_free(tmp_path):
    cache = LLMCache(tmp_path / "llm_cache")

    material = cache.key_material(
        feature="automapping",
        provider_fingerprint=FINGERPRINT,
        settings=_settings(),
        response_model=Answer,
        messages=_messages(),
        context={"a": 1},
    )

    assert set(material) == {
        "cache_format_version",
        "feature",
        "provider",
        "settings",
        "schema",
        "messages",
        "context",
    }
    assert "api_key" not in json.dumps(material)
