"""typed contracts for the optional LLM layer"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, Generic, Mapping, Optional, Sequence, TypeVar

from llm.errors import LLMConfigurationError, LLMCredentialsError

T = TypeVar("T")

# set default variables, overwrite through flags/.env
DEFAULT_API_KEY_ENV = "LLM_API_KEY"
DEFAULT_CACHE_DIRNAME = "llm_cache"
DEFAULT_TIMEOUT_S = 120.0
OPENAI_COMPATIBLE = "openai-compatible"

# keep legacy config/naming scheme intact
_PROVIDER_ALIASES = {
    "openai": OPENAI_COMPATIBLE,
    "openai-compatible": OPENAI_COMPATIBLE,
}


class StructuredOutputMode(str, Enum):
    """define possible structured output modes"""

    #: Server-side enforcement against the exact JSON Schema of the output model
    JSON_SCHEMA = "json-schema"
    #: Server guarantees syntactically valid JSON not the schema
    JSON_OBJECT = "json-object"
    #: No server-side constraint
    PROMPT = "prompt"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class LLMMessage:
    """plain message record"""

    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> "LLMMessage":
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str) -> "LLMMessage":
        return cls(Role.ASSISTANT, content)

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class ModelSettings:
    """define model parameters for provider calls"""

    model: str
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    seed: Optional[int] = None
    structured_output: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA

    def cache_fingerprint(self) -> Dict[str, Any]:
        """Keep used model information and parameters for documentation/reports later on"""

        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
            "structured_output": self.structured_output.value,
        }


@dataclass(frozen=True)
class LLMUsage:
    """Token estimation (if the provider reports it)"""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def as_dict(self) -> Dict[str, Optional[int]]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMResponse(Generic[T]):
    """A validated structured response plus the provenance needed to audit it."""

    value: T
    raw_text: str
    provider: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    cache_hit: bool = False
    finish_reason: Optional[str] = None


def _first_present(*values: Optional[Any]) -> Optional[Any]:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _as_int(value: Any, name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(f"'{name}' must be an integer, got {value!r}") from exc


def _as_float(value: Any, name: str, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(f"'{name}' must be a number, got {value!r}") from exc


def _as_bool(value: Any, name: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise LLMConfigurationError(f"'{name}' must be a boolean, got {value!r}")


def load_dotenv(path: Optional[Path] = None, *, env: Optional[Dict[str, str]] = None) -> None:
    """load .env parameters if necessary"""

    target = os.environ if env is None else env
    source = Path(path) if path is not None else Path.cwd() / ".env"
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in target:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        target[key] = value


@dataclass(frozen=True)
class LLMSettings:
    """defines provider configuration based based on environment or flags"""

    model: str
    base_url: Optional[str] = None
    provider: str = OPENAI_COMPATIBLE
    api_key_env: str = DEFAULT_API_KEY_ENV
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    seed: Optional[int] = None
    structured_output: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        *,
        env: Optional[Mapping[str, str]] = None,
        project_path: Optional[str] = None,
        dotenv: Optional[Path] = None,
    ) -> "LLMSettings":
        """resolve settings from matching LLM_* env variables"""

        if env is None:
            load_dotenv(dotenv)
            env = os.environ

        provider_raw = str(
            _first_present(env.get("LLM_BACKEND"), OPENAI_COMPATIBLE)
        ).lower()
        provider = _PROVIDER_ALIASES.get(provider_raw)
        if provider is None:
            raise LLMConfigurationError(
                f"Unsupported LLM provider '{provider_raw}' in LLM_BACKEND. "
                f"Supported: {sorted(set(_PROVIDER_ALIASES))}."
            )

        model = env.get("LLM_MODEL_NAME")
        if not model:
            raise LLMConfigurationError(
                "No LLM model configured. Set LLM_MODEL_NAME in .env or the "
                "environment. LLM settings are never read from conf/*.json."
            )

        mode_raw = _first_present(
            env.get("LLM_STRUCTURED_OUTPUT"), StructuredOutputMode.JSON_SCHEMA.value
        )
        try:
            structured_output = StructuredOutputMode(str(mode_raw))
        except ValueError as exc:
            raise LLMConfigurationError(
                f"Unsupported LLM_STRUCTURED_OUTPUT '{mode_raw}'. Supported: "
                f"{[m.value for m in StructuredOutputMode]}."
            ) from exc

        cache_dir = env.get("LLM_CACHE_DIR")
        if cache_dir:
            resolved_cache_dir: Optional[Path] = Path(str(cache_dir)).expanduser()
        else:
            # Keep LLM caches beside the project's other generated artifacts.
            resolved_cache_dir = (
                Path(str(project_path)).expanduser() / DEFAULT_CACHE_DIRNAME
                if project_path
                else Path(DEFAULT_CACHE_DIRNAME)
            )

        extra_body_raw = env.get("LLM_EXTRA_BODY")
        extra_body: Dict[str, Any] = {}
        if extra_body_raw:
            try:
                parsed = json.loads(extra_body_raw)
            except ValueError as exc:
                raise LLMConfigurationError(
                    f"LLM_EXTRA_BODY must be a JSON object: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise LLMConfigurationError("LLM_EXTRA_BODY must be a JSON object.")
            extra_body = parsed

        return cls(
            model=str(model),
            base_url=env.get("LLM_API_BASE_URL") or None,
            provider=provider,
            api_key_env=str(
                _first_present(env.get("LLM_API_KEY_ENV"), DEFAULT_API_KEY_ENV)
            ),
            temperature=_as_float(env.get("LLM_TEMPERATURE"), "LLM_TEMPERATURE", 0.0),
            max_output_tokens=_as_int(
                env.get("LLM_MAX_OUTPUT_TOKENS"), "LLM_MAX_OUTPUT_TOKENS"
            ),
            timeout_s=_as_float(env.get("LLM_TIMEOUT_S"), "LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S),
            seed=_as_int(env.get("LLM_SEED"), "LLM_SEED"),
            structured_output=structured_output,
            cache_enabled=_as_bool(env.get("LLM_CACHE_ENABLED"), "LLM_CACHE_ENABLED", True),
            cache_dir=resolved_cache_dir,
            extra_body=extra_body,
        )

    def resolve_api_key(self, env: Optional[Mapping[str, str]] = None) -> str:
        """retrieve API token from configured env variable"""

        env = os.environ if env is None else env
        api_key = env.get(self.api_key_env)
        if not api_key:
            raise LLMCredentialsError(self.api_key_env)
        return api_key

    def model_settings(self) -> ModelSettings:
        return ModelSettings(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            timeout_s=self.timeout_s,
            seed=self.seed,
            structured_output=self.structured_output,
        )

    def provider_fingerprint(self) -> Dict[str, Any]:
        """store relevant provider identities"""

        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "extra_body": dict(self.extra_body),
        }


def messages_as_dicts(messages: Sequence[LLMMessage]) -> list[Dict[str, str]]:
    return [message.as_dict() for message in messages]
