"""adapter for OpenAI-compatible chat-completions endpoints"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

from llm.errors import (
    LLMDependencyError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from llm.models import (
    OPENAI_COMPATIBLE,
    LLMMessage,
    LLMResponse,
    LLMSettings,
    LLMUsage,
    ModelSettings,
    messages_as_dicts,
)
from llm.structured_output import parse_structured_response, response_format_for

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


def _load_openai_module():
    """Import the optional OpenAI SDK, or raise an actionable dependency error."""

    try:
        import openai  # noqa: PLC0415  (deliberately lazy)
    except ImportError as exc:
        raise LLMDependencyError("openai") from exc
    return openai


class OpenAICompatibleClient:
    """Implements the :class:`~llm.backend.LLMClient` protocol."""

    provider = OPENAI_COMPATIBLE

    def __init__(
        self,
        settings: LLMSettings,
        *,
        env: Optional[Mapping[str, str]] = None,
        client: Any = None,
    ) -> None:
        """
        :param settings: resolved provider configuration.
        :param env: environment mapping used to resolve the API key.
        :param client: pre-built SDK client, used by tests to stand in for a live
            endpoint without patching module globals.
        """

        self.settings = settings
        self._openai = None
        if client is not None:
            self._client = client
            return

        self._openai = _load_openai_module()
        api_key = settings.resolve_api_key(env)
        self._client = self._openai.OpenAI(
            api_key=api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_s,
        )

    def _timeout_types(self) -> tuple:
        openai = self._openai
        if openai is None:
            return ()
        return tuple(
            t
            for t in (
                getattr(openai, "APITimeoutError", None),
                getattr(openai, "APIConnectionError", None),
            )
            if t is not None
        )

    def _provider_error_types(self) -> tuple:
        openai = self._openai
        if openai is None:
            return ()
        api_error = getattr(openai, "APIError", None)
        return (api_error,) if api_error is not None else ()

    def complete(
        self,
        *,
        messages: Sequence[LLMMessage],
        response_model: Type[TModel],
        settings: ModelSettings,
        feature: str = "llm",
        context: Any = None,
    ) -> LLMResponse[TModel]:
        request: dict[str, Any] = {
            "model": settings.model,
            "messages": messages_as_dicts(messages),
            "temperature": settings.temperature,
            "timeout": settings.timeout_s,
        }
        if settings.max_output_tokens is not None:
            request["max_tokens"] = settings.max_output_tokens
        if settings.seed is not None:
            request["seed"] = settings.seed

        response_format = response_format_for(response_model, settings.structured_output)
        if response_format is not None:
            request["response_format"] = response_format
        if self.settings.extra_body:
            request["extra_body"] = dict(self.settings.extra_body)

        logger.debug(
            "LLM call [%s] model=%s mode=%s messages=%d context=%s",
            feature,
            settings.model,
            settings.structured_output.value,
            len(messages),
            "yes" if context is not None else "no",
        )

        try:
            completion = self._client.chat.completions.create(**request)
        except Exception as exc:  # narrowed below against the SDK's own types
            timeout_types = self._timeout_types()
            if timeout_types and isinstance(exc, timeout_types):
                raise LLMTimeoutError(
                    f"The provider did not respond within {settings.timeout_s}s."
                ) from exc
            provider_types = self._provider_error_types()
            if provider_types and isinstance(exc, provider_types):
                raise LLMProviderError(
                    f"Provider rejected the request: {exc}",
                    status_code=getattr(exc, "status_code", None),
                ) from exc
            raise

        raw_text, finish_reason = _extract_message(completion)
        value = parse_structured_response(raw_text, response_model, settings.structured_output)

        return LLMResponse(
            value=value,
            raw_text=raw_text,
            provider=self.provider,
            model=getattr(completion, "model", settings.model) or settings.model,
            usage=_extract_usage(completion),
            cache_hit=False,
            finish_reason=finish_reason,
        )


def _extract_message(completion: Any) -> tuple[str, Optional[str]]:
    """Pull the assistant text and finish reason out of a completion object."""

    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise LLMResponseValidationError("The provider returned no choices.", None)
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    finish_reason = getattr(choice, "finish_reason", None)
    if content is None:
        raise LLMResponseValidationError(
            f"The provider returned no message content (finish_reason={finish_reason}).",
            None,
        )
    return content, finish_reason


def _extract_usage(completion: Any) -> LLMUsage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
