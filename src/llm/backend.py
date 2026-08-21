"""small LLM client for provider calls"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Protocol, Sequence, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

from llm.cache import CachingLLMClient, LLMCache
from llm.errors import LLMConfigurationError
from llm.models import (
    OPENAI_COMPATIBLE,
    LLMMessage,
    LLMResponse,
    LLMSettings,
    ModelSettings,
)
from llm.structured_output import messages_with_output_contract

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """handle structured llm calls"""

    @property
    def provider(self) -> str:
        """provide adapter identifier (e.g. for reports)"""
        ...

    def complete(
        self,
        *,
        messages: Sequence[LLMMessage],
        response_model: Type[TModel],
        settings: ModelSettings,
        feature: str = "llm",
        context: Any = None,
    ) -> LLMResponse[TModel]:
        """sending messages and return (validated) response models

        :param messages: plain message records; no provider or LangChain types.
        :param response_model: Pydantic model every response is validated against.
        :param settings: model, sampling parameters, timeout, and output mode.
        :param feature: caller identity, used for cache partitioning and logs.
        :param context: extra evidence that shaped the prompt
        """
        ...


class StructuredOutputClient:
    """applying the requested output"""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner

    @property
    def provider(self) -> str:
        return getattr(self.inner, "provider", "unknown")

    def complete(
        self,
        *,
        messages: Sequence[LLMMessage],
        response_model: Type[TModel],
        settings: ModelSettings,
        feature: str = "llm",
        context: Any = None,
    ) -> LLMResponse[TModel]:
        return self.inner.complete(
            messages=messages_with_output_contract(
                messages, response_model, settings.structured_output
            ),
            response_model=response_model,
            settings=settings,
            feature=feature,
            context=context,
        )


def build_client(
    settings: LLMSettings,
    *,
    env: Optional[Mapping[str, str]] = None,
    cache: Optional[LLMCache] = None,
) -> LLMClient:
    """construct a configured adapter, checks provider availability"""

    if settings.provider != OPENAI_COMPATIBLE:
        raise LLMConfigurationError(
            f"No adapter is registered for provider '{settings.provider}'."
        )

    # keep openai endpoint optional
    from llm.providers.openai_compat import OpenAICompatibleClient

    client: LLMClient = OpenAICompatibleClient(settings, env=env)

    if settings.cache_enabled:
        if cache is None:
            if settings.cache_dir is None:
                raise LLMConfigurationError(
                    "LLM caching is enabled but no cache directory could be resolved."
                )
            cache = LLMCache(settings.cache_dir)
        client = CachingLLMClient(client, cache, settings.provider_fingerprint())
    else:
        logger.info("LLM response cache disabled by configuration.")

    return StructuredOutputClient(client)
