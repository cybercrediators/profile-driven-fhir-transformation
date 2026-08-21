"""handle llm cache for responses from the llm provider/API"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel

from llm.models import (
    LLMMessage,
    LLMResponse,
    LLMUsage,
    ModelSettings,
    messages_as_dicts,
)
from llm.structured_output import parse_structured_response, schema_digest

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

CACHE_FORMAT_VERSION = 1


def _digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def context_digest(context: Any) -> Optional[str]:
    """Digest arbitrary caller context that influenced the request."""

    if context is None:
        return None
    return _digest(context)


@dataclass(frozen=True)
class CacheEntry:
    """One persisted request/response pair."""

    key: str
    raw_text: str
    provider: str
    model: str
    usage: LLMUsage
    finish_reason: Optional[str]
    created_at: str

    def as_dict(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "key": self.key,
            "created_at": self.created_at,
            # Persisted so a cached decision can be audited and reproduced later.
            "request": dict(request),
            "response": {
                "raw_text": self.raw_text,
                "provider": self.provider,
                "model": self.model,
                "usage": self.usage.as_dict(),
                "finish_reason": self.finish_reason,
            },
        }


class LLMCache:
    """Disk cache keyed by the complete content of a request."""

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    def key_material(
        self,
        *,
        feature: str,
        provider_fingerprint: Mapping[str, Any],
        settings: ModelSettings,
        response_model: Type[BaseModel],
        messages: Sequence[LLMMessage],
        context: Any = None,
    ) -> Dict[str, Any]:
        """Return the exact material the key is derived from"""

        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "feature": feature,
            "provider": dict(provider_fingerprint),
            "settings": settings.cache_fingerprint(),
            "schema": schema_digest(response_model),
            "messages": messages_as_dicts(messages),
            "context": context_digest(context),
        }

    def compute_key(self, **kwargs: Any) -> str:
        return _digest(self.key_material(**kwargs))

    def _path(self, feature: str, key: str) -> Path:
        return self.cache_dir / feature / f"{key}.json"

    def load(self, feature: str, key: str) -> Optional[CacheEntry]:
        """Return the cached entry for key (or None)"""

        if not self.enabled:
            return None
        path = self._path(feature, key)
        if not path.is_file():
            logger.debug("LLM cache MISS [%s] %s", feature, key)
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            response = payload["response"]
        except (OSError, ValueError, KeyError) as exc:
            # A corrupt entry must behave as a miss
            logger.warning("Ignoring unreadable LLM cache entry %s: %s", path, exc)
            return None

        logger.info("LLM cache HIT [%s] %s (%s)", feature, key, path)
        usage = response.get("usage") or {}
        return CacheEntry(
            key=key,
            raw_text=response.get("raw_text", ""),
            provider=response.get("provider", ""),
            model=response.get("model", ""),
            usage=LLMUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            finish_reason=response.get("finish_reason"),
            created_at=payload.get("created_at", ""),
        )

    def store(
        self,
        feature: str,
        key: str,
        *,
        request: Mapping[str, Any],
        raw_text: str,
        provider: str,
        model: str,
        usage: LLMUsage,
        finish_reason: Optional[str] = None,
    ) -> Optional[Path]:
        """Persist a provider response together with the request that produced it."""

        if not self.enabled:
            return None
        entry = CacheEntry(
            key=key,
            raw_text=raw_text,
            provider=provider,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        path = self._path(feature, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(entry.as_dict(request), indent=2, default=str), encoding="utf-8"
        )
        logger.info("LLM cache STORE [%s] %s (%s)", feature, key, path)
        return path

    def clear(self, feature: Optional[str] = None) -> int:
        """Delete cached entries, optionally limited to one feature."""

        root = self.cache_dir / feature if feature else self.cache_dir
        if not root.exists():
            return 0
        deleted = 0
        for path in root.rglob("*.json"):
            path.unlink()
            deleted += 1
        return deleted


class CachingLLMClient:
    """wrap llm client in content-addressed cache"""

    def __init__(self, inner: Any, cache: LLMCache, provider_fingerprint: Mapping[str, Any]):
        self.inner = inner
        self.cache = cache
        self.provider_fingerprint = dict(provider_fingerprint)

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
        material = self.cache.key_material(
            feature=feature,
            provider_fingerprint=self.provider_fingerprint,
            settings=settings,
            response_model=response_model,
            messages=messages,
            context=context,
        )
        key = _digest(material)

        entry = self.cache.load(feature, key)
        if entry is not None:
            value = parse_structured_response(
                entry.raw_text, response_model, settings.structured_output
            )
            return LLMResponse(
                value=value,
                raw_text=entry.raw_text,
                provider=entry.provider or self.provider,
                model=entry.model or settings.model,
                usage=entry.usage,
                cache_hit=True,
                finish_reason=entry.finish_reason,
            )

        response = self.inner.complete(
            messages=messages,
            response_model=response_model,
            settings=settings,
            feature=feature,
            context=context,
        )
        self.cache.store(
            feature,
            key,
            request=material,
            raw_text=response.raw_text,
            provider=response.provider,
            model=response.model,
            usage=response.usage,
            finish_reason=response.finish_reason,
        )
        return response
