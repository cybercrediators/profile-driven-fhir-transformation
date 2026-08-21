"""Error handling for llm layer"""

from __future__ import annotations

from typing import Any, Optional


class LLMError(Exception):
    """base error class"""


class LLMDependencyError(LLMError):
    """An optional package for one of the LLM-backed features is not installed."""

    def __init__(
        self, package: str, extra: str = "llm", feature: str = "LLM mode"
    ) -> None:
        super().__init__(
            f"The optional package '{package}' is required for {feature}. "
            f"Install it with: pip install -e '.[{extra}]'"
        )
        self.package = package
        self.extra = extra
        self.feature = feature


class LLMConfigurationError(LLMError):
    """Provider settings are missing, incomplete, or mutually inconsistent."""


class LLMCredentialsError(LLMConfigurationError):
    """No API key was available from the configured environment variable."""

    def __init__(self, env_var: str) -> None:
        super().__init__(
            f"No LLM API key found. Set the environment variable '{env_var}'. "
            "API keys are never read from the project configuration file."
        )
        self.env_var = env_var


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMProviderError(LLMError):
    """The provider rejected the request or returned a transport-level error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseValidationError(LLMError):
    """The response was not parseable, or did not satisfy the output schema.

    This is a normal, expected outcome: model output is an untrusted proposal.
    Callers are expected to reject the attempt, not to repair the payload.
    """

    def __init__(self, message: str, raw_text: Optional[str] = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text


class LLMSchemaError(LLMConfigurationError):
    """An output model cannot be expressed in the requested schema mode."""

    def __init__(self, model_name: str, reason: str, detail: Any = None) -> None:
        super().__init__(
            f"Output model '{model_name}' cannot be used in the requested "
            f"structured-output mode: {reason}"
        )
        self.model_name = model_name
        self.detail = detail
