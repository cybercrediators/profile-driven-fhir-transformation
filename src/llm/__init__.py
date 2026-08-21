"""Optional LLM infrastructure shared by LLM automapping and agent mode.

definition of different llm module-specific errors
"""

from llm.errors import (
    LLMConfigurationError,
    LLMCredentialsError,
    LLMDependencyError,
    LLMError,
    LLMProviderError,
    LLMResponseValidationError,
    LLMSchemaError,
    LLMTimeoutError,
)
from llm.models import (
    LLMMessage,
    LLMResponse,
    LLMSettings,
    LLMUsage,
    ModelSettings,
    Role,
    StructuredOutputMode,
)

__all__ = [
    "LLMConfigurationError",
    "LLMCredentialsError",
    "LLMDependencyError",
    "LLMError",
    "LLMMessage",
    "LLMProviderError",
    "LLMResponse",
    "LLMResponseValidationError",
    "LLMSchemaError",
    "LLMSettings",
    "LLMTimeoutError",
    "LLMUsage",
    "ModelSettings",
    "Role",
    "StructuredOutputMode",
]
