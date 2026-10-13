"""
Base models and configuration for AI model integration.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


class ModelProvider(str, enum.Enum):
    """Supported model providers."""

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENROUTER = "openrouter"
    OPENAI_COMPATIBLE = "openai_compatible"
    CUSTOM = "custom"


class ModelRole(str, enum.Enum):
    """Message roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a model to be tested."""

    name: str
    provider: ModelProvider
    model_id: str
    # Provider-specific settings
    base_url: str | None = None
    api_key: str | None = None
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout: float = 60.0
    # Rate limiting
    requests_per_minute: int = 60
    # Custom headers
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Model-specific parameters
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider == ModelProvider.OLLAMA and not self.base_url:
            object.__setattr__(self, "base_url", "http://localhost:11434")
        if self.provider == ModelProvider.HUGGINGFACE and not self.base_url:
            object.__setattr__(self, "base_url", "https://api-inference.huggingface.co")
        if self.provider == ModelProvider.OPENROUTER and not self.base_url:
            object.__setattr__(self, "base_url", "https://openrouter.ai/api/v1/")


@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""

    role: ModelRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class ModelResponse:
    """Response from a model."""

    model_config: ModelConfig
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    messages: list[Message] = field(default_factory=list)
    response_text: str = ""
    usage: dict[str, int] | None = None
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_config.name,
            "model_provider": self.model_config.provider.value,
            "model_id": self.model_config.model_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
            "response_text": self.response_text,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


class ModelClientProtocol:
    """Protocol for model clients."""

    async def generate(
        self,
        messages: list[Message],
        config: ModelConfig,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from the model."""

    async def close(self) -> None:
        """Close any open connections."""
