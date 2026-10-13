"""
Model client factory for creating appropriate clients based on provider.
"""

from __future__ import annotations

from .base import ModelClientProtocol, ModelConfig, ModelProvider
from .huggingface import HuggingFaceClient
from .ollama import OllamaClient
from .openrouter import OpenRouterClient


def create_client(config: ModelConfig) -> ModelClientProtocol:
    """Create a model client based on the provider."""
    match config.provider:
        case ModelProvider.OLLAMA:
            return OllamaClient(config)
        case ModelProvider.HUGGINGFACE:
            return HuggingFaceClient(config)
        case ModelProvider.OPENROUTER:
            return OpenRouterClient(config)
        case ModelProvider.OPENAI_COMPATIBLE:
            # OpenRouter client works for any OpenAI-compatible API
            return OpenRouterClient(config)
        case _:
            raise ValueError(f"Unsupported model provider: {config.provider}")


def create_clients(configs: list[ModelConfig]) -> dict[str, ModelClientProtocol]:
    """Create multiple clients, keyed by model name."""
    return {config.name: create_client(config) for config in configs}


__all__ = ["create_client", "create_clients"]
