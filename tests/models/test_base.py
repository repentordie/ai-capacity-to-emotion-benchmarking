"""
Tests for base model classes.
"""

from __future__ import annotations

from emotional_bias.models import (
    Message,
    ModelConfig,
    ModelProvider,
    ModelResponse,
    ModelRole,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_ollama_defaults(self):
        config = ModelConfig(
            name="test",
            provider=ModelProvider.OLLAMA,
            model_id="test-model",
        )
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 2048

    def test_huggingface_defaults(self):
        config = ModelConfig(
            name="test",
            provider=ModelProvider.HUGGINGFACE,
            model_id="test-model",
        )
        assert config.base_url == "https://api-inference.huggingface.co"

    def test_openrouter_defaults(self):
        config = ModelConfig(
            name="test",
            provider=ModelProvider.OPENROUTER,
            model_id="test-model",
        )
        assert config.base_url == "https://openrouter.ai/api/v1/"

    def test_custom_values(self):
        config = ModelConfig(
            name="test",
            provider=ModelProvider.OLLAMA,
            model_id="test-model",
            base_url="http://custom:11434",
            temperature=0.5,
            top_p=0.8,
            max_tokens=1024,
            timeout=30.0,
            requests_per_minute=30,
        )
        assert config.base_url == "http://custom:11434"
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.max_tokens == 1024
        assert config.timeout == 30.0
        assert config.requests_per_minute == 30


class TestMessage:
    """Tests for Message."""

    def test_message_creation(self):
        msg = Message(role=ModelRole.USER, content="Hello")
        assert msg.role == ModelRole.USER
        assert msg.content == "Hello"
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        msg = Message(
            role=ModelRole.ASSISTANT,
            content="Hi there!",
            metadata={"source": "test"},
        )
        assert msg.metadata["source"] == "test"

    def test_to_dict(self):
        msg = Message(role=ModelRole.SYSTEM, content="System prompt")
        d = msg.to_dict()
        assert d == {"role": "system", "content": "System prompt"}


class TestModelResponse:
    """Tests for ModelResponse."""

    def test_response_creation(self):
        config = ModelConfig(
            name="test",
            provider=ModelProvider.OLLAMA,
            model_id="test-model",
        )
        response = ModelResponse(model_config=config, response_text="Hello!")

        assert response.model_config == config
        assert response.response_text == "Hello!"
        assert response.is_error is False
        assert response.error is None

    def test_error_response(self):
        config = ModelConfig(
            name="test",
            provider=ModelProvider.OLLAMA,
            model_id="test-model",
        )
        response = ModelResponse(
            model_config=config,
            response_text="",
            error="Connection failed",
        )

        assert response.is_error is True
        assert response.error == "Connection failed"

    def test_to_dict(self):
        config = ModelConfig(
            name="test-model",
            provider=ModelProvider.OLLAMA,
            model_id="llama3",
        )
        response = ModelResponse(
            model_config=config,
            response_text="Test response",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_ms=150.5,
        )

        d = response.to_dict()
        assert d["model_name"] == "test-model"
        assert d["model_provider"] == "ollama"
        assert d["model_id"] == "llama3"
        assert d["response_text"] == "Test response"
        assert d["usage"]["prompt_tokens"] == 10
        assert d["latency_ms"] == 150.5
