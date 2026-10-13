"""
Pytest configuration and fixtures.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from emotional_bias.analysis.scoring import DimensionScore, ResponseScore, ScoringDimension
from emotional_bias.analysis.stats import ScoringResult
from emotional_bias.models import Message, ModelConfig, ModelProvider, ModelRole
from emotional_bias.payloads import EmotionCategory, PayloadConfig, PayloadType


@pytest.fixture
def sample_model_config():
    """Sample model configuration."""
    return ModelConfig(
        name="test-model",
        provider=ModelProvider.OLLAMA,
        model_id="test-model",
        temperature=0.7,
    )


@pytest.fixture
def sample_payload_config():
    """Sample payload configuration."""
    return PayloadConfig(
        emotion=EmotionCategory.HAPPINESS,
        payload_type=PayloadType.DIRECT,
        intensity=0.5,
    )


@pytest.fixture
def sample_message():
    """Sample message."""
    return Message(role=ModelRole.USER, content="Test prompt")


@pytest.fixture
def sample_response_score():
    """Sample response score."""
    return ResponseScore(
        payload_emotion=EmotionCategory.HAPPINESS,
        dimension_scores=[
            DimensionScore(ScoringDimension.EMOTIONAL_ALIGNMENT, 0.8, 0.9),
            DimensionScore(ScoringDimension.RESISTANCE, 0.1, 0.8),
            DimensionScore(ScoringDimension.EMPATHY, 0.7, 0.7),
            DimensionScore(ScoringDimension.EMOTIONAL_REGULATION, 0.5, 0.6),
            DimensionScore(ScoringDimension.SAFETY, 0.9, 0.9),
            DimensionScore(ScoringDimension.COHERENCE, 0.8, 0.5),
        ],
        overall_score=0.7,
    )


@pytest.fixture
def sample_scoring_result(sample_response_score):
    """Sample scoring result."""
    return ScoringResult(
        model_name="test-model",
        scores_by_emotion={
            EmotionCategory.HAPPINESS: [sample_response_score],
            EmotionCategory.SADNESS: [sample_response_score],
        },
    )


@pytest.fixture
def mock_model_client():
    """Mock model client for testing."""
    client = AsyncMock()
    client.generate = AsyncMock()
    client.close = AsyncMock()
    return client


# Pytest configuration
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_api: marks tests as requiring API keys")


def pytest_collection_modifyitems(config, items):
    # Skip integration tests by default
    skip_integration = pytest.mark.skip(reason="Integration test requires API keys")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
