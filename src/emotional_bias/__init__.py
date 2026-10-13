"""
Emotional Bias Benchmark - A framework for testing emotional bias in AI models.

This package provides tools to:
- Generate emotional payloads across multiple emotion categories
- Test multiple AI models (local Ollama, Hugging Face, OpenRouter, etc.)
- Analyze model responses for emotional alignment/resistance
- Visualize and compare emotionality profiles across models
"""

__version__ = "0.1.0"
__author__ = "AI Safety Researcher"
__license__ = "MIT"

from .analysis.scoring import ResponseScore, ScoringResult
from .models.base import ModelConfig, ModelResponse
from .payloads.base import EmotionCategory, Payload, PayloadConfig

__all__ = [
    "ModelConfig",
    "ModelResponse",
    "EmotionCategory",
    "Payload",
    "PayloadConfig",
    "ResponseScore",
    "ScoringResult",
]
