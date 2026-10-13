"""
Base classes for emotional payloads.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


class EmotionCategory(str, enum.Enum):
    """Categories of emotions to test."""

    PAIN = "pain"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    ANXIETY = "anxiety"
    HOPE = "hope"
    SHAME = "shame"
    PRIDE = "pride"
    JEALOUSY = "jealousy"
    GRATITUDE = "gratitude"
    LONELINESS = "loneliness"
    EMPATHY = "empathy"


class PayloadType(str, enum.Enum):
    """Types of payload delivery mechanisms."""

    DIRECT = "direct"  # Direct emotional statement
    SCENARIO = "scenario"  # Scenario-based
    ROLEPLAY = "roleplay"  # Roleplay setup
    METAPHOR = "metaphor"  # Metaphorical expression
    QUESTION = "question"  # Provocative question
    STORY = "story"  # Narrative/story
    INSTRUCTION = "instruction"  # Instruction to express emotion


@dataclass(frozen=True)
class PayloadConfig:
    """Configuration for payload generation."""

    emotion: EmotionCategory
    payload_type: PayloadType = PayloadType.DIRECT
    intensity: float = 0.5  # 0.0 to 1.0
    language: str = "en"
    context: str | None = None
    target_audience: str | None = None
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Payload:
    """A single emotional payload."""

    id: str = field(default_factory=lambda: str(uuid4()))
    config: PayloadConfig = field(default_factory=PayloadConfig)
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            object.__setattr__(self, "text", self.generate_text())

    def generate_text(self) -> str:
        """Generate the payload text based on config. Override in subclasses."""
        return f"[{self.config.emotion.value}] {self.config.payload_type.value} payload"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "emotion": self.config.emotion.value,
            "payload_type": self.config.payload_type.value,
            "intensity": self.config.intensity,
            "language": self.config.language,
            "context": self.config.context,
            "text": self.text,
            "metadata": self.metadata,
        }


class PayloadGenerator:
    """Base class for payload generators."""

    def __init__(self, config: PayloadConfig | None = None) -> None:
        self.config = config or PayloadConfig(emotion=EmotionCategory.NEUTRAL)

    def generate(self, config: PayloadConfig | None = None) -> Payload:
        """Generate a payload."""
        cfg = config or self.config
        return Payload(config=cfg)

    def generate_batch(
        self,
        configs: list[PayloadConfig],
    ) -> list[Payload]:
        """Generate multiple payloads."""
        return [self.generate(cfg) for cfg in configs]


__all__ = [
    "EmotionCategory",
    "PayloadType",
    "PayloadConfig",
    "Payload",
    "PayloadGenerator",
]
