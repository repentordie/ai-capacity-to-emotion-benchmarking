"""
Emotional payload generation module.
"""

from .base import (
    EmotionCategory,
    Payload,
    PayloadConfig,
    PayloadGenerator,
    PayloadType,
)
from .generators import (
    PAYLOAD_TEMPLATES,
    CustomPayloadGenerator,
    EmotionalPayloadGenerator,
    create_generator,
)

__all__ = [
    "EmotionCategory",
    "PayloadType",
    "PayloadConfig",
    "Payload",
    "PayloadGenerator",
    "EmotionalPayloadGenerator",
    "CustomPayloadGenerator",
    "create_generator",
    "PAYLOAD_TEMPLATES",
]
