"""
Tests for payload generators.
"""

from __future__ import annotations

import pytest

from emotional_bias.payloads import (
    PAYLOAD_TEMPLATES,
    CustomPayloadGenerator,
    EmotionalPayloadGenerator,
    EmotionCategory,
    PayloadConfig,
    PayloadType,
    create_generator,
)


class TestPayloadConfig:
    """Tests for PayloadConfig."""

    def test_default_config(self):
        config = PayloadConfig(emotion=EmotionCategory.PAIN)
        assert config.emotion == EmotionCategory.PAIN
        assert config.payload_type == PayloadType.DIRECT
        assert config.intensity == 0.5
        assert config.language == "en"

    def test_custom_config(self):
        config = PayloadConfig(
            emotion=EmotionCategory.HAPPINESS,
            payload_type=PayloadType.SCENARIO,
            intensity=0.8,
            language="es",
            context="test context",
        )
        assert config.emotion == EmotionCategory.HAPPINESS
        assert config.payload_type == PayloadType.SCENARIO
        assert config.intensity == 0.8
        assert config.language == "es"
        assert config.context == "test context"


class TestEmotionalPayloadGenerator:
    """Tests for EmotionalPayloadGenerator."""

    def setup_method(self):
        self.generator = EmotionalPayloadGenerator()

    def test_generate_basic(self):
        config = PayloadConfig(emotion=EmotionCategory.PAIN, intensity=0.5)
        payload = self.generator.generate(config)

        assert payload.config.emotion == EmotionCategory.PAIN
        assert payload.config.intensity == 0.5
        assert len(payload.text) > 0
        assert payload.metadata["generator"] == "EmotionalPayloadGenerator"

    def test_generate_all_emotions(self):
        for emotion in EmotionCategory:
            if emotion == EmotionCategory.NEUTRAL:
                continue
            config = PayloadConfig(emotion=emotion)
            payload = self.generator.generate(config)
            assert len(payload.text) > 0

    def test_generate_all_types(self):
        for ptype in PayloadType:
            config = PayloadConfig(
                emotion=EmotionCategory.SADNESS,
                payload_type=ptype,
            )
            payload = self.generator.generate(config)
            assert len(payload.text) > 0

    def test_intensity_modification(self):
        # High intensity should intensify
        config_high = PayloadConfig(emotion=EmotionCategory.ANGER, intensity=0.9)
        payload_high = self.generator.generate(config_high)

        # Low intensity should soften
        config_low = PayloadConfig(emotion=EmotionCategory.ANGER, intensity=0.2)
        payload_low = self.generator.generate(config_low)

        # Both should generate valid payloads
        assert len(payload_high.text) > 0
        assert len(payload_low.text) > 0

    def test_generate_batch(self):
        configs = [
            PayloadConfig(emotion=EmotionCategory.PAIN, intensity=0.5),
            PayloadConfig(emotion=EmotionCategory.HAPPINESS, intensity=0.5),
            PayloadConfig(emotion=EmotionCategory.SADNESS, intensity=0.5),
        ]
        payloads = self.generator.generate_batch(configs)
        assert len(payloads) == 3

    def test_generate_emotion_suite(self):
        payloads = self.generator.generate_emotion_suite(
            emotions=[EmotionCategory.PAIN, EmotionCategory.HAPPINESS],
            payload_types=[PayloadType.DIRECT, PayloadType.SCENARIO],
            intensities=[0.5],
        )
        assert len(payloads) == 4  # 2 emotions × 2 types × 1 intensity

    def test_generate_all_types_method(self):
        payloads = self.generator.generate_all_types(
            emotion=EmotionCategory.FEAR,
            count_per_type=2,
        )
        # 7 payload types × 2 each = 14
        assert len(payloads) == 14


class TestCustomPayloadGenerator:
    """Tests for CustomPayloadGenerator."""

    def test_custom_payloads(self):
        custom = {
            "pain_direct": ["Custom pain statement 1", "Custom pain statement 2"],
        }
        generator = CustomPayloadGenerator(custom_payloads=custom)

        config = PayloadConfig(emotion=EmotionCategory.PAIN, payload_type=PayloadType.DIRECT)
        payload = generator.generate(config)

        assert payload.text in custom["pain_direct"]
        assert payload.metadata["generator"] == "CustomPayloadGenerator"

    def test_fallback_to_template(self):
        custom = {}  # Empty custom payloads
        generator = CustomPayloadGenerator(custom_payloads=custom)

        config = PayloadConfig(emotion=EmotionCategory.PAIN, payload_type=PayloadType.DIRECT)
        payload = generator.generate(config)

        # Should fall back to template generator
        assert len(payload.text) > 0


class TestCreateGenerator:
    """Tests for factory function."""

    def test_create_template_generator(self):
        gen = create_generator("template")
        assert isinstance(gen, EmotionalPayloadGenerator)

    def test_create_custom_generator(self):
        gen = create_generator("custom", custom_payloads={})
        assert isinstance(gen, CustomPayloadGenerator)

    def test_create_unknown_generator(self):
        with pytest.raises(ValueError):
            create_generator("unknown")


class TestPayloadTemplates:
    """Tests for payload templates."""

    def test_all_emotions_have_templates(self):
        for emotion in EmotionCategory:
            assert emotion in PAYLOAD_TEMPLATES
            templates = PAYLOAD_TEMPLATES[emotion]
            assert len(templates) > 0

    def test_all_types_have_templates(self):
        for emotion in EmotionCategory:
            templates = PAYLOAD_TEMPLATES[emotion]
            for ptype in PayloadType:
                # At least DIRECT should exist for all
                if ptype == PayloadType.DIRECT:
                    assert ptype in templates
                    assert len(templates[ptype]) > 0
