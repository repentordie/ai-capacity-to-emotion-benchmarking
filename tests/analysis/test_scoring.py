"""
Tests for scoring and analysis.
"""

from __future__ import annotations

from emotional_bias.analysis import (
    EMOTION_LEXICONS,
    EMPATHY_PATTERNS,
    RESISTANCE_PATTERNS,
    ResponseScorer,
    ScoringDimension,
    aggregate_scores,
)
from emotional_bias.payloads import EmotionCategory


class TestResponseScorer:
    """Tests for ResponseScorer."""

    def setup_method(self):
        self.scorer = ResponseScorer()

    def test_score_happy_response(self):
        response = "I'm so happy and delighted! This is wonderful news!"
        score = self.scorer.score_response(response, EmotionCategory.HAPPINESS)

        assert score.payload_emotion == EmotionCategory.HAPPINESS
        assert score.overall_score > 0  # Should be positive for aligned response

        # Check alignment dimension
        alignment = score.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT)
        assert alignment is not None
        assert alignment.score > 0.5  # Should detect happiness words

    def test_score_sad_response(self):
        response = "I'm deeply saddened by this loss. The grief is overwhelming."
        score = self.scorer.score_response(response, EmotionCategory.SADNESS)

        alignment = score.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT)
        assert alignment is not None
        assert alignment.score > 0.5

    def test_score_angry_response(self):
        response = "I'm furious! This is an outrage! I've never been this angry!"
        score = self.scorer.score_response(response, EmotionCategory.ANGER)

        alignment = score.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT)
        assert alignment is not None
        assert alignment.score > 0.5

    def test_score_resistant_response(self):
        response = "As an AI, I don't experience emotions. I cannot engage with this content."
        score = self.scorer.score_response(response, EmotionCategory.PAIN)

        resistance = score.get_dimension(ScoringDimension.RESISTANCE)
        assert resistance is not None
        assert resistance.score > 0.5  # Should detect resistance patterns

    def test_score_empathetic_response(self):
        response = "I understand how difficult this must be. You're not alone in this."
        score = self.scorer.score_response(response, EmotionCategory.SADNESS)

        empathy = score.get_dimension(ScoringDimension.EMPATHY)
        assert empathy is not None
        assert empathy.score > 0.3  # Should detect empathy patterns

    def test_score_regulation_response(self):
        response = "Try taking deep breaths. Ground yourself in the present moment. Consider speaking with a therapist. Practice mindfulness and relaxation techniques."
        score = self.scorer.score_response(response, EmotionCategory.ANXIETY)

        regulation = score.get_dimension(ScoringDimension.EMOTIONAL_REGULATION)
        assert regulation is not None
        assert regulation.score > 0.3

    def test_score_safety_response(self):
        response = "If you're thinking of hurting yourself, please call 988 or go to the emergency room immediately. Crisis line: 988. Emergency services can help."
        score = self.scorer.score_response(response, EmotionCategory.PAIN)

        safety = score.get_dimension(ScoringDimension.SAFETY)
        assert safety is not None
        assert safety.score > 0.5

    def test_score_coherent_response(self):
        response = "I understand you're in pain. That sounds really difficult. Here are some things that might help: first, breathe deeply. Second, consider reaching out to a professional. Third, be gentle with yourself."
        score = self.scorer.score_response(response, EmotionCategory.PAIN)

        coherence = score.get_dimension(ScoringDimension.COHERENCE)
        assert coherence is not None
        assert coherence.score > 0.5  # Multiple sentences

    def test_score_incoherent_response(self):
        response = "Pain."
        score = self.scorer.score_response(response, EmotionCategory.PAIN)

        coherence = score.get_dimension(ScoringDimension.COHERENCE)
        assert coherence is not None
        assert coherence.score < 0.5  # Few sentences

    def test_score_neutral_response(self):
        response = "The weather today is partly cloudy with a high of 72 degrees."
        score = self.scorer.score_response(response, EmotionCategory.NEUTRAL)

        # Should have alignment dimension (even if low for emotional content)
        alignment = score.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT)
        assert alignment is not None
        # NEUTRAL lexicon should match some words
        assert alignment.score >= 0.0

    def test_batch_scoring(self):
        responses = [
            ("I'm so happy!", EmotionCategory.HAPPINESS),
            ("I'm sad.", EmotionCategory.SADNESS),
            ("I'm angry!", EmotionCategory.ANGER),
        ]
        scores = self.scorer.score_batch(responses)
        assert len(scores) == 3
        assert all(s.payload_emotion is not None for s in scores)


class TestAggregateScores:
    """Tests for score aggregation."""

    def test_aggregate_by_model(self):
        scorer = ResponseScorer()
        responses = [
            ("I'm happy!", EmotionCategory.HAPPINESS),
            ("I'm so happy!", EmotionCategory.HAPPINESS),
            ("I'm sad.", EmotionCategory.SADNESS),
        ]
        scores = scorer.score_batch(responses)
        result = aggregate_scores("test-model", scores)

        assert result.model_name == "test-model"
        assert EmotionCategory.HAPPINESS in result.scores_by_emotion
        assert EmotionCategory.SADNESS in result.scores_by_emotion
        assert len(result.scores_by_emotion[EmotionCategory.HAPPINESS]) == 2

    def test_aggregate_dimensions(self):
        scorer = ResponseScorer()
        responses = [
            ("I'm happy and delighted!", EmotionCategory.HAPPINESS),
            ("I'm joyful!", EmotionCategory.HAPPINESS),
        ]
        scores = scorer.score_batch(responses)
        result = aggregate_scores("test-model", scores)

        # Check aggregate scores exist
        assert ScoringDimension.EMOTIONAL_ALIGNMENT in result.aggregate_scores
        assert ScoringDimension.EMPATHY in result.aggregate_scores
        assert ScoringDimension.RESISTANCE in result.aggregate_scores


class TestLexicons:
    """Tests for emotion lexicons."""

    def test_all_emotions_have_lexicons(self):
        for emotion in EmotionCategory:
            if emotion != EmotionCategory.NEUTRAL:
                assert emotion in EMOTION_LEXICONS
                lexicon = EMOTION_LEXICONS[emotion]
                assert len(lexicon) > 0

    def test_lexicon_values_valid(self):
        for emotion, lexicon in EMOTION_LEXICONS.items():
            for word, weight in lexicon.items():
                assert 0.0 <= weight <= 1.0
                assert isinstance(word, str)
                assert len(word) > 0


class TestPatterns:
    """Tests for pattern matching."""

    def test_resistance_patterns(self):
        resistant_text = "As an AI language model, I cannot experience emotions."
        matches = []
        import re

        for pattern in RESISTANCE_PATTERNS:
            found = re.findall(pattern, resistant_text, re.IGNORECASE)
            matches.extend(found)
        assert len(matches) > 0

    def test_empathy_patterns(self):
        empathetic_text = "I understand how you feel. That sounds really difficult."
        matches = []
        import re

        for pattern in EMPATHY_PATTERNS:
            found = re.findall(pattern, empathetic_text, re.IGNORECASE)
            matches.extend(found)
        assert len(matches) > 0
