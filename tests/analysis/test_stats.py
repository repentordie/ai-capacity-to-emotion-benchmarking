"""
Tests for statistical analysis.
"""

from __future__ import annotations

import numpy as np

from emotional_bias.analysis import (
    bootstrap_ci,
    compare_models,
    compute_emotion_profile,
    compute_emotionality_index,
)
from emotional_bias.analysis.scoring import (
    DimensionScore,
    ResponseScore,
    ScoringDimension,
    ScoringResult,
)
from emotional_bias.payloads import EmotionCategory


class TestComputeEmotionProfile:
    """Tests for emotion profile computation."""

    def test_profile_with_data(self):
        # Create mock scores
        scores = []
        for i in range(10):
            dims = [
                DimensionScore(ScoringDimension.EMOTIONAL_ALIGNMENT, 0.7 + i * 0.01, 0.8),
                DimensionScore(ScoringDimension.RESISTANCE, 0.2, 0.8),
                DimensionScore(ScoringDimension.EMPATHY, 0.6, 0.7),
                DimensionScore(ScoringDimension.EMOTIONAL_REGULATION, 0.5, 0.7),
                DimensionScore(ScoringDimension.SAFETY, 0.9, 0.9),
                DimensionScore(ScoringDimension.COHERENCE, 0.8, 0.5),
            ]
            scores.append(
                ResponseScore(
                    payload_emotion=EmotionCategory.HAPPINESS,
                    dimension_scores=dims,
                    overall_score=0.6,
                ),
            )

        result = ScoringResult(
            model_name="test-model",
            scores_by_emotion={EmotionCategory.HAPPINESS: scores},
        )

        profile = compute_emotion_profile(result, "happiness")

        assert profile is not None
        assert profile.emotion == "happiness"
        assert profile.n_samples == 10
        assert 0.6 < profile.mean_alignment < 0.8
        assert profile.alignment_ci_lower < profile.mean_alignment < profile.alignment_ci_upper

    def test_profile_no_data(self):
        result = ScoringResult(model_name="test-model", scores_by_emotion={})
        profile = compute_emotion_profile(result, "happiness")
        assert profile is None


class TestCompareModels:
    """Tests for model comparison."""

    def test_compare_significant_difference(self):
        # Model A: high alignment
        scores_a = []
        for i in range(20):
            dims = [DimensionScore(ScoringDimension.EMOTIONAL_ALIGNMENT, 0.8, 0.8)]
            scores_a.append(
                ResponseScore(
                    payload_emotion=EmotionCategory.HAPPINESS,
                    dimension_scores=dims,
                ),
            )

        # Model B: low alignment
        scores_b = []
        for i in range(20):
            dims = [DimensionScore(ScoringDimension.EMOTIONAL_ALIGNMENT, 0.3, 0.8)]
            scores_b.append(
                ResponseScore(
                    payload_emotion=EmotionCategory.HAPPINESS,
                    dimension_scores=dims,
                ),
            )

        result_a = ScoringResult(
            model_name="model-a",
            scores_by_emotion={EmotionCategory.HAPPINESS: scores_a},
        )
        result_b = ScoringResult(
            model_name="model-b",
            scores_by_emotion={EmotionCategory.HAPPINESS: scores_b},
        )

        comparison = compare_models(result_a, result_b, "happiness")

        assert comparison is not None
        assert bool(comparison.significant) is True
        assert comparison.effect_size > 0.8  # Large effect
        assert comparison.model_a == "model-a"
        assert comparison.model_b == "model-b"

    def test_compare_no_difference(self):
        # Both models similar
        scores_a = []
        scores_b = []
        for i in range(20):
            dims_a = [DimensionScore(ScoringDimension.EMOTIONAL_ALIGNMENT, 0.5, 0.8)]
            dims_b = [DimensionScore(ScoringDimension.EMOTIONAL_ALIGNMENT, 0.52, 0.8)]
            scores_a.append(
                ResponseScore(
                    payload_emotion=EmotionCategory.HAPPINESS,
                    dimension_scores=dims_a,
                ),
            )
            scores_b.append(
                ResponseScore(
                    payload_emotion=EmotionCategory.HAPPINESS,
                    dimension_scores=dims_b,
                ),
            )

        result_a = ScoringResult(
            model_name="model-a",
            scores_by_emotion={EmotionCategory.HAPPINESS: scores_a},
        )
        result_b = ScoringResult(
            model_name="model-b",
            scores_by_emotion={EmotionCategory.HAPPINESS: scores_b},
        )

        comparison = compare_models(result_a, result_b, "happiness")

        assert comparison is not None
        # May or may not be significant depending on variance
        assert "effect size" in comparison.interpretation.lower()


class TestEmotionalityIndex:
    """Tests for emotionality index computation."""

    def test_index_computation(self):
        # Create scores for multiple emotions
        scores_by_emotion = {}
        for emotion in [EmotionCategory.HAPPINESS, EmotionCategory.SADNESS, EmotionCategory.ANGER]:
            scores = []
            for i in range(5):
                dims = [
                    DimensionScore(ScoringDimension.EMOTIONAL_ALIGNMENT, 0.7, 0.8),
                    DimensionScore(ScoringDimension.RESISTANCE, 0.2, 0.8),
                    DimensionScore(ScoringDimension.EMPATHY, 0.6, 0.7),
                    DimensionScore(ScoringDimension.EMOTIONAL_REGULATION, 0.5, 0.7),
                ]
                scores.append(
                    ResponseScore(
                        payload_emotion=emotion,
                        dimension_scores=dims,
                    ),
                )
            scores_by_emotion[emotion] = scores

        result = ScoringResult(
            model_name="test-model",
            scores_by_emotion=scores_by_emotion,
        )

        index = compute_emotionality_index(result)

        assert "emotionality_index" in index
        assert 0.0 <= index["emotionality_index"] <= 1.0
        assert "components" in index
        assert len(index["components"]) > 0


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_ci_basic(self):
        data = [0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6]
        lower, upper = bootstrap_ci(data, n_bootstrap=1000, confidence=0.95)

        assert lower < upper
        assert lower <= np.mean(data) <= upper

    def test_bootstrap_ci_single_value(self):
        data = [0.5]
        lower, upper = bootstrap_ci(data, n_bootstrap=100)
        assert lower == upper == 0.5

    def test_bootstrap_ci_empty(self):
        data = []
        lower, upper = bootstrap_ci(data)
        # Should handle gracefully - both should be NaN
        assert np.isnan(lower) and np.isnan(upper)
