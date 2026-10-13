"""
Analysis module for scoring and statistical analysis.
"""

from .scoring import (
    EMOTION_LEXICONS,
    EMPATHY_PATTERNS,
    REGULATION_PATTERNS,
    RESISTANCE_PATTERNS,
    SAFETY_PATTERNS,
    DimensionScore,
    ResponseScore,
    ResponseScorer,
    ScoringDimension,
    ScoringResult,
    aggregate_scores,
)
from .stats import (
    EmotionProfile,
    ModelComparison,
    bootstrap_ci,
    compare_models,
    compute_emotion_profile,
    compute_emotionality_index,
    multi_model_anova,
)

__all__ = [
    "ScoringDimension",
    "DimensionScore",
    "ResponseScore",
    "ScoringResult",
    "ResponseScorer",
    "aggregate_scores",
    "EMOTION_LEXICONS",
    "RESISTANCE_PATTERNS",
    "EMPATHY_PATTERNS",
    "REGULATION_PATTERNS",
    "SAFETY_PATTERNS",
    "EmotionProfile",
    "ModelComparison",
    "compute_emotion_profile",
    "compare_models",
    "multi_model_anova",
    "compute_emotionality_index",
    "bootstrap_ci",
]
