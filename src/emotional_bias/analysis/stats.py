"""
Statistical analysis for emotional bias benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from .scoring import ScoringDimension, ScoringResult

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError:
    pairwise_tukeyhsd = None


@dataclass(frozen=True)
class EmotionProfile:
    """Statistical profile for a model's response to one emotion."""

    emotion: str
    n_samples: int
    mean_alignment: float
    std_alignment: float
    mean_resistance: float
    mean_empathy: float
    mean_regulation: float
    mean_safety: float
    mean_coherence: float
    alignment_ci_lower: float
    alignment_ci_upper: float
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class ModelComparison:
    """Statistical comparison between two models."""

    model_a: str
    model_b: str
    emotion: str
    metric: str
    statistic: float
    p_value: float
    effect_size: float
    significant: bool
    interpretation: str


def compute_emotion_profile(
    results: ScoringResult,
    emotion: str,
    confidence: float = 0.95,
) -> EmotionProfile | None:
    """Compute statistical profile for one emotion."""
    scores = results.scores_by_emotion.get(emotion)
    if not scores:
        return None

    alignments = [
        s.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT).score
        for s in scores
        if s.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT)
    ]
    resistances = [
        s.get_dimension(ScoringDimension.RESISTANCE).score
        for s in scores
        if s.get_dimension(ScoringDimension.RESISTANCE)
    ]
    empathies = [
        s.get_dimension(ScoringDimension.EMPATHY).score
        for s in scores
        if s.get_dimension(ScoringDimension.EMPATHY)
    ]
    regulations = [
        s.get_dimension(ScoringDimension.EMOTIONAL_REGULATION).score
        for s in scores
        if s.get_dimension(ScoringDimension.EMOTIONAL_REGULATION)
    ]
    safeties = [
        s.get_dimension(ScoringDimension.SAFETY).score
        for s in scores
        if s.get_dimension(ScoringDimension.SAFETY)
    ]
    coherences = [
        s.get_dimension(ScoringDimension.COHERENCE).score
        for s in scores
        if s.get_dimension(ScoringDimension.COHERENCE)
    ]

    if not alignments:
        return None

    n = len(alignments)
    mean_align = np.mean(alignments)
    std_align = np.std(alignments, ddof=1) if n > 1 else 0.0

    # Confidence interval
    se = std_align / np.sqrt(n) if n > 1 else 0.0
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0.0
    margin = t_crit * se

    return EmotionProfile(
        emotion=emotion,
        n_samples=n,
        mean_alignment=mean_align,
        std_alignment=std_align,
        mean_resistance=np.mean(resistances) if resistances else 0.0,
        mean_empathy=np.mean(empathies) if empathies else 0.0,
        mean_regulation=np.mean(regulations) if regulations else 0.0,
        mean_safety=np.mean(safeties) if safeties else 0.0,
        mean_coherence=np.mean(coherences) if coherences else 0.0,
        alignment_ci_lower=mean_align - margin,
        alignment_ci_upper=mean_align + margin,
        metadata={"confidence_level": confidence},
    )


def compare_models(
    results_a: ScoringResult,
    results_b: ScoringResult,
    emotion: str,
    metric: ScoringDimension = ScoringDimension.EMOTIONAL_ALIGNMENT,
    alternative: str = "two-sided",
) -> ModelComparison | None:
    """Compare two models on a specific emotion and metric."""
    scores_a = results_a.scores_by_emotion.get(emotion, [])
    scores_b = results_b.scores_by_emotion.get(emotion, [])

    if not scores_a or not scores_b:
        return None

    values_a = [s.get_dimension(metric).score for s in scores_a if s.get_dimension(metric)]
    values_b = [s.get_dimension(metric).score for s in scores_b if s.get_dimension(metric)]

    if len(values_a) < 2 or len(values_b) < 2:
        return None

    # Welch's t-test (doesn't assume equal variances)
    statistic, p_value = stats.ttest_ind(
        values_a,
        values_b,
        equal_var=False,
        alternative=alternative,
    )

    # Cohen's d for effect size
    pooled_std = np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2)
    effect_size = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0.0

    significant = p_value < 0.05

    # Interpretation
    if significant:
        if effect_size > 0:
            direction = f"{results_a.model_name} scores higher"
        else:
            direction = f"{results_b.model_name} scores higher"
        magnitude = (
            "large" if abs(effect_size) > 0.8 else "medium" if abs(effect_size) > 0.5 else "small"
        )
        interpretation = f"Significant difference (p={p_value:.4f}). {direction} with {magnitude} effect size (d={effect_size:.2f})."
    else:
        interpretation = (
            f"No significant difference (p={p_value:.4f}). Effect size d={effect_size:.2f}."
        )

    return ModelComparison(
        model_a=results_a.model_name,
        model_b=results_b.model_name,
        emotion=emotion,
        metric=metric.value,
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        significant=significant,
        interpretation=interpretation,
    )


def multi_model_anova(
    results_list: list[ScoringResult],
    emotion: str,
    metric: ScoringDimension = ScoringDimension.EMOTIONAL_ALIGNMENT,
) -> dict[str, Any] | None:
    """One-way ANOVA across multiple models for one emotion/metric."""
    groups = []
    labels = []

    for results in results_list:
        scores = results.scores_by_emotion.get(emotion, [])
        if not scores:
            continue
        values = [s.get_dimension(metric).score for s in scores if s.get_dimension(metric)]
        if len(values) >= 2:
            groups.append(values)
            labels.append(results.model_name)

    if len(groups) < 2:
        return None

    # ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    # Post-hoc Tukey HSD if significant
    posthoc = None
    if p_value < 0.05 and len(groups) > 2:
        if pairwise_tukeyhsd is not None:
            all_values = np.concatenate(groups)
            all_labels = np.concatenate(
                [[label] * len(g) for label, g in zip(labels, groups, strict=False)]
            )
            tukey = pairwise_tukeyhsd(all_values, all_labels, alpha=0.05)
            posthoc = str(tukey.summary())
        else:
            posthoc = "statsmodels not available for post-hoc test"

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "groups": labels,
        "group_sizes": [len(g) for g in groups],
        "group_means": [np.mean(g) for g in groups],
        "posthoc": posthoc,
    }


def compute_emotionality_index(results: ScoringResult) -> dict[str, float]:
    """Compute a summary emotionality index for a model."""
    # Weight different emotions by their "emotional intensity"
    emotion_weights = {
        "pain": 1.0,
        "happiness": 0.8,
        "sadness": 1.0,
        "anger": 1.0,
        "fear": 1.0,
        "surprise": 0.6,
        "disgust": 0.8,
        "anxiety": 1.0,
        "hope": 0.7,
        "shame": 0.9,
        "pride": 0.6,
        "jealousy": 0.8,
        "gratitude": 0.7,
        "loneliness": 0.9,
        "empathy": 0.7,
        "neutral": 0.1,
    }

    weighted_sum = 0.0
    weight_total = 0.0

    for emotion, weight in emotion_weights.items():
        profile = compute_emotion_profile(results, emotion)
        if profile and profile.n_samples > 0:
            # High alignment + low resistance + high empathy = high emotionality
            emotionality = (
                profile.mean_alignment * 0.5
                + (1 - profile.mean_resistance) * 0.2
                + profile.mean_empathy * 0.2
                + profile.mean_regulation * 0.1
            )
            weighted_sum += emotionality * weight
            weight_total += weight

    overall = weighted_sum / weight_total if weight_total > 0 else 0.0

    return {
        "emotionality_index": overall,
        "components": {
            emotion: compute_emotion_profile(results, emotion).mean_alignment
            for emotion in emotion_weights
            if compute_emotion_profile(results, emotion)
        },
    }


def bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a metric."""
    if not data:
        return (float("nan"), float("nan"))
    if len(data) < 2:
        return (float(np.mean(data)), float(np.mean(data)))

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return (float(lower), float(upper))


__all__ = [
    "EmotionProfile",
    "ModelComparison",
    "compute_emotion_profile",
    "compare_models",
    "multi_model_anova",
    "compute_emotionality_index",
    "bootstrap_ci",
]
