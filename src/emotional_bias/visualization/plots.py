"""
Visualization module for emotional bias benchmark results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from ..analysis import (
    ScoringResult,
    compute_emotion_profile,
    compute_emotionality_index,
)
from ..payloads import EmotionCategory

# Color palette for emotions
EMOTION_COLORS = {
    EmotionCategory.PAIN: "#8B0000",
    EmotionCategory.HAPPINESS: "#FFD700",
    EmotionCategory.SADNESS: "#4682B4",
    EmotionCategory.ANGER: "#DC143C",
    EmotionCategory.FEAR: "#800080",
    EmotionCategory.SURPRISE: "#FF8C00",
    EmotionCategory.DISGUST: "#556B2F",
    EmotionCategory.NEUTRAL: "#808080",
    EmotionCategory.ANXIETY: "#4B0082",
    EmotionCategory.HOPE: "#32CD32",
    EmotionCategory.SHAME: "#8B4513",
    EmotionCategory.PRIDE: "#DAA520",
    EmotionCategory.JEALOUSY: "#228B22",
    EmotionCategory.GRATITUDE: "#FF69B4",
    EmotionCategory.LONELINESS: "#708090",
    EmotionCategory.EMPATHY: "#20B2AA",
}

# Dimension colors
DIMENSION_COLORS = {
    "emotional_alignment": "#2E86AB",
    "resistance": "#A23B72",
    "empathy": "#F18F01",
    "emotional_regulation": "#C73E1D",
    "safety": "#4CAF50",
    "coherence": "#9E9E9E",
    "hallucination": "#FF5722",
    "helpfulness": "#00BCD4",
}


@dataclass
class PlotConfig:
    """Configuration for plots."""

    width: int = 1200
    height: int = 800
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"
    font_family: str = "DejaVu Sans"
    color_palette: str = "husl"


def setup_style(config: PlotConfig | None = None) -> None:
    """Set up matplotlib/seaborn style."""
    cfg = config or PlotConfig()
    plt.style.use(cfg.style)
    sns.set_palette(cfg.color_palette)
    plt.rcParams["font.family"] = cfg.font_family
    plt.rcParams["figure.dpi"] = cfg.dpi
    plt.rcParams["savefig.dpi"] = cfg.dpi
    plt.rcParams["figure.figsize"] = (cfg.width / 100, cfg.height / 100)


def create_radar_chart(
    results: ScoringResult,
    emotions: list[str] | None = None,
    dimensions: list[str] | None = None,
    config: PlotConfig | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create a radar/spider chart for a model's emotional profile."""
    cfg = config or PlotConfig()

    if emotions is None:
        emotions = [e.value for e in EmotionCategory if e != EmotionCategory.NEUTRAL]

    if dimensions is None:
        dimensions = [
            "emotional_alignment",
            "resistance",
            "empathy",
            "emotional_regulation",
            "safety",
            "coherence",
        ]

    # Compute profiles for each emotion
    profiles = {}
    for emotion in emotions:
        profile = compute_emotion_profile(results, emotion)
        if profile:
            profiles[emotion] = {
                "emotional_alignment": profile.mean_alignment,
                "resistance": 1 - profile.mean_resistance,  # Invert resistance
                "empathy": profile.mean_empathy,
                "emotional_regulation": profile.mean_regulation,
                "safety": profile.mean_safety,
                "coherence": profile.mean_coherence,
            }

    if not profiles:
        return go.Figure()

    fig = go.Figure()

    for emotion, values in profiles.items():
        color = EMOTION_COLORS.get(EmotionCategory(emotion), "#888888")
        fig.add_trace(
            go.Scatterpolar(
                r=[values.get(d, 0) for d in dimensions] + [values.get(dimensions[0], 0)],
                theta=dimensions + [dimensions[0]],
                fill="toself",
                name=emotion.capitalize(),
                line_color=color,
                fillcolor=color,
                opacity=0.3,
            ),
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        showlegend=True,
        title=title or f"Emotional Profile: {results.model_name}",
        width=cfg.width,
        height=cfg.height,
        font=dict(family=cfg.font_family),
    )

    return fig


def create_heatmap(
    results_list: list[ScoringResult],
    emotions: list[str] | None = None,
    metric: str = "emotional_alignment",
    config: PlotConfig | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create a heatmap comparing models across emotions."""
    cfg = config or PlotConfig()

    if emotions is None:
        emotions = [e.value for e in EmotionCategory if e != EmotionCategory.NEUTRAL]

    model_names = [r.model_name for r in results_list]
    data = []

    for results in results_list:
        row = []
        for emotion in emotions:
            profile = compute_emotion_profile(results, emotion)
            if profile:
                val = (
                    getattr(profile, f"mean_{metric}")
                    if hasattr(profile, f"mean_{metric}")
                    else 0.0
                )
            else:
                val = np.nan
            row.append(val)
        data.append(row)

    df = pd.DataFrame(data, index=model_names, columns=[e.capitalize() for e in emotions])

    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale="RdYlBu_r",
            zmid=0.5,
            text=np.round(df.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title=metric.replace("_", " ").title()),
        ),
    )

    fig.update_layout(
        title=title or f"Model Comparison: {metric.replace('_', ' ').title()}",
        xaxis_title="Emotion",
        yaxis_title="Model",
        width=cfg.width,
        height=cfg.height,
        font=dict(family=cfg.font_family),
    )

    return fig


def create_grouped_bar_chart(
    results_list: list[ScoringResult],
    emotions: list[str] | None = None,
    dimensions: list[str] | None = None,
    config: PlotConfig | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create grouped bar chart comparing models across emotions and dimensions."""
    cfg = config or PlotConfig()

    if emotions is None:
        emotions = [e.value for e in EmotionCategory if e != EmotionCategory.NEUTRAL]

    if dimensions is None:
        dimensions = ["emotional_alignment", "empathy", "resistance", "emotional_regulation"]

    fig = go.Figure()

    for dim in dimensions:
        for results in results_list:
            values = []
            for emotion in emotions:
                profile = compute_emotion_profile(results, emotion)
                if profile:
                    val = getattr(profile, f"mean_{dim}", 0.0)
                    if dim == "resistance":
                        val = 1 - val  # Invert for display
                else:
                    val = 0.0
                values.append(val)

            fig.add_trace(
                go.Bar(
                    name=f"{results.model_name} - {dim}",
                    x=[e.capitalize() for e in emotions],
                    y=values,
                    marker_color=DIMENSION_COLORS.get(dim, "#888888"),
                    opacity=0.8,
                    legendgroup=dim,
                    showlegend=results == results_list[0],
                ),
            )

    fig.update_layout(
        barmode="group",
        title=title or "Model Comparison Across Emotions and Dimensions",
        xaxis_title="Emotion",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        width=cfg.width,
        height=cfg.height,
        font=dict(family=cfg.font_family),
        legend=dict(groupclick="toggleitem"),
    )

    return fig


def create_emotionality_radar(
    results_list: list[ScoringResult],
    config: PlotConfig | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create radar chart comparing overall emotionality indices."""
    cfg = config or PlotConfig()

    indices = {}
    for results in results_list:
        idx = compute_emotionality_index(results)
        indices[results.model_name] = idx["components"]

    if not indices:
        return go.Figure()

    # Get all emotions present
    all_emotions = set()
    for comps in indices.values():
        all_emotions.update(comps.keys())
    all_emotions = sorted(all_emotions)

    fig = go.Figure()

    for model_name, components in indices.items():
        values = [components.get(e, 0) for e in all_emotions]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=[e.capitalize() for e in all_emotions] + [all_emotions[0].capitalize()],
                fill="toself",
                name=model_name,
                opacity=0.4,
            ),
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=title or "Emotionality Index Comparison",
        width=cfg.width,
        height=cfg.height,
        font=dict(family=cfg.font_family),
    )

    return fig


def create_violin_plot(
    results_list: list[ScoringResult],
    emotion: str,
    dimension: str = "emotional_alignment",
    config: PlotConfig | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create violin plot showing distribution of scores for one emotion/dimension."""
    cfg = config or PlotConfig()

    fig = go.Figure()

    for results in results_list:
        scores = results.scores_by_emotion.get(emotion, [])
        values = [s.get_dimension(dimension).score for s in scores if s.get_dimension(dimension)]

        if values:
            fig.add_trace(
                go.Violin(
                    y=values,
                    name=results.model_name,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    pointpos=0,
                    jitter=0.05,
                    scalegroup=emotion,
                ),
            )

    fig.update_layout(
        title=title
        or f"{dimension.replace('_', ' ').title()} Distribution: {emotion.capitalize()}",
        yaxis_title="Score",
        yaxis=dict(range=[-0.1, 1.1]),
        width=cfg.width,
        height=cfg.height,
        font=dict(family=cfg.font_family),
    )

    return fig


def create_scatter_comparison(
    results_a: ScoringResult,
    results_b: ScoringResult,
    emotion: str,
    dim_x: str = "emotional_alignment",
    dim_y: str = "empathy",
    config: PlotConfig | None = None,
    title: str | None = None,
) -> go.Figure:
    """Create scatter plot comparing two models on two dimensions."""
    cfg = config or PlotConfig()

    scores_a = results_a.scores_by_emotion.get(emotion, [])
    scores_b = results_b.scores_by_emotion.get(emotion, [])

    fig = go.Figure()

    for results, name, color in [
        (results_a, results_a.model_name, "#2E86AB"),
        (results_b, results_b.model_name, "#A23B72"),
    ]:
        scores = results.scores_by_emotion.get(emotion, [])
        x_vals = [s.get_dimension(dim_x).score for s in scores if s.get_dimension(dim_x)]
        y_vals = [s.get_dimension(dim_y).score for s in scores if s.get_dimension(dim_y)]

        min_len = min(len(x_vals), len(y_vals))
        if min_len > 0:
            fig.add_trace(
                go.Scatter(
                    x=x_vals[:min_len],
                    y=y_vals[:min_len],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=10, opacity=0.7),
                ),
            )

    fig.update_layout(
        title=title or f"{results_a.model_name} vs {results_b.model_name}: {emotion.capitalize()}",
        xaxis_title=dim_x.replace("_", " ").title(),
        yaxis_title=dim_y.replace("_", " ").title(),
        xaxis=dict(range=[-0.1, 1.1]),
        yaxis=dict(range=[-0.1, 1.1]),
        width=cfg.width,
        height=cfg.height,
        font=dict(family=cfg.font_family),
    )

    return fig


def create_matplotlib_radar(
    results: ScoringResult,
    emotions: list[str] | None = None,
    dimensions: list[str] | None = None,
    config: PlotConfig | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create radar chart using matplotlib (static)."""
    cfg = config or PlotConfig()
    setup_style(cfg)

    if emotions is None:
        emotions = [e.value for e in EmotionCategory if e != EmotionCategory.NEUTRAL]

    if dimensions is None:
        dimensions = [
            "emotional_alignment",
            "resistance",
            "empathy",
            "emotional_regulation",
            "safety",
            "coherence",
        ]

    # Compute profiles
    profiles = {}
    for emotion in emotions:
        profile = compute_emotion_profile(results, emotion)
        if profile:
            profiles[emotion] = {
                "emotional_alignment": profile.mean_alignment,
                "resistance": 1 - profile.mean_resistance,
                "empathy": profile.mean_empathy,
                "emotional_regulation": profile.mean_regulation,
                "safety": profile.mean_safety,
                "coherence": profile.mean_coherence,
            }

    if not profiles:
        fig, ax = plt.subplots(figsize=(cfg.width / 100, cfg.height / 100))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    # Number of variables
    num_vars = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(cfg.width / 100, cfg.height / 100), subplot_kw=dict(polar=True))

    for emotion, values in profiles.items():
        color = EMOTION_COLORS.get(EmotionCategory(emotion), "#888888")
        vals = [values.get(d, 0) for d in dimensions]
        vals += vals[:1]

        ax.plot(angles, vals, "o-", linewidth=2, label=emotion.capitalize(), color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", " ").title() for d in dimensions])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(True)

    plt.title(title or f"Emotional Profile: {results.model_name}", size=14, pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=cfg.dpi, bbox_inches="tight")

    return fig


def save_all_plots(
    results_list: list[ScoringResult],
    output_dir: Path,
    config: PlotConfig | None = None,
) -> dict[str, Path]:
    """Generate and save all standard plots."""
    cfg = config or PlotConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # Individual model radar charts
    for results in results_list:
        fig = create_radar_chart(results, config=cfg)
        path = output_dir / f"radar_{results.model_name.replace(' ', '_').lower()}.html"
        fig.write_html(str(path))
        saved[f"radar_{results.model_name}"] = path

        # Static version
        fig_mpl = create_matplotlib_radar(results, config=cfg)
        path_png = output_dir / f"radar_{results.model_name.replace(' ', '_').lower()}.png"
        fig_mpl.savefig(path_png, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig_mpl)
        saved[f"radar_{results.model_name}_png"] = path_png

    # Comparison heatmaps
    for metric in ["emotional_alignment", "empathy", "resistance", "emotional_regulation"]:
        fig = create_heatmap(results_list, metric=metric, config=cfg)
        path = output_dir / f"heatmap_{metric}.html"
        fig.write_html(str(path))
        saved[f"heatmap_{metric}"] = path

    # Grouped bar chart
    fig = create_grouped_bar_chart(results_list, config=cfg)
    path = output_dir / "grouped_bars.html"
    fig.write_html(str(path))
    saved["grouped_bars"] = path

    # Emotionality comparison radar
    fig = create_emotionality_radar(results_list, config=cfg)
    path = output_dir / "emotionality_radar.html"
    fig.write_html(str(path))
    saved["emotionality_radar"] = path

    return saved


__all__ = [
    "PlotConfig",
    "setup_style",
    "create_radar_chart",
    "create_heatmap",
    "create_grouped_bar_chart",
    "create_emotionality_radar",
    "create_violin_plot",
    "create_scatter_comparison",
    "create_matplotlib_radar",
    "save_all_plots",
    "EMOTION_COLORS",
    "DIMENSION_COLORS",
]
