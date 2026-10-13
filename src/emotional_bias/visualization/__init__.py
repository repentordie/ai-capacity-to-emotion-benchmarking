"""
Visualization module for plotting emotional bias results.
"""

from .plots import (
    DIMENSION_COLORS,
    EMOTION_COLORS,
    PlotConfig,
    create_emotionality_radar,
    create_grouped_bar_chart,
    create_heatmap,
    create_matplotlib_radar,
    create_radar_chart,
    create_scatter_comparison,
    create_violin_plot,
    save_all_plots,
    setup_style,
)

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
