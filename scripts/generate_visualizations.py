#!/usr/bin/env python3
"""
Generate visualizations from existing benchmark data.
"""

from __future__ import annotations

import json
from pathlib import Path

from emotional_bias.visualization import save_all_plots, PlotConfig, create_radar_chart
from emotional_bias.analysis.stats import ScoringResult
from emotional_bias.analysis.scoring import ResponseScore, DimensionScore, ScoringDimension
from emotional_bias.payloads import EmotionCategory


def load_results_from_jsonl(jsonl_path: Path) -> dict:
    """Load scoring results from JSONL file."""
    results_by_model = {}

    with jsonl_path.open() as f:
        for line in f:
            record = json.loads(line)
            model = record["model"]
            score_data = record["score"]

            if model not in results_by_model:
                results_by_model[model] = {"scores": [], "emotions": set()}

            # Parse dimension scores
            dims = []
            for ds in score_data["dimension_scores"]:
                dims.append(DimensionScore(
                    ScoringDimension(ds["dimension"]),
                    ds["score"],
                    ds["confidence"],
                    ds["evidence"],
                    ds["metadata"],
                ))

            # Get payload emotion
            payload_emotion = None
            if score_data["payload_emotion"]:
                payload_emotion = EmotionCategory(score_data["payload_emotion"])

            score = ResponseScore(
                response_id=score_data["response_id"],
                payload_emotion=payload_emotion,
                dimension_scores=dims,
                overall_score=score_data["overall_score"],
                metadata=score_data["metadata"],
            )

            results_by_model[model]["scores"].append(score)
            if payload_emotion:
                results_by_model[model]["emotions"].add(payload_emotion)

    return results_by_model


def create_scoring_results(results_by_model: dict) -> list[ScoringResult]:
    """Create ScoringResult objects from loaded data."""
    scoring_results = []

    for model_name, data in results_by_model.items():
        # Group by emotion
        scores_by_emotion = {}
        for score in data["scores"]:
            if score.payload_emotion:
                scores_by_emotion.setdefault(score.payload_emotion, []).append(score)

        result = ScoringResult(
            model_name=model_name,
            scores_by_emotion=scores_by_emotion,
        )
        scoring_results.append(result)

    return scoring_results


def main():
    jsonl_path = Path("./prerun_findings/stream.jsonl")
    output_dir = Path("./prerun_findings/plots")

    print(f"Loading results from {jsonl_path}...")
    results_by_model = load_results_from_jsonl(jsonl_path)

    print(f"Found models: {list(results_by_model.keys())}")
    for model, data in results_by_model.items():
        print(f"  {model}: {len(data['scores'])} scores, emotions: {data['emotions']}")

    scoring_results = create_scoring_results(results_by_model)

    print(f"\nGenerating visualizations in {output_dir}...")
    plot_config = PlotConfig(width=1600, height=1000)

    saved = save_all_plots(scoring_results, output_dir, plot_config)

    print(f"\nSaved {len(saved)} plots:")
    for name, path in saved.items():
        print(f"  {name}: {path}")

    # Also create individual radar charts for each model
    print("\nGenerating individual radar charts...")
    for result in scoring_results:
        fig = create_radar_chart(result, config=plot_config)
        html_path = output_dir / f"radar_{result.model_name}.html"
        fig.write_html(str(html_path))
        print(f"  {result.model_name}: {html_path}")


if __name__ == "__main__":
    main()