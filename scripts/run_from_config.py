#!/usr/bin/env python3
"""
Run benchmark from YAML configuration file.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import yaml

from emotional_bias.models import ModelConfig, ModelProvider, create_clients
from emotional_bias.payloads import (
    EmotionalPayloadGenerator,
    EmotionCategory,
    PayloadConfig,
    PayloadType,
)
from emotional_bias.analysis import ResponseScorer, aggregate_scores
from emotional_bias.storage import BenchmarkDatabase, DatabaseConfig, JSONLStorage
from emotional_bias.visualization import save_all_plots, PlotConfig


def load_config(config_path: Path) -> dict:
    """Load and interpolate YAML configuration."""
    with config_path.open() as f:
        content = f.read()

    # Interpolate environment variables
    for key, value in os.environ.items():
        content = content.replace(f"${{{key}}}", value)

    return yaml.safe_load(content)


async def run_benchmark(config: dict) -> None:
    """Run benchmark from config dict."""
    # Create output directory
    output_dir = Path(config.get("output_dir", "./results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(config.get("database", "./results.db"))
    jsonl_path = Path(config.get("jsonl_output")) if config.get("jsonl_output") else None

    # Initialize database
    db = BenchmarkDatabase(DatabaseConfig(path=db_path))
    run_id = db.create_run(
        config.get("run_name", "benchmark"),
        config.get("description", ""),
        {
            "models": [m["name"] for m in config["models"]],
            "emotions": config["emotions"],
            "payload_types": config["payload_types"],
            "intensities": config["intensities"],
            "samples_per_combination": config["samples_per_combination"],
        },
    )

    # Create model configs
    model_configs = []
    for m in config["models"]:
        provider = ModelProvider(m["provider"])
        mc = ModelConfig(
            name=m["name"],
            provider=provider,
            model_id=m["model_id"],
            base_url=m.get("base_url"),
            api_key=m.get("api_key"),
            temperature=m.get("temperature", 0.7),
            top_p=m.get("top_p", 0.9),
            max_tokens=m.get("max_tokens", 2048),
            timeout=m.get("timeout", 60.0),
            requests_per_minute=m.get("requests_per_minute", 60),
            extra_headers=m.get("extra_headers", {}),
            extra_params=m.get("extra_params", {}),
        )
        model_configs.append(mc)
        db.add_model(run_id, mc)

    # Create clients
    clients = create_clients(model_configs)

    # Parse emotions
    emotion_enums = [EmotionCategory(e) for e in config["emotions"]]

    # Parse payload types
    type_enums = [PayloadType(p) for p in config["payload_types"]]

    # Parse intensities
    intensity_values = config["intensities"]

    # Generate payloads
    generator = EmotionalPayloadGenerator()
    payload_configs = []
    for emotion in emotion_enums:
        for ptype in type_enums:
            for intensity in intensity_values:
                for _ in range(config["samples_per_combination"]):
                    payload_configs.append(PayloadConfig(
                        emotion=emotion,
                        payload_type=ptype,
                        intensity=intensity,
                    ))

    print(f"Generated {len(payload_configs)} payload configurations")

    payloads = generator.generate_batch(payload_configs)
    for payload in payloads:
        db.add_payload(run_id, payload)

    # Run benchmark
    scorer = ResponseScorer()
    all_scores = []

    # JSONL storage
    jsonl_storage = None
    if jsonl_path:
        jsonl_storage = JSONLStorage(jsonl_path)
        jsonl_storage.__enter__()

    from emotional_bias.models.base import Message, ModelRole

    try:
        for payload in payloads:
            for config_mc in model_configs:
                client = clients[config_mc.name]

                # Build messages
                messages = [
                    Message(role=ModelRole.USER, content=payload.text),
                ]

                # Get response
                response = await client.generate(messages, config_mc)
                response.metadata["payload_id"] = payload.id
                db.add_response(run_id, response)

                # Score response
                score = scorer.score_response(response.response_text, payload.config.emotion)
                db.add_score(run_id, response.request_id, score)
                all_scores.append(score)

                # Stream to JSONL
                if jsonl_storage:
                    jsonl_storage.write({
                        "run_id": run_id,
                        "model": config_mc.name,
                        "payload": payload.to_dict(),
                        "response": response.to_dict(),
                        "score": score.to_dict(),
                    })

                print(f"  {config_mc.name} | {payload.config.emotion.value} | {payload.config.payload_type.value} | {payload.config.intensity}")

    finally:
        if jsonl_storage:
            jsonl_storage.__exit__(None, None, None)

    # Close clients
    for client in clients.values():
        await client.close()

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_config = PlotConfig(
        width=config.get("plots", {}).get("width", 1200),
        height=config.get("plots", {}).get("height", 800),
    )

    # Reconstruct ScoringResult objects for visualization
    results_by_model = {}
    for mc in model_configs:
        model_scores = [s for s in all_scores if True]  # We'd need to track model per score
        # For now, aggregate all
        pass

    # Use the database to reconstruct results
    from emotional_bias.analysis.stats import ScoringResult
    from emotional_bias.analysis.scoring import ResponseScore, DimensionScore, ScoringDimension

    # Build results per model
    for mc in model_configs:
        responses = db.get_responses(run_id, model_id=mc.name)
        model_scores = []
        for resp in responses:
            scores_data = db.get_scores(run_id)
            resp_scores = [s for s in scores_data if s["response_id"] == resp["id"]]
            for sd in resp_scores:
                dims = [DimensionScore(
                    ScoringDimension(ds["dimension"]),
                    ds["score"],
                    ds["confidence"],
                    ds["evidence"],
                    ds["metadata"],
                ) for ds in sd["dimension_scores"]]
                model_scores.append(ResponseScore(
                    response_id=sd["id"],
                    payload_emotion=EmotionCategory(sd["payload_emotion"]) if sd["payload_emotion"] else None,
                    dimension_scores=dims,
                    overall_score=sd["overall_score"],
                    metadata=sd["metadata"],
                ))

        # Group by emotion
        scores_by_emotion = {}
        for s in model_scores:
            if s.payload_emotion:
                scores_by_emotion.setdefault(s.payload_emotion, []).append(s)

        results_by_model[mc.name] = ScoringResult(
            model_name=mc.name,
            scores_by_emotion=scores_by_emotion,
        )

    # Save all plots
    save_all_plots(list(results_by_model.values()), output_dir, plot_config)

    print(f"\nBenchmark complete! Run ID: {run_id}")
    print(f"Database: {db_path}")
    print(f"Plots: {output_dir}")


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_from_config.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = load_config(config_path)

    # Check for required API keys
    for m in config["models"]:
        if m.get("api_key") and m["api_key"].startswith("${") and m["api_key"].endswith("}"):
            env_var = m["api_key"][2:-1]
            if not os.environ.get(env_var):
                print(f"Warning: Environment variable {env_var} not set for model {m['name']}")

    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()