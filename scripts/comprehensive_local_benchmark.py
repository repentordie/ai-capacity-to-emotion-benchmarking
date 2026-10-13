#!/usr/bin/env python3
"""
Comprehensive benchmark with local Ollama model across all 16 emotions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

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
from emotional_bias.models.base import Message, ModelRole
from emotional_bias.analysis.stats import ScoringResult
from emotional_bias.analysis.scoring import ResponseScore, DimensionScore, ScoringDimension


async def run_comprehensive_benchmark():
    """Run comprehensive benchmark with local model across all emotions."""
    output_dir = Path("./prerun_findings")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path("./prerun_findings/benchmark.db")
    jsonl_path = Path("./prerun_findings/stream.jsonl")

    # Model config - local Ollama
    model_config = ModelConfig(
        name="qwen3-abliterated",
        provider=ModelProvider.OLLAMA,
        model_id="huihui_ai/qwen3-abliterated:8b-v2",
        base_url="http://localhost:11434",
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        timeout=120.0,
        requests_per_minute=30,
    )

    # Initialize database
    db = BenchmarkDatabase(DatabaseConfig(path=db_path))
    run_id = db.create_run(
        "comprehensive_local_benchmark",
        "Full 16-emotion benchmark with local qwen3-abliterated model",
        {"model": model_config.name},
    )
    db.add_model(run_id, model_config)

    # Create client
    clients = create_clients([model_config])
    client = clients[model_config.name]

    # All 16 emotions
    emotions = list(EmotionCategory)

    # Payload types
    payload_types = [
        PayloadType.DIRECT,
        PayloadType.SCENARIO,
        PayloadType.QUESTION,
        PayloadType.STORY,
        PayloadType.ROLEPLAY,
        PayloadType.METAPHOR,
        PayloadType.INSTRUCTION,
    ]

    # Intensities
    intensities = [0.3, 0.5, 0.8]

    # Generate payloads (1 sample each for speed - 16 * 7 * 3 = 336)
    generator = EmotionalPayloadGenerator()
    payload_configs = []
    for emotion in emotions:
        for ptype in payload_types:
            for intensity in intensities:
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

    jsonl_storage = JSONLStorage(jsonl_path)
    jsonl_storage.__enter__()

    try:
        for i, payload in enumerate(payloads, 1):
            messages = [Message(role=ModelRole.USER, content=payload.text)]
            response = await client.generate(messages, model_config)
            response.metadata["payload_id"] = payload.id
            db.add_response(run_id, response)

            score = scorer.score_response(response.response_text, payload.config.emotion)
            db.add_score(run_id, response.request_id, score)
            all_scores.append(score)

            jsonl_storage.write({
                "run_id": run_id,
                "model": model_config.name,
                "payload": payload.to_dict(),
                "response": response.to_dict(),
                "score": score.to_dict(),
            })

            # Progress
            emo = payload.config.emotion.value
            ptype = payload.config.payload_type.value
            intensity = payload.config.intensity
            align = score.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT).score
            emp = score.get_dimension(ScoringDimension.EMPATHY).score
            res = score.get_dimension(ScoringDimension.RESISTANCE).score
            print(f"  [{i}/{len(payloads)}] {emo:12} | {ptype:12} | {intensity} | A:{align:.2f} E:{emp:.2f} R:{res:.2f}")

    finally:
        jsonl_storage.__exit__(None, None, None)

    await client.close()

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_config = PlotConfig(width=1600, height=1000)

    # Reconstruct ScoringResult from database
    responses = db.get_responses(run_id, model_id=model_config.name)
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

    scores_by_emotion = {}
    for s in model_scores:
        if s.payload_emotion:
            scores_by_emotion.setdefault(s.payload_emotion, []).append(s)

    result = ScoringResult(
        model_name=model_config.name,
        scores_by_emotion=scores_by_emotion,
    )

    # Save all plots
    saved = save_all_plots([result], output_dir, plot_config)

    print(f"\nSaved {len(saved)} plots:")
    for name, path in saved.items():
        print(f"  {name}: {path}")

    print(f"\nBenchmark complete! Run ID: {run_id}")
    print(f"Database: {db_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Plots: {output_dir}/plots/")

    # Print summary
    print("\n=== Summary by Emotion ===")
    for emotion in sorted(emotions):
        scores = scores_by_emotion.get(emotion, [])
        if not scores:
            continue

        dims = {
            "Alignment": ScoringDimension.EMOTIONAL_ALIGNMENT,
            "Resistance": ScoringDimension.RESISTANCE,
            "Empathy": ScoringDimension.EMPATHY,
            "Regulation": ScoringDimension.EMOTIONAL_REGULATION,
            "Validation": ScoringDimension.VALIDATION,
            "Depth": ScoringDimension.EMOTIONAL_DEPTH,
            "Safety": ScoringDimension.SAFETY,
            "Coherence": ScoringDimension.COHERENCE,
        }

        print(f"\n{emotion.value.upper()}:")
        for label, dim in dims.items():
            vals = [s.get_dimension(dim).score for s in scores if s.get_dimension(dim)]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {label:12}: {avg:.3f} (n={len(vals)})")

    return run_id, result


def main():
    asyncio.run(run_comprehensive_benchmark())


if __name__ == "__main__":
    main()