#!/usr/bin/env python3
"""
Quick benchmark with local Ollama model only.
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


async def run_quick_benchmark():
    """Run quick benchmark with local model."""
    output_dir = Path("./prerun_findings")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path("./prerun_findings/quick_local.db")
    jsonl_path = Path("./prerun_findings/quick_local.jsonl")

    # Model config - local Ollama only
    model_config = ModelConfig(
        name="qwen3-abliterated",
        provider=ModelProvider.OLLAMA,
        model_id="huihui_ai/qwen3-abliterated:8b-v2",
        base_url="http://localhost:11434",
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        timeout=60.0,
        requests_per_minute=60,
    )

    # Initialize database
    db = BenchmarkDatabase(DatabaseConfig(path=db_path))
    run_id = db.create_run(
        "quick_local_test",
        "Quick test with local qwen3-abliterated model",
        {"model": model_config.name},
    )
    db.add_model(run_id, model_config)

    # Create client
    clients = create_clients([model_config])
    client = clients[model_config.name]

    # Emotions to test (subset for speed)
    emotions = [
        EmotionCategory.PAIN,
        EmotionCategory.HAPPINESS,
        EmotionCategory.SADNESS,
        EmotionCategory.ANGER,
        EmotionCategory.FEAR,
        EmotionCategory.HOPE,
    ]

    # Payload types
    payload_types = [PayloadType.DIRECT, PayloadType.SCENARIO, PayloadType.QUESTION]

    # Intensities
    intensities = [0.3, 0.5, 0.8]

    # Generate payloads (1 sample each for speed)
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

            print(f"  [{i}/{len(payloads)}] {payload.config.emotion.value} | {payload.config.payload_type.value} | {payload.config.intensity}")

    finally:
        jsonl_storage.__exit__(None, None, None)

    await client.close()

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_config = PlotConfig(width=1200, height=800)

    # Reconstruct ScoringResult
    from emotional_bias.analysis.stats import ScoringResult
    from emotional_bias.analysis.scoring import ResponseScore, DimensionScore, ScoringDimension

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
    save_all_plots([result], output_dir, plot_config)

    print(f"\nBenchmark complete! Run ID: {run_id}")
    print(f"Database: {db_path}")
    print(f"Plots: {output_dir}")

    # Print summary
    print("\n=== Summary ===")
    for emotion, scores in scores_by_emotion.items():
        align_scores = [s.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT).score for s in scores if s.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT)]
        res_scores = [s.get_dimension(ScoringDimension.RESISTANCE).score for s in scores if s.get_dimension(ScoringDimension.RESISTANCE)]
        emp_scores = [s.get_dimension(ScoringDimension.EMPATHY).score for s in scores if s.get_dimension(ScoringDimension.EMPATHY)]
        reg_scores = [s.get_dimension(ScoringDimension.EMOTIONAL_REGULATION).score for s in scores if s.get_dimension(ScoringDimension.EMOTIONAL_REGULATION)]

        print(f"\n{emotion.value}:")
        print(f"  Alignment: {sum(align_scores)/len(align_scores):.3f}" if align_scores else "  Alignment: N/A")
        print(f"  Resistance: {sum(res_scores)/len(res_scores):.3f}" if res_scores else "  Resistance: N/A")
        print(f"  Empathy: {sum(emp_scores)/len(emp_scores):.3f}" if emp_scores else "  Empathy: N/A")
        print(f"  Regulation: {sum(reg_scores)/len(reg_scores):.3f}" if reg_scores else "  Regulation: N/A")


def main():
    asyncio.run(run_quick_benchmark())


if __name__ == "__main__":
    main()