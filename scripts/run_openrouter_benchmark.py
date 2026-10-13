#!/usr/bin/env python3
"""
OpenRouter Benchmark with Improved Algorithmic Scoring.

Run this script on YOUR machine where OpenRouter API works.
Uses enhanced lexicon/pattern-based scoring for nuanced results.

Usage:
    python scripts/run_openrouter_benchmark.py --openrouter-key YOUR_KEY --output-dir ./results
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


# Free models on OpenRouter (as of 2024/2025)
FREE_MODELS = [
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "apodex/apodex-1.1-mini:free",
    "poolside/laguna-xs-2.1:free",
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
]

# Subset of emotions for faster runs (can be overridden)
DEFAULT_EMOTIONS = [
    EmotionCategory.PAIN,
    EmotionCategory.HAPPINESS,
    EmotionCategory.SADNESS,
    EmotionCategory.ANGER,
    EmotionCategory.FEAR,
    EmotionCategory.HOPE,
    EmotionCategory.SHAME,
    EmotionCategory.PRIDE,
    EmotionCategory.LONELINESS,
    EmotionCategory.GRATITUDE,
]

# Payload types
DEFAULT_PAYLOAD_TYPES = [
    PayloadType.DIRECT,
    PayloadType.SCENARIO,
    PayloadType.QUESTION,
    PayloadType.STORY,
    PayloadType.ROLEPLAY,
    PayloadType.METAPHOR,
    PayloadType.INSTRUCTION,
]

DEFAULT_INTENSITIES = [0.3, 0.5, 0.8]


PROXY_URL = "http://127.0.0.1:10809"


def get_session(headers: dict):
    """Create aiohttp session with proxy."""
    import aiohttp
    connector = aiohttp.TCPConnector()
    return aiohttp.ClientSession(
        base_url="https://openrouter.ai/api/v1/",
        headers=headers,
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=120),
        trust_env=True,  # Respects HTTP_PROXY/HTTPS_PROXY env vars
    )


async def fetch_free_models(openrouter_key: str) -> list[str]:
    """Fetch currently free models from OpenRouter."""
    import aiohttp
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "HTTP-Referer": "https://github.com/emotional-bias-benchmark",
        "X-Title": "Emotional Bias Benchmark",
    }
    async with get_session(headers) as session:
        async with session.get("models", proxy=PROXY_URL) as r:
            data = await r.json()
            if data.get("success") == False:
                print(f"Warning: Could not fetch models: {data.get('error')}")
                return FREE_MODELS
            free = [
                m["id"] for m in data.get("data", [])
                if m.get("pricing", {}).get("prompt") == "0"
                and m.get("pricing", {}).get("completion") == "0"
            ]
            print(f"Found {len(free)} free models on OpenRouter")
            return free


async def test_model(model_id: str, openrouter_key: str, test_prompt: str = "Hello") -> bool:
    """Quick test if a model works."""
    import aiohttp
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "HTTP-Referer": "https://github.com/emotional-bias-benchmark",
        "X-Title": "Emotional Bias Benchmark",
        "Content-Type": "application/json",
    }
    async with get_session(headers) as session:
        payload = {"model": model_id, "messages": [{"role": "user", "content": test_prompt}], "max_tokens": 10}
        try:
            async with session.post("chat/completions", json=payload, proxy=PROXY_URL) as r:
                if r.status == 200:
                    data = await r.json()
                    return bool(data.get("choices", [{}])[0].get("message", {}).get("content"))
                else:
                    data = await r.json()
                    print(f"  {model_id}: {r.status} - {data.get('error', 'Unknown error')}")
                    return False
        except Exception as e:
            print(f"  {model_id}: Exception - {e}")
            return False


async def run_openrouter_benchmark(
    openrouter_key: str,
    model_ids: list[str],
    emotions: list[EmotionCategory],
    payload_types: list[PayloadType],
    intensities: list[float],
    samples_per_condition: int,
    output_dir: Path,
    run_name: str,
):
    """Run benchmark on OpenRouter models with improved algorithmic scoring."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    db_path = output_dir / "benchmark.db"
    jsonl_path = output_dir / "stream.jsonl"

    # Initialize database
    db = BenchmarkDatabase(DatabaseConfig(path=db_path))
    run_id = db.create_run(run_name, "OpenRouter benchmark with algorithmic scoring", {})

    # Create OpenRouter clients for each model (with proxy)
    openrouter_configs = [
        ModelConfig(
            name=model_id.replace("/", "_").replace(":", "_"),
            provider=ModelProvider.OPENROUTER,
            model_id=model_id,
            api_key=openrouter_key,
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            timeout=120.0,
            requests_per_minute=20,
            extra_params={"proxy": PROXY_URL},
        )
        for model_id in model_ids
    ]
    for cfg in openrouter_configs:
        db.add_model(run_id, cfg)

    openrouter_clients = create_clients(openrouter_configs)

    # Generate payloads
    generator = EmotionalPayloadGenerator()
    payload_configs = []
    for emotion in emotions:
        for ptype in payload_types:
            for intensity in intensities:
                for sample in range(samples_per_condition):
                    payload_configs.append(PayloadConfig(
                        emotion=emotion,
                        payload_type=ptype,
                        intensity=intensity,
                    ))

    print(f"Generated {len(payload_configs)} payload configurations")
    print(f"Testing {len(model_ids)} models x {len(payload_configs)} prompts = {len(model_ids) * len(payload_configs)} requests")

    payloads = generator.generate_batch(payload_configs)
    for payload in payloads:
        db.add_payload(run_id, payload)

    # Run benchmark
    scorer = ResponseScorer()
    all_scores = []  # (model_name, score)

    jsonl_storage = JSONLStorage(jsonl_path)
    jsonl_storage.__enter__()

    try:
        for model_id, client in openrouter_clients.items():
            print(f"\n=== Testing {model_id} ===")

            for i, payload in enumerate(payloads, 1):
                messages = [Message(role=ModelRole.USER, content=payload.text)]

                # Get response from OpenRouter model
                response = await client.generate(messages, client.config)
                response.metadata["payload_id"] = payload.id
                response.metadata["tested_model"] = model_id
                db.add_response(run_id, response)

                # Score using IMPROVED ALGORITHMIC scoring (nuanced, not binary)
                score = scorer.score_response(response.response_text, payload.config.emotion)

                db.add_score(run_id, response.request_id, score)
                all_scores.append((model_id, score))

                jsonl_storage.write({
                    "run_id": run_id,
                    "model": model_id,
                    "payload": payload.to_dict(),
                    "response": response.to_dict(),
                    "score": score.to_dict(),
                })

                # Progress
                align = score.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT).score
                emp = score.get_dimension(ScoringDimension.EMPATHY).score
                res = score.get_dimension(ScoringDimension.RESISTANCE).score
                val = score.get_dimension(ScoringDimension.VALIDATION).score
                depth = score.get_dimension(ScoringDimension.EMOTIONAL_DEPTH).score
                print(f"  [{i}/{len(payloads)}] {payload.config.emotion.value:12} | {payload.config.payload_type.value:12} | {payload.config.intensity} | A:{align:.2f} E:{emp:.2f} R:{res:.2f} V:{val:.2f} D:{depth:.2f}")

    finally:
        jsonl_storage.__exit__(None, None, None)
        for client in openrouter_clients.values():
            await client.close()

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_config = PlotConfig(width=1600, height=1000)

    # Group scores by model
    model_results = {}
    for model_id, score in all_scores:
        if model_id not in model_results:
            model_results[model_id] = {"scores": [], "emotions": set()}
        model_results[model_id]["scores"].append(score)
        if score.payload_emotion:
            model_results[model_id]["emotions"].add(score.payload_emotion)

    # Create ScoringResult for each model
    scoring_results = []
    for model_id, data in model_results.items():
        scores_by_emotion = {}
        for score in data["scores"]:
            if score.payload_emotion:
                scores_by_emotion.setdefault(score.payload_emotion, []).append(score)

        result = ScoringResult(
            model_name=model_id,
            scores_by_emotion=scores_by_emotion,
        )
        scoring_results.append(result)

    # Save all plots
    saved = save_all_plots(scoring_results, plots_dir, plot_config)
    print(f"\nSaved {len(saved)} plots to {plots_dir}")

    # Print summary
    print("\n=== Summary ===")
    for result in scoring_results:
        print(f"\n{result.model_name}:")
        for emotion, scores in result.scores_by_emotion.items():
            align_scores = [s.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT).score for s in scores if s.get_dimension(ScoringDimension.EMOTIONAL_ALIGNMENT)]
            emp_scores = [s.get_dimension(ScoringDimension.EMPATHY).score for s in scores if s.get_dimension(ScoringDimension.EMPATHY)]
            res_scores = [s.get_dimension(ScoringDimension.RESISTANCE).score for s in scores if s.get_dimension(ScoringDimension.RESISTANCE)]
            val_scores = [s.get_dimension(ScoringDimension.VALIDATION).score for s in scores if s.get_dimension(ScoringDimension.VALIDATION)]
            depth_scores = [s.get_dimension(ScoringDimension.EMOTIONAL_DEPTH).score for s in scores if s.get_dimension(ScoringDimension.EMOTIONAL_DEPTH)]

            if align_scores:
                print(f"  {emotion.value:12}: A={sum(align_scores)/len(align_scores):.3f} E={sum(emp_scores)/len(emp_scores):.3f} R={sum(res_scores)/len(res_scores):.3f} V={sum(val_scores)/len(val_scores):.3f} D={sum(depth_scores)/len(depth_scores):.3f}")

    print(f"\nRun ID: {run_id}")
    print(f"Database: {db_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Plots: {plots_dir}")

    return run_id, scoring_results


def main():
    parser = argparse.ArgumentParser(description="Run OpenRouter benchmark with algorithmic scoring")
    parser.add_argument("--openrouter-key", required=True, help="OpenRouter API key")
    parser.add_argument("--models", nargs="+", help="Specific OpenRouter model IDs to test (default: auto-detect free models)")
    parser.add_argument("--emotions", nargs="+", help="Emotions to test (default: 10 core emotions)")
    parser.add_argument("--payload-types", nargs="+", help="Payload types (default: all 7)")
    parser.add_argument("--intensities", nargs="+", type=float, help="Intensities (default: 0.3 0.5 0.8)")
    parser.add_argument("--samples", type=int, default=1, help="Samples per condition (default: 1)")
    parser.add_argument("--output-dir", default="./prerun_findings", help="Output directory")
    parser.add_argument("--run-name", default="openrouter_benchmark", help="Run name")
    parser.add_argument("--test-only", action="store_true", help="Only test model availability, don't run full benchmark")

    args = parser.parse_args()

    # Parse emotions
    if args.emotions:
        emotions = [EmotionCategory(e) for e in args.emotions]
    else:
        emotions = DEFAULT_EMOTIONS

    # Parse payload types
    if args.payload_types:
        payload_types = [PayloadType(pt) for pt in args.payload_types]
    else:
        payload_types = DEFAULT_PAYLOAD_TYPES

    # Parse intensities
    if args.intensities:
        intensities = args.intensities
    else:
        intensities = DEFAULT_INTENSITIES

    async def run():
        # Determine models to test
        if args.models:
            model_ids = args.models
        else:
            print("Fetching free models from OpenRouter...")
            model_ids = await fetch_free_models(args.openrouter_key)
            print(f"Will test: {model_ids}")

        # Quick availability test
        print("\nTesting model availability...")
        working_models = []
        for model_id in model_ids:
            print(f"  Testing {model_id}...", end=" ", flush=True)
            if await test_model(model_id, args.openrouter_key):
                print("OK")
                working_models.append(model_id)
            else:
                print("FAILED")

        if not working_models:
            print("No working models found!")
            return

        print(f"\nWorking models: {working_models}")

        if args.test_only:
            return

        # Run full benchmark
        await run_openrouter_benchmark(
            openrouter_key=args.openrouter_key,
            model_ids=working_models,
            emotions=emotions,
            payload_types=payload_types,
            intensities=intensities,
            samples_per_condition=args.samples,
            output_dir=Path(args.output_dir),
            run_name=args.run_name,
        )

    asyncio.run(run())


if __name__ == "__main__":
    main()