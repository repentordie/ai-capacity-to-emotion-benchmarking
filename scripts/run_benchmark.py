#!/usr/bin/env python3
"""
Batch benchmark runner script.

Run multiple benchmark configurations in sequence.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with config_path.open() as f:
        return yaml.safe_load(f)


def build_cli_args(config: dict[str, Any], run_name: str) -> list[str]:
    """Build CLI arguments from config."""
    args = [
        "ebb", "run",
        "--name", run_name,
    ]

    # Models
    for model in config.get("models", []):
        extra = ""
        if model.get("base_url"):
            extra += f" --base-url {model['base_url']}"
        if model.get("api_key"):
            extra += f" --api-key {model['api_key']}"
        args.extend(["--models", f"{model['name']}:{model['provider']}:{model['model_id']}"])

    # Emotions
    if emotions := config.get("emotions"):
        for e in emotions:
            args.extend(["--emotions", e])

    # Payload types
    if ptypes := config.get("payload_types"):
        for p in ptypes:
            args.extend(["--payload-types", p])

    # Intensities
    if intensities := config.get("intensities"):
        for i in intensities:
            args.extend(["--intensities", str(i)])

    # Samples
    if samples := config.get("samples_per_combination"):
        args.extend(["--samples", str(samples)])

    # Output
    if output := config.get("output_dir"):
        args.extend(["--output", output])

    if db := config.get("database"):
        args.extend(["--db", db])

    if jsonl := config.get("jsonl_output"):
        args.extend(["--jsonl", jsonl])

    return args


async def run_benchmark(config_path: Path, run_name: str | None = None) -> int:
    """Run a single benchmark."""
    config = load_config(config_path)

    name = run_name or config_path.stem
    args = build_cli_args(config, name)

    print(f"Running: {' '.join(args)}")
    result = subprocess.run(args, capture_output=False)
    return result.returncode


async def run_batch(config_dir: Path, pattern: str = "*.yaml") -> list[tuple[str, int]]:
    """Run all configs in a directory."""
    configs = sorted(config_dir.glob(pattern))
    results = []

    for config_path in configs:
        print(f"\n{'='*60}")
        print(f"Running config: {config_path.name}")
        print(f"{'='*60}\n")

        returncode = await run_benchmark(config_path)
        results.append((config_path.name, returncode))

        if returncode != 0:
            print(f"[WARNING] Config {config_path.name} failed with code {returncode}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch benchmarks")
    parser.add_argument("config", type=Path, help="Config file or directory")
    parser.add_argument("--name", help="Run name (for single config)")
    parser.add_argument("--pattern", default="*.yaml", help="Glob pattern for batch")
    args = parser.parse_args()

    if args.config.is_dir():
        results = asyncio.run(run_batch(args.config, args.pattern))
        print("\n\n=== BATCH SUMMARY ===")
        for name, code in results:
            status = "✓" if code == 0 else "✗"
            print(f"  {status} {name}: exit code {code}")
    else:
        returncode = asyncio.run(run_benchmark(args.config, args.name))
        sys.exit(returncode)


if __name__ == "__main__":
    main()