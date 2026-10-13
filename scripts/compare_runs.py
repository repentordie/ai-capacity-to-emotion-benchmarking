#!/usr/bin/env python3
"""
Compare multiple benchmark runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from emotional_bias.storage import BenchmarkDatabase, DatabaseConfig
from emotional_bias.analysis import (
    compute_emotion_profile,
    compare_models,
    compute_emotionality_index,
    multi_model_anova,
)
from emotional_bias.visualization import save_all_plots, PlotConfig


def load_run(run_id: str, db_path: Path) -> dict[str, Any] | None:
    """Load run data from database."""
    db = BenchmarkDatabase(DatabaseConfig(path=db_path))
    return db.get_run(run_id)


def compare_run_ids(run_ids: list[str], db_path: Path, output_dir: Path | None = None) -> None:
    """Compare multiple runs by ID."""
    db = BenchmarkDatabase(DatabaseConfig(path=db_path))

    print(f"Comparing {len(run_ids)} runs from {db_path}")

    # Load all runs
    runs = {}
    for run_id in run_ids:
        run = db.get_run(run_id)
        if run:
            runs[run_id] = run
            print(f"  Loaded: {run['name']} ({run_id})")
        else:
            print(f"  [WARNING] Run not found: {run_id}")

    if not runs:
        print("No valid runs found!")
        return

    # Export all to JSONL for analysis
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for run_id in runs:
            jsonl_path = output_dir / f"{run_id}.jsonl"
            count = db.export_run_jsonl(run_id, jsonl_path)
            print(f"  Exported {count} records to {jsonl_path}")

    # TODO: Implement full comparison logic
    # This would require reconstructing ScoringResult objects from DB
    print("\nComparison functionality - to be implemented")
    print("For now, use 'ebb analyze <run_id> --output' for individual run analysis")


def compare_jsonl_files(jsonl_paths: list[Path], output_dir: Path | None = None) -> None:
    """Compare results from JSONL files."""
    all_data = []

    for path in jsonl_paths:
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                record["source_file"] = path.stem
                all_data.append(record)

    if not all_data:
        print("No data found!")
        return

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} records from {len(jsonl_paths)} files")

    # Basic stats
    print("\n=== Models ===")
    print(df["model"].value_counts())

    print("\n=== Emotions ===")
    if "payload" in df.columns:
        emotions = df["payload"].apply(lambda x: x.get("emotion") if isinstance(x, dict) else None)
        print(emotions.value_counts())

    print("\n=== Response Lengths ===")
    if "response" in df.columns:
        lengths = df["response"].apply(lambda x: len(x.get("response_text", "")) if isinstance(x, dict) else 0)
        print(lengths.describe())

    # Score analysis
    if "score" in df.columns:
        scores_df = pd.DataFrame([s for s in df["score"] if s])
        if not scores_df.empty:
            print("\n=== Score Dimensions ===")
            dim_cols = [c for c in scores_df.columns if c.startswith("dimension_")]
            if dim_cols:
                print(scores_df[dim_cols].describe())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark runs")
    parser.add_argument("inputs", nargs="+", type=Path, help="Run IDs (from DB) or JSONL files")
    parser.add_argument("--db", type=Path, default=Path("results.db"), help="Database path")
    parser.add_argument("--output", "-o", type=Path, help="Output directory for plots")

    args = parser.parse_args()

    # Determine input type
    jsonl_files = [p for p in args.inputs if p.suffix == ".jsonl"]
    run_ids = [str(p) for p in args.inputs if p.suffix != ".jsonl"]

    if jsonl_files:
        compare_jsonl_files(jsonl_files, args.output)

    if run_ids:
        compare_run_ids(run_ids, args.db, args.output)


if __name__ == "__main__":
    main()