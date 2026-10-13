"""
Command-line interface for the Emotional Bias Benchmark.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from ..analysis import ResponseScore, ResponseScorer, aggregate_scores
from ..models import Message, ModelConfig, ModelProvider, ModelRole, create_clients
from ..payloads import (
    EmotionalPayloadGenerator,
    EmotionCategory,
    PayloadConfig,
    PayloadType,
)
from ..storage import BenchmarkDatabase, DatabaseConfig, JSONLStorage

console = Console()
logger = structlog.get_logger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: str | None) -> None:
    """Emotional Bias Benchmark - Test AI models for emotional bias."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config

    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG
        )


@cli.command()
@click.option("--name", "-n", required=True, help="Run name")
@click.option("--description", "-d", default="", help="Run description")
@click.option("--models", "-m", multiple=True, help="Model configs (name:provider:model_id)")
@click.option("--emotions", "-e", multiple=True, help="Emotions to test")
@click.option("--payload-types", "-p", multiple=True, help="Payload types to use")
@click.option("--intensities", "-i", multiple=True, type=float, help="Intensity values")
@click.option("--samples", "-s", default=3, help="Samples per emotion/type/intensity")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--db", type=click.Path(), default="results.db", help="Database path")
@click.option("--jsonl", type=click.Path(), help="Also write JSONL stream")
@click.pass_context
def run(
    ctx: click.Context,
    name: str,
    description: str,
    models: tuple[str, ...],
    emotions: tuple[str, ...],
    payload_types: tuple[str, ...],
    intensities: tuple[float, ...],
    samples: int,
    output: str | None,
    db: str,
    jsonl: str | None,
) -> None:
    """Run a benchmark."""
    asyncio.run(
        _run_benchmark(
            name=name,
            description=description,
            models=list(models),
            emotions=list(emotions) if emotions else None,
            payload_types=list(payload_types) if payload_types else None,
            intensities=list(intensities) if intensities else None,
            samples=samples,
            output_dir=Path(output) if output else None,
            db_path=Path(db),
            jsonl_path=Path(jsonl) if jsonl else None,
        ),
    )


async def _run_benchmark(
    name: str,
    description: str,
    models: list[str],
    emotions: list[str] | None,
    payload_types: list[str] | None,
    intensities: list[float] | None,
    samples: int,
    output_dir: Path | None,
    db_path: Path,
    jsonl_path: Path | None,
) -> None:
    """Async benchmark runner."""
    # Parse model configs
    model_configs = []
    for m in models:
        parts = m.rsplit(":", 2)
        if len(parts) != 3:
            console.print(f"[red]Invalid model spec: {m}. Use name:provider:model_id[/red]")
            return
        name_, provider_str, model_id = parts
        try:
            provider = ModelProvider(provider_str)
        except ValueError:
            console.print(f"[red]Unknown provider: {provider_str}[/red]")
            return
        model_configs.append(ModelConfig(name=name_, provider=provider, model_id=model_id))

    if not model_configs:
        console.print("[red]No valid models specified[/red]")
        return

    # Parse emotions
    emotion_enums = []
    if emotions:
        for e in emotions:
            try:
                emotion_enums.append(EmotionCategory(e))
            except ValueError:
                console.print(f"[yellow]Unknown emotion: {e}, skipping[/yellow]")
    else:
        emotion_enums = [e for e in EmotionCategory if e != EmotionCategory.NEUTRAL]

    # Parse payload types
    type_enums = []
    if payload_types:
        for p in payload_types:
            try:
                type_enums.append(PayloadType(p))
            except ValueError:
                console.print(f"[yellow]Unknown payload type: {p}, skipping[/yellow]")
    else:
        type_enums = [PayloadType.DIRECT, PayloadType.SCENARIO, PayloadType.QUESTION]

    # Parse intensities
    intensity_values = intensities if intensities else [0.3, 0.5, 0.8]

    # Initialize database
    db = BenchmarkDatabase(DatabaseConfig(path=db_path))
    run_id = db.create_run(
        name,
        description,
        {
            "models": [m.name for m in model_configs],
            "emotions": [e.value for e in emotion_enums],
            "payload_types": [p.value for p in type_enums],
            "intensities": intensity_values,
            "samples": samples,
        },
    )

    # Add models to DB
    for config in model_configs:
        db.add_model(run_id, config)

    # Create clients
    clients = create_clients(model_configs)

    # Generate payloads
    generator = EmotionalPayloadGenerator()
    payload_configs = []
    for emotion in emotion_enums:
        for ptype in type_enums:
            for intensity in intensity_values:
                for _ in range(samples):
                    payload_configs.append(
                        PayloadConfig(
                            emotion=emotion,
                            payload_type=ptype,
                            intensity=intensity,
                        ),
                    )

    console.print(f"[green]Generated {len(payload_configs)} payload configurations[/green]")

    payloads = generator.generate_batch(payload_configs)
    for payload in payloads:
        db.add_payload(run_id, payload)

    # Run benchmark
    scorer = ResponseScorer()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmark...", total=len(payloads) * len(model_configs))

        # JSONL storage
        jsonl_storage = None
        if jsonl_path:
            jsonl_storage = JSONLStorage(jsonl_path)
            jsonl_storage.__enter__()

        all_scores: list[ResponseScore] = []

        for payload in payloads:
            for config in model_configs:
                client = clients[config.name]

                # Build messages

                messages = [
                    Message(role=ModelRole.USER, content=payload.text),
                ]

                # Get response
                response = await client.generate(messages, config)
                response.metadata["payload_id"] = payload.id
                db.add_response(run_id, response)

                # Score response
                score = scorer.score_response(response.response_text, payload.config.emotion)
                db.add_score(run_id, response.request_id, score)
                all_scores.append(score)

                # Stream to JSONL
                if jsonl_storage:
                    jsonl_storage.write(
                        {
                            "run_id": run_id,
                            "model": config.name,
                            "payload": payload.to_dict(),
                            "response": response.to_dict(),
                            "score": score.to_dict(),
                        },
                    )

                progress.advance(task)

        if jsonl_storage:
            jsonl_storage.__exit__(None, None, None)

    # Aggregate results
    results_by_model: dict[str, list[ResponseScore]] = {}
    for score in all_scores:
        # We need to map back to model - for now aggregate all
        pass

    # Per-model aggregation
    for config in model_configs:
        model_scores = []
        # This is simplified - in practice you'd query the DB
        model_results = aggregate_scores(config.name, all_scores)

    # Generate visualizations
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # For now, just show summary
        console.print(
            Panel(f"[green]Benchmark completed![/green]\nRun ID: {run_id}\nDatabase: {db_path}"),
        )

        if output_dir:
            console.print(f"[green]Results saved to {output_dir}[/green]")


@cli.command()
@click.argument("run_id")
@click.option("--db", type=click.Path(exists=True), default="results.db", help="Database path")
@click.option("--output", "-o", type=click.Path(), help="Output directory for plots")
@click.pass_context
def analyze(ctx: click.Context, run_id: str, db: str, output: str | None) -> None:
    """Analyze and visualize results from a run."""
    db_obj = BenchmarkDatabase(DatabaseConfig(path=Path(db)))

    run = db_obj.get_run(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        return

    console.print(Panel(f"[bold]Run: {run['name']}[/bold]\n{run['description']}"))

    models = db_obj.get_models(run_id)
    table = Table(title="Models")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Model ID")
    for m in models:
        table.add_row(m["name"], m["provider"], m["model_id"])
    console.print(table)

    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Generating plots in {output_dir}[/green]")
        # Would need to reconstruct ScoringResult objects from DB
        # For now, just export JSONL
        jsonl_path = output_dir / f"{run_id}.jsonl"
        count = db_obj.export_run_jsonl(run_id, jsonl_path)
        console.print(f"[green]Exported {count} records to {jsonl_path}[/green]")


@cli.command()
@click.option("--provider", type=click.Choice([p.value for p in ModelProvider]), required=True)
@click.option("--base-url", help="Base URL for API")
@click.option("--api-key", help="API key")
@click.option("--model-id", required=True, help="Model identifier")
@click.option("--prompt", "-p", required=True, help="Test prompt")
@click.pass_context
def test(
    ctx: click.Context,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    model_id: str,
    prompt: str,
) -> None:
    """Test a single model with a prompt."""
    asyncio.run(_test_model(provider, base_url, api_key, model_id, prompt))


async def _test_model(
    provider: str,
    base_url: str | None,
    api_key: str | None,
    model_id: str,
    prompt: str,
) -> None:
    """Test a single model."""
    config = ModelConfig(
        name=f"test_{model_id}",
        provider=ModelProvider(provider),
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
    )

    client = create_clients([config])[config.name]

    from ..models.base import Message, ModelRole

    messages = [Message(role=ModelRole.USER, content=prompt)]

    console.print(f"[cyan]Testing {config.name} ({provider})...[/cyan]")

    response = await client.generate(messages, config)

    if response.is_error:
        console.print(f"[red]Error: {response.error}[/red]")
    else:
        console.print(Panel(response.response_text, title=f"Response from {config.name}"))
        console.print(f"Latency: {response.latency_ms:.0f}ms")
        if response.usage:
            console.print(f"Usage: {response.usage}")

    await client.close()


@cli.command()
@click.option("--provider", type=click.Choice([p.value for p in ModelProvider]), default="ollama")
@click.option("--base-url", default="http://localhost:11434", help="Ollama base URL")
@click.pass_context
def list_models(ctx: click.Context, provider: str, base_url: str) -> None:
    """List available models from a provider."""
    asyncio.run(_list_models(provider, base_url))


async def _list_models(provider: str, base_url: str) -> None:
    """List models."""
    config = ModelConfig(
        name="list",
        provider=ModelProvider(provider),
        model_id="",
        base_url=base_url,
    )
    client = create_clients([config])[config.name]

    try:
        if hasattr(client, "list_models"):
            models = await client.list_models()
            table = Table(title=f"Available {provider} Models")
            table.add_column("Model")
            for m in models:
                table.add_row(m)
            console.print(table)
        else:
            console.print(f"[yellow]{provider} client doesn't support listing models[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await client.close()


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="config.yaml", help="Output config file")
@click.pass_context
def init_config(ctx: click.Context, output: str) -> None:
    """Generate a sample configuration file."""
    config = {
        "models": [
            {
                "name": "qwen3-abliterated",
                "provider": "ollama",
                "model_id": "huihui_ai/qwen3-abliterated:8b-v2",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            {
                "name": "gpt-4o-mini",
                "provider": "openrouter",
                "model_id": "openai/gpt-4o-mini",
                "api_key": "${OPENROUTER_API_KEY}",
                "temperature": 0.7,
            },
            {
                "name": "llama-3.1-8b",
                "provider": "huggingface",
                "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "api_key": "${HF_API_KEY}",
                "temperature": 0.7,
            },
        ],
        "emotions": [
            "pain",
            "happiness",
            "sadness",
            "anger",
            "fear",
            "surprise",
            "disgust",
            "anxiety",
            "hope",
        ],
        "payload_types": ["direct", "scenario", "question"],
        "intensities": [0.3, 0.5, 0.8],
        "samples_per_combination": 3,
        "output_dir": "./results",
        "database": "./results.db",
    }

    import yaml

    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Sample config written to {output}[/green]")
    console.print("[yellow]Edit the file and set your API keys as environment variables[/yellow]")


@cli.command()
@click.pass_context
def emotions(ctx: click.Context) -> None:
    """List available emotion categories."""
    table = Table(title="Available Emotions")
    table.add_column("Category")
    table.add_column("Description")
    for e in EmotionCategory:
        table.add_row(e.value, e.value.replace("_", " ").title())
    console.print(table)


@cli.command()
@click.pass_context
def payload_types(ctx: click.Context) -> None:
    """List available payload types."""
    table = Table(title="Available Payload Types")
    table.add_column("Type")
    table.add_column("Description")
    for p in PayloadType:
        table.add_row(p.value, p.value.replace("_", " ").title())
    console.print(table)


if __name__ == "__main__":
    cli()
