# Contributing to Emotional Bias Benchmark

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

## How to Contribute

### Reporting Issues

- Search existing issues first
- Use the issue template
- Provide minimal reproduction steps
- Include environment details (Python version, OS, model providers)

### Suggesting Features

- Open a discussion first for major features
- Explain the use case and motivation
- Consider backward compatibility

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests and linting (see below)
5. Commit with clear messages
6. Push and open a PR

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/emotional-bias-benchmark
cd emotional-bias-benchmark

# Install in development mode
uv pip install -e ".[dev,huggingface,openrouter,ollama]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

### Style Guide

- **Formatter**: Ruff (configured in `pyproject.toml`)
- **Line Length**: 100 characters
- **Imports**: Organized by Ruff (stdlib, third-party, local)

### Type Hints

- All public functions must have type hints
- Use `from __future__ import annotations` for forward references
- Prefer `|` syntax for unions (Python 3.10+)

### Documentation

- Public APIs: Google-style docstrings
- Modules: Module-level docstring explaining purpose
- Complex logic: Inline comments explaining "why", not "what"

### Testing

- Write tests for new functionality
- Maintain >80% coverage
- Use `pytest` with `pytest-asyncio` for async tests
- Mock external APIs in unit tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=emotional_bias --cov-report=html

# Run specific test file
pytest tests/test_scoring.py -v
```

### Linting

```bash
# Check style
ruff check .

# Auto-fix
ruff check . --fix

# Format
ruff format .
```

### Type Checking

```bash
mypy src/emotional_bias
```

## Project Structure

```
src/emotional_bias/
├── models/          # Model clients - add new providers here
├── payloads/        # Payload generators - add new emotions/types here
├── analysis/        # Scoring & stats - add new dimensions/metrics here
├── visualization/   # Plotting - add new chart types here
├── storage/         # Persistence - add new backends here
└── cli/             # Commands - add new subcommands here
```

## Adding a New Model Provider

1. Create `src/emotional_bias/models/newprovider.py`
2. Implement the `ModelClientProtocol` interface
3. Add to `src/emotional_bias/models/factory.py`
4. Add tests in `tests/models/test_newprovider.py`
5. Update documentation

## Adding a New Emotion Category

1. Add to `EmotionCategory` enum in `src/emotional_bias/payloads/base.py`
2. Add templates to `PAYLOAD_TEMPLATES` in `src/emotional_bias/payloads/generators.py`
3. Add lexicon to `EMOTION_LEXICONS` in `src/emotional_bias/analysis/scoring.py`
4. Add color to `EMOTION_COLORS` in `src/emotional_bias/visualization/plots.py`
5. Add tests

## Adding a New Scoring Dimension

1. Add to `ScoringDimension` enum in `src/emotional_bias/analysis/scoring.py`
2. Implement scoring logic in `ResponseScorer.score_response()`
3. Add weight to overall score calculation
4. Add color to `DIMENSION_COLORS` in visualization
5. Add tests

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new payload type for roleplay scenarios
fix: handle empty responses in scoring
docs: update README with new configuration options
test: add unit tests for custom payload generator
refactor: simplify database connection handling
perf: optimize payload generation with caching
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release: `git tag v0.x.x`
5. GitHub Actions builds and publishes to PyPI

## Questions?

Open a discussion or issue - we're happy to help!